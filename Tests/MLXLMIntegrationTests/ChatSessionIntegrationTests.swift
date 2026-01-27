// Copyright © 2025 Apple Inc.

import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLLM
import MLXLMCommon
import MLXNN
import MLXOptimizers
import MLXVLM
import Tokenizers
import XCTest

/// Tests for the streamlined API using real models
public class ChatSessionIntegrationTests: XCTestCase {

    static let llmModelId = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
    static let vlmModelId = "mlx-community/Qwen3-VL-4B-Instruct-4bit"
    nonisolated(unsafe) static var llmContainer: ModelContainer!
    nonisolated(unsafe) static var vlmContainer: ModelContainer!

    override public class func setUp() {
        super.setUp()
        // Load models once for all tests
        let llmExpectation = XCTestExpectation(description: "Load LLM")
        let vlmExpectation = XCTestExpectation(description: "Load VLM")

        Task {
            llmContainer = try await LLMModelFactory.shared.loadContainer(
                configuration: .init(id: llmModelId)
            )
            llmExpectation.fulfill()
        }

        Task {
            vlmContainer = try await VLMModelFactory.shared.loadContainer(
                configuration: .init(id: vlmModelId)
            )
            vlmExpectation.fulfill()
        }

        _ = XCTWaiter.wait(for: [llmExpectation, vlmExpectation], timeout: 300)
    }

    func testOneShot() async throws {
        let session = ChatSession(Self.llmContainer)
        let result = try await session.respond(to: "What is 2+2? Reply with just the number.")
        print("One-shot result:", result)
        XCTAssertTrue(result.contains("4") || result.lowercased().contains("four"))
    }

    func testOneShotStream() async throws {
        let session = ChatSession(Self.llmContainer)
        var result = ""
        for try await token in session.streamResponse(
            to: "What is 2+2? Reply with just the number.")
        {
            print(token, terminator: "")
            result += token
        }
        print()  // newline
        XCTAssertTrue(result.contains("4") || result.lowercased().contains("four"))
    }

    func testMultiTurnConversation() async throws {
        let session = ChatSession(
            Self.llmContainer, instructions: "You are a helpful assistant. Keep responses brief.")

        let response1 = try await session.respond(to: "My name is Alice.")
        print("Response 1:", response1)

        let response2 = try await session.respond(to: "What is my name?")
        print("Response 2:", response2)

        // If multi-turn works, response2 should mention "Alice"
        XCTAssertTrue(
            response2.lowercased().contains("alice"),
            "Model should remember the name 'Alice' from previous turn")
    }

    func testVisionModel() async throws {
        let session = ChatSession(Self.vlmContainer)

        // Create a simple red image for testing
        let redImage = CIImage(color: .red).cropped(to: CGRect(x: 0, y: 0, width: 100, height: 100))

        let result = try await session.respond(
            to: "What color is this image? Reply with just the color name.",
            image: .ciImage(redImage))
        print("Vision result:", result)
        XCTAssertTrue(result.lowercased().contains("red"))
    }

    func testPromptRehydration() async throws {
        // Simulate a persisted history (e.g. loaded from SwiftData)
        let history: [Chat.Message] = [
            .system("You are a helpful assistant."),
            .user("My name is Bob."),
            .assistant("Hello Bob! How can I help you today?"),
        ]

        let session = ChatSession(Self.llmContainer, history: history)

        // Ask a question that requires the context
        let response = try await session.respond(to: "What is my name?")

        print("Rehydration result:", response)

        XCTAssertTrue(
            response.lowercased().contains("bob"),
            "Model should recognize the name 'Bob' from the injected history, proving successful prompt re-hydration."
        )
    }
}

public class Molmo2IntegrationTests: XCTestCase {
    static let molmo2ModelId = "mlx-community/Molmo2-8B-4bit"
    static let molmo2SnapshotEnvKey = "MOLMO2_SNAPSHOT_PATH"
    nonisolated(unsafe) static var molmo2Container: ModelContainer?

    private static func molmo2SnapshotURL() -> URL? {
        let environment = ProcessInfo.processInfo.environment
        if let override = environment[molmo2SnapshotEnvKey], !override.isEmpty {
            return URL(fileURLWithPath: override, isDirectory: true)
        }
        let repoPath = "models--" + molmo2ModelId.replacingOccurrences(of: "/", with: "--")
        let snapshotsRoot = FileManager.default.homeDirectoryForCurrentUser
            .appendingPathComponent(".cache/huggingface/hub")
            .appendingPathComponent(repoPath)
            .appendingPathComponent("snapshots")
        guard let candidates = try? FileManager.default.contentsOfDirectory(
            at: snapshotsRoot,
            includingPropertiesForKeys: [.isDirectoryKey],
            options: [.skipsHiddenFiles]
        ) else {
            return nil
        }
        let directories = candidates.filter {
            (try? $0.resourceValues(forKeys: [.isDirectoryKey]).isDirectory) ?? false
        }
        return directories.sorted { $0.lastPathComponent < $1.lastPathComponent }.last
    }

    private static func molmo2Configuration() -> ModelConfiguration? {
        guard let snapshotURL = molmo2SnapshotURL() else {
            return nil
        }
        return ModelConfiguration(
            directory: snapshotURL,
            defaultPrompt: "Describe the image in English"
        )
    }

    private func loadMolmo2Container() async throws -> ModelContainer {
        if let container = Self.molmo2Container {
            return container
        }
        guard let configuration = Self.molmo2Configuration() else {
            throw XCTSkip(
                "Molmo2 snapshot not found. Set MOLMO2_SNAPSHOT_PATH to a local snapshot directory."
            )
        }
        let container = try await VLMModelFactory.shared.loadContainer(configuration: configuration)
        Self.molmo2Container = container
        return container
    }

    private func makeSolidImage(color: CIColor) -> CIImage {
        CIImage(color: color).cropped(to: CGRect(x: 0, y: 0, width: 128, height: 128))
    }

    func testMolmo2ImageSmoke() async throws {
        let container = try await loadMolmo2Container()
        let session = ChatSession(
            container,
            generateParameters: .init(maxTokens: 32, temperature: 0),
            processing: .init()
        )
        let image = UserInput.Image.ciImage(
            makeSolidImage(color: CIColor(red: 1, green: 0, blue: 0))
        )
        let result = try await session.respond(
            to: "Describe the image.",
            image: image
        )
        XCTAssertFalse(result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }

    func testMolmo2VideoSmoke() async throws {
        let container = try await loadMolmo2Container()
        let session = ChatSession(
            container,
            generateParameters: .init(maxTokens: 32, temperature: 0),
            processing: .init()
        )
        let frame = UserInput.VideoFrame(
            frame: makeSolidImage(color: CIColor(red: 1, green: 0, blue: 0)),
            timeStamp: CMTime(seconds: 0, preferredTimescale: 30)
        )
        let result = try await session.respond(
            to: "Describe the video.",
            video: .frames([frame])
        )
        XCTAssertFalse(result.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
    }
}
