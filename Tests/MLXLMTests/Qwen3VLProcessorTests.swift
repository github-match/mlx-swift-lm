// Copyright © 2026 Apple Inc.

import XCTest

import MLX
import MLXLMCommon
@testable import MLXVLM
import Tokenizers

private struct Qwen3VLProcessorTestTokenizer: Tokenizer {
    private let endOfTextTokenId = 900
    private let newlineTokenId = 901
    private let doubleNewlineTokenId = 902

    func tokenize(text: String) -> [String] { [] }

    func encode(text: String) -> [Int] { encode(text: text, addSpecialTokens: false) }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        switch text {
        case "<|endoftext|>":
            return [endOfTextTokenId]
        case "\n":
            return [newlineTokenId]
        case "\n\n":
            return [doubleNewlineTokenId]
        default:
            return [101, doubleNewlineTokenId]
        }
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? { nil }

    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? = nil
    var bosTokenId: Int? = nil
    var eosToken: String? = nil
    var eosTokenId: Int? = nil
    var unknownToken: String? = nil
    var unknownTokenId: Int? = nil

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        [101, doubleNewlineTokenId]
    }
}

final class Qwen3VLProcessorTests: XCTestCase {
    private func makeProcessor() throws -> Qwen3VLProcessor {
        let json = """
            {
              "image_mean": [0.5, 0.5, 0.5],
              "image_std": [0.5, 0.5, 0.5],
              "merge_size": 2,
              "patch_size": 14,
              "temporal_patch_size": 2,
              "image_processor_type": "qwen3_vl"
            }
            """
        let data = Data(json.utf8)
        let config = try JSONDecoder().decode(Qwen3VLProcessorConfiguration.self, from: data)
        return Qwen3VLProcessor(config, tokenizer: Qwen3VLProcessorTestTokenizer())
    }

    func testPrepareDoesNotAttachMaskWhenSpecialTokensDisabled() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "hello",
            additionalContext: [
                "add_generation_prompt": false,
                "add_special_tokens": false,
            ]
        )

        let prepared = try await processor.prepare(input: input)
        XCTAssertNil(prepared.text.mask)
        XCTAssertEqual(prepared.text.tokens.asArray(Int32.self), [101, 901])
    }

    func testPrepareAttachesMaskWhenSpecialTokensEnabled() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "hello",
            additionalContext: [
                "add_generation_prompt": false,
                "add_special_tokens": true,
            ]
        )

        let prepared = try await processor.prepare(input: input)
        XCTAssertNotNil(prepared.text.mask)
        XCTAssertEqual(prepared.text.tokens.asArray(Int32.self), [101, 901, 900])
    }

    func testPrepareCanDisableMaskWithExplicitIncludeHiddenStatesOverride() async throws {
        let processor = try makeProcessor()
        let input = UserInput(
            prompt: "hello",
            additionalContext: [
                "add_generation_prompt": false,
                "add_special_tokens": true,
                "include_hidden_states": false,
            ]
        )

        let prepared = try await processor.prepare(input: input)
        XCTAssertNil(prepared.text.mask)
        XCTAssertEqual(prepared.text.tokens.asArray(Int32.self), [101, 901, 900])
    }
}
