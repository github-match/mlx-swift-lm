// Copyright © 2026 Apple Inc.

import XCTest

import MLXLMCommon
@testable import MLXVLM
import Tokenizers

private struct FixedTokenTokenizer: Tokenizer {
    private let tokenToId: [String: Int]

    init(tokenToId: [String: Int]) {
        self.tokenToId = tokenToId
    }

    func tokenize(text: String) -> [String] { [] }

    func encode(text: String) -> [Int] { [] }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] { [] }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String { "" }

    func convertTokenToId(_ token: String) -> Int? { tokenToId[token] }

    func convertIdToToken(_ id: Int) -> String? { nil }

    var bosToken: String? = nil
    var bosTokenId: Int? = nil
    var eosToken: String? = nil
    var eosTokenId: Int? = nil
    var unknownToken: String? = nil
    var unknownTokenId: Int? = nil

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] { [] }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws
        -> [Int]
    {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        []
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        []
    }
}

final class QwenVLTokenReplacementTests: XCTestCase {
    private let imagePad = "<|image_pad|>"
    private let imagePadId = 151_655
    private let visionStart = "<|vision_start|>"
    private let visionStartId = 151_652
    private let visionEnd = "<|vision_end|>"
    private let visionEndId = 151_653

    func testReplacePaddingTokensExpandsSinglePlaceholder() throws {
        let tokenizer = FixedTokenTokenizer(tokenToId: [imagePad: imagePadId])

        let promptTokens = [101, imagePadId, 102]
        let frames = [THW(1, 4, 4)]  // 16 / (2*2) = 4 image tokens

        let updated = try QwenVL.replacePaddingTokens(
            in: promptTokens,
            frames: frames,
            paddingToken: imagePad,
            mergeSize: 2,
            tokenizer: tokenizer
        )

        XCTAssertEqual(updated, [101, imagePadId, imagePadId, imagePadId, imagePadId, 102])
    }

    func testReplacePaddingTokensThrowsOnPlaceholderFrameMismatch() {
        let tokenizer = FixedTokenTokenizer(tokenToId: [imagePad: imagePadId])

        XCTAssertThrowsError(
            try QwenVL.replacePaddingTokens(
                in: [101, 102],
                frames: [THW(1, 4, 4)],
                paddingToken: imagePad,
                mergeSize: 2,
                tokenizer: tokenizer
            )
        )
    }

    func testReplacePaddingTokensThrowsOnNonDivisibleFrameGrid() {
        let tokenizer = FixedTokenTokenizer(tokenToId: [imagePad: imagePadId])

        XCTAssertThrowsError(
            try QwenVL.replacePaddingTokens(
                in: [101, imagePadId, 102],
                frames: [THW(1, 3, 3)],  // 9 is not divisible by 4
                paddingToken: imagePad,
                mergeSize: 2,
                tokenizer: tokenizer
            )
        )
    }

    func testReplacePaddingTokensScopesToVisionMarkersWhenPresent() throws {
        let tokenizer = FixedTokenTokenizer(
            tokenToId: [
                imagePad: imagePadId,
                visionStart: visionStartId,
                visionEnd: visionEndId,
            ]
        )

        let promptTokens = [101, imagePadId, visionStartId, imagePadId, visionEndId, 102]
        let frames = [THW(1, 4, 4)]  // 16 / (2*2) = 4 placeholder tokens

        let updated = try QwenVL.replacePaddingTokens(
            in: promptTokens,
            frames: frames,
            paddingToken: imagePad,
            mergeSize: 2,
            tokenizer: tokenizer
        )

        XCTAssertEqual(
            updated,
            [101, imagePadId, visionStartId, imagePadId, imagePadId, imagePadId, imagePadId, visionEndId, 102]
        )
    }
}
