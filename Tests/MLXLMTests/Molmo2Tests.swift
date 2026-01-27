import AVFoundation
import CoreImage
import MLX
import MLXLMCommon
import MLXVLM
import Tokenizers
import XCTest

private enum Molmo2TestTokenIds {
    static let bos = 9000
    static let imagePlaceholder = 9001
    static let videoPlaceholder = 9002
    static let imagePatch = 9003
    static let imageStart = 9004
    static let lowResImageStart = 9005
    static let imageEnd = 9006
    static let imageCol = 9007
    static let frameStart = 9008
    static let frameEnd = 9009
    static let text = 9010
}

private struct Molmo2TestTokenizer: Tokenizer {
    let vocabulary: [Int: String]
    let tokenMap: [String: Int]
    let templateTokens: [Int]

    let bosTokenId: Int? = Molmo2TestTokenIds.bos
    let eosTokenId: Int? = Molmo2TestTokenIds.bos
    let bosToken: String? = "<bos>"
    let eosToken: String? = "<eos>"
    let unknownToken: String? = "<unk>"
    let unknownTokenId: Int? = Molmo2TestTokenIds.text
    let fuseUnknownTokens: Bool = false

    init(templateTokens: [Int]) {
        let tokens: [String: Int] = [
            "<|image|>": Molmo2TestTokenIds.imagePlaceholder,
            "<|video|>": Molmo2TestTokenIds.videoPlaceholder,
            "<im_patch>": Molmo2TestTokenIds.imagePatch,
            "<im_start>": Molmo2TestTokenIds.imageStart,
            "<low_res_im_start>": Molmo2TestTokenIds.lowResImageStart,
            "<im_end>": Molmo2TestTokenIds.imageEnd,
            "<im_col>": Molmo2TestTokenIds.imageCol,
            "<frame_start>": Molmo2TestTokenIds.frameStart,
            "<frame_end>": Molmo2TestTokenIds.frameEnd,
            "<bos>": Molmo2TestTokenIds.bos,
            "<text>": Molmo2TestTokenIds.text,
        ]
        self.tokenMap = tokens
        self.vocabulary = Dictionary(uniqueKeysWithValues: tokens.map { ($0.value, $0.key) })
        self.templateTokens = templateTokens
    }

    func tokenize(text: String) -> [String] {
        text.split(separator: " ").map(String.init)
    }

    func encode(text: String) -> [Int] {
        if let id = tokenMap[text] {
            return [id]
        }
        return [Molmo2TestTokenIds.text]
    }

    func encode(text: String, addSpecialTokens: Bool) -> [Int] {
        encode(text: text)
    }

    func decode(tokens: [Int], skipSpecialTokens: Bool) -> String {
        tokens.map { convertIdToToken($0) ?? "<unk>" }.joined(separator: " ")
    }

    func convertTokenToId(_ token: String) -> Int? {
        tokenMap[token]
    }

    func convertIdToToken(_ id: Int) -> String? {
        vocabulary[id]
    }

    func applyChatTemplate(messages: [Tokenizers.Message]) throws -> [Int] {
        templateTokens
    }

    func applyChatTemplate(messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?) throws -> [Int] {
        templateTokens
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        templateTokens
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument
    ) throws -> [Int] {
        templateTokens
    }

    func applyChatTemplate(messages: [Tokenizers.Message], chatTemplate: String) throws -> [Int] {
        templateTokens
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?
    ) throws -> [Int] {
        templateTokens
    }

    func applyChatTemplate(
        messages: [Tokenizers.Message], chatTemplate: Tokenizers.ChatTemplateArgument?,
        addGenerationPrompt: Bool, truncation: Bool, maxLength: Int?, tools: [Tokenizers.ToolSpec]?,
        additionalContext: [String: any Sendable]?
    ) throws -> [Int] {
        templateTokens
    }
}

final class Molmo2Tests: XCTestCase {
    private func makeProcessorConfiguration() throws -> Molmo2ProcessorConfiguration {
        try JSONDecoder().decode(Molmo2ProcessorConfiguration.self, from: Data("{}".utf8))
    }

    private func makeSolidImage() -> CIImage {
        CIImage(color: CIColor(red: 0, green: 0, blue: 0))
            .cropped(to: CGRect(x: 0, y: 0, width: 10, height: 10))
    }

    private func countTokens(_ tokens: [Int], tokenId: Int) -> Int {
        tokens.filter { $0 == tokenId }.count
    }

    func testMolmo2ImageTokenExpansion() async throws {
        let tokenizer = Molmo2TestTokenizer(templateTokens: [Molmo2TestTokenIds.imagePlaceholder])
        let processor = Molmo2Processor(try makeProcessorConfiguration(), tokenizer: tokenizer)

        let input = UserInput(chat: [.user("describe", images: [.ciImage(makeSolidImage())])])
        let output = try await processor.prepare(input: input)

        let processedImage = try XCTUnwrap(output.image)
        let gridValues = try XCTUnwrap(processedImage.grids)
            .asArray(Int32.self)
            .map(Int.init)
        XCTAssertEqual(gridValues.count, 4)

        let expectedPatchTokens = gridValues[0] * gridValues[1] + gridValues[2] * gridValues[3]

        let tokens = output.text.tokens.asArray(Int.self)
        XCTAssertFalse(tokens.contains(Molmo2TestTokenIds.imagePlaceholder))
        XCTAssertEqual(countTokens(tokens, tokenId: Molmo2TestTokenIds.imagePatch), expectedPatchTokens)

        let tokenPooling = try XCTUnwrap(processedImage.tokenPooling)
        XCTAssertEqual(tokenPooling.dim(0), expectedPatchTokens)
        XCTAssertEqual(tokenPooling.dim(1), 4)
        XCTAssertEqual(processedImage.grids?.dim(0), 1)
        XCTAssertEqual(processedImage.grids?.dim(1), 4)
        XCTAssertEqual(processedImage.numCrops?.dim(0), 1)
    }

    func testMolmo2VideoTokenExpansion() async throws {
        let tokenizer = Molmo2TestTokenizer(templateTokens: [Molmo2TestTokenIds.videoPlaceholder])
        let processor = Molmo2Processor(try makeProcessorConfiguration(), tokenizer: tokenizer)

        let frame = UserInput.VideoFrame(
            frame: makeSolidImage(),
            timeStamp: CMTime(seconds: 0, preferredTimescale: 30))
        let input = UserInput(chat: [.user("describe", videos: [.frames([frame])])])
        let output = try await processor.prepare(input: input)

        let processedVideo = try XCTUnwrap(output.video)
        let gridValues = try XCTUnwrap(processedVideo.grids)
            .asArray(Int32.self)
            .map(Int.init)
        XCTAssertEqual(gridValues.count, 3)

        let frameCount = gridValues[0]
        let pooledH = gridValues[1]
        let pooledW = gridValues[2]
        let expectedPatchTokens = frameCount * pooledH * pooledW

        let tokens = output.text.tokens.asArray(Int.self)
        XCTAssertFalse(tokens.contains(Molmo2TestTokenIds.videoPlaceholder))
        XCTAssertEqual(countTokens(tokens, tokenId: Molmo2TestTokenIds.imagePatch), expectedPatchTokens)
        XCTAssertEqual(countTokens(tokens, tokenId: Molmo2TestTokenIds.frameStart), frameCount)
        XCTAssertEqual(countTokens(tokens, tokenId: Molmo2TestTokenIds.frameEnd), frameCount)

        let tokenPooling = try XCTUnwrap(processedVideo.tokenPooling)
        XCTAssertEqual(tokenPooling.dim(0), expectedPatchTokens)
        XCTAssertEqual(tokenPooling.dim(1), 9)
        XCTAssertEqual(processedVideo.grids?.dim(0), 1)
        XCTAssertEqual(processedVideo.grids?.dim(1), 3)
    }

    func testMolmo2VitLayerClamp() throws {
        let json = """
        {
          "adapter_config": {
            "head_dim": 64,
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_attention_heads": 8,
            "num_key_value_heads": 8,
            "pooling_attention_mask": true,
            "text_hidden_size": 128,
            "vit_layers": [0],
            "float32_attention": true
          },
          "text_config": {
            "additional_vocab_size": 0,
            "attention_dropout": 0.0,
            "embedding_dropout": 0.0,
            "head_dim": 64,
            "hidden_act": "silu",
            "hidden_size": 128,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-6,
            "max_position_embeddings": 2048,
            "model_type": "molmo2_text",
            "num_attention_heads": 8,
            "num_hidden_layers": 2,
            "num_key_value_heads": 8,
            "qkv_bias": false,
            "rope_theta": 1000000,
            "use_cache": true,
            "use_qk_norm": true,
            "vocab_size": 32000
          },
          "vit_config": {
            "head_dim": 64,
            "hidden_act": "gelu_pytorch_tanh",
            "hidden_size": 128,
            "image_default_input_size": [378, 378],
            "image_num_pos": 729,
            "image_patch_size": 14,
            "intermediate_size": 256,
            "layer_norm_eps": 1e-6,
            "num_attention_heads": 8,
            "num_hidden_layers": 27,
            "num_key_value_heads": 8,
            "float32_attention": true
          },
          "image_patch_id": 151938,
          "image_end_token_id": 151937,
          "frame_end_token_id": 151944,
          "frame_start_token_id": 151943,
          "image_col_id": 151939,
          "image_start_token_id": 151936,
          "low_res_image_start_token_id": 151940,
          "model_type": "molmo2",
          "tie_word_embeddings": false,
          "use_frame_special_tokens": true
        }
        """

        let configuration = try JSONDecoder().decode(
            Molmo2Configuration.self,
            from: Data(json.utf8))

        XCTAssertEqual(configuration.vitConfig.numHiddenLayers, 25)
    }
}
