import AVFoundation
import CoreImage
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers

public struct Molmo2ProcessorConfiguration: Codable, Sendable {
    public struct Size: Codable, Sendable {
        public let height: Int
        public let width: Int
    }

    private struct ImageProcessor: Decodable, Sendable {
        let size: Size?
        let imageMean: [CGFloat]?
        let imageStd: [CGFloat]?
        let maxCrops: Int?
        let overlapMargins: [Int]?
        let patchSize: Int?
        let poolingSize: [Int]?
        let resample: Int?

        enum CodingKeys: String, CodingKey {
            case size
            case imageMean = "image_mean"
            case imageStd = "image_std"
            case maxCrops = "max_crops"
            case overlapMargins = "overlap_margins"
            case patchSize = "patch_size"
            case poolingSize = "pooling_size"
            case resample
        }
    }

    public let size: Size
    public let imageMean: [CGFloat]
    public let imageStd: [CGFloat]
    public let maxCrops: Int
    public let overlapMargins: [Int]
    public let patchSize: Int
    public let poolingSize: [Int]
    public let resample: Int?
    public let imageUseColTokens: Bool
    public let useSingleCropColTokens: Bool?
    public let useSingleCropStartToken: Bool
    public let videoUseColTokens: Bool
    public let useFrameSpecialTokens: Bool

    enum CodingKeys: String, CodingKey {
        case size
        case imageMean = "image_mean"
        case imageStd = "image_std"
        case maxCrops = "max_crops"
        case overlapMargins = "overlap_margins"
        case patchSize = "patch_size"
        case poolingSize = "pooling_size"
        case resample
        case imageUseColTokens = "image_use_col_tokens"
        case useSingleCropColTokens = "use_single_crop_col_tokens"
        case useSingleCropStartToken = "use_single_crop_start_token"
        case videoUseColTokens = "video_use_col_tokens"
        case useFrameSpecialTokens = "use_frame_special_tokens"
        case imageProcessor = "image_processor"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        let imageProcessor = try container.decodeIfPresent(ImageProcessor.self, forKey: .imageProcessor)

        let size = try imageProcessor?.size ?? container.decodeIfPresent(Size.self, forKey: .size)
        self.size = size ?? Size(height: 378, width: 378)

        self.imageMean = try imageProcessor?.imageMean
            ?? container.decodeIfPresent([CGFloat].self, forKey: .imageMean)
            ?? [0.5, 0.5, 0.5]
        self.imageStd = try imageProcessor?.imageStd
            ?? container.decodeIfPresent([CGFloat].self, forKey: .imageStd)
            ?? [0.5, 0.5, 0.5]
        self.maxCrops = try imageProcessor?.maxCrops
            ?? container.decodeIfPresent(Int.self, forKey: .maxCrops)
            ?? 8
        self.overlapMargins = try imageProcessor?.overlapMargins
            ?? container.decodeIfPresent([Int].self, forKey: .overlapMargins)
            ?? [4, 4]
        self.patchSize = try imageProcessor?.patchSize
            ?? container.decodeIfPresent(Int.self, forKey: .patchSize)
            ?? 14
        self.poolingSize = try imageProcessor?.poolingSize
            ?? container.decodeIfPresent([Int].self, forKey: .poolingSize)
            ?? [2, 2]
        self.resample = try imageProcessor?.resample ?? container.decodeIfPresent(Int.self, forKey: .resample)

        self.imageUseColTokens = try container.decodeIfPresent(Bool.self, forKey: .imageUseColTokens) ?? true
        self.useSingleCropColTokens = try container.decodeIfPresent(
            Bool.self, forKey: .useSingleCropColTokens)
        self.useSingleCropStartToken = try container.decodeIfPresent(
            Bool.self, forKey: .useSingleCropStartToken) ?? true
        self.videoUseColTokens = try container.decodeIfPresent(Bool.self, forKey: .videoUseColTokens) ?? false
        self.useFrameSpecialTokens = try container.decodeIfPresent(
            Bool.self, forKey: .useFrameSpecialTokens) ?? true
    }

    public func encode(to encoder: Encoder) throws {
        var container = encoder.container(keyedBy: CodingKeys.self)
        try container.encode(size, forKey: .size)
        try container.encode(imageMean, forKey: .imageMean)
        try container.encode(imageStd, forKey: .imageStd)
        try container.encode(maxCrops, forKey: .maxCrops)
        try container.encode(overlapMargins, forKey: .overlapMargins)
        try container.encode(patchSize, forKey: .patchSize)
        try container.encode(poolingSize, forKey: .poolingSize)
        try container.encodeIfPresent(resample, forKey: .resample)
        try container.encode(imageUseColTokens, forKey: .imageUseColTokens)
        try container.encodeIfPresent(useSingleCropColTokens, forKey: .useSingleCropColTokens)
        try container.encode(useSingleCropStartToken, forKey: .useSingleCropStartToken)
        try container.encode(videoUseColTokens, forKey: .videoUseColTokens)
        try container.encode(useFrameSpecialTokens, forKey: .useFrameSpecialTokens)
    }

    public var imageMeanTuple: (CGFloat, CGFloat, CGFloat) {
        let mean = imageMean.count >= 3 ? imageMean : [0.5, 0.5, 0.5]
        return (mean[0], mean[1], mean[2])
    }

    public var imageStdTuple: (CGFloat, CGFloat, CGFloat) {
        let std = imageStd.count >= 3 ? imageStd : [0.5, 0.5, 0.5]
        return (std[0], std[1], std[2])
    }

    public var imageSize: CGSize {
        CGSize(width: size.width, height: size.height)
    }

    public var imagePooling: (Int, Int) {
        let height = poolingSize.first ?? 2
        let width = poolingSize.count > 1 ? poolingSize[1] : height
        return (height, width)
    }
}

public struct Molmo2MessageGenerator: MessageGenerator {
    public init() {}

    public func generate(message: Chat.Message) -> MLXLMCommon.Message {
        let imageContent = message.images.map { _ in
            ["type": "image"]
        }
        let videoContent = message.videos.map { _ in
            ["type": "video"]
        }
        let textContent = [["type": "text", "text": message.content]]

        return [
            "role": message.role.rawValue,
            "content": imageContent + videoContent + textContent,
        ]
    }
}

public struct Molmo2Processor: UserInputProcessor {
    private let config: Molmo2ProcessorConfiguration
    private let tokenizer: any Tokenizer
    private let videoConfig = Molmo2VideoConfiguration()

    public init(_ config: Molmo2ProcessorConfiguration, tokenizer: any Tokenizer) {
        self.config = config
        self.tokenizer = tokenizer
    }

    public func prepare(input: UserInput) async throws -> LMInput {
        let messages = Molmo2MessageGenerator().generate(from: input)
        var promptTokens = try tokenizer.applyChatTemplate(messages: messages, tools: input.tools)

        if input.images.isEmpty && input.videos.isEmpty {
            promptTokens = insertBosIfNeeded(promptTokens)
            let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
            let mask = ones(like: promptArray).asType(.int8)
            return LMInput(text: .init(tokens: promptArray, mask: mask))
        }

        let tokenIds = try Molmo2TokenIds(tokenizer: tokenizer)

        var processedImage: LMInput.ProcessedImage?
        var imageTokens: [[Int]] = []
        var expectedImageTokenCount = 0

        if !input.images.isEmpty {
            let imageResult = try processImages(input.images, processing: input.processing, tokenIds: tokenIds)
            processedImage = imageResult.processedImage
            imageTokens = imageResult.tokenSequences
            expectedImageTokenCount = imageResult.expectedTokenCount

            if let tokenPooling = processedImage?.tokenPooling, tokenPooling.dim(0) != expectedImageTokenCount {
                throw VLMError.processing(
                    "Image pooled token count mismatch (expected \(expectedImageTokenCount), got \(tokenPooling.dim(0)))"
                )
            }
        }

        var processedVideo: LMInput.ProcessedVideo?
        var videoTokens: [[Int]] = []
        var expectedVideoTokenCount = 0

        if !input.videos.isEmpty {
            if input.videos.count > 1 {
                throw VLMError.singleVideoAllowed
            }
            let videoResult = try await processVideos(
                input.videos, processing: input.processing, tokenIds: tokenIds)
            processedVideo = videoResult.processedVideo
            videoTokens = videoResult.tokenSequences
            expectedVideoTokenCount = videoResult.expectedTokenCount

            if let tokenPooling = processedVideo?.tokenPooling, tokenPooling.dim(0) != expectedVideoTokenCount {
                throw VLMError.processing(
                    "Video pooled token count mismatch (expected \(expectedVideoTokenCount), got \(tokenPooling.dim(0)))"
                )
            }
        }

        promptTokens = try expandPromptTokens(
            promptTokens,
            imageTokens: imageTokens,
            videoTokens: videoTokens,
            tokenIds: tokenIds)

        promptTokens = insertBosIfNeeded(promptTokens)
        let promptArray = MLXArray(promptTokens).expandedDimensions(axis: 0)
        let mask = ones(like: promptArray).asType(.int8)
        return LMInput(
            text: .init(tokens: promptArray, mask: mask),
            image: processedImage,
            video: processedVideo)
    }
}

private struct Molmo2VideoConfiguration: Sendable {
    let size = CGSize(width: 378, height: 378)
    let patchSize = 14
    let poolingSize: (Int, Int) = (3, 3)
    let maxFps: Double = 2
    let numFrames = 384
    let frameSampleMode: FrameSampleMode = .uniformLastFrame
    let samplingFps: Double = 2

    enum FrameSampleMode {
        case uniformLastFrame
    }
}

private struct Molmo2TokenIds {
    let imagePlaceholder: Int
    let videoPlaceholder: Int
    let imagePatch: Int
    let imageStart: Int
    let lowResImageStart: Int
    let imageEnd: Int
    let imageCol: Int
    let frameStart: Int
    let frameEnd: Int

    init(tokenizer: Tokenizer) throws {
        imagePlaceholder = try Self.tokenId("<|image|>", tokenizer: tokenizer)
        videoPlaceholder = try Self.tokenId("<|video|>", tokenizer: tokenizer)
        imagePatch = try Self.tokenId("<im_patch>", tokenizer: tokenizer)
        imageStart = try Self.tokenId("<im_start>", tokenizer: tokenizer)
        lowResImageStart = try Self.tokenId("<low_res_im_start>", tokenizer: tokenizer)
        imageEnd = try Self.tokenId("<im_end>", tokenizer: tokenizer)
        imageCol = try Self.tokenId("<im_col>", tokenizer: tokenizer)
        frameStart = try Self.tokenId("<frame_start>", tokenizer: tokenizer)
        frameEnd = try Self.tokenId("<frame_end>", tokenizer: tokenizer)
    }

    private static func tokenId(_ token: String, tokenizer: Tokenizer) throws -> Int {
        if let id = tokenizer.convertTokenToId(token) {
            return id
        }
        let encoded = tokenizer.encode(text: token)
        guard encoded.count == 1, let id = encoded.first else {
            throw VLMError.processing("Token \(token) did not map to a single id")
        }
        return id
    }
}

private struct Molmo2ImageResult {
    let processedImage: LMInput.ProcessedImage
    let tokenSequences: [[Int]]
    let expectedTokenCount: Int
}

private struct Molmo2VideoResult {
    let processedVideo: LMInput.ProcessedVideo
    let tokenSequences: [[Int]]
    let expectedTokenCount: Int
}

private extension Molmo2Processor {
    func processImages(
        _ images: [UserInput.Image],
        processing: UserInput.Processing?,
        tokenIds: Molmo2TokenIds
    ) throws -> Molmo2ImageResult {
        var grids: [Int] = []
        var numCrops: [Int] = []
        var pooledIndices: [MLXArray] = []
        var cropBatches: [MLXArray] = []
        var tokenSequences: [[Int]] = []
        var expectedTokenCount = 0

        for image in images {
            let ciImage = try image.asCIImage()
            let processed = MediaProcessing.apply(ciImage, processing: processing)
            let result = try imageToPatchesAndGrids(processed)

            grids.append(contentsOf: result.grid)
            numCrops.append(result.numCrops)
            pooledIndices.append(result.tokenPooling)
            cropBatches.append(result.crops)

            let tokenCount = result.grid[0] * result.grid[1] + result.grid[2] * result.grid[3]
            expectedTokenCount += tokenCount
            tokenSequences.append(
                imageTokenSequence(grid: result.grid, tokenIds: tokenIds)
            )
        }

        let pixelValues = concatenated(cropBatches, axis: 0)
        let imageTokenPooling = concatenated(pooledIndices, axis: 0)
        let imageGrids = MLXArray(grids.map(Int32.init)).reshaped(images.count, 4)
        let imageNumCrops = MLXArray(numCrops.map(Int32.init))

        let processedImage = LMInput.ProcessedImage(
            pixels: pixelValues,
            tokenPooling: imageTokenPooling,
            grids: imageGrids,
            numCrops: imageNumCrops)

        return Molmo2ImageResult(
            processedImage: processedImage,
            tokenSequences: tokenSequences,
            expectedTokenCount: expectedTokenCount)
    }

    func processVideos(
        _ videos: [UserInput.Video],
        processing: UserInput.Processing?,
        tokenIds: Molmo2TokenIds
    ) async throws -> Molmo2VideoResult {
        var grids: [Int] = []
        var pooledIndices: [MLXArray] = []
        var cropBatches: [MLXArray] = []
        var tokenSequences: [[Int]] = []
        var expectedTokenCount = 0

        for video in videos {
            let framesResult = try await loadVideoFrames(video, processing: processing)
            let videoResult = try videoToPatchesAndGrids(framesResult.frames)

            grids.append(contentsOf: videoResult.grid)
            pooledIndices.append(videoResult.tokenPooling)
            cropBatches.append(videoResult.crops)

            let tokenCount = videoResult.grid[0] * videoResult.grid[1] * videoResult.grid[2]
            expectedTokenCount += tokenCount
            tokenSequences.append(
                try videoTokenSequence(
                    grid: videoResult.grid,
                    timestamps: framesResult.timestamps,
                    tokenIds: tokenIds))
        }

        let pixelValues = concatenated(cropBatches, axis: 0)
        let videoTokenPooling = concatenated(pooledIndices, axis: 0)
        let videoGrids = MLXArray(grids.map(Int32.init)).reshaped(videos.count, 3)

        let processedVideo = LMInput.ProcessedVideo(
            pixels: pixelValues,
            tokenPooling: videoTokenPooling,
            grids: videoGrids)

        return Molmo2VideoResult(
            processedVideo: processedVideo,
            tokenSequences: tokenSequences,
            expectedTokenCount: expectedTokenCount)
    }

    func expandPromptTokens(
        _ tokens: [Int],
        imageTokens: [[Int]],
        videoTokens: [[Int]],
        tokenIds: Molmo2TokenIds
    ) throws -> [Int] {
        var expanded: [Int] = []
        expanded.reserveCapacity(tokens.count)

        var imageIndex = 0
        var videoIndex = 0

        for token in tokens {
            if token == tokenIds.imagePlaceholder {
                guard imageIndex < imageTokens.count else {
                    throw VLMError.processing("Not enough image tokens for prompt placeholders")
                }
                expanded.append(contentsOf: imageTokens[imageIndex])
                imageIndex += 1
            } else if token == tokenIds.videoPlaceholder {
                guard videoIndex < videoTokens.count else {
                    throw VLMError.processing("Not enough video tokens for prompt placeholders")
                }
                expanded.append(contentsOf: videoTokens[videoIndex])
                videoIndex += 1
            } else {
                expanded.append(token)
            }
        }

        if imageIndex != imageTokens.count {
            throw VLMError.processing("Unused image tokens after prompt expansion")
        }
        if videoIndex != videoTokens.count {
            throw VLMError.processing("Unused video tokens after prompt expansion")
        }

        return expanded
    }

    func insertBosIfNeeded(_ tokens: [Int]) -> [Int] {
        guard let bosId = tokenizer.bosTokenId ?? tokenizer.eosTokenId else {
            return tokens
        }
        if tokens.first == bosId {
            return tokens
        }
        return [bosId] + tokens
    }

    func imageTokenSequence(grid: [Int], tokenIds: Molmo2TokenIds) -> [Int] {
        let loH = grid[0]
        let loW = grid[1]
        let hiH = grid[2]
        let hiW = grid[3]

        let useLowResCols = config.useSingleCropColTokens ?? config.imageUseColTokens
        let lowResRowLength = loW + (useLowResCols ? 1 : 0)
        let hiResRowLength = hiW + (config.imageUseColTokens ? 1 : 0)

        let lowResTokensCount = 1 + loH * lowResRowLength + 1
        let hiResTokensCount = 1 + hiH * hiResRowLength + 1

        var tokens: [Int] = []
        tokens.reserveCapacity(lowResTokensCount + hiResTokensCount)

        let lowResStart = config.useSingleCropStartToken ? tokenIds.lowResImageStart : tokenIds.imageStart
        tokens.append(lowResStart)
        for _ in 0 ..< loH {
            for _ in 0 ..< loW {
                tokens.append(tokenIds.imagePatch)
            }
            if useLowResCols {
                tokens.append(tokenIds.imageCol)
            }
        }
        tokens.append(tokenIds.imageEnd)

        tokens.append(tokenIds.imageStart)
        for _ in 0 ..< hiH {
            for _ in 0 ..< hiW {
                tokens.append(tokenIds.imagePatch)
            }
            if config.imageUseColTokens {
                tokens.append(tokenIds.imageCol)
            }
        }
        tokens.append(tokenIds.imageEnd)

        return tokens
    }

    func videoTokenSequence(
        grid: [Int],
        timestamps: [Double],
        tokenIds: Molmo2TokenIds
    ) throws -> [Int] {
        let frameCount = grid[0]
        let pooledH = grid[1]
        let pooledW = grid[2]

        if timestamps.count != frameCount {
            throw VLMError.processing(
                "Video timestamp count mismatch (expected \(frameCount), got \(timestamps.count))"
            )
        }

        let startToken = config.useFrameSpecialTokens ? tokenIds.frameStart : tokenIds.imageStart
        let endToken = config.useFrameSpecialTokens ? tokenIds.frameEnd : tokenIds.imageEnd
        let locale = Locale(identifier: "en_US_POSIX")

        var tokens: [Int] = []

        for (index, timestamp) in timestamps.enumerated() {
            let prefix = (index > 0 ? " " : "")
                + String(format: "%.1f ", locale: locale, timestamp)
            tokens.append(contentsOf: tokenizer.encode(text: prefix))

            tokens.append(startToken)
            for _ in 0 ..< pooledH {
                for _ in 0 ..< pooledW {
                    tokens.append(tokenIds.imagePatch)
                }
                if config.videoUseColTokens {
                    tokens.append(tokenIds.imageCol)
                }
            }
            tokens.append(endToken)
        }

        return tokens
    }

    func imageToPatchesAndGrids(_ image: CIImage) throws -> (grid: [Int], crops: MLXArray, tokenPooling: MLXArray, numCrops: Int) {
        let baseSize = config.imageSize
        let patchSize = config.patchSize
        let (poolH, poolW) = config.imagePooling

        let (highResCrops, patchIdxGrid) = buildOverlappingCrops(image, baseSize: baseSize)
        let pooledIdx = arangeForPooling(patchIdxGrid, poolH: poolH, poolW: poolW)
        let hiH = pooledIdx.dim(0)
        let hiW = pooledIdx.dim(1)
        let pooledFlat = pooledIdx.reshaped(hiH * hiW, poolH * poolW)

        let (lowResCrop, resizeIdx) = buildResizedImage(image, baseSize: baseSize)
        let resizePooling = arangeForPooling(resizeIdx, poolH: poolH, poolW: poolW)
        let loH = resizePooling.dim(0)
        let loW = resizePooling.dim(1)
        let resizeFlat = resizePooling.reshaped(loH * loW, poolH * poolW)

        let cropPatchH = Int(baseSize.height) / patchSize
        let cropPatchW = Int(baseSize.width) / patchSize
        let offsetValue = MLXArray(Int32(cropPatchH * cropPatchW))
        let pooledOffset = MLX.where(pooledFlat .>= 0, pooledFlat + offsetValue, pooledFlat)

        let tokenPooling = concatenated([resizeFlat, pooledOffset], axis: 0)
        let allCrops = concatenated([lowResCrop, highResCrops], axis: 0)
        let patches = patchify(allCrops, patchSize: patchSize)

        let grid = [loH, loW, hiH, hiW]
        let numCrops = allCrops.dim(0)

        return (grid, patches, tokenPooling, numCrops)
    }

    func videoToPatchesAndGrids(_ frames: [CIImage]) throws -> (grid: [Int], crops: MLXArray, tokenPooling: MLXArray) {
        guard !frames.isEmpty else {
            throw VLMError.processing("No video frames provided")
        }

        let baseSize = videoConfig.size
        let patchSize = videoConfig.patchSize
        let (poolH, poolW) = videoConfig.poolingSize

        var allCrops: [MLXArray] = []
        var pooledIndices: [MLXArray] = []
        var patchOffset = 0

        var pooledGrid: (Int, Int)?

        for frame in frames {
            let resized = frame
                .resampled(to: baseSize, method: .bicubic)
                .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
            let frameArray = resized.asMLXArray().transposed(0, 2, 3, 1)
            let crops = patchify(frameArray, patchSize: patchSize)

            let resizeIdx = makeSequentialIndexGrid(
                height: Int(baseSize.height) / patchSize,
                width: Int(baseSize.width) / patchSize)
            let pooled = arangeForPooling(resizeIdx, poolH: poolH, poolW: poolW)

            let pooledH = pooled.dim(0)
            let pooledW = pooled.dim(1)
            pooledGrid = pooledGrid ?? (pooledH, pooledW)

            let pooledFlat = pooled.reshaped(pooledH * pooledW, poolH * poolW)
            let offsetArray = MLXArray(Int32(patchOffset))
            let pooledOffset = MLX.where(pooledFlat .>= 0, pooledFlat + offsetArray, pooledFlat)

            pooledIndices.append(pooledOffset)
            allCrops.append(crops)
            patchOffset += crops.dim(0) * crops.dim(1)
        }

        let gridValues = pooledGrid ?? (0, 0)
        let videoGrid = [frames.count, gridValues.0, gridValues.1]
        let pixelValues = concatenated(allCrops, axis: 0)
        let tokenPooling = concatenated(pooledIndices, axis: 0)

        return (videoGrid, pixelValues, tokenPooling)
    }

    func buildResizedImage(
        _ image: CIImage,
        baseSize: CGSize
    ) -> (MLXArray, MLXArray) {
        let resized = image
            .toSRGB()
            .resampled(to: baseSize, method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)
        let array = resized.asMLXArray().transposed(0, 2, 3, 1)

        let cropPatchH = Int(baseSize.height) / config.patchSize
        let cropPatchW = Int(baseSize.width) / config.patchSize
        let resizeIdx = makeSequentialIndexGrid(height: cropPatchH, width: cropPatchW)

        return (array, resizeIdx)
    }

    func buildOverlappingCrops(
        _ image: CIImage,
        baseSize: CGSize
    ) -> (MLXArray, MLXArray) {
        let patchSize = config.patchSize
        let cropSize = Int(baseSize.height)
        let leftMargin = config.overlapMargins.first ?? 0
        let rightMargin = config.overlapMargins.count > 1 ? config.overlapMargins[1] : leftMargin

        let totalMarginPixels = patchSize * (leftMargin + rightMargin)
        let cropPatches = cropSize / patchSize
        let cropWindowPatches = cropPatches - (leftMargin + rightMargin)
        let cropWindowSize = cropWindowPatches * patchSize

        let extent = image.extent
        let originalHeight = Int(extent.height)
        let originalWidth = Int(extent.width)

        let tiling = selectTiling(
            height: originalHeight - totalMarginPixels,
            width: originalWidth - totalMarginPixels,
            patchSize: cropWindowSize,
            maxCrops: config.maxCrops)

        let resizedHeight = tiling.0 * cropWindowSize + totalMarginPixels
        let resizedWidth = tiling.1 * cropWindowSize + totalMarginPixels

        let resized = image
            .toSRGB()
            .resampled(to: CGSize(width: resizedWidth, height: resizedHeight), method: .bicubic)
            .normalized(mean: config.imageMeanTuple, std: config.imageStdTuple)

        let nCrops = tiling.0 * tiling.1
        var crops: [MLXArray] = []
        crops.reserveCapacity(nCrops)

        let cropPatchSize = cropPatches * cropPatches
        var patchIdx = Array(repeating: 0, count: nCrops * cropPatchSize)

        for row in 0 ..< tiling.0 {
            let y0 = row * cropWindowSize
            for col in 0 ..< tiling.1 {
                let x0 = col * cropWindowSize
                let crop = cropImage(resized, x: x0, y: y0, size: cropSize)
                let cropArray = crop.asMLXArray().transposed(0, 2, 3, 1)
                crops.append(cropArray)

                let cropIndex = row * tiling.1 + col
                let baseOffset = cropIndex * cropPatchSize
                for y in 0 ..< cropPatches {
                    for x in 0 ..< cropPatches {
                        var value = y * cropPatches + x + cropIndex * cropPatchSize
                        if row != 0, y < leftMargin {
                            value = -1
                        }
                        if col != 0, x < leftMargin {
                            value = -1
                        }
                        if row != tiling.0 - 1, y >= cropPatches - rightMargin {
                            value = -1
                        }
                        if col != tiling.1 - 1, x >= cropPatches - rightMargin {
                            value = -1
                        }
                        patchIdx[baseOffset + y * cropPatches + x] = value
                    }
                }
            }
        }

        let fullHeight = tiling.0 * cropWindowPatches + leftMargin + rightMargin
        let fullWidth = tiling.1 * cropWindowPatches + leftMargin + rightMargin
        var patchIdxFull = Array(repeating: 0, count: fullHeight * fullWidth)

        for row in 0 ..< tiling.0 {
            for col in 0 ..< tiling.1 {
                let cropIndex = row * tiling.1 + col
                let yStart = row * cropWindowPatches
                let xStart = col * cropWindowPatches
                let baseOffset = cropIndex * cropPatchSize

                for y in 0 ..< cropPatches {
                    for x in 0 ..< cropPatches {
                        let value = patchIdx[baseOffset + y * cropPatches + x]
                        if value >= 0 {
                            let targetIndex = (yStart + y) * fullWidth + (xStart + x)
                            patchIdxFull[targetIndex] = value
                        }
                    }
                }
            }
        }

        let cropsArray = concatenated(crops, axis: 0)
        let patchIdxArray = MLXArray(patchIdxFull.map(Int32.init)).reshaped(fullHeight, fullWidth)
        return (cropsArray, patchIdxArray)
    }

    func cropImage(_ image: CIImage, x: Int, y: Int, size: Int) -> CIImage {
        let extent = image.extent
        let cropY = extent.height - CGFloat(size) - CGFloat(y)
        let cropRect = CGRect(x: CGFloat(x), y: cropY, width: CGFloat(size), height: CGFloat(size))
        let cropped = image.cropped(to: cropRect)
        return cropped.transformed(
            by: CGAffineTransform(translationX: -cropRect.minX, y: -cropRect.minY))
    }

    func makeSequentialIndexGrid(height: Int, width: Int) -> MLXArray {
        MLXArray(0 ..< (height * width)).asType(.int32).reshaped(height, width)
    }

    func arangeForPooling(_ indexArray: MLXArray, poolH: Int, poolW: Int) -> MLXArray {
        let height = indexArray.dim(0)
        let width = indexArray.dim(1)

        let padH = poolH * ((height + poolH - 1) / poolH) - height
        let padW = poolW * ((width + poolW - 1) / poolW) - width

        let padTop = padH / 2
        let padBottom = (padH + 1) / 2
        let padLeft = padW / 2
        let padRight = (padW + 1) / 2

        let paddedArray = padded(
            indexArray,
            widths: [[padTop, padBottom], [padLeft, padRight]],
            mode: .constant,
            value: MLXArray(Int32(-1))
        )

        let paddedH = paddedArray.dim(0)
        let paddedW = paddedArray.dim(1)
        let outH = paddedH / poolH
        let outW = paddedW / poolW

        let reshaped = paddedArray.reshaped(outH, poolH, outW, poolW)
        let transposed = reshaped.transposed(0, 2, 1, 3)
        return transposed.reshaped(outH, outW, poolH * poolW)
    }

    func patchify(_ crops: MLXArray, patchSize: Int) -> MLXArray {
        let nCrops = crops.dim(0)
        let height = crops.dim(1)
        let width = crops.dim(2)
        let channels = crops.dim(3)

        let patchesH = height / patchSize
        let patchesW = width / patchSize
        let patchDim = patchSize * patchSize * channels

        var patches = crops.reshaped(
            nCrops,
            patchesH,
            patchSize,
            patchesW,
            patchSize,
            channels)
        patches = patches.transposed(0, 1, 3, 2, 4, 5)
        return patches.reshaped(nCrops, patchesH * patchesW, patchDim)
    }

    func selectTiling(height: Int, width: Int, patchSize: Int, maxCrops: Int) -> (Int, Int) {
        if height <= 0 || width <= 0 {
            return (1, 1)
        }

        var tilings: [(Int, Int)] = []
        for i in 1 ... maxCrops {
            for j in 1 ... maxCrops where i * j <= maxCrops {
                tilings.append((i, j))
            }
        }
        tilings.sort { lhs, rhs in
            let lhsArea = lhs.0 * lhs.1
            let rhsArea = rhs.0 * rhs.1
            if lhsArea == rhsArea {
                return lhs.0 < rhs.0
            }
            return lhsArea < rhsArea
        }

        let originalH = Double(height)
        let originalW = Double(width)
        let scales = tilings.map { tiling -> Double in
            let heightScale = Double(tiling.0 * patchSize) / originalH
            let widthScale = Double(tiling.1 * patchSize) / originalW
            return min(heightScale, widthScale)
        }

        if scales.allSatisfy({ $0 < 1 }) {
            let maxScale = scales.enumerated().max(by: { $0.element < $1.element })
            return tilings[maxScale?.offset ?? 0]
        }

        var bestIndex = 0
        var bestScale = Double.greatestFiniteMagnitude
        for (index, scale) in scales.enumerated() {
            let candidate = scale < 1 ? Double.greatestFiniteMagnitude : scale
            if candidate < bestScale {
                bestScale = candidate
                bestIndex = index
            }
        }

        return tilings[bestIndex]
    }

    func loadVideoFrames(
        _ video: UserInput.Video,
        processing: UserInput.Processing?
    ) async throws -> (frames: [CIImage], timestamps: [Double]) {
        switch video {
        case .frames(let frames):
            return sampleFrames(frames, processing: processing)
        case .avAsset(let asset):
            return try await decodeAsset(asset, processing: processing)
        case .url(let url):
            return try await decodeAsset(AVAsset(url: url), processing: processing)
        }
    }

    func sampleFrames(
        _ frames: [UserInput.VideoFrame],
        processing: UserInput.Processing?
    ) -> (frames: [CIImage], timestamps: [Double]) {
        guard let first = frames.first else {
            return ([], [])
        }

        let sortedFrames = frames.sorted { $0.timeStamp.seconds < $1.timeStamp.seconds }
        let start = sortedFrames.first?.timeStamp.seconds ?? 0
        let end = sortedFrames.last?.timeStamp.seconds ?? 0
        let duration = max(end - start, 0)

        let targetTimes = uniformLastFrameTimes(
            duration: duration,
            numFrames: videoConfig.numFrames,
            maxFps: videoConfig.maxFps)

        var sampled: [CIImage] = []
        var timestamps: [Double] = []
        var frameIndex = sortedFrames.startIndex

        for time in targetTimes {
            let target = CMTime(seconds: time, preferredTimescale: first.timeStamp.timescale)
            var selectedIndex: Int?

            while frameIndex < sortedFrames.endIndex {
                if sortedFrames[frameIndex].timeStamp.seconds > target.seconds {
                    break
                }
                selectedIndex = frameIndex
                frameIndex += 1
            }

            if let index = selectedIndex {
                let frame = sortedFrames[index]
                let processedFrame = MediaProcessing.apply(frame.frame, processing: processing).toSRGB()
                sampled.append(processedFrame)
                timestamps.append(time)
            }
        }

        if sampled.isEmpty {
            let processedFrame = MediaProcessing.apply(first.frame, processing: processing).toSRGB()
            sampled.append(processedFrame)
            timestamps.append(0)
        }

        return (sampled, timestamps)
    }

    func decodeAsset(
        _ asset: AVAsset,
        processing: UserInput.Processing?
    ) async throws -> (frames: [CIImage], timestamps: [Double]) {
        try await validateAsset(asset)
        let duration = try await asset.load(.duration)
        let targetTimes = uniformLastFrameTimes(
            duration: duration.seconds,
            numFrames: videoConfig.numFrames,
            maxFps: videoConfig.maxFps)

        let generator = AVAssetImageGenerator(asset: asset)
        generator.appliesPreferredTrackTransform = true
        generator.requestedTimeToleranceBefore = .zero
        generator.requestedTimeToleranceAfter = .zero

        let timescale = duration.timescale
        let sampledTimes = targetTimes.map { CMTime(seconds: $0, preferredTimescale: timescale) }

        var frames: [CIImage] = []
        var timestamps: [Double] = []

        for await result in generator.images(for: sampledTimes) {
            switch result {
            case .success(requestedTime: _, let image, actualTime: let actual):
                let ciImage = CIImage(
                    cgImage: image, options: [.colorSpace: CGColorSpace(name: CGColorSpace.sRGB)!])
                let processed = MediaProcessing.apply(ciImage, processing: processing).toSRGB()
                frames.append(processed)
                timestamps.append(actual.seconds)
            case .failure(requestedTime: _, _):
                break
            }
        }

        if frames.isEmpty {
            throw VLMError.videoNotDecodable
        }

        return (frames, timestamps)
    }

    func uniformLastFrameTimes(
        duration: Double,
        numFrames: Int,
        maxFps: Double?
    ) -> [Double] {
        guard numFrames > 1 else {
            return [0]
        }

        let durationValue = max(duration, 0)
        if let maxFps, maxFps > 0 {
            let maxDuration = Double(numFrames - 1) / maxFps
            if maxDuration < durationValue {
                return linspace(start: 0, end: durationValue, count: numFrames)
            }
            let step = 1.0 / maxFps
            var times = stride(from: 0.0, to: durationValue, by: step).map { $0 }
            if times.last != durationValue {
                times.append(durationValue)
            }
            if times.count > numFrames {
                times = Array(times.prefix(numFrames))
            }
            return times
        }

        return linspace(start: 0, end: durationValue, count: numFrames)
    }

    func linspace(start: Double, end: Double, count: Int) -> [Double] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Double(count - 1)
        return (0 ..< count).map { start + Double($0) * step }
    }

    func validateAsset(_ asset: AVAsset) async throws {
        let tracks = try await asset.loadTracks(withMediaType: .video)
        guard let videoTrack = tracks.first else {
            throw VLMError.noVideoTrackFound
        }
        let isDecodable = try await videoTrack.load(.isDecodable)
        if !isDecodable {
            throw VLMError.videoNotDecodable
        }
    }
}

// MARK: - Model

private enum Molmo2ModelError: Error {
    case featureTokenMismatch(expected: Int, actual: Int)
    case unsupportedMixedMedia
    case missingImageMetadata
    case missingVideoMetadata
}

public struct Molmo2Configuration: Codable, Sendable {
    public struct TextConfiguration: Codable, Sendable {
        public let additionalVocabSize: Int
        public let attentionDropout: Double
        public let embeddingDropout: Double
        public let headDim: Int
        public let hiddenAct: String
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let layerNormEps: Float
        public let maxPositionEmbeddings: Int
        public let modelType: String
        public let numAttentionHeads: Int
        public let numHiddenLayers: Int
        public let numKeyValueHeads: Int
        public let qkvBias: Bool
        public let ropeTheta: Float
        public let useCache: Bool
        public let useQkNorm: Bool
        public let vocabSize: Int

        enum CodingKeys: String, CodingKey {
            case additionalVocabSize = "additional_vocab_size"
            case attentionDropout = "attention_dropout"
            case embeddingDropout = "embedding_dropout"
            case headDim = "head_dim"
            case hiddenAct = "hidden_act"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case layerNormEps = "layer_norm_eps"
            case maxPositionEmbeddings = "max_position_embeddings"
            case modelType = "model_type"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case numKeyValueHeads = "num_key_value_heads"
            case qkvBias = "qkv_bias"
            case ropeTheta = "rope_theta"
            case useCache = "use_cache"
            case useQkNorm = "use_qk_norm"
            case vocabSize = "vocab_size"
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.additionalVocabSize = try container.decodeIfPresent(Int.self, forKey: .additionalVocabSize) ?? 0
            self.attentionDropout = try container.decodeIfPresent(Double.self, forKey: .attentionDropout) ?? 0
            self.embeddingDropout = try container.decodeIfPresent(Double.self, forKey: .embeddingDropout) ?? 0
            self.headDim = try container.decode(Int.self, forKey: .headDim)
            self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "silu"
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.layerNormEps = Float(try container.decodeIfPresent(Double.self, forKey: .layerNormEps) ?? 1e-6)
            self.maxPositionEmbeddings = try container.decodeIfPresent(Int.self, forKey: .maxPositionEmbeddings) ?? 0
            self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo2_text"
            self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
            self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
            self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
            self.qkvBias = try container.decodeIfPresent(Bool.self, forKey: .qkvBias) ?? false
            self.ropeTheta = try container.decodeIfPresent(Float.self, forKey: .ropeTheta) ?? 1_000_000
            self.useCache = try container.decodeIfPresent(Bool.self, forKey: .useCache) ?? true
            self.useQkNorm = try container.decodeIfPresent(Bool.self, forKey: .useQkNorm) ?? true
            self.vocabSize = try container.decode(Int.self, forKey: .vocabSize)
        }
    }

    public struct AdapterConfiguration: Codable, Sendable {
        public let headDim: Int
        public let hiddenSize: Int
        public let intermediateSize: Int
        public let numAttentionHeads: Int
        public let numKeyValueHeads: Int
        public let poolingAttentionMask: Bool
        public let textHiddenSize: Int
        public let vitLayers: [Int]
        public let float32Attention: Bool

        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case hiddenSize = "hidden_size"
            case intermediateSize = "intermediate_size"
            case numAttentionHeads = "num_attention_heads"
            case numKeyValueHeads = "num_key_value_heads"
            case poolingAttentionMask = "pooling_attention_mask"
            case textHiddenSize = "text_hidden_size"
            case vitLayers = "vit_layers"
            case float32Attention = "float32_attention"
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.headDim = try container.decode(Int.self, forKey: .headDim)
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
            self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
            self.poolingAttentionMask = try container.decodeIfPresent(Bool.self, forKey: .poolingAttentionMask) ?? true
            self.textHiddenSize = try container.decode(Int.self, forKey: .textHiddenSize)
            self.vitLayers = try container.decodeIfPresent([Int].self, forKey: .vitLayers) ?? []
            self.float32Attention = try container.decodeIfPresent(Bool.self, forKey: .float32Attention) ?? true
        }
    }

    public struct VitConfiguration: Codable, Sendable {
        public let headDim: Int
        public let hiddenAct: String
        public let hiddenSize: Int
        public let imageDefaultInputSize: [Int]
        public let imageNumPos: Int
        public let imagePatchSize: Int
        public let intermediateSize: Int
        public let layerNormEps: Float
        public let numAttentionHeads: Int
        public var numHiddenLayers: Int
        public let numKeyValueHeads: Int
        public let float32Attention: Bool

        enum CodingKeys: String, CodingKey {
            case headDim = "head_dim"
            case hiddenAct = "hidden_act"
            case hiddenSize = "hidden_size"
            case imageDefaultInputSize = "image_default_input_size"
            case imageNumPos = "image_num_pos"
            case imagePatchSize = "image_patch_size"
            case intermediateSize = "intermediate_size"
            case layerNormEps = "layer_norm_eps"
            case numAttentionHeads = "num_attention_heads"
            case numHiddenLayers = "num_hidden_layers"
            case numKeyValueHeads = "num_key_value_heads"
            case float32Attention = "float32_attention"
        }

        public init(from decoder: any Swift.Decoder) throws {
            let container = try decoder.container(keyedBy: CodingKeys.self)
            self.headDim = try container.decode(Int.self, forKey: .headDim)
            self.hiddenAct = try container.decodeIfPresent(String.self, forKey: .hiddenAct) ?? "gelu_pytorch_tanh"
            self.hiddenSize = try container.decode(Int.self, forKey: .hiddenSize)
            self.imageDefaultInputSize = try container.decodeIfPresent([Int].self, forKey: .imageDefaultInputSize) ?? [378, 378]
            self.imageNumPos = try container.decodeIfPresent(Int.self, forKey: .imageNumPos) ?? 729
            self.imagePatchSize = try container.decodeIfPresent(Int.self, forKey: .imagePatchSize) ?? 14
            self.intermediateSize = try container.decode(Int.self, forKey: .intermediateSize)
            self.layerNormEps = Float(try container.decodeIfPresent(Double.self, forKey: .layerNormEps) ?? 1e-6)
            self.numAttentionHeads = try container.decode(Int.self, forKey: .numAttentionHeads)
            self.numHiddenLayers = try container.decode(Int.self, forKey: .numHiddenLayers)
            self.numKeyValueHeads = try container.decode(Int.self, forKey: .numKeyValueHeads)
            self.float32Attention = try container.decodeIfPresent(Bool.self, forKey: .float32Attention) ?? true
        }

        public var imageNumPatch: (Int, Int) {
            let height = imageDefaultInputSize.first ?? 378
            let width = imageDefaultInputSize.count > 1 ? imageDefaultInputSize[1] : height
            return (height / imagePatchSize, width / imagePatchSize)
        }
    }

    public let adapterConfig: AdapterConfiguration
    public let textConfig: TextConfiguration
    public var vitConfig: VitConfiguration

    public let imagePatchId: Int
    public let imageEndTokenId: Int
    public let frameEndTokenId: Int

    public let frameStartTokenId: Int
    public let imageColId: Int
    public let imageStartTokenId: Int
    public let lowResImageStartTokenId: Int

    public let modelType: String
    public let tieWordEmbeddings: Bool
    public let useFrameSpecialTokens: Bool

    enum CodingKeys: String, CodingKey {
        case adapterConfig = "adapter_config"
        case textConfig = "text_config"
        case vitConfig = "vit_config"
        case imagePatchId = "image_patch_id"
        case imageEndTokenId = "image_end_token_id"
        case frameEndTokenId = "frame_end_token_id"
        case frameStartTokenId = "frame_start_token_id"
        case imageColId = "image_col_id"
        case imageStartTokenId = "image_start_token_id"
        case lowResImageStartTokenId = "low_res_image_start_token_id"
        case modelType = "model_type"
        case tieWordEmbeddings = "tie_word_embeddings"
        case useFrameSpecialTokens = "use_frame_special_tokens"
    }

    public init(from decoder: any Swift.Decoder) throws {
        let container = try decoder.container(keyedBy: CodingKeys.self)
        self.adapterConfig = try container.decode(AdapterConfiguration.self, forKey: .adapterConfig)
        self.textConfig = try container.decode(TextConfiguration.self, forKey: .textConfig)
        self.vitConfig = try container.decode(VitConfiguration.self, forKey: .vitConfig)

        self.imagePatchId = try container.decode(Int.self, forKey: .imagePatchId)
        self.imageEndTokenId = try container.decode(Int.self, forKey: .imageEndTokenId)
        self.frameEndTokenId = try container.decode(Int.self, forKey: .frameEndTokenId)
        self.frameStartTokenId = try container.decode(Int.self, forKey: .frameStartTokenId)
        self.imageColId = try container.decode(Int.self, forKey: .imageColId)
        self.imageStartTokenId = try container.decode(Int.self, forKey: .imageStartTokenId)
        self.lowResImageStartTokenId = try container.decode(Int.self, forKey: .lowResImageStartTokenId)

        self.modelType = try container.decodeIfPresent(String.self, forKey: .modelType) ?? "molmo2"
        self.tieWordEmbeddings = try container.decodeIfPresent(Bool.self, forKey: .tieWordEmbeddings) ?? false
        self.useFrameSpecialTokens = try container.decodeIfPresent(Bool.self, forKey: .useFrameSpecialTokens) ?? false

        // Snapshot weights contain 25 ViT blocks even if config.json reports more.
        self.vitConfig.numHiddenLayers = min(self.vitConfig.numHiddenLayers, 25)
    }
}

fileprivate final class Molmo2Embedding: Module {
    @ParameterInfo(key: "embedding") var embedding: MLXArray
    @ParameterInfo(key: "new_embedding") var newEmbedding: MLXArray

    init(numEmbeddings: Int, numNewEmbeddings: Int, features: Int) {
        self._embedding.wrappedValue = MLXArray.zeros([numEmbeddings, features])
        self._newEmbedding.wrappedValue = MLXArray.zeros([numNewEmbeddings, features])
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let table = concatenated([embedding, newEmbedding], axis: 0)
        return table[x]
    }
}

fileprivate final class Molmo2MLP: Module, UnaryLayer {
    @ModuleInfo(key: "ff_proj") var ffProj: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear

    init(inputDim: Int, intermediateSize: Int) {
        self._ffProj.wrappedValue = Linear(inputDim, intermediateSize * 2, bias: false)
        self._ffOut.wrappedValue = Linear(intermediateSize, inputDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let projected = ffProj(x)
        let pieces = projected.split(parts: 2, axis: -1)
        let value = pieces[0]
        let gate = pieces[1]
        return ffOut(silu(gate) * value)
    }
}

fileprivate final class Molmo2Attention: Module {
    private let config: Molmo2Configuration.TextConfiguration
    private let scale: Float
    private let fusedDims: (Int, Int, Int)

    @ModuleInfo(key: "att_proj") var attProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    @ModuleInfo(key: "attn_out") var attnOut: Linear

    private let rope: RoPE

    init(_ config: Molmo2Configuration.TextConfiguration) {
        self.config = config
        self.scale = pow(Float(config.headDim), -0.5)

        fusedDims = (
            config.numAttentionHeads * config.headDim,
            config.numKeyValueHeads * config.headDim,
            config.numKeyValueHeads * config.headDim
        )

        self._attProj.wrappedValue = Linear(
            config.hiddenSize,
            fusedDims.0 + fusedDims.1 + fusedDims.2,
            bias: config.qkvBias)
        self._qNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.layerNormEps)
        self._kNorm.wrappedValue = RMSNorm(dimensions: config.headDim, eps: config.layerNormEps)
        self._attnOut.wrappedValue = Linear(
            config.numAttentionHeads * config.headDim,
            config.hiddenSize,
            bias: false)

        self.rope = RoPE(dimensions: config.headDim, traditional: false, base: config.ropeTheta)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        let (B, L) = (hiddenStates.dim(0), hiddenStates.dim(1))

        let qkv = attProj(hiddenStates)
        let splits = split(
            qkv,
            indices: [fusedDims.0, fusedDims.0 + fusedDims.1],
            axis: -1)

        var q = splits[0]
        var k = splits[1]
        var v = splits[2]

        q = qNorm(q.reshaped(B, L, config.numAttentionHeads, -1)).transposed(0, 2, 1, 3)
        k = kNorm(k.reshaped(B, L, config.numKeyValueHeads, -1)).transposed(0, 2, 1, 3)
        v = v.reshaped(B, L, config.numKeyValueHeads, -1).transposed(0, 2, 1, 3)

        if let cache {
            q = rope(q, offset: cache.offset)
            k = rope(k, offset: cache.offset)
        } else {
            q = rope(q)
            k = rope(k)
        }

        let attn = attentionWithCacheUpdate(
            queries: q,
            keys: k,
            values: v,
            cache: cache,
            scale: scale,
            mask: mask)

        let output = attn
            .transposed(0, 2, 1, 3)
            .reshaped(B, L, -1)

        return attnOut(output)
    }
}

fileprivate final class Molmo2DecoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Molmo2Attention
    @ModuleInfo(key: "attn_norm") var attnNorm: RMSNorm
    @ModuleInfo(key: "ff_norm") var ffNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: Molmo2MLP

    init(_ config: Molmo2Configuration.TextConfiguration) {
        self._selfAttn.wrappedValue = Molmo2Attention(config)
        self._attnNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._ffNorm.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._mlp.wrappedValue = Molmo2MLP(inputDim: config.hiddenSize, intermediateSize: config.intermediateSize)
    }

    func callAsFunction(
        _ hiddenStates: MLXArray,
        mask: MLXFast.ScaledDotProductAttentionMaskMode,
        cache: KVCache?
    ) -> MLXArray {
        var hiddenStates = hiddenStates + selfAttn(attnNorm(hiddenStates), mask: mask, cache: cache)
        hiddenStates = hiddenStates + mlp(ffNorm(hiddenStates))
        return hiddenStates
    }
}

fileprivate final class Molmo2Transformer: Module {
    private let config: Molmo2Configuration.TextConfiguration

    @ModuleInfo(key: "wte") var wte: Molmo2Embedding
    @ModuleInfo(key: "blocks") var blocks: [Molmo2DecoderLayer]
    @ModuleInfo(key: "ln_f") var lnF: RMSNorm
    @ModuleInfo(key: "emb_drop") var embDrop: Dropout

    init(_ config: Molmo2Configuration.TextConfiguration) {
        self.config = config
        self._wte.wrappedValue = Molmo2Embedding(
            numEmbeddings: config.vocabSize,
            numNewEmbeddings: config.additionalVocabSize,
            features: config.hiddenSize)
        self._blocks.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Molmo2DecoderLayer(config)
        }
        self._lnF.wrappedValue = RMSNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._embDrop.wrappedValue = Dropout(p: Float(config.embeddingDropout))
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil,
        cache: [KVCache]?
    ) -> MLXArray {
        var hiddenStates = inputsEmbeds ?? wte(inputIds)

        let computedMask = mask ?? createAttentionMask(h: hiddenStates, cache: cache?.first)
        hiddenStates = embDrop(hiddenStates)

        for (index, block) in blocks.enumerated() {
            hiddenStates = block(hiddenStates, mask: computedMask, cache: cache?[safe: index])
        }

        return lnF(hiddenStates)
    }
}

fileprivate final class Molmo2LanguageModel: Module {
    let config: Molmo2Configuration

    @ModuleInfo(key: "model") var model: Molmo2Transformer
    @ModuleInfo(key: "lm_head") var lmHead: Linear

    let kvHeads: [Int]

    init(_ config: Molmo2Configuration) {
        self.config = config
        self._model.wrappedValue = Molmo2Transformer(config.textConfig)
        self._lmHead.wrappedValue = Linear(config.textConfig.hiddenSize, config.textConfig.vocabSize, bias: false)
        self.kvHeads = (0 ..< config.textConfig.numHiddenLayers).map { _ in config.textConfig.numKeyValueHeads }
    }

    var layers: [Module] {
        model.blocks
    }

    func embedTokens(_ inputIds: MLXArray) -> MLXArray {
        model.wte(inputIds)
    }

    func callAsFunction(
        _ inputIds: MLXArray,
        cache: [KVCache]?,
        inputsEmbeds: MLXArray? = nil,
        mask: MLXFast.ScaledDotProductAttentionMaskMode? = nil
    ) -> LMOutput {
        let hiddenStates = model(inputIds, inputsEmbeds: inputsEmbeds, mask: mask, cache: cache)
        let logits = lmHead(hiddenStates)
        return LMOutput(logits: logits)
    }
}

fileprivate final class Molmo2ViTMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo var activation: GELU

    init(hiddenSize: Int, intermediateSize: Int, hiddenAct: String) {
        self._w1.wrappedValue = Linear(hiddenSize, intermediateSize, bias: true)
        self._w2.wrappedValue = Linear(intermediateSize, hiddenSize, bias: true)
        if hiddenAct == "gelu_pytorch_tanh" {
            self.activation = GELU(approximation: .fast)
        } else {
            self.activation = GELU(approximation: .fast)
        }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(activation(w1(x)))
    }
}

fileprivate final class Molmo2ViTMultiHeadDotProductAttention: Module {
    private let hiddenSize: Int
    private let numHeads: Int
    private let numKeyValueHeads: Int
    private let headDim: Int
    private let numKeyValueGroups: Int
    private let scale: Float
    private let float32Attention: Bool

    @ModuleInfo(key: "wq") var wq: Linear
    @ModuleInfo(key: "wk") var wk: Linear
    @ModuleInfo(key: "wv") var wv: Linear
    @ModuleInfo(key: "wo") var wo: Linear

    init(
        hiddenSize: Int,
        numHeads: Int,
        numKeyValueHeads: Int,
        headDim: Int,
        inputDim: Int? = nil,
        useBias: Bool = true,
        float32Attention: Bool = true
    ) {
        self.hiddenSize = hiddenSize
        self.numHeads = numHeads
        self.numKeyValueHeads = numKeyValueHeads
        self.headDim = headDim
        self.numKeyValueGroups = max(1, numHeads / max(1, numKeyValueHeads))
        self.scale = pow(Float(headDim), -0.5)
        self.float32Attention = float32Attention

        let inputDim = inputDim ?? hiddenSize
        self._wq.wrappedValue = Linear(inputDim, numHeads * headDim, bias: useBias)
        self._wk.wrappedValue = Linear(inputDim, numKeyValueHeads * headDim, bias: useBias)
        self._wv.wrappedValue = Linear(inputDim, numKeyValueHeads * headDim, bias: useBias)
        self._wo.wrappedValue = Linear(numHeads * headDim, hiddenSize, bias: true)
    }

    func callAsFunction(
        _ inputsQ: MLXArray,
        inputsKV: MLXArray? = nil,
        attnMask: MLXArray? = nil
    ) -> MLXArray {
        let inputsK = inputsKV ?? inputsQ
        let inputsV = inputsKV ?? inputsQ

        var xq = wq(inputsQ)
        var xk = wk(inputsK)
        var xv = wv(inputsV)

        let (B, qLen) = (xq.dim(0), xq.dim(1))
        let kvLen = xk.dim(1)

        xq = xq.reshaped(B, qLen, numHeads, headDim)
        xk = xk.reshaped(B, kvLen, numKeyValueHeads, headDim)
        xv = xv.reshaped(B, kvLen, numKeyValueHeads, headDim)

        if numHeads != numKeyValueHeads {
            xk = repeated(xk, count: numKeyValueGroups, axis: 2)
            xv = repeated(xv, count: numKeyValueGroups, axis: 2)
        }

        var q = xq.transposed(0, 2, 1, 3)
        var k = xk.transposed(0, 2, 1, 3)
        var v = xv.transposed(0, 2, 1, 3)

        let originalDtype = q.dtype
        if float32Attention {
            q = q.asType(.float32)
            k = k.asType(.float32)
            v = v.asType(.float32)
        }

        var scores = MLX.matmul(q, k.transposed(0, 1, 3, 2)) * scale
        if let attnMask {
            let negative = MLXArray(-1e9).asType(scores.dtype)
            scores = MLX.where(attnMask, scores, negative)
        }

        let weights = softmax(scores, axis: -1)
        var out = MLX.matmul(weights, v)
        if float32Attention {
            out = out.asType(originalDtype)
        }

        out = out.transposed(0, 2, 1, 3).reshaped(B, qLen, -1)
        return wo(out)
    }
}

fileprivate final class Molmo2VisionBlock: Module {
    @ModuleInfo(key: "attention") var attention: Molmo2ViTMultiHeadDotProductAttention
    @ModuleInfo(key: "feed_forward") var feedForward: Molmo2ViTMLP
    @ModuleInfo(key: "attention_norm") var attentionNorm: LayerNorm
    @ModuleInfo(key: "ffn_norm") var ffnNorm: LayerNorm

    init(_ config: Molmo2Configuration.VitConfiguration) {
        self._attention.wrappedValue = Molmo2ViTMultiHeadDotProductAttention(
            hiddenSize: config.hiddenSize,
            numHeads: config.numAttentionHeads,
            numKeyValueHeads: config.numKeyValueHeads,
            headDim: config.headDim,
            inputDim: config.hiddenSize,
            useBias: true,
            float32Attention: config.float32Attention)
        self._feedForward.wrappedValue = Molmo2ViTMLP(
            hiddenSize: config.hiddenSize,
            intermediateSize: config.intermediateSize,
            hiddenAct: config.hiddenAct)
        self._attentionNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
        self._ffnNorm.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: config.layerNormEps)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var hidden = x + attention(attentionNorm(x))
        hidden = hidden + feedForward(ffnNorm(hidden))
        return hidden
    }
}

fileprivate final class Molmo2VisionTransformer: Module {
    private let config: Molmo2Configuration.VitConfiguration

    @ParameterInfo(key: "positional_embedding") var positionalEmbedding: MLXArray
    @ModuleInfo(key: "patch_embedding") var patchEmbedding: Linear
    @ModuleInfo(key: "transformer") var transformer: [Molmo2VisionBlock]

    init(_ config: Molmo2Configuration.VitConfiguration) {
        self.config = config
        self._positionalEmbedding.wrappedValue = MLXArray.zeros([config.imageNumPos, config.hiddenSize])
        let patchDim = config.imagePatchSize * config.imagePatchSize * 3
        self._patchEmbedding.wrappedValue = Linear(patchDim, config.hiddenSize, bias: true)
        self._transformer.wrappedValue = (0 ..< config.numHiddenLayers).map { _ in
            Molmo2VisionBlock(config)
        }
        super.init()
    }

    private func addPosEmb(_ x: MLXArray, patchNum: (Int, Int)) -> MLXArray {
        let posEmbSize = Int(Double(positionalEmbedding.dim(0)).squareRoot())
        let hidden = positionalEmbedding.dim(1)
        var posEmb = positionalEmbedding.reshaped(posEmbSize, posEmbSize, hidden)

        let (patchH, patchW) = patchNum
        if posEmb.dim(0) != patchH || posEmb.dim(1) != patchW {
            var nchw = expandedDimensions(posEmb, axis: 0).transposed(0, 3, 1, 2)
            nchw = bicubicInterpolate(nchw, size: (patchH, patchW), alignCorners: false)
            posEmb = nchw.transposed(0, 2, 3, 1)[0]
        }

        let flattened = posEmb.reshaped(-1, hidden)
        return x + expandedDimensions(flattened, axis: 0).asType(x.dtype)
    }

    func forwardSelectedLayers(
        _ x: MLXArray,
        patchNum: (Int, Int)? = nil,
        layers: [Int]
    ) -> [MLXArray] {
        let patchNum = patchNum ?? config.imageNumPatch

        var hiddenStates = patchEmbedding(x)
        hiddenStates = addPosEmb(hiddenStates, patchNum: patchNum)

        let wanted = Set(layers)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(layers.count)

        for (index, block) in transformer.enumerated() {
            hiddenStates = block(hiddenStates)
            if wanted.contains(index) {
                outputs.append(hiddenStates)
            }
        }

        return outputs
    }
}

fileprivate final class Molmo2ImageProjectorMLP: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w2") var w2: Linear
    @ModuleInfo(key: "w3") var w3: Linear

    init(inputDim: Int, hiddenDim: Int, outputDim: Int) {
        self._w1.wrappedValue = Linear(inputDim, hiddenDim, bias: false)
        self._w2.wrappedValue = Linear(hiddenDim, outputDim, bias: false)
        self._w3.wrappedValue = Linear(inputDim, hiddenDim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        w2(silu(w1(x)) * w3(x))
    }
}

fileprivate final class Molmo2VisionTower: Module {
    private let config: Molmo2Configuration
    private let vitLayers: [Int]

    @ModuleInfo(key: "image_vit") var imageVit: Molmo2VisionTransformer
    @ModuleInfo(key: "image_pooling_2d") var imagePooling2d: Molmo2ViTMultiHeadDotProductAttention
    @ModuleInfo(key: "image_projector") var imageProjector: Molmo2ImageProjectorMLP

    init(_ config: Molmo2Configuration) {
        self.config = config

        self._imageVit.wrappedValue = Molmo2VisionTransformer(config.vitConfig)

        let totalLayers = config.vitConfig.numHiddenLayers
        let mappedLayers = config.adapterConfig.vitLayers.map { layer in
            layer >= 0 ? layer : layer + totalLayers
        }
        self.vitLayers = mappedLayers.isEmpty ? [max(totalLayers - 1, 0)] : mappedLayers

        let poolDim = config.vitConfig.hiddenSize * vitLayers.count

        self._imagePooling2d.wrappedValue = Molmo2ViTMultiHeadDotProductAttention(
            hiddenSize: config.adapterConfig.hiddenSize,
            numHeads: config.adapterConfig.numAttentionHeads,
            numKeyValueHeads: config.adapterConfig.numKeyValueHeads,
            headDim: config.adapterConfig.headDim,
            inputDim: poolDim,
            useBias: true,
            float32Attention: config.adapterConfig.float32Attention)

        self._imageProjector.wrappedValue = Molmo2ImageProjectorMLP(
            inputDim: config.adapterConfig.hiddenSize,
            hiddenDim: config.adapterConfig.intermediateSize,
            outputDim: config.adapterConfig.textHiddenSize)
    }

    private func encodeImage(_ images: MLXArray) -> MLXArray {
        let batchSize = images.dim(0)
        let numCrops = images.dim(1)
        let numPatch = images.dim(2)
        let patchDim = images.dim(3)

        let flat = images.reshaped(batchSize * numCrops, numPatch, patchDim)
        let selectedLayers = imageVit.forwardSelectedLayers(flat, layers: vitLayers)

        let features = concatenated(selectedLayers, axis: -1)
        return features.reshaped(batchSize, numCrops, numPatch, -1)
    }

    func callAsFunction(
        _ images: MLXArray,
        pooledPatchesIdx: MLXArray
    ) -> MLXArray {
        let batchSize = images.dim(0)
        let numPooled = pooledPatchesIdx.dim(1)
        let tokenDim = pooledPatchesIdx.dim(2)

        let imageFeatures = encodeImage(images)
        let dim = imageFeatures.dim(3)

        let valid = pooledPatchesIdx .>= 0
        let validToken = valid.asType(.int32).sum(axis: -1) .> 0

        let flatFeatures = imageFeatures.reshaped(batchSize, -1, dim)
        let flatLen = flatFeatures.dim(1)

        let idx = MLX.where(valid, pooledPatchesIdx, MLXArray(Int32(0))).asType(.int32)
        let batchIndexBase = MLXArray(0 ..< batchSize).asType(.int32).reshaped(batchSize, 1, 1)
        let batchIndices = broadcast(batchIndexBase, to: idx.shape)
        let linear = batchIndices * MLXArray(Int32(flatLen)) + idx
        let linearFlat = linear.flattened().asType(.uint32)

        let flat2d = flatFeatures.reshaped(-1, dim)
        let gathered = flat2d[linearFlat]
        var toPool = gathered.reshaped(batchSize, numPooled, tokenDim, dim)

        let validExpanded = expandedDimensions(valid.asType(toPool.dtype), axis: -1)
        toPool = toPool * validExpanded
        toPool = toPool.reshaped(-1, tokenDim, dim)

        let pooled: MLXArray
        if config.adapterConfig.poolingAttentionMask {
            let mask = valid.reshaped(-1, 1, 1, tokenDim).asType(.bool)
            let denom = valid.reshaped(-1, tokenDim).asType(.float32).sum(axis: -1)
            let safeDenom = MLX.where(denom .== 0, ones(like: denom), denom)
            let query = toPool.sum(axis: -2, keepDims: true) / expandedDimensions(expandedDimensions(safeDenom, axis: -1), axis: -1).asType(toPool.dtype)
            pooled = imagePooling2d(query, inputsKV: toPool, attnMask: mask)
        } else {
            let query = toPool.mean(axis: -2, keepDims: true)
            pooled = imagePooling2d(query, inputsKV: toPool, attnMask: nil)
        }

        let pooledHidden = pooled.squeezed(axis: 1)
            .reshaped(batchSize, numPooled, -1)

        let projected = imageProjector(pooledHidden)
        let flattened = projected.reshaped(-1, projected.dim(2))

        let indices = Molmo2.nonZero(validToken.flattened().asType(.bool))
        let indexArray = MLXArray(indices.map(UInt32.init))
        return flattened[indexArray]
    }
}

public final class Molmo2: Module, VLMModel, KVCacheDimensionProvider {
    @ModuleInfo(key: "vision_tower") private var visionTower: Molmo2VisionTower
    @ModuleInfo(key: "language_model") private var languageModel: Molmo2LanguageModel

    public let config: Molmo2Configuration

    public init(_ config: Molmo2Configuration) {
        self.config = config
        self._languageModel.wrappedValue = Molmo2LanguageModel(config)
        self._visionTower.wrappedValue = Molmo2VisionTower(config)
    }

    public var vocabularySize: Int { config.textConfig.vocabSize }
    public var kvHeads: [Int] { languageModel.kvHeads }

    public var loraLayers: [Module] {
        languageModel.layers
    }

    public func prepare(
        _ input: LMInput,
        cache: [any KVCache],
        windowSize _: Int?
    ) throws -> PrepareResult {
        if input.image != nil && input.video != nil {
            throw Molmo2ModelError.unsupportedMixedMedia
        }

        let inputIds = input.text.tokens
        let typedCache = castCache(cache)

        var inputsEmbeds: MLXArray? = nil

        if let image = input.image {
            guard let pixels = image.pixels.nilIfEmpty else {
                throw Molmo2ModelError.missingImageMetadata
            }
            guard let tokenPooling = image.tokenPooling, let grids = image.grids, let numCrops = image.numCrops else {
                throw Molmo2ModelError.missingImageMetadata
            }
            let (images, pooling) = try buildBatchedImages(
                inputIds: inputIds,
                pixelValues: pixels,
                tokenPooling: tokenPooling,
                imageGrids: grids,
                imageNumCrops: numCrops)
            inputsEmbeds = try buildInputEmbeddings(
                inputIds: inputIds,
                images: images,
                tokenPooling: pooling)
        } else if let video = input.video {
            guard let tokenPooling = video.tokenPooling, let grids = video.grids else {
                throw Molmo2ModelError.missingVideoMetadata
            }
            let (videos, pooling) = try buildBatchedVideos(
                inputIds: inputIds,
                pixelValues: video.pixels,
                tokenPooling: tokenPooling,
                videoGrids: grids)
            inputsEmbeds = try buildInputEmbeddings(
                inputIds: inputIds,
                images: videos,
                tokenPooling: pooling)
        }

        let output = languageModel(
            inputIds,
            cache: typedCache,
            inputsEmbeds: inputsEmbeds)

        return .logits(output)
    }

    public func callAsFunction(_ inputs: MLXArray, cache: [any KVCache]?) -> MLXArray {
        let typedCache = castCacheOptional(cache)
        return languageModel(inputs, cache: typedCache, inputsEmbeds: nil).logits
    }

    private func buildInputEmbeddings(
        inputIds: MLXArray,
        images: MLXArray,
        tokenPooling: MLXArray
    ) throws -> MLXArray {
        let mask = inputIds .!= MLXArray(-1)
        let safeInputIds = inputIds * mask.asType(inputIds.dtype)

        let textEmbeds = languageModel.embedTokens(safeInputIds)

        let dtype = visionTower.imageVit.patchEmbedding.weight.dtype
        let imageFeatures = visionTower(images.asType(dtype), pooledPatchesIdx: tokenPooling).asType(textEmbeds.dtype)

        let flatIds = safeInputIds.flattened()
        let isPatch = flatIds .== MLXArray(config.imagePatchId)
        let positions = Molmo2.nonZero(isPatch.asType(.bool))

        if positions.count != imageFeatures.dim(0) {
            throw Molmo2ModelError.featureTokenMismatch(expected: positions.count, actual: imageFeatures.dim(0))
        }

        let flatEmbeds = textEmbeds.reshaped(-1, textEmbeds.dim(2))
        if !positions.isEmpty {
            let indexArray = MLXArray(positions.map(UInt32.init))
            flatEmbeds[indexArray] = flatEmbeds[indexArray] + imageFeatures
        }

        return flatEmbeds.reshaped(textEmbeds.shape)
    }

    private func buildBatchedImages(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        tokenPooling: MLXArray,
        imageGrids: MLXArray,
        imageNumCrops: MLXArray
    ) throws -> (MLXArray, MLXArray) {
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(1)

        let flatIds = inputIds.asArray(Int.self)
        var counts: [Int] = Array(repeating: 0, count: batchSize)
        for b in 0 ..< batchSize {
            let start = b * seqLength
            let end = start + seqLength
            let rawCount = flatIds[start..<end].filter { $0 == config.imageEndTokenId }.count
            counts[b] = rawCount / 2
        }

        let numImages = counts.reduce(0, +)
        guard imageGrids.dim(0) == numImages, imageNumCrops.dim(0) == numImages else {
            throw Molmo2ModelError.missingImageMetadata
        }

        let gridsFlat = imageGrids.asArray(Int32.self)
        let cropsFlat = imageNumCrops.asArray(Int32.self)

        var pooledPerImage: [Int] = []
        pooledPerImage.reserveCapacity(numImages)
        var cropsPerImage: [Int] = []
        cropsPerImage.reserveCapacity(numImages)

        for i in 0 ..< numImages {
            let base = i * 4
            let loH = Int(gridsFlat[base])
            let loW = Int(gridsFlat[base + 1])
            let hiH = Int(gridsFlat[base + 2])
            let hiW = Int(gridsFlat[base + 3])
            pooledPerImage.append(loH * loW + hiH * hiW)
            cropsPerImage.append(Int(cropsFlat[i]))
        }

        let (nCrops, nPatches, patchDim) = (pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2))

        var cropsPerExample = Array(repeating: 0, count: batchSize)
        var pooledPerExample = Array(repeating: 0, count: batchSize)

        var imageIndex = 0
        for ex in 0 ..< batchSize {
            for _ in 0 ..< counts[ex] {
                cropsPerExample[ex] += cropsPerImage[imageIndex]
                pooledPerExample[ex] += pooledPerImage[imageIndex]
                imageIndex += 1
            }
        }

        if cropsPerExample.reduce(0, +) != nCrops {
            throw Molmo2ModelError.missingImageMetadata
        }
        if pooledPerExample.reduce(0, +) != tokenPooling.dim(0) {
            throw Molmo2ModelError.missingImageMetadata
        }

        let maxCrops = cropsPerExample.max() ?? 0
        let images = full(
            [batchSize, maxCrops, nPatches, patchDim],
            values: Float(-1)
        ).asType(pixelValues.dtype)

        var cropOffset = 0
        for ex in 0 ..< batchSize {
            let num = cropsPerExample[ex]
            if num > 0 {
                images[ex, 0 ..< num, 0..., 0...] = pixelValues[cropOffset ..< cropOffset + num, 0..., 0...]
                cropOffset += num
            }
        }

        let maxPooled = pooledPerExample.max() ?? 0
        let tokenDim = tokenPooling.dim(1)
        let newTokenPooling = full(
            [batchSize, maxPooled, tokenDim],
            values: Int32(-1)
        ).asType(tokenPooling.dtype)

        let patchesPerImage = cropsPerImage.map { $0 * nPatches }

        var pooledOffset = 0
        imageIndex = 0
        for ex in 0 ..< batchSize {
            let c = counts[ex]
            let numPooled = pooledPerExample[ex]
            if numPooled == 0 { continue }

            let cur = tokenPooling[pooledOffset ..< pooledOffset + numPooled, 0...]

            let perImgPatches = patchesPerImage[imageIndex ..< imageIndex + c]
            var indexOffsets: [Int] = [0]
            indexOffsets.reserveCapacity(c)
            var running = 0
            for p in perImgPatches.dropLast() {
                running += p
                indexOffsets.append(running)
            }

            let perImgPooled = pooledPerImage[imageIndex ..< imageIndex + c]

            var localOffset = 0
            for j in 0 ..< c {
                let n = perImgPooled[perImgPooled.index(perImgPooled.startIndex, offsetBy: j)]
                let idxOff = indexOffsets[j]
                if n > 0 {
                    let slice = cur[localOffset ..< localOffset + n, 0...]
                    cur[localOffset ..< localOffset + n, 0...] = MLX.where(
                        slice .>= 0,
                        slice + MLXArray(Int32(idxOff)),
                        slice)
                    localOffset += n
                }
            }

            newTokenPooling[ex, 0 ..< numPooled, 0...] = cur

            pooledOffset += numPooled
            imageIndex += c
        }

        return (images, newTokenPooling)
    }

    private func buildBatchedVideos(
        inputIds: MLXArray,
        pixelValues: MLXArray,
        tokenPooling: MLXArray,
        videoGrids: MLXArray
    ) throws -> (MLXArray, MLXArray) {
        let batchSize = inputIds.dim(0)
        let seqLength = inputIds.dim(1)

        let idsFlat = inputIds.asArray(Int.self)
        let frameEnd = config.frameEndTokenId
        let imageEnd = config.imageEndTokenId
        let usesFrameEnd = idsFlat.contains(frameEnd)
        let endTokenId = usesFrameEnd ? frameEnd : imageEnd

        var counts: [Int] = Array(repeating: 0, count: batchSize)
        for b in 0 ..< batchSize {
            let start = b * seqLength
            let end = start + seqLength
            let hasVideo = idsFlat[start..<end].contains(endTokenId)
            counts[b] = hasVideo ? 1 : 0
        }

        let numVideos = counts.reduce(0, +)
        guard videoGrids.dim(0) == numVideos else {
            throw Molmo2ModelError.missingVideoMetadata
        }

        let gridsFlat = videoGrids.asArray(Int32.self)
        let (nFrames, nPatches, patchDim) = (pixelValues.dim(0), pixelValues.dim(1), pixelValues.dim(2))

        var framesPerExample = Array(repeating: 0, count: batchSize)
        var pooledPerExample = Array(repeating: 0, count: batchSize)

        var videoIndex = 0
        for ex in 0 ..< batchSize {
            if counts[ex] == 1 {
                let base = videoIndex * 3
                let t = Int(gridsFlat[base])
                let h = Int(gridsFlat[base + 1])
                let w = Int(gridsFlat[base + 2])
                framesPerExample[ex] = t
                pooledPerExample[ex] = t * h * w
                videoIndex += 1
            }
        }

        let maxFrames = framesPerExample.max() ?? 0
        let videos = full([batchSize, maxFrames, nPatches, patchDim], values: Float(-1)).asType(pixelValues.dtype)

        var frameOffset = 0
        for ex in 0 ..< batchSize {
            let num = framesPerExample[ex]
            if num > 0 {
                videos[ex, 0 ..< num, 0..., 0...] = pixelValues[frameOffset ..< frameOffset + num, 0..., 0...]
                frameOffset += num
            }
        }

        let maxPooled = pooledPerExample.max() ?? 0
        let tokenDim = tokenPooling.dim(1)
        let newTokenPooling = full([batchSize, maxPooled, tokenDim], values: Int32(-1)).asType(tokenPooling.dtype)

        var pooledOffset = 0
        for ex in 0 ..< batchSize {
            let num = pooledPerExample[ex]
            if num > 0 {
                newTokenPooling[ex, 0 ..< num, 0...] = tokenPooling[pooledOffset ..< pooledOffset + num, 0...]
                pooledOffset += num
            }
        }

        if frameOffset != nFrames || pooledOffset != tokenPooling.dim(0) {
            throw Molmo2ModelError.missingVideoMetadata
        }

        return (videos, newTokenPooling)
    }

    fileprivate static func nonZero(_ mask: MLXArray) -> [Int] {
        let values = mask.asArray(Bool.self)
        var indices: [Int] = []
        indices.reserveCapacity(values.count)
        for (idx, value) in values.enumerated() where value {
            indices.append(idx)
        }
        return indices
    }
}

extension Molmo2 {
    fileprivate func castCache(_ cache: [any KVCache]) -> [KVCache]? {
        guard !cache.isEmpty else { return nil }
        return cache.map { $0 }
    }

    fileprivate func castCacheOptional(_ cache: [any KVCache]?) -> [KVCache]? {
        guard let cache else { return nil }
        return castCache(cache)
    }
}

fileprivate extension Collection {
    subscript(safe index: Index) -> Element? {
        indices.contains(index) ? self[index] : nil
    }
}

fileprivate extension MLXArray {
    var nilIfEmpty: MLXArray? { size == 0 ? nil : self }
}
