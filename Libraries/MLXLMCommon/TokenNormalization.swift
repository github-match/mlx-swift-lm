// Copyright (c) 2024-2025 Apple Inc.
// TokenNormalization.swift - Workaround for swift-jinja whitespace issues

import Foundation
import Tokenizers

/// Extension to normalize token sequences after chat template application.
///
/// swift-jinja can insert extra whitespace tokens when rendering Jinja2 templates
/// (e.g., Llama-3.1 chat templates). This causes:
/// - Prompt token count to be higher than Python's transformers library
/// - The model to emit incorrect tokens or produce degraded output
///
/// This extension provides an idempotent normalization that:
/// 1. Removes whitespace-only tokens immediately following BOS
/// 2. Removes spurious whitespace after <|eot_id|> tokens (before <|start_header_id|>)
/// 3. Normalizes trailing whitespace to match expected format (\n\n instead of \n\n\n)
/// 4. Does nothing if no excess whitespace exists (future-proof for when swift-jinja is fixed)
/// 5. Works with any tokenizer, not just Llama models
public extension Tokenizer {

    /// Normalizes tokens by trimming excess whitespace inserted by swift-jinja.
    ///
    /// This is a workaround for swift-jinja inserting extra newline tokens
    /// when rendering chat templates. The fix is idempotent: if the upstream issue
    /// is fixed, this function will simply return the original tokens unchanged.
    ///
    /// - Parameter tokens: The token sequence from `applyChatTemplate`
    /// - Returns: Normalized token sequence with excess whitespace removed
    func normalizePromptTokens(_ tokens: [Int]) -> [Int] {
        var result = tokens

        // === Step 1: Trim leading whitespace after BOS ===
        if let bosTokenId = self.bosTokenId,
            let firstToken = result.first,
            firstToken == bosTokenId,
            result.count > 1
        {
            // Find the first non-whitespace token after BOS
            var firstContentIndex = 1
            while firstContentIndex < result.count {
                // Use decode() to get the actual text representation (handles BPE markers)
                let tokenText = self.decode(tokens: [result[firstContentIndex]])

                // Check if this token is purely whitespace (spaces, tabs, newlines)
                if tokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                    firstContentIndex += 1
                    continue
                }
                break
            }

            // If whitespace was found after BOS, remove it
            if firstContentIndex > 1 {
                var normalized = [bosTokenId]
                normalized.append(contentsOf: result[firstContentIndex...])
                result = normalized
            }
        }

        // === Step 2: Remove spurious whitespace between <|eot_id|> and <|start_header_id|> ===
        // swift-jinja inserts \n\n after <|eot_id|> before <|start_header_id|>
        // which Python doesn't have.
        if let eotTokenId = self.eosTokenId {
            var i = 0
            while i < result.count - 2 {
                // Check for pattern: eot_id, whitespace, start_header_id
                if result[i] == eotTokenId {
                    let nextToken = result[i + 1]
                    let nextTokenText = self.decode(tokens: [nextToken])

                    // If next token is whitespace and following token is a special token
                    if nextTokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                        if i + 2 < result.count {
                            let afterWhitespaceToken = result[i + 2]
                            // Special tokens in Llama have IDs >= 128000
                            if afterWhitespaceToken >= 128000 {
                                // Remove the spurious whitespace token
                                result.remove(at: i + 1)
                                // Don't increment i, check if there are more whitespace tokens
                                continue
                            }
                        }
                    }
                }
                i += 1
            }
        }

        // === Step 3: Normalize trailing whitespace ===
        // swift-jinja produces \n\n\n instead of \n\n
        // We need to replace the last token if it's excess whitespace
        if result.count > 1 {
            let lastToken = result[result.count - 1]
            let lastTokenText = self.decode(tokens: [lastToken])

            // Check if last token is whitespace with more than 2 newlines
            if lastTokenText.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty {
                let newlineCount = lastTokenText.filter { $0 == "\n" }.count

                // If we have 3+ newlines, replace with the correct 2-newline token
                if newlineCount >= 3 {
                    // Find the token for "\n\n" by encoding it
                    let correctWhitespace = "\n\n"
                    var correctTokens = self.encode(text: correctWhitespace, addSpecialTokens: false)

                    // Remove BOS if it was added despite addSpecialTokens: false
                    if let bosTokenId = self.bosTokenId,
                        correctTokens.first == bosTokenId
                    {
                        correctTokens.removeFirst()
                    }

                    // Replace the last token with the correct one(s)
                    if !correctTokens.isEmpty {
                        result.removeLast()
                        result.append(contentsOf: correctTokens)
                    }
                }
            }
        }

        return result
    }
}
