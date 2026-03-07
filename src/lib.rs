//! # ALICE-Token — Ultra-fast BPE Tokenizer
//!
//! ALICEエコシステム向け超高速バイトペアエンコーディング（BPE）トークナイザー。
//!
//! ## アーキテクチャ
//!
//! ```text
//! [入力テキスト &[u8]]
//!     │
//!     ├─ special::split_with_special() ← 特殊トークン分離
//!     │
//!     ├─ pretokenizer::split()  ← GPT-4/GPT-2正規表現分割
//!     │
//!     ├─ simd::find_boundaries()  ← NEON/AVX2 intrinsics
//!     │
//!     ├─ parallel::chunk_tokenize() ← Rayon 並列チャンク分割
//!     │   │
//!     │   └─ bpe::encode_chunk()  ← 優先度キュー BPE マージ
//!     │       │
//!     │       └─ vocab::lookup()  ← FxHashMap O(1) 語彙検索
//!     │
//!     └─ [トークンID列 Vec<u32>]
//! ```
//!
//! ## 全領域カバレッジ
//!
//! 1. **BPEアルゴリズム**: 連結リスト + 優先度キュー O(n log n) マージ
//! 2. **正規表現プリトークナイザー**: GPT-4 (cl100k) / GPT-2 (r50k) パターン対応
//! 3. **特殊トークン**: `<|endoftext|>` 等の登録・検出・エンコード制御
//! 4. **語彙ファイル I/O**: tiktoken `.tiktoken` 形式 + バイナリ (bincode)
//! 5. **BPEトレーニング**: コーパスからのマージルール学習
//! 6. **SIMD intrinsics**: ARM64 NEON / `x86_64` AVX2 によるバイト検索・境界検出
//! 7. **Rayon並列化**: UTF-8安全境界でのチャンク分割並列トークナイズ
//! 8. **`FxHashMap`**: rustc内部使用の高速ハッシュ語彙テーブル
//! 9. **C FFI / `PyO3`**: C言語 + Python バインディング

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::inline_always,
    clippy::too_many_lines
)]

mod bpe;
mod decoder;
mod encoder;
pub mod io;
mod parallel;
pub mod pretokenizer;
pub mod simd;
pub mod special;
pub mod trainer;
mod vocab;

#[cfg(feature = "ffi")]
pub mod ffi;

pub use bpe::BpeMerge;
pub use decoder::Decoder;
pub use encoder::Encoder;
pub use io::{load_binary, load_tiktoken, save_binary, save_tiktoken, TokenIoError};
pub use parallel::ParallelTokenizer;
pub use pretokenizer::{PreTokenizer, PreTokenizerPattern};
pub use special::{SpecialTokenPolicy, SpecialTokens, TextChunk};
pub use trainer::{Trainer, TrainerConfig};
pub use vocab::{byte_level_builder, Vocab, VocabBuilder};

/// ALICE-Token トークナイザー本体
///
/// 語彙、BPEマージルール、プリトークナイザー、特殊トークンを統合し、
/// テキストの完全なエンコード/デコードパイプラインを提供する。
pub struct Tokenizer {
    vocab: Vocab,
    encoder: Encoder,
    decoder: Decoder,
    pre_tok: Option<PreTokenizer>,
    special_tokens: SpecialTokens,
}

impl Tokenizer {
    /// 語彙のみからトークナイザーを構築（プリトークナイザーなし）
    #[must_use]
    pub fn new(vocab: Vocab) -> Self {
        let encoder = Encoder::new(&vocab);
        let decoder = Decoder::new(&vocab);
        Self {
            vocab,
            encoder,
            decoder,
            pre_tok: None,
            special_tokens: SpecialTokens::new(),
        }
    }

    /// 全コンポーネントを指定してトークナイザーを構築
    #[must_use]
    pub fn with_config(
        vocab: Vocab,
        pre_tok: Option<PreTokenizer>,
        special_tokens: SpecialTokens,
    ) -> Self {
        let encoder = Encoder::new(&vocab);
        let decoder = Decoder::new(&vocab);
        Self {
            vocab,
            encoder,
            decoder,
            pre_tok,
            special_tokens,
        }
    }

    /// バイト列をトークンID列にエンコード
    ///
    /// パイプライン:
    /// 1. 特殊トークンで分割
    /// 2. 通常テキスト部分をプリトークナイズ
    /// 3. 各チャンクにBPEを適用
    #[must_use]
    pub fn encode(&self, text: &[u8]) -> Vec<u32> {
        self.encode_with_policy(text, &SpecialTokenPolicy::AllAllowed)
    }

    /// 特殊トークンポリシーを指定してエンコード
    #[must_use]
    pub fn encode_with_policy(&self, text: &[u8], policy: &SpecialTokenPolicy) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        let use_special =
            !self.special_tokens.is_empty() && !matches!(policy, SpecialTokenPolicy::NoneAllowed);

        if !use_special {
            return self.encode_text_chunk(text);
        }

        let chunks = self.special_tokens.split_with_special(text);
        let mut ids = Vec::new();

        for chunk in &chunks {
            if chunk.is_special {
                if let Some(token_id) = chunk.token_id {
                    match policy {
                        SpecialTokenPolicy::AllAllowed => {
                            ids.push(token_id);
                        }
                        SpecialTokenPolicy::Whitelist(allowed) => {
                            let text_str = std::str::from_utf8(chunk.bytes).unwrap_or_default();
                            if allowed.iter().any(|a| a == text_str) {
                                ids.push(token_id);
                            } else {
                                ids.extend(self.encode_text_chunk(chunk.bytes));
                            }
                        }
                        SpecialTokenPolicy::NoneAllowed => {
                            ids.extend(self.encode_text_chunk(chunk.bytes));
                        }
                    }
                }
            } else {
                ids.extend(self.encode_text_chunk(chunk.bytes));
            }
        }

        ids
    }

    /// 通常テキストチャンクをエンコード（プリトークナイザー適用）
    fn encode_text_chunk(&self, text: &[u8]) -> Vec<u32> {
        self.pre_tok.as_ref().map_or_else(
            || self.encoder.encode(text, &self.vocab),
            |pt| {
                let chunks = pt.split(text);
                let mut ids = Vec::new();
                for chunk in chunks {
                    ids.extend(self.encoder.encode(chunk, &self.vocab));
                }
                ids
            },
        )
    }

    /// トークンID列をバイト列にデコード
    ///
    /// 特殊トークンIDも正しくデコードする。
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut result = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            // 特殊トークンか確認
            if let Some(text) = self.special_tokens.get_text(id) {
                result.extend_from_slice(text.as_bytes());
            } else {
                result.extend(self.decoder.decode(&[id]));
            }
        }
        result
    }

    /// Rayon並列エンコード（大規模テキスト向け）
    #[inline]
    #[must_use]
    pub fn encode_parallel(&self, text: &[u8], chunk_size: usize) -> Vec<u32> {
        ParallelTokenizer::encode(text, chunk_size, &self.encoder, &self.vocab)
    }

    /// 語彙サイズを返す
    #[inline]
    #[must_use]
    pub const fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// 語彙への参照を返す
    #[inline]
    #[must_use]
    pub const fn vocab(&self) -> &Vocab {
        &self.vocab
    }

    /// 特殊トークンへの参照を返す
    #[inline]
    #[must_use]
    pub const fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_vocab() -> Vocab {
        let mut builder = VocabBuilder::new();
        for b in 0u16..=255 {
            builder.add_token(vec![b as u8]);
        }
        builder.add_merge(b"a", b"b");
        builder.add_merge(b"c", b"d");
        builder.build()
    }

    #[test]
    fn test_roundtrip_ascii() {
        let tok = Tokenizer::new(make_test_vocab());
        let text = b"hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(&decoded, text);
    }

    #[test]
    fn test_roundtrip_utf8() {
        let tok = Tokenizer::new(make_test_vocab());
        let text = "こんにちは世界".as_bytes();
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(&decoded, text);
    }

    #[test]
    fn test_empty_input() {
        let tok = Tokenizer::new(make_test_vocab());
        assert!(tok.encode(b"").is_empty());
        assert!(tok.decode(&[]).is_empty());
    }

    #[test]
    fn test_vocab_size() {
        let tok = Tokenizer::new(make_test_vocab());
        assert_eq!(tok.vocab_size(), 258);
    }

    #[test]
    fn test_merge_applied() {
        let tok = Tokenizer::new(make_test_vocab());
        let ids = tok.encode(b"ab");
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_with_pretokenizer() {
        let vocab = make_test_vocab();
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let tok = Tokenizer::with_config(vocab, Some(pt), SpecialTokens::new());
        let text = b"hello world";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(&decoded, text);
    }

    #[test]
    fn test_with_special_tokens() {
        let vocab = make_test_vocab();
        let mut special = SpecialTokens::new();
        special.add("<|test|>".to_string(), 500);
        let tok = Tokenizer::with_config(vocab, None, special);

        let text = b"hello<|test|>world";
        let ids = tok.encode(text);
        assert!(ids.contains(&500));
        let decoded = tok.decode(&ids);
        assert_eq!(&decoded, text);
    }

    #[test]
    fn test_special_token_policy_none() {
        let vocab = make_test_vocab();
        let mut special = SpecialTokens::new();
        special.add("<|test|>".to_string(), 500);
        let tok = Tokenizer::with_config(vocab, None, special);

        let text = b"hello<|test|>world";
        let ids = tok.encode_with_policy(text, &SpecialTokenPolicy::NoneAllowed);
        // 特殊トークンが通常テキストとしてエンコードされるため500は含まない
        assert!(!ids.contains(&500));
    }

    #[test]
    fn test_special_token_policy_whitelist() {
        let vocab = make_test_vocab();
        let mut special = SpecialTokens::new();
        special.add("<|a|>".to_string(), 500);
        special.add("<|b|>".to_string(), 501);
        let tok = Tokenizer::with_config(vocab, None, special);

        let allowed = vec!["<|a|>".to_string()];
        let text = b"x<|a|>y<|b|>z";
        let ids = tok.encode_with_policy(text, &SpecialTokenPolicy::Whitelist(allowed));
        assert!(ids.contains(&500)); // <|a|> は許可
        assert!(!ids.contains(&501)); // <|b|> は不許可
    }

    #[test]
    fn test_special_token_decode() {
        let vocab = make_test_vocab();
        let mut special = SpecialTokens::new();
        special.add("<|endoftext|>".to_string(), 500);
        let tok = Tokenizer::with_config(vocab, None, special);

        let decoded = tok.decode(&[500]);
        assert_eq!(&decoded, b"<|endoftext|>");
    }

    #[test]
    fn test_full_pipeline() {
        let vocab = make_test_vocab();
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let mut special = SpecialTokens::new();
        special.add("<|end|>".to_string(), 999);
        let tok = Tokenizer::with_config(vocab, Some(pt), special);

        let text = b"hello world<|end|>goodbye";
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(&decoded, text);
    }
}
