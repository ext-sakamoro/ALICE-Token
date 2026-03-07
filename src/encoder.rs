//! エンコーダー — テキスト（バイト列）→ トークンID列
//!
//! BPEアルゴリズムを適用してバイト列をトークンID列に変換する。

use crate::bpe;
use crate::vocab::Vocab;

/// エンコーダー
#[derive(Debug, Clone)]
pub struct Encoder {
    _priv: (),
}

impl Encoder {
    /// 語彙からエンコーダーを構築
    #[must_use]
    pub const fn new(_vocab: &Vocab) -> Self {
        Self { _priv: () }
    }

    /// バイト列をトークンID列にエンコード
    ///
    /// 1. 入力バイト列を初期トークンに変換
    /// 2. BPEマージルールを優先度順に適用
    /// 3. 最終的なトークンID列を返す
    #[inline]
    #[must_use]
    pub fn encode(&self, input: &[u8], vocab: &Vocab) -> Vec<u32> {
        bpe::bpe_encode(input, vocab)
    }

    /// 複数のバイト列を一括エンコード
    #[must_use]
    pub fn encode_batch(&self, inputs: &[&[u8]], vocab: &Vocab) -> Vec<Vec<u32>> {
        inputs
            .iter()
            .map(|input| self.encode(input, vocab))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::byte_level_builder;

    fn make_vocab() -> Vocab {
        let mut builder = byte_level_builder();
        builder.add_merge(b"t", b"h");
        builder.add_merge(b"th", b"e");
        builder.build()
    }

    #[test]
    fn test_encode_basic() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = enc.encode(b"the", &vocab);
        // "t"+"h"→"th", "th"+"e"→"the" = 1トークン
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_encode_no_merge() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = enc.encode(b"xyz", &vocab);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_encode_empty() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = enc.encode(b"", &vocab);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_encode_batch() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let results = enc.encode_batch(&[b"the", b"xyz", b""], &vocab);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].len(), 1); // "the" merged
        assert_eq!(results[1].len(), 3); // no merge
        assert!(results[2].is_empty()); // empty
    }

    #[test]
    fn test_encode_repeated() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = enc.encode(b"thethe", &vocab);
        // "the" × 2 = 2トークン
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_encode_partial_match() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = enc.encode(b"th", &vocab);
        // "t"+"h" → "th" = 1トークン
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_encode_single_byte() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = enc.encode(b"a", &vocab);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_encode_utf8_passthrough() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let input = "漢字".as_bytes();
        let ids = enc.encode(input, &vocab);
        assert_eq!(ids.len(), input.len());
    }
}
