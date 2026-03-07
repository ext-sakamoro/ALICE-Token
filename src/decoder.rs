//! デコーダー — トークンID列 → バイト列
//!
//! トークンIDを語彙テーブルで逆引きし、元のバイト列を復元する。

use crate::vocab::Vocab;

/// デコーダー
///
/// 語彙テーブルの `id_to_token` マッピングを保持し、
/// トークンID列をバイト列に復元する。
#[derive(Debug, Clone)]
pub struct Decoder {
    /// トークンID → バイト列のマッピング（Vocabからコピー）
    id_to_bytes: Vec<Vec<u8>>,
}

impl Decoder {
    /// 語彙からデコーダーを構築
    #[must_use]
    pub fn new(vocab: &Vocab) -> Self {
        let mut id_to_bytes = Vec::with_capacity(vocab.len());
        for id in 0..vocab.len() as u32 {
            if let Some(token) = vocab.get_token(id) {
                id_to_bytes.push(token.to_vec());
            }
        }
        Self { id_to_bytes }
    }

    /// トークンID列をバイト列にデコード
    ///
    /// 各トークンIDに対応するバイト列を連結して返す。
    /// 不明なIDは空バイト列として扱う。
    #[inline]
    #[must_use]
    pub fn decode(&self, ids: &[u32]) -> Vec<u8> {
        let mut result = Vec::with_capacity(ids.len() * 4);
        for &id in ids {
            if let Some(bytes) = self.id_to_bytes.get(id as usize) {
                result.extend_from_slice(bytes);
            }
        }
        result
    }

    /// 単一トークンIDをバイト列にデコード
    #[inline]
    #[must_use]
    pub fn decode_single(&self, id: u32) -> Option<&[u8]> {
        self.id_to_bytes
            .get(id as usize)
            .map(std::vec::Vec::as_slice)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::byte_level_builder;

    fn make_vocab() -> Vocab {
        let mut builder = byte_level_builder();
        builder.add_merge(b"h", b"e");
        builder.add_merge(b"l", b"l");
        builder.add_merge(b"he", b"ll");
        builder.add_merge(b"hell", b"o");
        builder.build()
    }

    #[test]
    fn test_decode_single_byte() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        let result = dec.decode(&[b'A' as u32]);
        assert_eq!(result, b"A");
    }

    #[test]
    fn test_decode_merged_token() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        let hello_id = vocab.get_id(b"hello").unwrap();
        let result = dec.decode(&[hello_id]);
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_decode_empty() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        let result = dec.decode(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_unknown_id() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        // 存在しないID → スキップ
        let result = dec.decode(&[99999]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_decode_multiple() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        let he_id = vocab.get_id(b"he").unwrap();
        let ll_id = vocab.get_id(b"ll").unwrap();
        let o_id = vocab.get_id(b"o").unwrap();
        let result = dec.decode(&[he_id, ll_id, o_id]);
        assert_eq!(result, b"hello");
    }

    #[test]
    fn test_decode_single_fn() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        let he_id = vocab.get_id(b"he").unwrap();
        assert_eq!(dec.decode_single(he_id), Some(b"he".as_slice()));
    }

    #[test]
    fn test_decode_single_unknown() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        assert!(dec.decode_single(99999).is_none());
    }

    #[test]
    fn test_roundtrip() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        let hello_id = vocab.get_id(b"hello").unwrap();
        let space_id = vocab.get_id(b" ").unwrap();
        // "hello hello"
        let result = dec.decode(&[hello_id, space_id, hello_id]);
        assert_eq!(result, b"hello hello");
    }

    #[test]
    fn test_decode_byte_range() {
        let vocab = make_vocab();
        let dec = Decoder::new(&vocab);
        // バイト 0-255 が正しくデコードされるか
        for b in 0u32..=255 {
            let result = dec.decode(&[b]);
            assert_eq!(result.len(), 1);
            assert_eq!(result[0], b as u8);
        }
    }
}
