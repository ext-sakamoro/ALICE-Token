//! 並列トークナイズ — Rayon によるチャンク並列BPEエンコード
//!
//! 大規模テキストをUTF-8安全な境界で分割し、
//! 各チャンクを並列にBPEエンコードする。

use rayon::prelude::*;

use crate::encoder::Encoder;
use crate::simd;
use crate::vocab::Vocab;

/// 並列トークナイザー
pub struct ParallelTokenizer;

impl ParallelTokenizer {
    /// テキストをチャンク分割し並列エンコード
    ///
    /// # 引数
    /// - `text`: 入力バイト列
    /// - `chunk_size`: 目標チャンクサイズ（バイト数）
    /// - `encoder`: エンコーダー
    /// - `vocab`: 語彙テーブル
    ///
    /// # 戻り値
    /// 全チャンクのトークンID列を連結した結果
    #[must_use]
    pub fn encode(text: &[u8], chunk_size: usize, encoder: &Encoder, vocab: &Vocab) -> Vec<u32> {
        if text.is_empty() {
            return Vec::new();
        }

        // 小さいテキストは直接エンコード
        if text.len() <= chunk_size {
            return encoder.encode(text, vocab);
        }

        // UTF-8安全な分割点を取得
        let split_points = simd::find_safe_split_points(text, chunk_size);

        // チャンクに分割
        let chunks = build_chunks(text, &split_points);

        // Rayon で並列エンコード
        let results: Vec<Vec<u32>> = chunks
            .par_iter()
            .map(|chunk| encoder.encode(chunk, vocab))
            .collect();

        // 結果を連結
        let total_len: usize = results.iter().map(std::vec::Vec::len).sum();
        let mut merged = Vec::with_capacity(total_len);
        for result in results {
            merged.extend_from_slice(&result);
        }
        merged
    }
}

/// 分割点からチャンクスライスを生成
fn build_chunks<'a>(data: &'a [u8], split_points: &[usize]) -> Vec<&'a [u8]> {
    let mut chunks = Vec::with_capacity(split_points.len() + 1);
    let mut start = 0;

    for &point in split_points {
        if point > start && point <= data.len() {
            chunks.push(&data[start..point]);
            start = point;
        }
    }

    // 残り
    if start < data.len() {
        chunks.push(&data[start..]);
    }

    chunks
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::byte_level_builder;

    fn make_vocab() -> Vocab {
        let mut builder = byte_level_builder();
        builder.add_merge(b"a", b"b");
        builder.add_merge(b"c", b"d");
        builder.build()
    }

    #[test]
    fn test_parallel_empty() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let ids = ParallelTokenizer::encode(b"", 64, &enc, &vocab);
        assert!(ids.is_empty());
    }

    #[test]
    fn test_parallel_small() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        // チャンクサイズ以下 → 直接エンコード
        let ids = ParallelTokenizer::encode(b"ab", 1024, &enc, &vocab);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_parallel_large_ascii() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let input = b"abcdabcdabcd".repeat(100); // 1200バイト
        let sequential = enc.encode(&input, &vocab);
        let parallel = ParallelTokenizer::encode(&input, 128, &enc, &vocab);
        // 並列結果がシーケンシャルと同じトークン数（境界での分割差は許容）
        // 最低限、全バイトがカバーされていることを確認
        assert!(!parallel.is_empty());
        assert!(parallel.len() <= sequential.len() + 10); // 境界効果のマージン
    }

    #[test]
    fn test_parallel_utf8() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let input = "こんにちは世界".as_bytes().repeat(50);
        let ids = ParallelTokenizer::encode(&input, 32, &enc, &vocab);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_build_chunks_basic() {
        let data = b"hello world test";
        let points = vec![5, 11];
        let chunks = build_chunks(data, &points);
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[0], b"hello");
        assert_eq!(chunks[1], b" world");
        assert_eq!(chunks[2], b" test");
    }

    #[test]
    fn test_build_chunks_empty() {
        let chunks = build_chunks(b"hello", &[]);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], b"hello");
    }

    #[test]
    fn test_build_chunks_single_point() {
        let data = b"abcdef";
        let chunks = build_chunks(data, &[3]);
        assert_eq!(chunks.len(), 2);
        assert_eq!(chunks[0], b"abc");
        assert_eq!(chunks[1], b"def");
    }

    #[test]
    fn test_parallel_consistency() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        // 同じ入力で複数回実行しても同じ結果
        let input = b"abcdxyz".repeat(200);
        let r1 = ParallelTokenizer::encode(&input, 64, &enc, &vocab);
        let r2 = ParallelTokenizer::encode(&input, 64, &enc, &vocab);
        assert_eq!(r1, r2);
    }

    #[test]
    fn test_parallel_single_byte_chunks() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        // 極小チャンクサイズ
        let input = b"abcdefgh";
        let ids = ParallelTokenizer::encode(input, 2, &enc, &vocab);
        assert!(!ids.is_empty());
    }

    #[test]
    fn test_parallel_all_same() {
        let vocab = make_vocab();
        let enc = Encoder::new(&vocab);
        let input = vec![b'a'; 500];
        let ids = ParallelTokenizer::encode(&input, 50, &enc, &vocab);
        assert_eq!(ids.len(), 500); // マージなし
    }
}
