//! 語彙ファイル I/O
//!
//! tiktoken `.tiktoken` 形式、バイナリ形式の読み書きを提供。

use base64::Engine;
use rustc_hash::FxHashMap;
use serde::{Deserialize, Serialize};

use crate::vocab::{byte_level_builder, Vocab, VocabBuilder};

/// tiktoken形式の語彙ファイル（.tiktoken）を読み込む
///
/// フォーマット: 各行が `base64(token_bytes) rank` の形式
///
/// # Errors
/// パース失敗時にエラーを返す。
pub fn load_tiktoken(data: &str) -> Result<Vocab, TokenIoError> {
    let mut builder = byte_level_builder();
    let mut ranks: Vec<(Vec<u8>, usize)> = Vec::new();

    for (line_no, line) in data.lines().enumerate() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }

        let mut parts = line.split_whitespace();
        let b64 = parts
            .next()
            .ok_or_else(|| TokenIoError::Parse(format!("line {}: missing base64", line_no + 1)))?;
        let rank_str = parts
            .next()
            .ok_or_else(|| TokenIoError::Parse(format!("line {}: missing rank", line_no + 1)))?;

        let token_bytes = base64::engine::general_purpose::STANDARD
            .decode(b64)
            .map_err(|e| TokenIoError::Parse(format!("line {}: base64: {e}", line_no + 1)))?;
        let rank: usize = rank_str
            .parse()
            .map_err(|e| TokenIoError::Parse(format!("line {}: rank: {e}", line_no + 1)))?;

        builder.add_token(token_bytes.clone());
        ranks.push((token_bytes, rank));
    }

    // ランクからマージルールを再構成
    // ランク順にソートし、マルチバイトトークンをマージとして登録
    ranks.sort_by_key(|(_, r)| *r);

    // 既存トークンのバイト列→IDマッピングを構築（マージルール再構成用）
    let temp_vocab = builder.build();
    let mut token_map: FxHashMap<Vec<u8>, u32> = FxHashMap::default();
    for (token_bytes, _) in &ranks {
        if let Some(id) = temp_vocab.get_id(token_bytes) {
            token_map.insert(token_bytes.clone(), id);
        }
    }

    // マージルール再構成: マルチバイトトークンを左右に分割
    // ranks を消費して builder2 に移動し、clone を排除する
    let mut builder2 = byte_level_builder();
    for (token_bytes, _) in ranks.iter() {
        // 分割点の探索は token_map への参照で完結するため先にスキャンする
        if token_bytes.len() >= 2 {
            for split_pos in 1..token_bytes.len() {
                let left = &token_bytes[..split_pos];
                let right = &token_bytes[split_pos..];
                if token_map.contains_key(left) && token_map.contains_key(right) {
                    builder2.add_merge(left, right);
                    break;
                }
            }
        }
    }
    // トークンは所有権ごと builder2 に渡して clone を回避する
    for (token_bytes, _) in ranks {
        builder2.add_token(token_bytes);
    }

    Ok(builder2.build())
}

/// tiktoken形式で語彙を文字列にエクスポート
#[must_use]
pub fn save_tiktoken(vocab: &Vocab) -> String {
    let mut lines = Vec::with_capacity(vocab.len());
    for id in 0..vocab.len() as u32 {
        if let Some(token_bytes) = vocab.get_token(id) {
            let b64 = base64::engine::general_purpose::STANDARD.encode(token_bytes);
            lines.push(format!("{b64} {id}"));
        }
    }
    lines.join("\n")
}

/// バイナリシリアライズ用の中間構造体
#[derive(Serialize, Deserialize)]
struct VocabBinary {
    tokens: Vec<Vec<u8>>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
}

/// 語彙をバイナリ形式（bincode）にシリアライズ
///
/// # Errors
/// シリアライズ失敗時にエラーを返す。
pub fn save_binary(vocab: &Vocab) -> Result<Vec<u8>, TokenIoError> {
    let mut tokens = Vec::with_capacity(vocab.len());
    for id in 0..vocab.len() as u32 {
        if let Some(t) = vocab.get_token(id) {
            tokens.push(t.to_vec());
        }
    }

    let mut merges = Vec::new();
    for &(left_id, right_id) in vocab.merge_list() {
        if let (Some(left), Some(right)) = (vocab.get_token(left_id), vocab.get_token(right_id)) {
            merges.push((left.to_vec(), right.to_vec()));
        }
    }

    let binary = VocabBinary { tokens, merges };
    bincode::serialize(&binary).map_err(|e| TokenIoError::Serialize(e.to_string()))
}

/// バイナリ形式から語彙をデシリアライズ
///
/// # Errors
/// デシリアライズ失敗時にエラーを返す。
pub fn load_binary(data: &[u8]) -> Result<Vocab, TokenIoError> {
    let binary: VocabBinary =
        bincode::deserialize(data).map_err(|e| TokenIoError::Deserialize(e.to_string()))?;

    let mut builder = VocabBuilder::new();
    for (left, right) in &binary.merges {
        builder.add_merge(left, right);
    }
    for token in binary.tokens {
        builder.add_token(token);
    }

    Ok(builder.build())
}

/// ファイルI/Oエラー
#[derive(Debug, Clone)]
pub enum TokenIoError {
    /// パースエラー
    Parse(String),
    /// シリアライズエラー
    Serialize(String),
    /// デシリアライズエラー
    Deserialize(String),
}

impl std::fmt::Display for TokenIoError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Parse(s) => write!(f, "parse error: {s}"),
            Self::Serialize(s) => write!(f, "serialize error: {s}"),
            Self::Deserialize(s) => write!(f, "deserialize error: {s}"),
        }
    }
}

impl std::error::Error for TokenIoError {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_tiktoken_roundtrip() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"h", b"e");
        builder.add_merge(b"l", b"l");
        let vocab = builder.build();

        let exported = save_tiktoken(&vocab);
        assert!(!exported.is_empty());

        // 再読み込みで語彙サイズが一致
        let reloaded = load_tiktoken(&exported).unwrap();
        assert_eq!(reloaded.len(), vocab.len());
    }

    #[test]
    fn test_tiktoken_parse_basic() {
        // "hello" = aGVsbG8= rank 0
        // "world" = d29ybGQ= rank 1
        let data = "aGVsbG8= 0\nd29ybGQ= 1\n";
        let vocab = load_tiktoken(data).unwrap();
        assert!(vocab.get_id(b"hello").is_some());
        assert!(vocab.get_id(b"world").is_some());
    }

    #[test]
    fn test_tiktoken_empty() {
        let vocab = load_tiktoken("").unwrap();
        // バイトレベル語彙のみ
        assert_eq!(vocab.len(), 256);
    }

    #[test]
    fn test_tiktoken_skip_empty_lines() {
        let data = "\n\naGVsbG8= 0\n\n";
        let vocab = load_tiktoken(data).unwrap();
        assert!(vocab.get_id(b"hello").is_some());
    }

    #[test]
    fn test_tiktoken_invalid_base64() {
        let data = "!!!invalid 0\n";
        assert!(load_tiktoken(data).is_err());
    }

    #[test]
    fn test_tiktoken_missing_rank() {
        let data = "aGVsbG8=\n";
        assert!(load_tiktoken(data).is_err());
    }

    #[test]
    fn test_binary_roundtrip() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"a", b"b");
        builder.add_merge(b"c", b"d");
        builder.add_merge(b"ab", b"cd");
        let vocab = builder.build();

        let binary = save_binary(&vocab).unwrap();
        let reloaded = load_binary(&binary).unwrap();

        assert_eq!(reloaded.len(), vocab.len());
        assert_eq!(reloaded.merge_count(), vocab.merge_count());

        // トークンの一致確認
        for id in 0..vocab.len() as u32 {
            assert_eq!(vocab.get_token(id), reloaded.get_token(id));
        }
    }

    #[test]
    fn test_binary_empty_vocab() {
        let vocab = VocabBuilder::new().build();
        let binary = save_binary(&vocab).unwrap();
        let reloaded = load_binary(&binary).unwrap();
        assert!(reloaded.is_empty());
    }

    #[test]
    fn test_binary_invalid_data() {
        assert!(load_binary(b"not valid bincode").is_err());
    }

    #[test]
    fn test_binary_large_vocab() {
        let mut builder = byte_level_builder();
        // 1000個のマージルール
        for i in 0u8..250 {
            let j = i.wrapping_add(1);
            builder.add_merge(&[i], &[j]);
        }
        let vocab = builder.build();

        let binary = save_binary(&vocab).unwrap();
        let reloaded = load_binary(&binary).unwrap();
        assert_eq!(reloaded.len(), vocab.len());
    }

    #[test]
    fn test_error_display() {
        let e = TokenIoError::Parse("test".to_string());
        assert_eq!(format!("{e}"), "parse error: test");
    }

    #[test]
    fn test_tiktoken_single_byte_tokens() {
        // 単一バイトトークンのみ
        let data = "AA== 0\nAQ== 1\n"; // 0x00, 0x01
        let vocab = load_tiktoken(data).unwrap();
        assert!(vocab.get_id(&[0x00]).is_some());
        assert!(vocab.get_id(&[0x01]).is_some());
    }

    #[test]
    fn test_save_tiktoken_preserves_all() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"t", b"h");
        builder.add_merge(b"th", b"e");
        let vocab = builder.build();

        let exported = save_tiktoken(&vocab);
        // 全トークンが出力されているか
        let line_count = exported.lines().count();
        assert_eq!(line_count, vocab.len());
    }
}
