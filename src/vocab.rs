//! 語彙テーブル — `FxHashMap` ベースの O(1) トークン検索
//!
//! バイト列 → トークンID、トークンID → バイト列の双方向マッピングを提供。
//! BPEマージルールも保持する。

use rustc_hash::FxHashMap;

/// BPEマージルール（ペア → マージ後トークンID）
#[derive(Debug, Clone)]
pub struct BpeMergeRule {
    /// マージ元の左トークンID
    pub left: u32,
    /// マージ元の右トークンID
    pub right: u32,
    /// マージ後のトークンID
    pub merged: u32,
    /// マージ優先度（低い = 高優先）
    pub rank: u32,
}

/// 語彙テーブル
///
/// - `token_to_id`: バイト列 → トークンID（エンコード用）
/// - `id_to_token`: トークンID → バイト列（デコード用）
/// - `merges`: BPEマージルール（ペアキー → マージルール）
#[derive(Debug, Clone)]
pub struct Vocab {
    token_to_id: FxHashMap<Vec<u8>, u32>,
    id_to_token: Vec<Vec<u8>>,
    /// (`left_id`, `right_id`) → `BpeMergeRule`
    merges: FxHashMap<u64, BpeMergeRule>,
    merge_list: Vec<(u32, u32)>,
}

/// ペアキーの生成（2つのu32を1つのu64に結合）
#[inline(always)]
fn pair_key(left: u32, right: u32) -> u64 {
    (u64::from(left) << 32) | u64::from(right)
}

impl Vocab {
    /// 語彙サイズ
    #[inline]
    #[must_use]
    pub const fn len(&self) -> usize {
        self.id_to_token.len()
    }

    /// 語彙が空か
    #[inline]
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.id_to_token.is_empty()
    }

    /// バイト列からトークンIDを検索
    #[inline]
    #[must_use]
    pub fn get_id(&self, token: &[u8]) -> Option<u32> {
        self.token_to_id.get(token).copied()
    }

    /// トークンIDからバイト列を取得
    #[inline]
    #[must_use]
    pub fn get_token(&self, id: u32) -> Option<&[u8]> {
        self.id_to_token
            .get(id as usize)
            .map(std::vec::Vec::as_slice)
    }

    /// マージルールを検索
    #[inline]
    #[must_use]
    pub fn get_merge(&self, left: u32, right: u32) -> Option<&BpeMergeRule> {
        self.merges.get(&pair_key(left, right))
    }

    /// マージルール一覧（優先度順）
    #[inline]
    #[must_use]
    pub fn merge_list(&self) -> &[(u32, u32)] {
        &self.merge_list
    }

    /// マージルール数
    #[inline]
    #[must_use]
    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }
}

/// 語彙ビルダー
pub struct VocabBuilder {
    tokens: Vec<Vec<u8>>,
    token_to_id: FxHashMap<Vec<u8>, u32>,
    merges: Vec<(Vec<u8>, Vec<u8>)>,
}

impl VocabBuilder {
    /// 新規ビルダー
    #[must_use]
    pub fn new() -> Self {
        Self {
            tokens: Vec::new(),
            token_to_id: FxHashMap::default(),
            merges: Vec::new(),
        }
    }

    /// トークンを追加（IDは自動採番）
    pub fn add_token(&mut self, token: Vec<u8>) -> u32 {
        if let Some(&existing) = self.token_to_id.get(&token) {
            return existing;
        }
        let id = self.tokens.len() as u32;
        self.token_to_id.insert(token.clone(), id);
        self.tokens.push(token);
        id
    }

    /// BPEマージルールを追加
    pub fn add_merge(&mut self, left: &[u8], right: &[u8]) {
        self.merges.push((left.to_vec(), right.to_vec()));
    }

    /// 語彙を構築
    #[must_use]
    pub fn build(mut self) -> Vocab {
        let mut merge_map = FxHashMap::default();
        let mut merge_list = Vec::with_capacity(self.merges.len());

        // イテレーション中の借用問題を回避するためクローン
        let merges_snapshot = self.merges.clone();

        for (rank, (left_bytes, right_bytes)) in merges_snapshot.iter().enumerate() {
            let Some(&left_id) = self.token_to_id.get(left_bytes.as_slice()) else {
                continue;
            };
            let Some(&right_id) = self.token_to_id.get(right_bytes.as_slice()) else {
                continue;
            };

            // マージ後のバイト列を生成
            let mut merged_bytes = left_bytes.clone();
            merged_bytes.extend_from_slice(right_bytes);

            let merged_id = self.add_token(merged_bytes);

            let key = pair_key(left_id, right_id);
            merge_map.insert(
                key,
                BpeMergeRule {
                    left: left_id,
                    right: right_id,
                    merged: merged_id,
                    rank: rank as u32,
                },
            );
            merge_list.push((left_id, right_id));
        }

        Vocab {
            token_to_id: self.token_to_id,
            id_to_token: self.tokens,
            merges: merge_map,
            merge_list,
        }
    }
}

impl Default for VocabBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// バイトレベル基本語彙（0-255）を持つビルダーを生成
#[must_use]
pub fn byte_level_builder() -> VocabBuilder {
    let mut builder = VocabBuilder::new();
    for b in 0u16..=255 {
        builder.add_token(vec![b as u8]);
    }
    builder
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_byte_level_vocab() {
        let builder = byte_level_builder();
        let vocab = builder.build();
        assert_eq!(vocab.len(), 256);
    }

    #[test]
    fn test_add_token_dedup() {
        let mut builder = VocabBuilder::new();
        let id1 = builder.add_token(vec![0x41]);
        let id2 = builder.add_token(vec![0x41]);
        assert_eq!(id1, id2);
    }

    #[test]
    fn test_get_id_roundtrip() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"h", b"e");
        let vocab = builder.build();
        let id = vocab.get_id(b"he").unwrap();
        let token = vocab.get_token(id).unwrap();
        assert_eq!(token, b"he");
    }

    #[test]
    fn test_merge_rule_lookup() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"a", b"b");
        let vocab = builder.build();
        let a_id = vocab.get_id(b"a").unwrap();
        let b_id = vocab.get_id(b"b").unwrap();
        let merge = vocab.get_merge(a_id, b_id).unwrap();
        assert_eq!(merge.rank, 0);
        assert_eq!(vocab.get_token(merge.merged).unwrap(), b"ab");
    }

    #[test]
    fn test_merge_count() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"a", b"b");
        builder.add_merge(b"c", b"d");
        let vocab = builder.build();
        assert_eq!(vocab.merge_count(), 2);
    }

    #[test]
    fn test_merge_list_order() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"x", b"y");
        builder.add_merge(b"a", b"b");
        let vocab = builder.build();
        let list = vocab.merge_list();
        assert_eq!(list.len(), 2);
        // 最初のマージ: x+y
        let merge0 = vocab.get_merge(list[0].0, list[0].1).unwrap();
        assert_eq!(merge0.rank, 0);
    }

    #[test]
    fn test_empty_vocab() {
        let builder = VocabBuilder::new();
        let vocab = builder.build();
        assert!(vocab.is_empty());
        assert_eq!(vocab.len(), 0);
    }

    #[test]
    fn test_get_nonexistent_id() {
        let vocab = byte_level_builder().build();
        assert!(vocab.get_id(b"nonexistent").is_none());
    }

    #[test]
    fn test_get_nonexistent_token() {
        let vocab = byte_level_builder().build();
        assert!(vocab.get_token(9999).is_none());
    }

    #[test]
    fn test_pair_key_no_collision() {
        // 異なるペアが異なるキーを生成することを検証
        let k1 = pair_key(1, 2);
        let k2 = pair_key(2, 1);
        let k3 = pair_key(1, 3);
        assert_ne!(k1, k2);
        assert_ne!(k1, k3);
        assert_ne!(k2, k3);
    }

    #[test]
    fn test_multibyte_token() {
        let mut builder = VocabBuilder::new();
        let id = builder.add_token("あ".as_bytes().to_vec());
        let vocab = builder.build();
        assert_eq!(vocab.get_id("あ".as_bytes()), Some(id));
    }

    #[test]
    fn test_chained_merges() {
        let mut builder = byte_level_builder();
        builder.add_merge(b"a", b"b"); // "ab" → 256
        builder.add_merge(b"ab", b"c"); // "abc" → 257
        let vocab = builder.build();
        assert_eq!(vocab.len(), 258); // 256 bytes + "ab" + "abc"
        assert!(vocab.get_id(b"abc").is_some());
    }

    #[test]
    fn test_default_builder() {
        let builder = VocabBuilder::default();
        let vocab = builder.build();
        assert!(vocab.is_empty());
    }
}
