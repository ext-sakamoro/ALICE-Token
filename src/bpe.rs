//! BPE（Byte Pair Encoding）アルゴリズム本体
//!
//! 優先度キュー方式で O(n log n) のマージを実現。
//! tiktoken の greedy 方式とは異なり、ペアのランク（優先度）に基づいて
//! 最小ランクのペアから順にマージする。

use std::cmp::Reverse;
use std::collections::BinaryHeap;

use crate::vocab::Vocab;

/// BPEマージ操作の記録
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BpeMerge {
    /// マージ位置（トークン列内のインデックス）
    pub position: usize,
    /// マージ前の左トークンID
    pub left: u32,
    /// マージ前の右トークンID
    pub right: u32,
    /// マージ後のトークンID
    pub merged: u32,
}

/// 優先度キュー内のエントリ
#[derive(Debug, Clone, Eq, PartialEq)]
struct MergeCandidate {
    rank: u32,
    position: usize,
    left: u32,
    right: u32,
    merged: u32,
}

impl Ord for MergeCandidate {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse: 低ランク = 高優先度
        self.rank
            .cmp(&other.rank)
            .then_with(|| self.position.cmp(&other.position))
    }
}

impl PartialOrd for MergeCandidate {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

/// BPEエンコーディングをバイト列に適用
///
/// 1. 各バイトを初期トークンIDに変換
/// 2. 隣接ペアをスキャンし、マージ可能なペアを優先度キューに投入
/// 3. 最小ランクのペアからマージを繰り返す
///
/// # 戻り値
/// マージ適用後のトークンID列
pub fn bpe_encode(input: &[u8], vocab: &Vocab) -> Vec<u32> {
    if input.is_empty() {
        return Vec::new();
    }

    // Step 1: 各バイトを初期トークンIDに変換
    let mut tokens: Vec<u32> = input
        .iter()
        .map(|&b| vocab.get_id(&[b]).unwrap_or(0))
        .collect();

    if tokens.len() <= 1 {
        return tokens;
    }

    // 連結リスト的な管理（削除マーク用）
    // next[i] = i の次の有効トークンのインデックス
    let n = tokens.len();
    let mut next: Vec<usize> = (1..=n).collect(); // next[i] = i+1, next[n-1] = n (sentinel)
    let mut prev: Vec<usize> = Vec::with_capacity(n);
    prev.push(usize::MAX); // prev[0] = sentinel
    for i in 1..n {
        prev.push(i - 1);
    }

    // Step 2: 初期ペアをキューに投入
    let mut heap: BinaryHeap<Reverse<MergeCandidate>> = BinaryHeap::new();

    for i in 0..n - 1 {
        let j = next[i];
        if j < n {
            if let Some(rule) = vocab.get_merge(tokens[i], tokens[j]) {
                heap.push(Reverse(MergeCandidate {
                    rank: rule.rank,
                    position: i,
                    left: tokens[i],
                    right: tokens[j],
                    merged: rule.merged,
                }));
            }
        }
    }

    // Step 3: マージループ
    // deleted[i] = true ならそのスロットは既にマージで消された
    let mut deleted = vec![false; n];

    while let Some(Reverse(candidate)) = heap.pop() {
        let i = candidate.position;

        // 無効化チェック: 既に削除済み or トークンが変わっている
        if deleted[i] {
            continue;
        }
        let j = next[i];
        if j >= n || deleted[j] {
            continue;
        }
        if tokens[i] != candidate.left || tokens[j] != candidate.right {
            continue;
        }

        // マージ実行: tokens[i] = merged, tokens[j] を削除
        tokens[i] = candidate.merged;
        deleted[j] = true;

        // リンク更新
        let k = next[j]; // j の次
        next[i] = k;
        if k < n {
            prev[k] = i;
        }

        // 新しいペア (prev[i], i) をチェック
        if prev[i] < n && !deleted[prev[i]] {
            let pi = prev[i];
            if let Some(rule) = vocab.get_merge(tokens[pi], tokens[i]) {
                heap.push(Reverse(MergeCandidate {
                    rank: rule.rank,
                    position: pi,
                    left: tokens[pi],
                    right: tokens[i],
                    merged: rule.merged,
                }));
            }
        }

        // 新しいペア (i, next[i]) をチェック
        if next[i] < n && !deleted[next[i]] {
            let ni = next[i];
            if let Some(rule) = vocab.get_merge(tokens[i], tokens[ni]) {
                heap.push(Reverse(MergeCandidate {
                    rank: rule.rank,
                    position: i,
                    left: tokens[i],
                    right: tokens[ni],
                    merged: rule.merged,
                }));
            }
        }
    }

    // Step 4: 削除されていないトークンを収集
    let mut result = Vec::with_capacity(n);
    let mut i = 0;
    while i < n {
        if !deleted[i] {
            result.push(tokens[i]);
        }
        i = if next[i] > i { next[i] } else { i + 1 };
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vocab::byte_level_builder;

    fn test_vocab() -> Vocab {
        let mut builder = byte_level_builder();
        // rank 0: "a"+"b" → "ab" (id=256)
        builder.add_merge(b"a", b"b");
        // rank 1: "c"+"d" → "cd" (id=257)
        builder.add_merge(b"c", b"d");
        // rank 2: "ab"+"cd" → "abcd" (id=258)
        builder.add_merge(b"ab", b"cd");
        builder.build()
    }

    #[test]
    fn test_empty() {
        let vocab = test_vocab();
        assert!(bpe_encode(b"", &vocab).is_empty());
    }

    #[test]
    fn test_single_byte() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"x", &vocab);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_no_merge() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"xyz", &vocab);
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_simple_merge() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"ab", &vocab);
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], vocab.get_id(b"ab").unwrap());
    }

    #[test]
    fn test_two_merges() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"abcd", &vocab);
        // "ab" + "cd" → "abcd" (3段階マージ)
        assert_eq!(ids.len(), 1);
        assert_eq!(ids[0], vocab.get_id(b"abcd").unwrap());
    }

    #[test]
    fn test_partial_merge() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"abx", &vocab);
        // "ab" はマージされるが "x" はそのまま
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_merge_in_context() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"xaby", &vocab);
        // "x" + "ab"(merged) + "y" = 3トークン
        assert_eq!(ids.len(), 3);
    }

    #[test]
    fn test_repeated_merge() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"ababab", &vocab);
        // "ab" が3回マージ → 3トークン
        assert_eq!(ids.len(), 3);
        let ab_id = vocab.get_id(b"ab").unwrap();
        assert!(ids.iter().all(|&id| id == ab_id));
    }

    #[test]
    fn test_rank_priority() {
        // rank 0 の "ab" が rank 1 の "cd" より先にマージされる
        let vocab = test_vocab();
        let ids = bpe_encode(b"cdab", &vocab);
        assert_eq!(ids.len(), 2);
    }

    #[test]
    fn test_chained_merge() {
        let vocab = test_vocab();
        // "abcd" → "ab"+"cd" → "abcd"
        let ids = bpe_encode(b"abcd", &vocab);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_long_input() {
        let vocab = test_vocab();
        let input = b"abcdabcdxyzabcd";
        let ids = bpe_encode(input, &vocab);
        // "abcd" × 3 + "xyz" = 3 + 3 = 6 トークン
        assert!(ids.len() <= 6);
    }

    #[test]
    fn test_all_same_byte() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"aaaa", &vocab);
        assert_eq!(ids.len(), 4); // "a" は自身とマージしない
    }

    #[test]
    fn test_utf8_bytes() {
        let vocab = test_vocab();
        let input = "日本語".as_bytes();
        let ids = bpe_encode(input, &vocab);
        assert_eq!(ids.len(), input.len()); // マージルールなし → バイト数 = トークン数
    }

    #[test]
    fn test_two_bytes() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"ab", &vocab);
        assert_eq!(ids.len(), 1);
    }

    #[test]
    fn test_preserves_order() {
        let vocab = test_vocab();
        let ids = bpe_encode(b"xabcdy", &vocab);
        // x + abcd + y = 3 tokens, 順序保持
        assert_eq!(ids.len(), 3);
        let x_id = vocab.get_id(b"x").unwrap();
        let y_id = vocab.get_id(b"y").unwrap();
        assert_eq!(ids[0], x_id);
        assert_eq!(ids[2], y_id);
    }
}
