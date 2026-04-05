//! BPEトレーニング
//!
//! コーパスからBPEマージルールを学習する。
//! バイトレベルBPE: 初期語彙は0-255の各バイト。
//! 最頻出の隣接ペアを繰り返しマージしてマージルールを生成する。

use rustc_hash::FxHashMap;

use crate::vocab::{byte_level_builder, Vocab};

/// BPEトレーナー設定
#[derive(Debug, Clone)]
pub struct TrainerConfig {
    /// 目標語彙サイズ（バイトレベル256 + マージ数）
    pub vocab_size: usize,
    /// 最小ペア出現頻度（これ未満のペアはマージしない）
    pub min_frequency: u64,
}

impl Default for TrainerConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32_768,
            min_frequency: 2,
        }
    }
}

/// BPEトレーナー
///
/// コーパスからバイトペアエンコーディングのマージルールを学習する。
pub struct Trainer {
    config: TrainerConfig,
}

impl Trainer {
    /// 設定からトレーナーを構築
    #[must_use]
    pub const fn new(config: TrainerConfig) -> Self {
        Self { config }
    }

    /// コーパスからBPE語彙を学習
    ///
    /// 1. 各単語をバイト列に変換
    /// 2. 最頻出隣接ペアを検出
    /// 3. ペアをマージして新トークンを追加
    /// 4. 目標語彙サイズに達するまで繰り返す
    #[must_use]
    pub fn train(&self, corpus: &[&[u8]]) -> Vocab {
        // 単語→出現頻度のマッピング
        let mut word_freqs: FxHashMap<Vec<Vec<u8>>, u64> = FxHashMap::default();

        // 各入力を単語（バイト列のシーケンス）に変換
        for &text in corpus {
            let word: Vec<Vec<u8>> = text.iter().map(|&b| vec![b]).collect();
            if !word.is_empty() {
                *word_freqs.entry(word).or_insert(0) += 1;
            }
        }

        let mut builder = byte_level_builder();
        let max_merges = self.config.vocab_size.saturating_sub(256);

        for _ in 0..max_merges {
            // 隣接ペアの頻度をカウント
            let pair_counts = count_pairs(&word_freqs);

            // 最頻出ペアを所有権ごと取り出す（clone 不要）
            let best_pair = pair_counts
                .into_iter()
                .max_by_key(|(_, count)| *count)
                .filter(|(_, count)| *count >= self.config.min_frequency);

            let Some(((left, right), _)) = best_pair else {
                break;
            };

            // マージルールを追加
            builder.add_merge(&left, &right);

            // 全単語でペアをマージ
            let mut new_freqs = FxHashMap::default();
            for (word, freq) in &word_freqs {
                let merged = merge_pair(word, &left, &right);
                *new_freqs.entry(merged).or_insert(0) += freq;
            }
            word_freqs = new_freqs;
        }

        builder.build()
    }
}

/// 隣接ペアの出現頻度をカウント
fn count_pairs(word_freqs: &FxHashMap<Vec<Vec<u8>>, u64>) -> FxHashMap<(Vec<u8>, Vec<u8>), u64> {
    let mut counts = FxHashMap::default();

    for (word, &freq) in word_freqs {
        if word.len() < 2 {
            continue;
        }
        for i in 0..word.len() - 1 {
            let pair = (word[i].clone(), word[i + 1].clone());
            *counts.entry(pair).or_insert(0) += freq;
        }
    }

    counts
}

/// 単語内の指定ペアをマージ
fn merge_pair(word: &[Vec<u8>], left: &[u8], right: &[u8]) -> Vec<Vec<u8>> {
    let mut result = Vec::with_capacity(word.len());
    let mut i = 0;

    while i < word.len() {
        if i + 1 < word.len() && word[i] == left && word[i + 1] == right {
            // マージ: left + right → concatenated
            let mut merged = left.to_vec();
            merged.extend_from_slice(right);
            result.push(merged);
            i += 2;
        } else {
            result.push(word[i].clone());
            i += 1;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Tokenizer;

    #[test]
    fn test_train_basic() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 260,
            min_frequency: 2,
        });

        let corpus: Vec<&[u8]> = vec![b"aaabdaaabac", b"aabdaabac", b"aaabdaaabac"];
        let vocab = trainer.train(&corpus);

        // 256バイト + いくつかのマージ
        assert!(vocab.len() > 256);
        assert!(vocab.merge_count() > 0);
    }

    #[test]
    fn test_train_empty_corpus() {
        let trainer = Trainer::new(TrainerConfig::default());
        let corpus: Vec<&[u8]> = vec![];
        let vocab = trainer.train(&corpus);
        // バイトレベル語彙のみ
        assert_eq!(vocab.len(), 256);
    }

    #[test]
    fn test_train_single_byte() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 257,
            min_frequency: 1,
        });
        let corpus: Vec<&[u8]> = vec![b"a"];
        let vocab = trainer.train(&corpus);
        assert_eq!(vocab.len(), 256); // 単一バイト → マージなし
    }

    #[test]
    fn test_train_frequent_pair() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 258,
            min_frequency: 1,
        });

        // "ab" が最頻出ペア
        let corpus: Vec<&[u8]> = vec![b"ababab", b"ababab", b"ababab"];
        let vocab = trainer.train(&corpus);
        // "ab" がマージされているはず
        assert!(vocab.get_id(b"ab").is_some());
    }

    #[test]
    fn test_train_min_frequency_filter() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 300,
            min_frequency: 100,
        });

        // 各ペアが1回しか出現しない → min_frequency=100 でフィルタされる
        let corpus: Vec<&[u8]> = vec![b"abcdefgh"];
        let vocab = trainer.train(&corpus);
        assert_eq!(vocab.len(), 256); // マージなし
    }

    #[test]
    fn test_train_roundtrip() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 280,
            min_frequency: 2,
        });

        let text = b"the quick brown fox jumps over the lazy dog";
        let corpus: Vec<&[u8]> = std::iter::repeat_n(text.as_slice(), 10).collect();
        let vocab = trainer.train(&corpus);

        let tok = Tokenizer::new(vocab);
        let ids = tok.encode(text);
        let decoded = tok.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn test_train_utf8() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 280,
            min_frequency: 2,
        });

        let text = "こんにちは".as_bytes();
        let corpus: Vec<&[u8]> = std::iter::repeat_n(text, 10).collect();
        let vocab = trainer.train(&corpus);

        // UTF-8バイト列のマージが学習されている
        assert!(vocab.len() > 256);
    }

    #[test]
    fn test_count_pairs() {
        let mut freqs = FxHashMap::default();
        freqs.insert(vec![vec![b'a'], vec![b'b'], vec![b'c']], 3);
        let counts = count_pairs(&freqs);
        assert_eq!(counts.get(&(vec![b'a'], vec![b'b'])), Some(&3));
        assert_eq!(counts.get(&(vec![b'b'], vec![b'c'])), Some(&3));
    }

    #[test]
    fn test_merge_pair() {
        let word = vec![vec![b'a'], vec![b'b'], vec![b'c'], vec![b'a'], vec![b'b']];
        let result = merge_pair(&word, b"a", b"b");
        // "ab" + "c" + "ab" = 3要素
        assert_eq!(result.len(), 3);
        assert_eq!(result[0], vec![b'a', b'b']);
        assert_eq!(result[1], vec![b'c']);
        assert_eq!(result[2], vec![b'a', b'b']);
    }

    #[test]
    fn test_merge_pair_no_match() {
        let word = vec![vec![b'x'], vec![b'y']];
        let result = merge_pair(&word, b"a", b"b");
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_trainer_default_config() {
        let config = TrainerConfig::default();
        assert_eq!(config.vocab_size, 32_768);
        assert_eq!(config.min_frequency, 2);
    }

    #[test]
    fn test_train_respects_vocab_size() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 258, // 256 + 2 merges max
            min_frequency: 1,
        });

        let corpus: Vec<&[u8]> = vec![b"abababcdcdcd"; 10];
        let vocab = trainer.train(&corpus);
        // 最大258トークン
        assert!(vocab.len() <= 258);
    }

    #[test]
    fn test_train_large_corpus() {
        let trainer = Trainer::new(TrainerConfig {
            vocab_size: 270,
            min_frequency: 2,
        });

        let text = b"the quick brown fox jumps over the lazy dog ";
        let corpus: Vec<&[u8]> = std::iter::repeat_n(text.as_slice(), 100).collect();
        let vocab = trainer.train(&corpus);

        assert!(vocab.len() > 256);
        assert!(vocab.merge_count() > 0);
    }
}
