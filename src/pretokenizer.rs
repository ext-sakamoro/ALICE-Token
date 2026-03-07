//! 正規表現プリトークナイザー
//!
//! BPEエンコード前にテキストをチャンクに分割する。
//! tiktoken互換のGPT-4/GPT-2パターンを提供。
//! 先読み(`(?!\S)`)対応のため `fancy-regex` を使用。

use fancy_regex::Regex;

/// プリトークナイザーパターン定義
#[derive(Debug, Clone)]
pub enum PreTokenizerPattern {
    /// GPT-4 (`cl100k_base`) パターン
    Gpt4,
    /// GPT-2 (`r50k_base` / `p50k_base`) パターン
    Gpt2,
    /// カスタム正規表現パターン
    Custom(String),
}

/// プリトークナイザー
pub struct PreTokenizer {
    regex: Regex,
    pattern: PreTokenizerPattern,
}

/// GPT-4 (`cl100k_base`) の正規表現パターン
const GPT4_PATTERN: &str = concat!(
    r"(?i:'s|'t|'re|'ve|'m|'ll|'d)",
    r"|[^\r\n\p{L}\p{N}]?\p{L}+",
    r"|\p{N}{1,3}",
    r"| ?[^\s\p{L}\p{N}]+[\r\n]*",
    r"|\s*[\r\n]",
    r"|\s+(?!\S)",
    r"|\s+",
);

/// GPT-2 (`r50k_base`) の正規表現パターン
const GPT2_PATTERN: &str = concat!(
    r"'s|'t|'re|'ve|'m|'ll|'d",
    r"| ?\p{L}+",
    r"| ?\p{N}+",
    r"| ?[^\s\p{L}\p{N}]+",
    r"|\s+(?!\S)",
    r"|\s+",
);

impl PreTokenizer {
    /// 指定パターンでプリトークナイザーを構築
    ///
    /// # Panics
    /// カスタムパターンが不正な正規表現の場合パニックする。
    #[must_use]
    pub fn new(pattern: PreTokenizerPattern) -> Self {
        let regex_str = match &pattern {
            PreTokenizerPattern::Gpt4 => GPT4_PATTERN,
            PreTokenizerPattern::Gpt2 => GPT2_PATTERN,
            PreTokenizerPattern::Custom(s) => s.as_str(),
        };
        let regex = Regex::new(regex_str).expect("invalid pretokenizer regex pattern");
        Self { regex, pattern }
    }

    /// テキストをプリトークナイズし、各チャンクのバイト列を返す
    #[must_use]
    pub fn split<'a>(&self, text: &'a [u8]) -> Vec<&'a [u8]> {
        let Ok(text_str) = std::str::from_utf8(text) else {
            return vec![text];
        };

        self.regex
            .find_iter(text_str)
            .filter_map(std::result::Result::ok)
            .map(|m| m.as_str().as_bytes())
            .collect()
    }

    /// 使用中のパターンを返す
    #[must_use]
    pub const fn pattern(&self) -> &PreTokenizerPattern {
        &self.pattern
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpt4_basic_english() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"Hello world");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"Hello world");
    }

    #[test]
    fn test_gpt4_contractions() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"I'm don't you'll they've we're");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"I'm don't you'll they've we're");
    }

    #[test]
    fn test_gpt4_numbers() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"test 12345 ok");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"test 12345 ok");
    }

    #[test]
    fn test_gpt4_unicode() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let text = "こんにちは世界 hello".as_bytes();
        let chunks = pt.split(text);
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn test_gpt4_punctuation() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"hello, world! how are you?");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"hello, world! how are you?");
    }

    #[test]
    fn test_gpt4_newlines() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"line1\nline2\r\nline3");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"line1\nline2\r\nline3");
    }

    #[test]
    fn test_gpt2_basic() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt2);
        let chunks = pt.split(b"Hello world");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"Hello world");
    }

    #[test]
    fn test_gpt2_contractions() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt2);
        let chunks = pt.split(b"I'm don't");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"I'm don't");
    }

    #[test]
    fn test_custom_pattern() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Custom(r"\w+|\s+".to_string()));
        let chunks = pt.split(b"hello world");
        assert!(chunks.len() >= 2);
    }

    #[test]
    fn test_empty_input() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"");
        assert!(chunks.is_empty());
    }

    #[test]
    fn test_whitespace_only() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"   ");
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, b"   ");
    }

    #[test]
    fn test_single_char() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let chunks = pt.split(b"a");
        assert_eq!(chunks.len(), 1);
    }

    #[test]
    fn test_emoji() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let text = "hello 🌍 world".as_bytes();
        let chunks = pt.split(text);
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn test_mixed_scripts() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let text = "English日本語العربية123".as_bytes();
        let chunks = pt.split(text);
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, text);
    }

    #[test]
    fn test_long_text() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let text = "the quick brown fox ".repeat(1000);
        let chunks = pt.split(text.as_bytes());
        let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
        assert_eq!(joined, text.as_bytes());
    }

    #[test]
    fn test_invalid_utf8_fallback() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        let data = vec![0xFF, 0xFE, 0x80];
        let chunks = pt.split(&data);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0], &data[..]);
    }

    #[test]
    fn test_pattern_accessor() {
        let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
        assert!(matches!(pt.pattern(), PreTokenizerPattern::Gpt4));
    }
}
