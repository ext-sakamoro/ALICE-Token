//! 特殊トークン管理
//!
//! `<|endoftext|>`, `<|im_start|>` 等の特殊トークンの
//! 登録、検出、エンコード/デコード制御を行う。

use rustc_hash::FxHashMap;

/// 特殊トークンの許可ポリシー
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SpecialTokenPolicy {
    /// 全ての特殊トークンを許可
    AllAllowed,
    /// 指定されたもののみ許可
    Whitelist(Vec<String>),
    /// 全て禁止（特殊トークンがあればエラー）
    NoneAllowed,
}

/// 特殊トークン管理
///
/// テキスト中の特殊トークン文字列を検出し、
/// 対応するトークンIDに変換する。
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// 特殊トークン文字列 → トークンID
    token_to_id: FxHashMap<String, u32>,
    /// トークンID → 特殊トークン文字列
    id_to_token: FxHashMap<u32, String>,
}

impl SpecialTokens {
    /// 空の特殊トークン管理を生成
    #[must_use]
    pub fn new() -> Self {
        Self {
            token_to_id: FxHashMap::default(),
            id_to_token: FxHashMap::default(),
        }
    }

    /// 特殊トークン一覧から構築
    #[must_use]
    pub fn from_map(tokens: &[(String, u32)]) -> Self {
        let mut st = Self::new();
        for (text, id) in tokens {
            st.add(text.clone(), *id);
        }
        st
    }

    /// 特殊トークンを追加
    pub fn add(&mut self, text: String, id: u32) {
        self.id_to_token.insert(id, text.clone());
        self.token_to_id.insert(text, id);
    }

    /// 特殊トークン文字列からIDを取得
    #[inline]
    #[must_use]
    pub fn get_id(&self, text: &str) -> Option<u32> {
        self.token_to_id.get(text).copied()
    }

    /// IDから特殊トークン文字列を取得
    #[inline]
    #[must_use]
    pub fn get_text(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(&id).map(String::as_str)
    }

    /// 特殊トークンが存在するか
    #[inline]
    #[must_use]
    pub fn contains(&self, text: &str) -> bool {
        self.token_to_id.contains_key(text)
    }

    /// IDが特殊トークンか
    #[inline]
    #[must_use]
    pub fn is_special_id(&self, id: u32) -> bool {
        self.id_to_token.contains_key(&id)
    }

    /// 登録数
    #[inline]
    #[must_use]
    pub fn len(&self) -> usize {
        self.token_to_id.len()
    }

    /// 空か
    #[inline]
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.token_to_id.is_empty()
    }

    /// テキスト内の特殊トークンを検出し、(position, length, id) のリストを返す
    ///
    /// 長いマッチを優先する（longest match first）。
    #[must_use]
    pub fn find_all(&self, text: &[u8]) -> Vec<(usize, usize, u32)> {
        let Ok(text_str) = std::str::from_utf8(text) else {
            return Vec::new();
        };

        let mut results = Vec::new();
        // 長いトークンを先にマッチさせるため、長さ降順でソート
        let mut sorted_tokens: Vec<(&str, u32)> = self
            .token_to_id
            .iter()
            .map(|(k, &v)| (k.as_str(), v))
            .collect();
        sorted_tokens.sort_by(|a, b| b.0.len().cmp(&a.0.len()));

        let mut pos = 0;
        while pos < text_str.len() {
            let mut found = false;
            for &(token_text, token_id) in &sorted_tokens {
                if text_str[pos..].starts_with(token_text) {
                    results.push((pos, token_text.len(), token_id));
                    pos += token_text.len();
                    found = true;
                    break;
                }
            }
            if !found {
                pos += 1;
            }
        }

        results
    }

    /// テキストを特殊トークンの境界で分割する
    ///
    /// 返り値: `(chunk, is_special, optional_token_id)` のリスト
    #[must_use]
    pub fn split_with_special<'a>(&self, text: &'a [u8]) -> Vec<TextChunk<'a>> {
        if self.is_empty() {
            return vec![TextChunk {
                bytes: text,
                is_special: false,
                token_id: None,
            }];
        }

        let matches = self.find_all(text);
        if matches.is_empty() {
            return vec![TextChunk {
                bytes: text,
                is_special: false,
                token_id: None,
            }];
        }

        let mut chunks = Vec::new();
        let mut pos = 0;

        for (match_pos, match_len, token_id) in &matches {
            // 特殊トークン前の通常テキスト
            if pos < *match_pos {
                chunks.push(TextChunk {
                    bytes: &text[pos..*match_pos],
                    is_special: false,
                    token_id: None,
                });
            }
            // 特殊トークン自体
            chunks.push(TextChunk {
                bytes: &text[*match_pos..*match_pos + *match_len],
                is_special: true,
                token_id: Some(*token_id),
            });
            pos = *match_pos + *match_len;
        }

        // 残りの通常テキスト
        if pos < text.len() {
            chunks.push(TextChunk {
                bytes: &text[pos..],
                is_special: false,
                token_id: None,
            });
        }

        chunks
    }

    /// 全トークンのイテレータ
    pub fn iter(&self) -> impl Iterator<Item = (&str, u32)> {
        self.token_to_id.iter().map(|(k, &v)| (k.as_str(), v))
    }
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self::new()
    }
}

/// テキストチャンク（特殊トークン分割結果）
#[derive(Debug, Clone)]
pub struct TextChunk<'a> {
    /// バイト列
    pub bytes: &'a [u8],
    /// 特殊トークンか
    pub is_special: bool,
    /// 特殊トークンの場合のID
    pub token_id: Option<u32>,
}

/// tiktoken `cl100k_base` のデフォルト特殊トークン
#[must_use]
pub fn cl100k_special_tokens() -> SpecialTokens {
    let mut st = SpecialTokens::new();
    st.add("<|endoftext|>".to_string(), 100_257);
    st.add("<|fim_prefix|>".to_string(), 100_258);
    st.add("<|fim_middle|>".to_string(), 100_259);
    st.add("<|fim_suffix|>".to_string(), 100_260);
    st.add("<|endofprompt|>".to_string(), 100_276);
    st
}

/// tiktoken `o200k_base` のデフォルト特殊トークン
#[must_use]
pub fn o200k_special_tokens() -> SpecialTokens {
    let mut st = SpecialTokens::new();
    st.add("<|endoftext|>".to_string(), 199_999);
    st.add("<|endofprompt|>".to_string(), 200_018);
    st
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_special() -> SpecialTokens {
        let mut st = SpecialTokens::new();
        st.add("<|endoftext|>".to_string(), 50256);
        st.add("<|im_start|>".to_string(), 50257);
        st.add("<|im_end|>".to_string(), 50258);
        st
    }

    #[test]
    fn test_add_and_lookup() {
        let st = make_test_special();
        assert_eq!(st.get_id("<|endoftext|>"), Some(50256));
        assert_eq!(st.get_text(50256), Some("<|endoftext|>"));
    }

    #[test]
    fn test_contains() {
        let st = make_test_special();
        assert!(st.contains("<|endoftext|>"));
        assert!(!st.contains("<|unknown|>"));
    }

    #[test]
    fn test_is_special_id() {
        let st = make_test_special();
        assert!(st.is_special_id(50256));
        assert!(!st.is_special_id(0));
    }

    #[test]
    fn test_len() {
        let st = make_test_special();
        assert_eq!(st.len(), 3);
    }

    #[test]
    fn test_empty() {
        let st = SpecialTokens::new();
        assert!(st.is_empty());
    }

    #[test]
    fn test_find_all() {
        let st = make_test_special();
        let text = b"hello<|endoftext|>world";
        let matches = st.find_all(text);
        assert_eq!(matches.len(), 1);
        assert_eq!(matches[0].0, 5); // position
        assert_eq!(matches[0].1, 13); // length
        assert_eq!(matches[0].2, 50256); // id
    }

    #[test]
    fn test_find_all_multiple() {
        let st = make_test_special();
        let text = b"<|im_start|>hello<|im_end|>";
        let matches = st.find_all(text);
        assert_eq!(matches.len(), 2);
        assert_eq!(matches[0].2, 50257); // im_start
        assert_eq!(matches[1].2, 50258); // im_end
    }

    #[test]
    fn test_find_all_none() {
        let st = make_test_special();
        let matches = st.find_all(b"hello world");
        assert!(matches.is_empty());
    }

    #[test]
    fn test_split_with_special() {
        let st = make_test_special();
        let text = b"hello<|endoftext|>world";
        let chunks = st.split_with_special(text);
        assert_eq!(chunks.len(), 3);
        assert!(!chunks[0].is_special);
        assert_eq!(chunks[0].bytes, b"hello");
        assert!(chunks[1].is_special);
        assert_eq!(chunks[1].token_id, Some(50256));
        assert!(!chunks[2].is_special);
        assert_eq!(chunks[2].bytes, b"world");
    }

    #[test]
    fn test_split_no_special() {
        let st = make_test_special();
        let text = b"hello world";
        let chunks = st.split_with_special(text);
        assert_eq!(chunks.len(), 1);
        assert!(!chunks[0].is_special);
    }

    #[test]
    fn test_split_only_special() {
        let st = make_test_special();
        let text = b"<|endoftext|>";
        let chunks = st.split_with_special(text);
        assert_eq!(chunks.len(), 1);
        assert!(chunks[0].is_special);
    }

    #[test]
    fn test_split_empty_special_tokens() {
        let st = SpecialTokens::new();
        let text = b"hello<|endoftext|>world";
        let chunks = st.split_with_special(text);
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].bytes, text.as_slice());
    }

    #[test]
    fn test_from_map() {
        let tokens = vec![
            ("<|test|>".to_string(), 100u32),
            ("<|foo|>".to_string(), 101u32),
        ];
        let st = SpecialTokens::from_map(&tokens);
        assert_eq!(st.len(), 2);
        assert_eq!(st.get_id("<|test|>"), Some(100));
    }

    #[test]
    fn test_cl100k_special() {
        let st = cl100k_special_tokens();
        assert_eq!(st.get_id("<|endoftext|>"), Some(100_257));
        assert_eq!(st.len(), 5);
    }

    #[test]
    fn test_o200k_special() {
        let st = o200k_special_tokens();
        assert_eq!(st.get_id("<|endoftext|>"), Some(199_999));
        assert_eq!(st.len(), 2);
    }

    #[test]
    fn test_iter() {
        let st = make_test_special();
        let items: Vec<_> = st.iter().collect();
        assert_eq!(items.len(), 3);
    }

    #[test]
    fn test_default() {
        let st = SpecialTokens::default();
        assert!(st.is_empty());
    }

    #[test]
    fn test_adjacent_special_tokens() {
        let st = make_test_special();
        let text = b"<|im_start|><|im_end|>";
        let chunks = st.split_with_special(text);
        assert_eq!(chunks.len(), 2);
        assert!(chunks[0].is_special);
        assert!(chunks[1].is_special);
    }

    #[test]
    fn test_special_at_boundaries() {
        let st = make_test_special();
        let text = b"<|endoftext|>middle<|endoftext|>";
        let chunks = st.split_with_special(text);
        assert_eq!(chunks.len(), 3);
        assert!(chunks[0].is_special);
        assert!(!chunks[1].is_special);
        assert!(chunks[2].is_special);
    }
}
