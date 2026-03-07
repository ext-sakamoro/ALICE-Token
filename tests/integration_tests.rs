//! ALICE-Token 統合テスト — 全領域カバレッジ

use alice_token::{
    byte_level_builder, load_binary, load_tiktoken, save_binary, save_tiktoken, PreTokenizer,
    PreTokenizerPattern, SpecialTokenPolicy, SpecialTokens, Tokenizer, Trainer, TrainerConfig,
    Vocab, VocabBuilder,
};

// =========================================
// テスト用語彙
// =========================================

fn build_test_vocab() -> Vocab {
    let mut builder = VocabBuilder::new();
    for b in 0u8..=255 {
        builder.add_token(vec![b]);
    }
    builder.add_merge(b"t", b"h");
    builder.add_merge(b"th", b"e");
    builder.add_merge(b"i", b"n");
    builder.add_merge(b"a", b"n");
    builder.add_merge(b"an", b"d");
    builder.add_merge(b"e", b"r");
    builder.add_merge(b"o", b"n");
    builder.add_merge(b"r", b"e");
    builder.add_merge(b"i", b"s");
    builder.add_merge(b"o", b"f");
    builder.build()
}

// =========================================
// 1. BPEアルゴリズム — ラウンドトリップ
// =========================================

#[test]
fn roundtrip_ascii() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = b"the quick brown fox";
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn roundtrip_utf8_japanese() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = "東京タワーは日本の観光名所です".as_bytes();
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn roundtrip_utf8_emoji() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = "Hello 🌍🎉 World".as_bytes();
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn roundtrip_mixed_scripts() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = "English日本語العربيةहिन्दी".as_bytes();
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn roundtrip_empty() {
    let tok = Tokenizer::new(build_test_vocab());
    assert!(tok.encode(b"").is_empty());
    assert!(tok.decode(&[]).is_empty());
}

#[test]
fn roundtrip_all_bytes() {
    let tok = Tokenizer::new(build_test_vocab());
    let text: Vec<u8> = (0..=255).collect();
    assert_eq!(tok.decode(&tok.encode(&text)), text);
}

#[test]
fn merge_basic() {
    let tok = Tokenizer::new(build_test_vocab());
    assert_eq!(tok.encode(b"the").len(), 1);
}

#[test]
fn merge_repeated() {
    let tok = Tokenizer::new(build_test_vocab());
    let ids = tok.encode(b"thethethe");
    assert_eq!(ids.len(), 3);
    assert!(ids.iter().all(|&id| id == ids[0]));
}

#[test]
fn no_merge_unmatched() {
    let tok = Tokenizer::new(build_test_vocab());
    assert_eq!(tok.encode(b"xyz").len(), 3);
}

// =========================================
// 2. プリトークナイザー
// =========================================

#[test]
fn pretokenizer_gpt4_roundtrip() {
    let vocab = build_test_vocab();
    let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
    let tok = Tokenizer::with_config(vocab, Some(pt), SpecialTokens::new());
    let text = b"Hello, world! How are you?";
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn pretokenizer_gpt2_roundtrip() {
    let vocab = build_test_vocab();
    let pt = PreTokenizer::new(PreTokenizerPattern::Gpt2);
    let tok = Tokenizer::with_config(vocab, Some(pt), SpecialTokens::new());
    let text = b"I'm don't you'll they've";
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn pretokenizer_preserves_text() {
    let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
    let text = "Hello world 123 🎉 日本語".as_bytes();
    let chunks = pt.split(text);
    let joined: Vec<u8> = chunks.iter().flat_map(|c| c.iter().copied()).collect();
    assert_eq!(joined, text);
}

#[test]
fn pretokenizer_custom_pattern() {
    let pt = PreTokenizer::new(PreTokenizerPattern::Custom(r"\w+|\s+".to_string()));
    let chunks = pt.split(b"hello world");
    assert!(chunks.len() >= 2);
}

// =========================================
// 3. 特殊トークン
// =========================================

#[test]
fn special_tokens_encode_decode() {
    let vocab = build_test_vocab();
    let mut special = SpecialTokens::new();
    special.add("<|endoftext|>".to_string(), 50256);
    let tok = Tokenizer::with_config(vocab, None, special);

    let text = b"hello<|endoftext|>world";
    let ids = tok.encode(text);
    assert!(ids.contains(&50256));
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn special_tokens_policy_all_allowed() {
    let vocab = build_test_vocab();
    let mut special = SpecialTokens::new();
    special.add("<|end|>".to_string(), 500);
    let tok = Tokenizer::with_config(vocab, None, special);

    let ids = tok.encode_with_policy(b"test<|end|>", &SpecialTokenPolicy::AllAllowed);
    assert!(ids.contains(&500));
}

#[test]
fn special_tokens_policy_none_allowed() {
    let vocab = build_test_vocab();
    let mut special = SpecialTokens::new();
    special.add("<|end|>".to_string(), 500);
    let tok = Tokenizer::with_config(vocab, None, special);

    let ids = tok.encode_with_policy(b"test<|end|>", &SpecialTokenPolicy::NoneAllowed);
    assert!(!ids.contains(&500));
}

#[test]
fn special_tokens_policy_whitelist() {
    let vocab = build_test_vocab();
    let mut special = SpecialTokens::new();
    special.add("<|a|>".to_string(), 500);
    special.add("<|b|>".to_string(), 501);
    let tok = Tokenizer::with_config(vocab, None, special);

    let policy = SpecialTokenPolicy::Whitelist(vec!["<|a|>".to_string()]);
    let ids = tok.encode_with_policy(b"x<|a|>y<|b|>z", &policy);
    assert!(ids.contains(&500));
    assert!(!ids.contains(&501));
}

#[test]
fn special_tokens_adjacent() {
    let vocab = build_test_vocab();
    let mut special = SpecialTokens::new();
    special.add("<|a|>".to_string(), 500);
    special.add("<|b|>".to_string(), 501);
    let tok = Tokenizer::with_config(vocab, None, special);

    let ids = tok.encode(b"<|a|><|b|>");
    assert_eq!(ids, vec![500, 501]);
}

#[test]
fn special_tokens_cl100k_preset() {
    let st = alice_token::special::cl100k_special_tokens();
    assert_eq!(st.get_id("<|endoftext|>"), Some(100_257));
    assert_eq!(st.len(), 5);
}

// =========================================
// 4. 語彙ファイル I/O
// =========================================

#[test]
fn io_tiktoken_roundtrip() {
    let mut builder = byte_level_builder();
    builder.add_merge(b"h", b"e");
    builder.add_merge(b"l", b"l");
    let vocab = builder.build();

    let exported = save_tiktoken(&vocab);
    let reloaded = load_tiktoken(&exported).unwrap();
    assert_eq!(reloaded.len(), vocab.len());
}

#[test]
fn io_tiktoken_parse() {
    let data = "aGVsbG8= 0\nd29ybGQ= 1\n";
    let vocab = load_tiktoken(data).unwrap();
    assert!(vocab.get_id(b"hello").is_some());
    assert!(vocab.get_id(b"world").is_some());
}

#[test]
fn io_tiktoken_invalid() {
    assert!(load_tiktoken("!!!invalid 0\n").is_err());
    assert!(load_tiktoken("aGVsbG8=\n").is_err());
}

#[test]
fn io_binary_roundtrip() {
    let mut builder = byte_level_builder();
    builder.add_merge(b"a", b"b");
    builder.add_merge(b"c", b"d");
    builder.add_merge(b"ab", b"cd");
    let vocab = builder.build();

    let binary = save_binary(&vocab).unwrap();
    let reloaded = load_binary(&binary).unwrap();
    assert_eq!(reloaded.len(), vocab.len());
    assert_eq!(reloaded.merge_count(), vocab.merge_count());

    #[allow(clippy::cast_possible_truncation)]
    for id in 0..vocab.len() as u32 {
        assert_eq!(vocab.get_token(id), reloaded.get_token(id));
    }
}

#[test]
fn io_binary_invalid() {
    assert!(load_binary(b"not valid bincode").is_err());
}

// =========================================
// 5. BPEトレーニング
// =========================================

#[test]
fn trainer_basic() {
    let trainer = Trainer::new(TrainerConfig {
        vocab_size: 260,
        min_frequency: 2,
    });
    let corpus: Vec<&[u8]> = vec![b"aaabdaaabac", b"aabdaabac", b"aaabdaaabac"];
    let vocab = trainer.train(&corpus);
    assert!(vocab.len() > 256);
    assert!(vocab.merge_count() > 0);
}

#[test]
fn trainer_roundtrip() {
    let trainer = Trainer::new(TrainerConfig {
        vocab_size: 280,
        min_frequency: 2,
    });
    let text = b"the quick brown fox jumps over the lazy dog";
    let corpus: Vec<&[u8]> = std::iter::repeat_n(text.as_slice(), 10).collect();
    let vocab = trainer.train(&corpus);
    let tok = Tokenizer::new(vocab);
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn trainer_empty_corpus() {
    let trainer = Trainer::new(TrainerConfig::default());
    let vocab = trainer.train(&[]);
    assert_eq!(vocab.len(), 256);
}

#[test]
fn trainer_respects_min_frequency() {
    let trainer = Trainer::new(TrainerConfig {
        vocab_size: 300,
        min_frequency: 1000,
    });
    let vocab = trainer.train(&[b"abcdef"]);
    assert_eq!(vocab.len(), 256); // マージなし
}

#[test]
fn trainer_respects_vocab_size() {
    let trainer = Trainer::new(TrainerConfig {
        vocab_size: 258,
        min_frequency: 1,
    });
    let corpus: Vec<&[u8]> = vec![b"abababcdcdcd"; 10];
    let vocab = trainer.train(&corpus);
    assert!(vocab.len() <= 258);
}

// =========================================
// 6. 並列エンコード
// =========================================

#[test]
fn parallel_roundtrip() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = b"the quick brown fox jumps over the lazy dog".repeat(100);
    let seq_decoded = tok.decode(&tok.encode(&text));
    let par_decoded = tok.decode(&tok.encode_parallel(&text, 128));
    assert_eq!(seq_decoded, par_decoded);
}

#[test]
fn parallel_utf8_safe() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = "あいうえおかきくけこ".as_bytes().repeat(100);
    let ids = tok.encode_parallel(&text, 32);
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn parallel_deterministic() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = b"hello world test".repeat(50);
    assert_eq!(
        tok.encode_parallel(&text, 64),
        tok.encode_parallel(&text, 64)
    );
}

// =========================================
// 7. 統合パイプライン
// =========================================

#[test]
fn full_pipeline_pretok_special() {
    let vocab = build_test_vocab();
    let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
    let mut special = SpecialTokens::new();
    special.add("<|end|>".to_string(), 999);
    let tok = Tokenizer::with_config(vocab, Some(pt), special);

    let text = b"Hello, world!<|end|>Goodbye.";
    let ids = tok.encode(text);
    assert!(ids.contains(&999));
    assert_eq!(tok.decode(&ids), text);
}

#[test]
fn full_pipeline_train_then_encode() {
    let trainer = Trainer::new(TrainerConfig {
        vocab_size: 300,
        min_frequency: 2,
    });
    let corpus_text = b"the quick brown fox jumps over the lazy dog ";
    let corpus: Vec<&[u8]> = std::iter::repeat_n(corpus_text.as_slice(), 50).collect();
    let vocab = trainer.train(&corpus);

    let pt = PreTokenizer::new(PreTokenizerPattern::Gpt4);
    let tok = Tokenizer::with_config(vocab, Some(pt), SpecialTokens::new());

    let text = b"the quick brown fox";
    let ids = tok.encode(text);
    assert_eq!(tok.decode(&ids), text);
    // トレーニング後はマージにより元のバイト数より少ないトークンに
    assert!(ids.len() < text.len());
}

#[test]
fn full_pipeline_io_train_encode() {
    // トレーニング → 保存 → 読み込み → エンコード
    let trainer = Trainer::new(TrainerConfig {
        vocab_size: 270,
        min_frequency: 2,
    });
    let corpus: Vec<&[u8]> = vec![b"hello world hello world"; 20];
    let vocab = trainer.train(&corpus);

    let binary = save_binary(&vocab).unwrap();
    let reloaded = load_binary(&binary).unwrap();

    let tok = Tokenizer::new(reloaded);
    let text = b"hello world";
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

// =========================================
// 8. エッジケース
// =========================================

#[test]
fn edge_long_input() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = vec![b'a'; 1_000_000];
    let decoded = tok.decode(&tok.encode(&text));
    assert_eq!(decoded.len(), 1_000_000);
}

#[test]
fn edge_null_bytes() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = b"\x00\x00\x00";
    assert_eq!(tok.decode(&tok.encode(text)), text);
}

#[test]
fn edge_binary_data() {
    let tok = Tokenizer::new(build_test_vocab());
    let text: Vec<u8> = (0..=255).cycle().take(512).collect();
    assert_eq!(tok.decode(&tok.encode(&text)), text);
}

// =========================================
// 9. パフォーマンス特性（回帰検知用）
// =========================================

#[test]
fn perf_encode_1kb() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = b"the quick brown fox jumps over the lazy dog ".repeat(25);
    for _ in 0..100 {
        let ids = tok.encode(&text);
        assert!(!ids.is_empty());
    }
}

#[test]
fn perf_parallel_10kb() {
    let tok = Tokenizer::new(build_test_vocab());
    let text = b"the quick brown fox jumps over the lazy dog ".repeat(250);
    for _ in 0..10 {
        let ids = tok.encode_parallel(&text, 1024);
        assert!(!ids.is_empty());
    }
}
