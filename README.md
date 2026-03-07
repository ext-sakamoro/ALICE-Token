# ALICE-Token

Ultra-fast BPE tokenizer for the ALICE ecosystem.

## Modules

| Module | Description |
|--------|-------------|
| `encoder` | BPE encoding (text to token IDs) |
| `decoder` | BPE decoding (token IDs to text) |
| `vocab` | Vocabulary management, byte-level builder |
| `parallel` | Rayon-based parallel tokenization |
| `trainer` | BPE vocabulary trainer from corpus |
| `pretokenizer` | Regex-based pre-tokenization (GPT-4 compatible) |
| `special` | Special token handling (BOS, EOS, PAD) |
| `io` | tiktoken / binary format load/save |
| `simd` | SIMD-accelerated byte operations |
| `ffi` | C-ABI FFI exports (feature `ffi`) |

## Features

| Feature | Default | Description |
|---------|---------|-------------|
| `std` | Yes | Standard library support |
| `ffi` | No | C-ABI FFI exports |
| `python` | No | PyO3 Python bindings |
| `simd_bridge` | No | ALICE-SIMD integration |
| `text_bridge` | No | ALICE-Text integration |
| `ml_bridge` | No | ALICE-ML integration |
| `search_bridge` | No | ALICE-Search integration |

## Example

```rust
use alice_token::{Tokenizer, Vocab};

// Load tiktoken vocabulary
let vocab = alice_token::load_tiktoken("cl100k_base.tiktoken").unwrap();
let tokenizer = Tokenizer::new(vocab);

// Encode
let tokens = tokenizer.encode("Hello, world!");

// Decode
let text = tokenizer.decode(&tokens);

// Parallel tokenization
use alice_token::ParallelTokenizer;
let par = ParallelTokenizer::new(tokenizer);
let batch_results = par.encode_batch(&["text1", "text2", "text3"]);
```

## Quality

| Metric | Value |
|--------|-------|
| clippy (pedantic+nursery) | 0 warnings |
| Tests | 150 |
| fmt | clean |

## License

MIT
