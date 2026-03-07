//! FFI / Python バインディング
//!
//! C FFI と PyO3 による Python バインディングを提供。
//! `ffi` feature または `python` feature で有効化。

use crate::vocab::byte_level_builder;
use crate::Tokenizer;

// --- C FFI ---

/// C FFI: トークナイザーを生成（バイトレベル語彙）
///
/// # Safety
/// 返されたポインタは `alice_token_free` で解放すること。
#[no_mangle]
pub unsafe extern "C" fn alice_token_new() -> *mut Tokenizer {
    let vocab = byte_level_builder().build();
    let tokenizer = Tokenizer::new(vocab);
    Box::into_raw(Box::new(tokenizer))
}

/// C FFI: トークナイザーを解放
///
/// # Safety
/// `ptr` は `alice_token_new` で生成されたポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_token_free(ptr: *mut Tokenizer) {
    if !ptr.is_null() {
        drop(Box::from_raw(ptr));
    }
}

/// C FFI: エンコード
///
/// # Safety
/// - `tokenizer` は有効なポインタであること
/// - `data` は `len` バイトの有効なバッファであること
/// - `out_ids` は `out_cap` 個の u32 を格納できるバッファであること
/// - 戻り値: 書き込んだトークン数（`out_cap` 超過時は必要数を負値で返す）
#[no_mangle]
pub unsafe extern "C" fn alice_token_encode(
    tokenizer: *const Tokenizer,
    data: *const u8,
    len: usize,
    out_ids: *mut u32,
    out_cap: usize,
) -> i64 {
    if tokenizer.is_null() || data.is_null() || out_ids.is_null() {
        return -1;
    }

    let tok = &*tokenizer;
    let input = std::slice::from_raw_parts(data, len);
    let ids = tok.encode(input);

    if ids.len() > out_cap {
        return -(ids.len() as i64);
    }

    let out_slice = std::slice::from_raw_parts_mut(out_ids, out_cap);
    out_slice[..ids.len()].copy_from_slice(&ids);
    ids.len() as i64
}

/// C FFI: デコード
///
/// # Safety
/// - `tokenizer` は有効なポインタであること
/// - `ids` は `ids_len` 個の u32 配列であること
/// - `out_buf` は `out_cap` バイトのバッファであること
/// - 戻り値: 書き込んだバイト数（`out_cap` 超過時は必要数を負値で返す）
#[no_mangle]
pub unsafe extern "C" fn alice_token_decode(
    tokenizer: *const Tokenizer,
    ids: *const u32,
    ids_len: usize,
    out_buf: *mut u8,
    out_cap: usize,
) -> i64 {
    if tokenizer.is_null() || ids.is_null() || out_buf.is_null() {
        return -1;
    }

    let tok = &*tokenizer;
    let id_slice = std::slice::from_raw_parts(ids, ids_len);
    let bytes = tok.decode(id_slice);

    if bytes.len() > out_cap {
        return -(bytes.len() as i64);
    }

    let out_slice = std::slice::from_raw_parts_mut(out_buf, out_cap);
    out_slice[..bytes.len()].copy_from_slice(&bytes);
    bytes.len() as i64
}

/// C FFI: 語彙サイズ取得
///
/// # Safety
/// `tokenizer` は有効なポインタであること。
#[no_mangle]
pub unsafe extern "C" fn alice_token_vocab_size(tokenizer: *const Tokenizer) -> usize {
    if tokenizer.is_null() {
        return 0;
    }
    (*tokenizer).vocab_size()
}

// --- PyO3 Python バインディング ---

#[cfg(feature = "python")]
mod python {
    use pyo3::prelude::*;
    use pyo3::types::PyBytes;

    use crate::pretokenizer::{PreTokenizer, PreTokenizerPattern};
    use crate::special::SpecialTokens;
    use crate::vocab::byte_level_builder;
    use crate::{io, Tokenizer};

    #[pyclass(name = "AliceTokenizer")]
    struct PyAliceTokenizer {
        inner: Tokenizer,
    }

    #[pymethods]
    impl PyAliceTokenizer {
        #[new]
        #[pyo3(signature = (tiktoken_data=None, use_gpt4_pattern=true))]
        fn new(tiktoken_data: Option<&str>, use_gpt4_pattern: bool) -> PyResult<Self> {
            let vocab = if let Some(data) = tiktoken_data {
                io::load_tiktoken(data)
                    .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?
            } else {
                byte_level_builder().build()
            };

            let pt = if use_gpt4_pattern {
                Some(PreTokenizer::new(PreTokenizerPattern::Gpt4))
            } else {
                None
            };

            Ok(Self {
                inner: Tokenizer::with_config(vocab, pt, SpecialTokens::new()),
            })
        }

        fn encode(&self, text: &[u8]) -> Vec<u32> {
            self.inner.encode(text)
        }

        fn decode<'py>(&self, py: Python<'py>, ids: Vec<u32>) -> Bound<'py, PyBytes> {
            let bytes = self.inner.decode(&ids);
            PyBytes::new(py, &bytes)
        }

        fn encode_parallel(&self, text: &[u8], chunk_size: usize) -> Vec<u32> {
            self.inner.encode_parallel(text, chunk_size)
        }

        #[getter]
        fn vocab_size(&self) -> usize {
            self.inner.vocab_size()
        }
    }

    #[pymodule]
    fn alice_token(m: &Bound<'_, PyModule>) -> PyResult<()> {
        m.add_class::<PyAliceTokenizer>()?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ffi_new_and_free() {
        unsafe {
            let tok = alice_token_new();
            assert!(!tok.is_null());
            let size = alice_token_vocab_size(tok);
            assert_eq!(size, 256);
            alice_token_free(tok);
        }
    }

    #[test]
    fn test_ffi_encode_decode() {
        unsafe {
            let tok = alice_token_new();
            let input = b"hello";
            let mut ids = vec![0u32; 256];
            let n = alice_token_encode(tok, input.as_ptr(), input.len(), ids.as_mut_ptr(), 256);
            assert!(n > 0);

            let n = n as usize;
            let mut out = vec![0u8; 256];
            let m = alice_token_decode(tok, ids.as_ptr(), n, out.as_mut_ptr(), 256);
            assert_eq!(m, input.len() as i64);
            assert_eq!(&out[..input.len()], input);

            alice_token_free(tok);
        }
    }

    #[test]
    fn test_ffi_null_safety() {
        unsafe {
            assert_eq!(
                alice_token_encode(
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0
                ),
                -1
            );
            assert_eq!(
                alice_token_decode(
                    std::ptr::null(),
                    std::ptr::null(),
                    0,
                    std::ptr::null_mut(),
                    0
                ),
                -1
            );
            assert_eq!(alice_token_vocab_size(std::ptr::null()), 0);
            alice_token_free(std::ptr::null_mut());
        }
    }

    #[test]
    fn test_ffi_buffer_too_small() {
        unsafe {
            let tok = alice_token_new();
            let input = b"hello world";
            let mut ids = vec![0u32; 2];
            let n = alice_token_encode(tok, input.as_ptr(), input.len(), ids.as_mut_ptr(), 2);
            assert!(n < 0);
            alice_token_free(tok);
        }
    }
}
