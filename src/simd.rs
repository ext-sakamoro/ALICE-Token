//! SIMD最適化層 — NEON/AVX2 intrinsicsによるバイト処理
//!
//! UTF-8境界検出、バイト検索をハードウェアSIMD命令で高速化。
//! フォールバック（スカラー）パスも提供。

// プラットフォーム別SIMD幅（テスト・ベンチマークで使用）
#[cfg(target_arch = "aarch64")]
pub const SIMD_WIDTH: usize = 16; // NEON: 128-bit

#[cfg(target_arch = "x86_64")]
pub const SIMD_WIDTH: usize = 32; // AVX2: 256-bit

#[cfg(not(any(target_arch = "aarch64", target_arch = "x86_64")))]
pub const SIMD_WIDTH: usize = 16; // フォールバック

// ============================================================
// NEON intrinsics (aarch64)
// ============================================================
#[cfg(target_arch = "aarch64")]
mod neon {
    use std::arch::aarch64::{
        vandq_u8, vceqq_u8, vdupq_n_u8, vget_lane_u64, vld1q_u8, vreinterpret_u64_u8,
        vreinterpretq_u16_u8, vshrn_n_u16,
    };

    /// NEON: 16バイト中で needle と一致するバイトのビットマスクを返す
    ///
    /// # Safety
    /// `data` は16バイト以上のスライスであること。
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn find_byte_mask_16(data: &[u8], needle: u8) -> u16 {
        let chunk = vld1q_u8(data.as_ptr());
        let needle_vec = vdupq_n_u8(needle);
        let cmp = vceqq_u8(chunk, needle_vec);

        // 各レーンの最上位ビットを集めてビットマスク化
        // NEON には movemask がないため、vshrn + 手動集約
        let high_bits = vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4);
        let narrowed = vreinterpret_u64_u8(high_bits);
        let val = vget_lane_u64(narrowed, 0);

        // 各4bit中の最上位bitを抽出
        let mut mask: u16 = 0;
        for i in 0..16 {
            if (val >> (i * 4)) & 0x8 != 0 {
                mask |= 1 << i;
            }
        }
        mask
    }

    /// NEON: 16バイト中でUTF-8継続バイト(10xxxxxx)のビットマスクを返す
    ///
    /// # Safety
    /// `data` は16バイト以上のスライスであること。
    #[inline]
    #[target_feature(enable = "neon")]
    pub unsafe fn utf8_continuation_mask_16(data: &[u8]) -> u16 {
        let chunk = vld1q_u8(data.as_ptr());
        // (b & 0xC0) == 0x80 → 継続バイト
        let mask_c0 = vdupq_n_u8(0xC0);
        let val_80 = vdupq_n_u8(0x80);
        let masked = vandq_u8(chunk, mask_c0);
        let cmp = vceqq_u8(masked, val_80);

        let high_bits = vshrn_n_u16(vreinterpretq_u16_u8(cmp), 4);
        let narrowed = vreinterpret_u64_u8(high_bits);
        let val = vget_lane_u64(narrowed, 0);

        let mut result: u16 = 0;
        for i in 0..16 {
            if (val >> (i * 4)) & 0x8 != 0 {
                result |= 1 << i;
            }
        }
        result
    }
}

// ============================================================
// AVX2 intrinsics (x86_64)
// ============================================================
#[cfg(target_arch = "x86_64")]
mod avx2 {
    #[cfg(target_arch = "x86_64")]
    use std::arch::x86_64::*;

    /// AVX2: 32バイト中で needle と一致するバイトのビットマスクを返す
    ///
    /// # Safety
    /// `data` は32バイト以上のスライスであること。AVX2対応CPUであること。
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn find_byte_mask_32(data: &[u8], needle: u8) -> u32 {
        let chunk = _mm256_loadu_si256(data.as_ptr().cast());
        let needle_vec = _mm256_set1_epi8(needle as i8);
        let cmp = _mm256_cmpeq_epi8(chunk, needle_vec);
        _mm256_movemask_epi8(cmp) as u32
    }

    /// AVX2: 32バイト中でUTF-8継続バイト(10xxxxxx)のビットマスクを返す
    ///
    /// # Safety
    /// `data` は32バイト以上のスライスであること。AVX2対応CPUであること。
    #[inline]
    #[target_feature(enable = "avx2")]
    pub unsafe fn utf8_continuation_mask_32(data: &[u8]) -> u32 {
        let chunk = _mm256_loadu_si256(data.as_ptr().cast());
        let mask_c0 = _mm256_set1_epi8(0xC0_u8 as i8);
        let val_80 = _mm256_set1_epi8(0x80_u8 as i8);
        let masked = _mm256_and_si256(chunk, mask_c0);
        let cmp = _mm256_cmpeq_epi8(masked, val_80);
        _mm256_movemask_epi8(cmp) as u32
    }
}

// ============================================================
// 公開API（ディスパッチ）
// ============================================================

/// 指定バイトの全出現位置を高速検索
#[inline]
#[must_use]
pub fn find_byte(data: &[u8], needle: u8) -> Vec<usize> {
    let mut positions = Vec::new();
    let len = data.len();
    let mut offset = 0;

    // SIMD パス
    #[cfg(target_arch = "aarch64")]
    {
        while offset + 16 <= len {
            let mask = unsafe { neon::find_byte_mask_16(&data[offset..], needle) };
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                positions.push(offset + bit);
                m &= m - 1;
            }
            offset += 16;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            while offset + 32 <= len {
                let mask = unsafe { avx2::find_byte_mask_32(&data[offset..], needle) };
                let mut m = mask;
                while m != 0 {
                    let bit = m.trailing_zeros() as usize;
                    positions.push(offset + bit);
                    m &= m - 1;
                }
                offset += 32;
            }
        }
    }

    // スカラーフォールバック（残り）
    for (i, &byte) in data[offset..len].iter().enumerate() {
        if byte == needle {
            positions.push(offset + i);
        }
    }

    positions
}

/// UTF-8文字境界を検出（各バイトが先頭バイトなら true）
#[inline]
#[must_use]
pub fn utf8_char_boundaries(data: &[u8]) -> Vec<bool> {
    let len = data.len();
    let mut result = vec![true; len];
    let mut offset = 0;

    // SIMD パス: 継続バイトを検出して false にする
    #[cfg(target_arch = "aarch64")]
    {
        while offset + 16 <= len {
            let mask = unsafe { neon::utf8_continuation_mask_16(&data[offset..]) };
            let mut m = mask;
            while m != 0 {
                let bit = m.trailing_zeros() as usize;
                result[offset + bit] = false;
                m &= m - 1;
            }
            offset += 16;
        }
    }

    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            while offset + 32 <= len {
                let mask = unsafe { avx2::utf8_continuation_mask_32(&data[offset..]) };
                let mut m = mask;
                while m != 0 {
                    let bit = m.trailing_zeros() as usize;
                    result[offset + bit] = false;
                    m &= m - 1;
                }
                offset += 32;
            }
        }
    }

    // スカラーフォールバック
    for i in offset..len {
        result[i] = !is_continuation_byte(data[i]);
    }

    result
}

/// UTF-8継続バイトかどうか判定
#[inline(always)]
#[must_use]
pub const fn is_continuation_byte(b: u8) -> bool {
    (b & 0xC0) == 0x80
}

/// テキストをUTF-8文字境界で安全に分割する位置を検出
#[inline]
#[must_use]
pub fn find_safe_split_points(data: &[u8], chunk_size: usize) -> Vec<usize> {
    if data.is_empty() || chunk_size == 0 {
        return Vec::new();
    }

    let mut points = Vec::new();
    let mut pos = chunk_size;

    while pos < data.len() {
        let split = find_nearest_boundary(data, pos);
        if split > 0 && split < data.len() {
            points.push(split);
        }
        pos = split + chunk_size;
    }

    points
}

/// 指定位置の最も近いUTF-8文字境界を探す
#[inline]
fn find_nearest_boundary(data: &[u8], pos: usize) -> usize {
    let pos = pos.min(data.len());

    // 前方に最大4バイト探索
    for offset in 0..4 {
        let p = pos.saturating_sub(offset);
        if p < data.len() && !is_continuation_byte(data[p]) {
            return p;
        }
    }

    // 後方探索
    for (p, &byte) in data[pos..data.len().min(pos + 4)].iter().enumerate() {
        if !is_continuation_byte(byte) {
            return pos + p;
        }
    }

    pos
}

/// 空白文字の出現位置を高速検索
#[inline]
#[must_use]
pub fn find_whitespace(data: &[u8]) -> Vec<usize> {
    // スペースの位置を SIMD で検出し、他の空白はスカラーで追加
    let mut positions = find_byte(data, b' ');

    // タブ、改行、CRも追加
    for (i, &b) in data.iter().enumerate() {
        if matches!(b, b'\t' | b'\n' | b'\r') {
            positions.push(i);
        }
    }

    positions.sort_unstable();
    positions
}

/// ASCII空白文字判定
#[inline(always)]
#[must_use]
pub const fn is_whitespace(b: u8) -> bool {
    matches!(b, b' ' | b'\t' | b'\n' | b'\r')
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_byte_basic() {
        let data = b"hello world";
        let positions = find_byte(data, b'l');
        assert_eq!(positions, vec![2, 3, 9]);
    }

    #[test]
    fn test_find_byte_not_found() {
        assert!(find_byte(b"hello", b'z').is_empty());
    }

    #[test]
    fn test_find_byte_empty() {
        assert!(find_byte(b"", b'a').is_empty());
    }

    #[test]
    fn test_find_byte_all_match() {
        let data = vec![0x42u8; 100];
        let positions = find_byte(&data, 0x42);
        assert_eq!(positions.len(), 100);
    }

    #[test]
    fn test_find_byte_simd_boundary() {
        // SIMD幅の境界をまたぐテスト
        let mut data = vec![0u8; SIMD_WIDTH * 3 + 7];
        data[0] = 0xFF;
        data[SIMD_WIDTH - 1] = 0xFF;
        data[SIMD_WIDTH] = 0xFF;
        data[SIMD_WIDTH * 2 + 3] = 0xFF;
        let last = data.len() - 1;
        data[last] = 0xFF;

        let positions = find_byte(&data, 0xFF);
        assert_eq!(positions.len(), 5);
        assert_eq!(positions[0], 0);
        assert_eq!(positions[1], SIMD_WIDTH - 1);
        assert_eq!(positions[2], SIMD_WIDTH);
    }

    #[test]
    fn test_find_byte_large() {
        let data: Vec<u8> = (0..=255).cycle().take(4096).collect();
        let positions = find_byte(&data, 42);
        assert_eq!(positions.len(), 16); // 4096/256 = 16
        for &p in &positions {
            assert_eq!(data[p], 42);
        }
    }

    #[test]
    fn test_utf8_boundaries_ascii() {
        let data = b"hello";
        let bounds = utf8_char_boundaries(data);
        assert_eq!(bounds.len(), 5);
        assert!(bounds.iter().all(|&b| b));
    }

    #[test]
    fn test_utf8_boundaries_3byte() {
        let data = "あいう".as_bytes(); // 各3バイト = 9バイト
        let bounds = utf8_char_boundaries(data);
        assert_eq!(bounds.len(), 9);
        assert!(bounds[0]); // あ先頭
        assert!(!bounds[1]);
        assert!(!bounds[2]);
        assert!(bounds[3]); // い先頭
        assert!(!bounds[4]);
        assert!(!bounds[5]);
        assert!(bounds[6]); // う先頭
    }

    #[test]
    fn test_utf8_boundaries_4byte() {
        let data = "🎉".as_bytes();
        let bounds = utf8_char_boundaries(data);
        assert_eq!(bounds.len(), 4);
        assert!(bounds[0]);
        assert!(!bounds[1]);
        assert!(!bounds[2]);
        assert!(!bounds[3]);
    }

    #[test]
    fn test_utf8_boundaries_mixed() {
        let data = "aあb".as_bytes(); // 1+3+1 = 5
        let bounds = utf8_char_boundaries(data);
        assert_eq!(bounds.len(), 5);
        assert!(bounds[0]); // a
        assert!(bounds[1]); // あ先頭
        assert!(!bounds[2]);
        assert!(!bounds[3]);
        assert!(bounds[4]); // b
    }

    #[test]
    fn test_utf8_boundaries_empty() {
        assert!(utf8_char_boundaries(b"").is_empty());
    }

    #[test]
    fn test_utf8_boundaries_simd_sized() {
        // SIMD幅ちょうどのデータ
        let text = "a".repeat(SIMD_WIDTH);
        let bounds = utf8_char_boundaries(text.as_bytes());
        assert_eq!(bounds.len(), SIMD_WIDTH);
        assert!(bounds.iter().all(|&b| b));
    }

    #[test]
    fn test_utf8_boundaries_large() {
        let text = "こんにちは世界 hello 🎉".repeat(100);
        let data = text.as_bytes();
        let bounds = utf8_char_boundaries(data);
        assert_eq!(bounds.len(), data.len());
        // 先頭バイトの数 = 文字数
        let char_count = text.chars().count();
        let boundary_count = bounds.iter().filter(|&&b| b).count();
        assert_eq!(boundary_count, char_count);
    }

    #[test]
    fn test_continuation_byte() {
        assert!(!is_continuation_byte(b'A'));
        assert!(!is_continuation_byte(0xC0));
        assert!(!is_continuation_byte(0xE0));
        assert!(!is_continuation_byte(0xF0));
        assert!(is_continuation_byte(0x80));
        assert!(is_continuation_byte(0xBF));
    }

    #[test]
    fn test_safe_split_ascii() {
        let data = b"hello world test data";
        let points = find_safe_split_points(data, 5);
        assert!(!points.is_empty());
        for &p in &points {
            assert!(p <= data.len());
        }
    }

    #[test]
    fn test_safe_split_utf8() {
        let data = "あいうえお".as_bytes();
        let points = find_safe_split_points(data, 6);
        for &p in &points {
            assert!(!is_continuation_byte(data[p]));
        }
    }

    #[test]
    fn test_safe_split_empty() {
        assert!(find_safe_split_points(b"", 10).is_empty());
        assert!(find_safe_split_points(b"test", 0).is_empty());
    }

    #[test]
    fn test_safe_split_large_chunk() {
        assert!(find_safe_split_points(b"short", 100).is_empty());
    }

    #[test]
    fn test_find_whitespace() {
        let data = b"hello world\tfoo\nbar";
        let positions = find_whitespace(data);
        assert_eq!(positions, vec![5, 11, 15]);
    }

    #[test]
    fn test_find_whitespace_none() {
        assert!(find_whitespace(b"helloworld").is_empty());
    }

    #[test]
    fn test_is_whitespace() {
        assert!(is_whitespace(b' '));
        assert!(is_whitespace(b'\t'));
        assert!(is_whitespace(b'\n'));
        assert!(is_whitespace(b'\r'));
        assert!(!is_whitespace(b'a'));
        assert!(!is_whitespace(0));
    }

    #[test]
    fn test_find_nearest_boundary_at_start() {
        let data = "あ".as_bytes();
        assert_eq!(find_nearest_boundary(data, 0), 0);
    }
}
