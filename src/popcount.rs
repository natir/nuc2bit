#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn popcount(bits: &[u64], len: usize) -> usize {
    if len / 64 <= 8 {
        return popcount_scalar(bits, len);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { popcount_avx(bits, len) };
        } else if is_x86_feature_detected!("ssse3") {
            return unsafe { popcount_sse(bits, len) };
        }
    }

    popcount_scalar_fast(bits, len)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn popcount_avx(bits: &[u64], len: usize) -> usize {
    let end_idx = (((len / 64) / 8) / 2) / 4;

    let mut res = _mm256_setzero_si256();
    let mut acc = [_mm256_setzero_si256(); 2];
    let ptr = bits.as_ptr() as *const __m256i;

    // bit patterns:
    // 0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000, 0b0111, 0b0110, 0b0101,
    // 0b0100, 0b0011, 0b0010, 0b0001, 0b0000
    // lut stores the popcount for each bit pattern:
    let lut = _mm256_set_epi64x(0x0403030203020201, 0x0302020102010100, 0x0403030203020201, 0x0302020102010100);
    let mask = _mm256_set1_epi8(0x0F);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn internal_popcount(lut: __m256i, mask: __m256i, a: __m256i) -> __m256i {
        let lo_nybbles_lut = _mm256_shuffle_epi8(lut, _mm256_and_si256(a, mask));
        let hi_nybbles_lut = _mm256_shuffle_epi8(lut, _mm256_and_si256(_mm256_srli_epi16(a, 4), mask));
        _mm256_add_epi8(lo_nybbles_lut, hi_nybbles_lut)
    }

    let mut idx = 0;

    for _i in 0..end_idx {
        for _j in 0..8 {
            acc[0] = _mm256_add_epi8(acc[0], internal_popcount(lut, mask, _mm256_loadu_si256(ptr.offset(idx + 0))));
            acc[1] = _mm256_add_epi8(acc[1], internal_popcount(lut, mask, _mm256_loadu_si256(ptr.offset(idx + 1))));
            idx += 2;
        }

        let sum = _mm256_add_epi8(acc[0], acc[1]);
        let sad = _mm256_sad_epu8(sum, _mm256_setzero_si256());
        res = _mm256_add_epi64(res, sad);

        acc[0] = _mm256_setzero_si256();
        acc[1] = _mm256_setzero_si256();
    }

    union A {
        v: __m256i,
        a: [u64; 4]
    }

    let a = A { v: res };
    let res = (a.a[0] + a.a[1]) + (a.a[2] + a.a[3]);

    let end = end_idx * 8 * 2 * 4;

    res as usize + popcount_scalar_fast(&bits[end..], len - (end * 64))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn popcount_sse(bits: &[u64], len: usize) -> usize {
    let end_idx = (((len / 64) / 8) / 2) / 2;

    let mut res = _mm_setzero_si128();
    let mut acc = [_mm_setzero_si128(); 2];
    let ptr = bits.as_ptr() as *const __m128i;

    let lut = _mm_set_epi64x(0x0403030203020201, 0x0302020102010100);
    let mask = _mm_set1_epi8(0x0F);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn internal_popcount(lut: __m128i, mask: __m128i, a: __m128i) -> __m128i {
        let lo_nybbles_lut = _mm_shuffle_epi8(lut, _mm_and_si128(a, mask));
        let hi_nybbles_lut = _mm_shuffle_epi8(lut, _mm_and_si128(_mm_srli_epi16(a, 4), mask));
        _mm_add_epi8(lo_nybbles_lut, hi_nybbles_lut)
    }

    let mut idx = 0;

    for _i in 0..end_idx {
        for _j in 0..8 {
            acc[0] = _mm_add_epi8(acc[0], internal_popcount(lut, mask, _mm_loadu_si128(ptr.offset(idx + 0))));
            acc[1] = _mm_add_epi8(acc[1], internal_popcount(lut, mask, _mm_loadu_si128(ptr.offset(idx + 1))));
            idx += 2;
        }

        let sum = _mm_add_epi8(acc[0], acc[1]);
        let sad = _mm_sad_epu8(sum, _mm_setzero_si128());
        res = _mm_add_epi64(res, sad);

        acc[0] = _mm_setzero_si128();
        acc[1] = _mm_setzero_si128();
    }

    union A {
        v: __m128i,
        a: [u64; 2]
    }

    let a = A { v: res };
    let res = a.a[0] + a.a[1];

    let end = end_idx * 8 * 2 * 2;

    res as usize + popcount_scalar_fast(&bits[end..], len - (end * 64))
}

// likely faster than popcount_scalar for long sequences
fn popcount_scalar_fast(bits: &[u64], len: usize) -> usize {
    let mut res = [0usize; 4];
    let end = len / 64;
    let end_idx = end / 4;
    let mut idx = 0;

    for _i in 0..end_idx {
        unsafe {
            res[0] += (*bits.get_unchecked(idx + 0)).count_ones() as usize;
            res[1] += (*bits.get_unchecked(idx + 1)).count_ones() as usize;
            res[2] += (*bits.get_unchecked(idx + 2)).count_ones() as usize;
            res[3] += (*bits.get_unchecked(idx + 3)).count_ones() as usize;
        }

        idx += 4;
    }

    for i in (end_idx * 4)..end {
        res[0] += unsafe { (*bits.get_unchecked(i)).count_ones() as usize };
    }

    let leftover = len % 64;

    if leftover > 0 {
        res[1] += unsafe { (*bits.get_unchecked(end) & ((1 << leftover) - 1)).count_ones() as usize };
    }

    (res[0] + res[1]) + (res[2] + res[3])
}

fn popcount_scalar(bits: &[u64], len: usize) -> usize {
    let mut res = 0;
    let end_idx = len / 64;

    for i in 0..end_idx {
        res += unsafe { (*bits.get_unchecked(i)).count_ones() as usize };
    }

    let leftover = len % 64;

    if leftover > 0 {
        res += unsafe { (*bits.get_unchecked(end_idx) & ((1 << leftover) - 1)).count_ones() as usize };
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_popcount_scalar() {
        assert_eq!(popcount_scalar(&vec![0x0101010101010101; 128], 8192), 1024);
        assert_eq!(popcount_scalar_fast(&vec![0x0101010101010101; 128], 8192), 1024);
        assert_eq!(popcount_scalar(&vec![0b010101], 6), 3);
        assert_eq!(popcount_scalar_fast(&vec![0b010101], 6), 3);
    }

    #[test]
    fn test_popcount_avx() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { popcount_avx(&vec![0x0101010101010101; 128], 8192) }, 1024);
                assert_eq!(unsafe { popcount_avx(&vec![0b010101], 6) }, 3);
            }
        }
    }

    #[test]
    fn test_popcount_sse() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("ssse3") {
                assert_eq!(unsafe { popcount_sse(&vec![0x0101010101010101; 128], 8192) }, 1024);
                assert_eq!(unsafe { popcount_sse(&vec![0b010101], 6) }, 3);
            }
        }
    }
}
