#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn hamming(a: &[u64], b: &[u64], len: usize) -> usize {
    if len / 64 <= 8 {
        return hamming_scalar(a, b, len);
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { hamming_avx(a, b, len) };
        } else if is_x86_feature_detected!("ssse3") {
            return unsafe { hamming_sse(a, b, len) };
        }
    }

    hamming_scalar_fast(a, b, len)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn hamming_avx(a: &[u64], b: &[u64], len: usize) -> usize {
    let end_idx = (((len / 32) / 8) / 2) / 4;

    let mut res = _mm256_setzero_si256();
    let mut acc = [_mm256_setzero_si256(); 2];
    let a_ptr = a.as_ptr() as *const __m256i;
    let b_ptr = b.as_ptr() as *const __m256i;

    // bit patterns:
    // 0b1111, 0b1110, 0b1101, 0b1100, 0b1011, 0b1010, 0b1001, 0b1000, 0b0111, 0b0110, 0b0101,
    // 0b0100, 0b0011, 0b0010, 0b0001, 0b0000
    // lut stores the hamming distance for each bit pattern:
    let lut = _mm256_set_epi64x(0x0202020102020201, 0x0202020101010100, 0x0202020102020201, 0x0202020101010100);
    let mask = _mm256_set1_epi8(0x0F);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "avx2")]
    #[inline]
    unsafe fn internal_hamming(lut: __m256i, mask: __m256i, a: __m256i, b: __m256i) -> __m256i {
        let xor = _mm256_xor_si256(a, b);
        let lo_nybbles_lut = _mm256_shuffle_epi8(lut, _mm256_and_si256(xor, mask));
        let hi_nybbles_lut = _mm256_shuffle_epi8(lut, _mm256_srli_epi16(xor, 4));
        _mm256_add_epi8(lo_nybbles_lut, hi_nybbles_lut)
    }

    let mut idx = 0;

    for _i in 0..end_idx {
        for _j in 0..8 {
            let d = internal_hamming(lut, mask, _mm256_loadu_si256(a_ptr.offset(idx + 0)), _mm256_loadu_si256(b_ptr.offset(idx + 0)));
            acc[0] = _mm256_add_epi8(acc[0], d);

            let d = internal_hamming(lut, mask, _mm256_loadu_si256(a_ptr.offset(idx + 1)), _mm256_loadu_si256(b_ptr.offset(idx + 1)));
            acc[1] = _mm256_add_epi8(acc[1], d);

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

    let arr = A { v: res };
    let res = (arr.a[0] + arr.a[1]) + (arr.a[2] + arr.a[3]);

    let end = end_idx * 8 * 2 * 4;

    res as usize + hamming_scalar_fast(&a[end..], &b[end..], len - (end * 32))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn hamming_sse(a: &[u64], b: &[u64], len: usize) -> usize {
    let end_idx = (((len / 32) / 8) / 2) / 2;

    let mut res = _mm_setzero_si128();
    let mut acc = [_mm_setzero_si128(); 2];
    let a_ptr = a.as_ptr() as *const __m128i;
    let b_ptr = b.as_ptr() as *const __m128i;

    let lut = _mm_set_epi64x(0x0202020102020201, 0x0202020101010100);
    let mask = _mm_set1_epi8(0x0F);

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    #[target_feature(enable = "ssse3")]
    #[inline]
    unsafe fn internal_hamming(lut: __m128i, mask: __m128i, a: __m128i, b: __m128i) -> __m128i {
        let xor = _mm_xor_si128(a, b);
        let lo_nybbles_lut = _mm_shuffle_epi8(lut, _mm_and_si128(xor, mask));
        let hi_nybbles_lut = _mm_shuffle_epi8(lut, _mm_srli_epi16(xor, 4));
        _mm_add_epi8(lo_nybbles_lut, hi_nybbles_lut)
    }

    let mut idx = 0;

    for _i in 0..end_idx {
        for _j in 0..8 {
            let d = internal_hamming(lut, mask, _mm_loadu_si128(a_ptr.offset(idx + 0)), _mm_loadu_si128(b_ptr.offset(idx + 0)));
            acc[0] = _mm_add_epi8(acc[0], d);

            let d = internal_hamming(lut, mask, _mm_loadu_si128(a_ptr.offset(idx + 1)), _mm_loadu_si128(b_ptr.offset(idx + 1)));
            acc[1] = _mm_add_epi8(acc[1], d);

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

    let arr = A { v: res };
    let res = arr.a[0] + arr.a[1];

    let end = end_idx * 8 * 2 * 2;

    res as usize + hamming_scalar_fast(&a[end..], &b[end..], len - (end * 32))
}

// likely faster than hamming_scalar for long sequences
fn hamming_scalar_fast(a: &[u64], b: &[u64], len: usize) -> usize {
    let mut res = [0usize; 4];
    let end = len / 32;
    let end_idx = end / 4;
    let mut idx = 0;

    let mask = 0x5555555555555555u64; // 0b...01010101

    for _i in 0..end_idx {
        unsafe {
            let xor = *a.get_unchecked(idx + 0) ^ *b.get_unchecked(idx + 0);
            res[0] += (((xor >> 1) | xor) & mask).count_ones() as usize;

            let xor = *a.get_unchecked(idx + 1) ^ *b.get_unchecked(idx + 1);
            res[1] += (((xor >> 1) | xor) & mask).count_ones() as usize;

            let xor = *a.get_unchecked(idx + 2) ^ *b.get_unchecked(idx + 2);
            res[2] += (((xor >> 1) | xor) & mask).count_ones() as usize;

            let xor = *a.get_unchecked(idx + 3) ^ *b.get_unchecked(idx + 3);
            res[3] += (((xor >> 1) | xor) & mask).count_ones() as usize;
        }

        idx += 4;
    }

    for i in (end_idx * 4)..end {
        let xor = unsafe { *a.get_unchecked(i) ^ *b.get_unchecked(i) };
        res[0] += (((xor >> 1) | xor) & mask).count_ones() as usize;
    }

    let leftover = (len % 32) * 2;

    if leftover > 0 {
        let xor = unsafe { (*a.get_unchecked(end) ^ *b.get_unchecked(end)) & ((1 << leftover) - 1) };
        res[1] += (((xor >> 1) | xor) & mask).count_ones() as usize;
    }

    (res[0] + res[1]) + (res[2] + res[3])
}

fn hamming_scalar(a: &[u64], b: &[u64], len: usize) -> usize {
    let mut res = 0;
    let end_idx = len / 32;

    let mask = 0x5555555555555555u64; // 0b...01010101

    for i in 0..end_idx {
        let xor = unsafe { *a.get_unchecked(i) ^ *b.get_unchecked(i) };
        res += (((xor >> 1) | xor) & mask).count_ones() as usize;
    }

    let leftover = (len % 32) * 2;

    if leftover > 0 {
        let xor = unsafe { (*a.get_unchecked(end_idx) ^ *b.get_unchecked(end_idx)) & ((1 << leftover) - 1) };
        res += (((xor >> 1) | xor) & mask).count_ones() as usize;
    }

    res
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hamming_scalar() {
        assert_eq!(hamming_scalar(&vec![0x0101010101010101; 128], &vec![0x0101010101010100; 128], 4096), 128);
        assert_eq!(hamming_scalar_fast(&vec![0x0101010101010101; 128], &vec![0x0101010101010100; 128], 4096), 128);
        assert_eq!(hamming_scalar(&vec![0b010101], &vec![0b010100], 3), 1);
        assert_eq!(hamming_scalar_fast(&vec![0b010101], &vec![0b010100], 3), 1);
    }

    #[test]
    fn test_hamming_avx() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { hamming_avx(&vec![0x0101010101010101; 128], &vec![0x0101010101010100; 128], 4096) }, 128);
                assert_eq!(unsafe { hamming_avx(&vec![0b010101], &vec![0b010100], 3) }, 1);
            }
        }
    }

    #[test]
    fn test_hamming_sse() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("ssse3") {
                assert_eq!(unsafe { hamming_sse(&vec![0x0101010101010101; 128], &vec![0x0101010101010100; 128], 4096) }, 128);
                assert_eq!(unsafe { hamming_sse(&vec![0b010101], &vec![0b010100], 3) }, 1);
            }
        }
    }
}
