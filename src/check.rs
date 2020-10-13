#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

pub fn check(nuc: &[u8]) -> bool {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { check_avx(nuc) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { check_sse(nuc) };
        }
    }

    check_scalar(nuc)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn check_avx(nuc: &[u8]) -> bool {
    let ptr = nuc.as_ptr() as *const __m256i;
    let end_idx = nuc.len() / 32;

    let lut = {
        let mut lut_hi = 0i64;
        lut_hi |= 1i64 << ((b'A' as i64) - 64);
        lut_hi |= 1i64 << ((b'T' as i64) - 64);
        lut_hi |= 1i64 << ((b'U' as i64) - 64);
        lut_hi |= 1i64 << ((b'C' as i64) - 64);
        lut_hi |= 1i64 << ((b'G' as i64) - 64);
        lut_hi |= 1i64 << ((b'a' as i64) - 64);
        lut_hi |= 1i64 << ((b't' as i64) - 64);
        lut_hi |= 1i64 << ((b'u' as i64) - 64);
        lut_hi |= 1i64 << ((b'c' as i64) - 64);
        lut_hi |= 1i64 << ((b'g' as i64) - 64);
        _mm256_set_epi64x(!lut_hi, -1i64, !lut_hi, -1i64)
    };
    let shift_lut = _mm256_set_epi64x(0i64, 0x8040201008040201u64 as i64, 0i64, 0x8040201008040201u64 as i64);
    let mask = _mm256_set1_epi8(0b00000111);

    for i in 0..end_idx as isize {
        let v = _mm256_loadu_si256(ptr.offset(i));
        let zero_neg = _mm256_max_epi8(v, _mm256_setzero_si256());
        let lo_lut = _mm256_shuffle_epi8(lut, zero_neg);
        let hi = _mm256_and_si256(_mm256_srli_epi16(v, 4), mask);
        // if MSB of a byte in v is 1, then the corresponding byte in
        // lo_lut should be all ones, guaranteeing that the check fails
        let hi_lut = _mm256_shuffle_epi8(shift_lut, hi); // convert byte x into (1 << x)

        // check if (lo_lut & hi_lut) has any ones
        if _mm256_testz_si256(lo_lut, hi_lut) != 0 {
            return false;
        }
    }

    if nuc.len() % 32 > 0 {
        let end = end_idx * 32;
        return check_scalar(&nuc[end..]);
    }

    true
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn check_sse(nuc: &[u8]) -> bool {
    let ptr = nuc.as_ptr() as *const __m128i;
    let end_idx = nuc.len() / 16;

    let lut = {
        let mut lut_hi = 0i64;
        lut_hi |= 1i64 << ((b'A' as i64) - 64);
        lut_hi |= 1i64 << ((b'T' as i64) - 64);
        lut_hi |= 1i64 << ((b'U' as i64) - 64);
        lut_hi |= 1i64 << ((b'C' as i64) - 64);
        lut_hi |= 1i64 << ((b'G' as i64) - 64);
        lut_hi |= 1i64 << ((b'a' as i64) - 64);
        lut_hi |= 1i64 << ((b't' as i64) - 64);
        lut_hi |= 1i64 << ((b'u' as i64) - 64);
        lut_hi |= 1i64 << ((b'c' as i64) - 64);
        lut_hi |= 1i64 << ((b'g' as i64) - 64);
        _mm_set_epi64x(!lut_hi, -1i64)
    };
    let shift_lut = _mm_set_epi64x(0i64, 0x8040201008040201u64 as i64);
    let mask = _mm_set1_epi8(0b00000111);

    for i in 0..end_idx as isize {
        let v = _mm_loadu_si128(ptr.offset(i));
        let zero_neg = _mm_max_epi8(v, _mm_setzero_si128());
        let lo_lut = _mm_shuffle_epi8(lut, zero_neg);
        let hi = _mm_and_si128(_mm_srli_epi16(v, 4), mask);
        // if MSB of a byte in v is 1, then the corresponding byte in
        // lo_lut should be all ones, guaranteeing that the check fails
        let hi_lut = _mm_shuffle_epi8(shift_lut, hi); // convert byte x into (1 << x)

        // check if (lo_lut & hi_lut) has any ones
        if _mm_testz_si128(lo_lut, hi_lut) != 0 {
            return false;
        }
    }

    if nuc.len() % 16 > 0 {
        let end = end_idx * 16;
        return check_scalar(&nuc[end..]);
    }

    true
}

static CHECK_LUT: [bool; 256] = {
    let mut lut = [true; 256];
    lut[b'A' as usize] = false;
    lut[b'T' as usize] = false;
    lut[b'U' as usize] = false;
    lut[b'C' as usize] = false;
    lut[b'G' as usize] = false;
    lut[b'a' as usize] = false;
    lut[b't' as usize] = false;
    lut[b'u' as usize] = false;
    lut[b'c' as usize] = false;
    lut[b'g' as usize] = false;
    lut
};

fn check_scalar(nuc: &[u8]) -> bool {
    unsafe {
        let end_idx = nuc.len() / 4;

        for i in 0..end_idx {
            let a = (*CHECK_LUT.get_unchecked(*nuc.get_unchecked(i * 4 + 0) as usize))
                | (*CHECK_LUT.get_unchecked(*nuc.get_unchecked(i * 4 + 1) as usize));
            let b = (*CHECK_LUT.get_unchecked(*nuc.get_unchecked(i * 4 + 2) as usize))
                | (*CHECK_LUT.get_unchecked(*nuc.get_unchecked(i * 4 + 3) as usize));

            if a | b {
                return false;
            }
        }

        for i in (end_idx * 4)..nuc.len() {
            if *CHECK_LUT.get_unchecked(*nuc.get_unchecked(i) as usize) {
                return false;
            }
        }

        true
    }
}

#[cfg(feature = "bench-internals")]
pub fn pub_check_scalar(nuc: &[u8]) -> bool {
    check_scalar(nuc)
}

#[cfg(feature = "bench-internals")]
#[target_feature(enable = "avx2")]
pub unsafe fn pub_check_avx(nuc: &[u8]) -> bool {
    check_avx(nuc)
}

#[cfg(feature = "bench-internals")]
#[target_feature(enable = "sse4.1")]
pub unsafe fn pub_check_sse(nuc: &[u8]) -> bool {
    check_sse(nuc)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_check_scalar() {
        assert_eq!(check_scalar(b"AUCGATCGATCGATCGATCGATCGATCGATCG"), true);
        assert_eq!(check_scalar(b"AUCGATCGATCGATCGATCGATCGATCGATCGb"), false);
        assert_eq!(check_scalar(b"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"), false);
        assert_eq!(check_scalar(b"ATUCG"), true);
        assert_eq!(check_scalar(b"ATUCG "), false);
    }

    #[test]
    fn test_check_avx() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                assert_eq!(unsafe { check_avx(b"AUCGATCGATCGATCGATCGATCGATCGATCG") }, true);
                assert_eq!(unsafe { check_avx(b"AUCGATCGATCGATCGATCGATCGATCGATCGb") }, false);
                assert_eq!(unsafe { check_avx(b"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB") }, false);
                assert_eq!(unsafe { check_avx(b"ATUCG") }, true);
                assert_eq!(unsafe { check_avx(b"ATUCG ") }, false);
            }
        }
    }

    #[test]
    fn test_check_sse() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                assert_eq!(unsafe { check_sse(b"AUCGATCGATCGATCGATCGATCGATCGATCG") }, true);
                assert_eq!(unsafe { check_sse(b"AUCGATCGATCGATCGATCGATCGATCGATCGb") }, false);
                assert_eq!(unsafe { check_sse(b"BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB") }, false);
                assert_eq!(unsafe { check_sse(b"ATUCG") }, true);
                assert_eq!(unsafe { check_sse(b"ATUCG ") }, false);
            }
        }
    }
}
