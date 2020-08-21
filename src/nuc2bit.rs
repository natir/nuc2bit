#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::alloc;

pub fn nuc2bit(nuc: &[u8]) -> Vec<u64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { nuc2bit_movemask_avx(nuc) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { nuc2bit_movemask_sse(nuc) };
        }
    }

    nuc2bit_lut(nuc)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn nuc2bit_movemask_avx(nuc: &[u8]) -> Vec<u64> {
    let ptr = nuc.as_ptr() as *const __m256i;
    let end_idx = nuc.len() / 32;
    let len = end_idx + if nuc.len() % 32 == 0 {0} else {1};

    let layout = alloc::Layout::from_size_align_unchecked(len * 8, 8);
    let res_ptr = alloc::alloc(layout) as *mut u64;

    for i in 0..end_idx as isize {
        let v = _mm256_loadu_si256(ptr.offset(i));

        // permute because unpacks works on the low/high 64 bits in each lane
        let v = _mm256_permute4x64_epi64(v, 0b11011000);

        // shift each group of two bits for each nucleotide to the end of each byte
        let lo = _mm256_slli_epi64(v, 6);
        let hi = _mm256_slli_epi64(v, 5);

        // interleave bytes then extract the bit at the end of each byte
        let a = _mm256_unpackhi_epi8(lo, hi);
        let b = _mm256_unpacklo_epi8(lo, hi);

        // zero extend after movemask
        let a = (_mm256_movemask_epi8(a) as u32) as u64;
        let b = (_mm256_movemask_epi8(b) as u32) as u64;

        *res_ptr.offset(i) = (a << 32) | b;
    }

    if nuc.len() % 32 > 0 {
        *res_ptr.offset(end_idx as isize) = *nuc2bit_lut(&nuc[(end_idx * 32)..]).get_unchecked(0);
    }

    Vec::from_raw_parts(res_ptr, len, len)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "ssse3")]
unsafe fn nuc2bit_movemask_sse(nuc: &[u8]) -> Vec<u64> {
    let ptr = nuc.as_ptr() as *const __m128i;
    let end_idx = nuc.len() / 16;
    let len = nuc.len() / 32 + if nuc.len() % 32 == 0 {0} else {1};

    let layout = alloc::Layout::from_size_align_unchecked(len * 8, 8);
    let res_ptr = alloc::alloc(layout) as *mut u32;

    for i in 0..end_idx as isize {
        let v = _mm_loadu_si128(ptr.offset(i));

        // shift each group of two bits for each nucleotide to the end of each byte
        let lo = _mm_slli_epi64(v, 6);
        let hi = _mm_slli_epi64(v, 5);

        // interleave bytes then extract the bit at the end of each byte
        let a = _mm_unpackhi_epi8(lo, hi);
        let b = _mm_unpacklo_epi8(lo, hi);
        let a = _mm_movemask_epi8(a);
        let b = _mm_movemask_epi8(b);

        *res_ptr.offset(i) = ((a << 16) | b) as u32;
    }

    if nuc.len() % 16 > 0 {
        *res_ptr.offset(end_idx as isize) = *nuc2bit_lut(&nuc[(end_idx * 16)..]).get_unchecked(0) as u32;
    }

    Vec::from_raw_parts(res_ptr as *mut u64, len, len)
}

static BYTE_LUT: [u8; 128] = {
    let mut lut = [0u8; 128];
    lut[b'a' as usize] = 0b00;
    lut[b't' as usize] = 0b11;
    lut[b'u' as usize] = 0b11;
    lut[b'c' as usize] = 0b01;
    lut[b'g' as usize] = 0b10;
    lut[b'A' as usize] = 0b00;
    lut[b'T' as usize] = 0b11;
    lut[b'U' as usize] = 0b11;
    lut[b'C' as usize] = 0b01;
    lut[b'G' as usize] = 0b10;
    lut
};

fn nuc2bit_lut(nuc: &[u8]) -> Vec<u64> {
    let mut res = vec![0u64; (nuc.len() / 32) + if nuc.len() % 32 == 0 {0} else {1}];

    for i in 0..nuc.len() {
        let offset = i / 32;
        let shift = (i % 32) << 1;

        unsafe {
            *res.get_unchecked_mut(offset) = *res.get_unchecked(offset)
                | ((*BYTE_LUT.get_unchecked(*nuc.get_unchecked(i) as usize) as u64) << shift);
        }
    }

    res
}
