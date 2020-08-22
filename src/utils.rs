#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::{alloc, ptr};

pub fn complement(bits: &[u64]) -> Vec<u64> {
    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { complement_avx(bits) };
        } else if is_x86_feature_detected!("sse2") {
            return unsafe { complement_sse(bits) };
        }
    }

    complement_scalar(bits)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn complement_avx(bits: &[u64]) -> Vec<u64> {
    let ptr = bits.as_ptr() as *const __m256i;
    let end_idx = bits.len() / 4;

    let layout = alloc::Layout::from_size_align_unchecked(bits.len() * 8, 32);
    let res_ptr = alloc::alloc(layout) as *mut __m256i;

    let mask = _mm256_set1_epi64x(0xAAAAAAAAAAAAAAAAu64 as i64);

    for i in 0..end_idx as isize {
        let v = _mm256_loadu_si256(ptr.offset(i));
        let v = _mm256_xor_si256(v, mask);
        _mm256_store_si256(res_ptr.offset(i), v);
    }

    if bits.len() % 4 > 0 {
        let end = end_idx * 4;

        ptr::copy_nonoverlapping(
            complement_scalar(&bits[end..]).as_ptr(),
            res_ptr.offset(end_idx as isize) as *mut u64,
            bits.len() - end
        );
    }

    Vec::from_raw_parts(res_ptr as *mut u64, bits.len(), bits.len())
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse2")]
unsafe fn complement_sse(bits: &[u64]) -> Vec<u64> {
    let ptr = bits.as_ptr() as *const __m128i;
    let end_idx = bits.len() / 2;

    let layout = alloc::Layout::from_size_align_unchecked(bits.len() * 8, 16);
    let res_ptr = alloc::alloc(layout) as *mut __m128i;

    let mask = _mm_set1_epi64x(0xAAAAAAAAAAAAAAAAu64 as i64);

    for i in 0..end_idx as isize {
        let v = _mm_loadu_si128(ptr.offset(i));
        let v = _mm_xor_si128(v, mask);
        _mm_store_si128(res_ptr.offset(i), v);
    }

    if bits.len() % 2 > 0 {
        *(res_ptr.offset(end_idx as isize) as *mut u64) = *complement_scalar(&bits[(end_idx * 2)..]).get_unchecked(0);
    }

    Vec::from_raw_parts(res_ptr as *mut u64, bits.len(), bits.len())
}

fn complement_scalar(bits: &[u64]) -> Vec<u64> {
    unsafe {
        let layout = alloc::Layout::from_size_align_unchecked(bits.len() * 8, 8);
        let res_ptr = alloc::alloc(layout) as *mut u64;

        for i in 0..bits.len() {
            // XOR 0b...10101010 to complement
            *res_ptr.offset(i as isize) = *bits.get_unchecked(i) ^ 0xAAAAAAAAAAAAAAAAu64;
        }

        Vec::from_raw_parts(res_ptr, bits.len(), bits.len())
    }
}
