#[cfg(target_arch = "x86")]
use std::arch::x86::*;

#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

use std::alloc;

pub fn decode(bits: &[u64], len: usize) -> Vec<u8> {
    if len > (bits.len() * 32) {
        panic!(
            "The length {} is greater than the number of nucleotides!",
            len
        );
    }

    #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return unsafe { decode_shuffle_avx(bits, len) };
        } else if is_x86_feature_detected!("sse4.1") {
            return unsafe { decode_shuffle_sse(bits, len) };
        }
    }

    decode_lut(bits, len)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn decode_shuffle_avx(bits: &[u64], len: usize) -> Vec<u8> {
    let layout = alloc::Layout::from_size_align_unchecked(bits.len() * 32, 32);
    let ptr = alloc::alloc(layout) as *mut __m256i;

    let shuffle_mask = _mm256_set_epi32(
        0x07070707, 0x06060606, 0x05050505, 0x04040404, 0x03030303, 0x02020202, 0x01010101,
        0x00000000,
    );
    let lo_mask = _mm256_set1_epi16(0b0000110000000011);
    let lut_i32 =
        (b'A' as i32) | ((b'C' as i32) << 8) | ((b'T' as i32) << 16) | ((b'G' as i32) << 24);
    let lut = _mm256_set_epi32(
        b'G' as i32,
        b'T' as i32,
        b'C' as i32,
        lut_i32,
        b'G' as i32,
        b'T' as i32,
        b'C' as i32,
        lut_i32,
    );

    for i in 0..bits.len() {
        let curr = *bits.get_unchecked(i) as i64;
        let v = _mm256_set1_epi64x(curr);

        // duplicate each byte four times
        let v1 = _mm256_shuffle_epi8(v, shuffle_mask);

        // separately right shift each 16-bit chunk by 0 or 4 bits
        let v2 = _mm256_srli_epi16(v1, 4);

        // merge together shifted chunks
        let v = _mm256_blend_epi16(v1, v2, 0b10101010i32);

        // only keep two bits in each byte
        // either 0b0011 or 0b1100
        let v = _mm256_and_si256(v, lo_mask);

        // use lookup table to convert nucleotide bits to bytes
        let v = _mm256_shuffle_epi8(lut, v);
        _mm256_store_si256(ptr.offset(i as isize), v);
    }

    Vec::from_raw_parts(ptr as *mut u8, len, bits.len() * 32)
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "sse4.1")]
unsafe fn decode_shuffle_sse(bits: &[u64], len: usize) -> Vec<u8> {
    let layout = alloc::Layout::from_size_align_unchecked(bits.len() * 32, 16);
    let ptr = alloc::alloc(layout) as *mut __m128i;

    let bits_ptr = bits.as_ptr() as *const i32;

    let shuffle_mask = _mm_set_epi32(0x03030303, 0x02020202, 0x01010101, 0x00000000);
    let lo_mask = _mm_set1_epi16(0b0000110000000011);
    let lut_i32 =
        (b'A' as i32) | ((b'C' as i32) << 8) | ((b'T' as i32) << 16) | ((b'G' as i32) << 24);
    let lut = _mm_set_epi32(b'G' as i32, b'T' as i32, b'C' as i32, lut_i32);

    for i in 0..(bits.len() * 2) as isize {
        let curr = *bits_ptr.offset(i);
        let v = _mm_set1_epi32(curr);

        // duplicate each byte four times
        let v1 = _mm_shuffle_epi8(v, shuffle_mask);

        // separately right shift each 16-bit chunk by 0 or 4 bits
        let v2 = _mm_srli_epi16(v1, 4);

        // merge together shifted chunks
        let v = _mm_blend_epi16(v1, v2, 0b10101010i32);

        // only keep two bits in each byte
        // either 0b0011 or 0b1100
        let v = _mm_and_si128(v, lo_mask);

        // use lookup table to convert nucleotide bits to bytes
        let v = _mm_shuffle_epi8(lut, v);
        _mm_store_si128(ptr.offset(i), v);
    }

    Vec::from_raw_parts(ptr as *mut u8, len, bits.len() * 32)
}

static BITS_LUT: [u8; 4] = {
    let mut lut = [0u8; 4];
    lut[0b00] = b'A';
    lut[0b10] = b'T';
    lut[0b01] = b'C';
    lut[0b11] = b'G';
    lut
};

fn decode_lut(bits: &[u64], len: usize) -> Vec<u8> {
    let layout = unsafe { alloc::Layout::from_size_align_unchecked(len, 1) };
    let res_ptr = unsafe { alloc::alloc(layout) };

    for i in 0..len {
        let offset = i >> 5;
        let shift = (i & 31) << 1;
        let curr = unsafe { *bits.get_unchecked(offset) };

        unsafe {
            *res_ptr.offset(i as isize) =
                *BITS_LUT.get_unchecked(((curr >> shift) & 0b11) as usize);
        }
    }

    unsafe { Vec::from_raw_parts(res_ptr, len, len) }
}

#[cfg(feature = "bench-internals")]
pub fn pub_decode_lut(bits: &[u64], len: usize) -> Vec<u8> {
    decode_lut(bits, len)
}

#[cfg(feature = "bench-internals")]
#[target_feature(enable = "avx2")]
pub unsafe fn pub_decode_avx(bits: &[u64], len: usize) -> Vec<u8> {
    decode_shuffle_avx(bits, len)
}

#[cfg(feature = "bench-internals")]
#[target_feature(enable = "sse2")]
pub unsafe fn pub_decode_sse(bits: &[u64], len: usize) -> Vec<u8> {
    decode_shuffle_sse(bits, len)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_decode_lut() {
        assert_eq!(
            decode_lut(
                &vec![0b1101100011011000110110001101100011011000110110001101100011011000],
                32
            ),
            b"ATCGATCGATCGATCGATCGATCGATCGATCG"
        );
        assert_eq!(decode_lut(&vec![0b11011000], 4), b"ATCG");
    }

    #[test]
    fn test_decode_avx() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("avx2") {
                assert_eq!(
                    unsafe {
                        decode_shuffle_avx(
                            &vec![
                                0b1101100011011000110110001101100011011000110110001101100011011000,
                            ],
                            32,
                        )
                    },
                    b"ATCGATCGATCGATCGATCGATCGATCGATCG"
                );
                assert_eq!(unsafe { decode_shuffle_avx(&vec![0b11011000], 4) }, b"ATCG");
            }
        }
    }

    #[test]
    fn test_decode_sse() {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        {
            if is_x86_feature_detected!("sse4.1") {
                assert_eq!(
                    unsafe {
                        decode_shuffle_sse(
                            &vec![
                                0b1101100011011000110110001101100011011000110110001101100011011000,
                            ],
                            32,
                        )
                    },
                    b"ATCGATCGATCGATCGATCGATCGATCGATCG"
                );
                assert_eq!(unsafe { decode_shuffle_sse(&vec![0b11011000], 4) }, b"ATCG");
            }
        }
    }
}
