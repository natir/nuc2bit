pub fn encoding_equals(a_bits: &[u64], b_bits: &[u64], len: usize) -> bool {
    let mut equals = true;
    let end_idx = len / 32;

    for i in 0..end_idx {
        unsafe {
            equals &= *a_bits.get_unchecked(i) == *b_bits.get_unchecked(i);
        }
    }

    let leftover = len % 32;

    if leftover > 0 {
        let mask = (1 << (leftover * 2)) - 1;

        unsafe {
            equals &= (*a_bits.get_unchecked(end_idx) & mask) == (*b_bits.get_unchecked(end_idx) & mask);
        }
    }

    equals
}
