use rand::distributions::Distribution;

pub fn get_nuc(length: usize, gc: f64) -> Vec<u8> {
    let mut rng = rand::thread_rng();

    let dna = [b'A', b'T', b'C', b'G'];
    let prob = [1.0 - gc, 1.0 - gc, gc, gc];

    let dist = rand::distributions::WeightedIndex::new(&prob).unwrap(); // value of weight can't be negative

    let mut seq = Vec::with_capacity(length);

    for _ in 0..length {
        seq.push(dna[dist.sample(&mut rng)]);
    }

    seq
}

pub fn get_bit(length: usize, gc: f64) -> Vec<u64> {
    nuc2bit::nuc2bit::encode(&get_nuc(length, gc))
}
