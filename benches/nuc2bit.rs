use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

mod utils;

fn on_length(c: &mut Criterion) {
    let mut g = c.benchmark_group("nuc2bit/length");

    g.sample_size(100);
    g.warm_up_time(std::time::Duration::from_secs(1));

    for len in (1_000..10_000)
        .step_by(1_000)
        .chain((10_000..=100_000).step_by(10_000))
    {
        let seq = utils::get_nuc(len, 0.5);

        g.bench_with_input(BenchmarkId::new("lut", len), &seq, |b, seq| {
            b.iter(|| {
                nuc2bit::nuc2bit::pub_encode_lut(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("avx", len), &seq, |b, seq| {
            b.iter(|| unsafe {
                nuc2bit::nuc2bit::pub_encode_avx(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("sse", len), &seq, |b, seq| {
            b.iter(|| unsafe {
                nuc2bit::nuc2bit::pub_encode_sse(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("pub", len), &seq, |b, seq| {
            b.iter(|| {
                nuc2bit::nuc2bit::encode(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("iterator", len), &seq, |b, seq| {
            b.iter(|| for _ in nuc2bit::nuc2bit::Encode::new(seq) {})
        });
    }
}

fn on_gc(c: &mut Criterion) {
    let mut g = c.benchmark_group("nuc2bit/GC%");

    g.sample_size(100);
    g.warm_up_time(std::time::Duration::from_secs(1));

    for gc_prev in 0..11 {
        let gc = gc_prev as f64 / 10.0;
        let seq = utils::get_nuc(20000, gc);

        g.bench_with_input(BenchmarkId::new("lut", gc), &seq, |b, seq| {
            b.iter(|| {
                nuc2bit::nuc2bit::pub_encode_lut(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("avx", gc), &seq, |b, seq| {
            b.iter(|| unsafe {
                nuc2bit::nuc2bit::pub_encode_avx(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("sse", gc), &seq, |b, seq| {
            b.iter(|| unsafe {
                nuc2bit::nuc2bit::pub_encode_sse(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("pub", gc), &seq, |b, seq| {
            b.iter(|| {
                nuc2bit::nuc2bit::encode(seq);
            })
        });

        g.bench_with_input(BenchmarkId::new("iterator", gc), &seq, |b, seq| {
            b.iter(|| for _ in nuc2bit::nuc2bit::Encode::new(seq) {})
        });
    }
}

criterion_group!(benches, on_length, on_gc);
criterion_main!(benches);
