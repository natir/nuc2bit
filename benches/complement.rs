use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};

mod utils;

fn on_length(c: &mut Criterion) {
    let mut g = c.benchmark_group("Length");

    g.sample_size(100);
    g.warm_up_time(std::time::Duration::from_secs(1));

    for len in (1_000..10_000)
        .step_by(1_000)
        .chain((10_000..=100_000).step_by(10_000))
    {
        let bits = utils::get_bit(len, 0.5);

        g.bench_with_input(BenchmarkId::new("scalar", len), &bits, |b, bits| {
            b.iter(|| {
                nuc2bit::complement::pub_complement_scalar(bits);
            })
        });

        g.bench_with_input(BenchmarkId::new("avx", len), &bits, |b, bits| {
            b.iter(|| unsafe {
                nuc2bit::complement::pub_complement_avx(bits);
            })
        });

        g.bench_with_input(BenchmarkId::new("sse", len), &bits, |b, bits| {
            b.iter(|| unsafe {
                nuc2bit::complement::pub_complement_sse(bits);
            })
        });

        g.bench_with_input(BenchmarkId::new("pub", len), &bits, |b, bits| {
            b.iter(|| {
                nuc2bit::complement::complement(bits);
            })
        });
    }
}

fn on_gc(c: &mut Criterion) {
    let mut g = c.benchmark_group("GC%");

    g.sample_size(100);
    g.warm_up_time(std::time::Duration::from_secs(1));

    for gc_prev in 0..11 {
        let gc = gc_prev as f64 / 10.0;
        let bits = utils::get_bit(20000, gc);

        g.bench_with_input(BenchmarkId::new("scalar", gc), &bits, |b, bits| {
            b.iter(|| {
                nuc2bit::complement::pub_complement_scalar(bits);
            })
        });

        g.bench_with_input(BenchmarkId::new("avx", gc), &bits, |b, bits| {
            b.iter(|| unsafe {
                nuc2bit::complement::pub_complement_avx(bits);
            })
        });

        g.bench_with_input(BenchmarkId::new("sse", gc), &bits, |b, bits| {
            b.iter(|| unsafe {
                nuc2bit::complement::pub_complement_sse(bits);
            })
        });

        g.bench_with_input(BenchmarkId::new("pub", gc), &bits, |b, bits| {
            b.iter(|| {
                nuc2bit::complement::complement(bits);
            })
        });
    }
}

criterion_group!(benches, on_length, on_gc);
criterion_main!(benches);
