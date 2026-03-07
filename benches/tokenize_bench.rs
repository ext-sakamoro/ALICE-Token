//! ALICE-Token ベンチマーク

use alice_token::{Tokenizer, VocabBuilder};
use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};

fn build_bench_vocab() -> alice_token::Vocab {
    let mut builder = VocabBuilder::new();
    for b in 0u16..=255 {
        builder.add_token(vec![b as u8]);
    }
    // 頻出英語バイグラム
    let merges: &[(&[u8], &[u8])] = &[
        (b"t", b"h"),
        (b"th", b"e"),
        (b"i", b"n"),
        (b"a", b"n"),
        (b"an", b"d"),
        (b"e", b"r"),
        (b"o", b"n"),
        (b"r", b"e"),
        (b"i", b"s"),
        (b"o", b"f"),
        (b"t", b"o"),
        (b"i", b"t"),
        (b"h", b"a"),
        (b"ha", b"t"),
        (b"a", b"t"),
        (b"s", b"t"),
    ];
    for &(l, r) in merges {
        builder.add_merge(l, r);
    }
    builder.build()
}

fn bench_encode(c: &mut Criterion) {
    let vocab = build_bench_vocab();
    let tok = Tokenizer::new(vocab);
    let base = b"the quick brown fox jumps over the lazy dog and the cat ";

    let mut group = c.benchmark_group("encode");
    for size in [64, 256, 1024, 4096, 16384] {
        let input = base.repeat(size / base.len() + 1);
        let input = &input[..size];
        group.bench_with_input(BenchmarkId::from_parameter(size), &input, |b, input| {
            b.iter(|| tok.encode(black_box(input)));
        });
    }
    group.finish();
}

fn bench_decode(c: &mut Criterion) {
    let vocab = build_bench_vocab();
    let tok = Tokenizer::new(vocab);
    let base = b"the quick brown fox jumps over the lazy dog and the cat ";

    let mut group = c.benchmark_group("decode");
    for size in [64, 256, 1024, 4096] {
        let input = base.repeat(size / base.len() + 1);
        let ids = tok.encode(&input[..size]);
        group.bench_with_input(BenchmarkId::from_parameter(size), &ids, |b, ids| {
            b.iter(|| tok.decode(black_box(ids)));
        });
    }
    group.finish();
}

fn bench_parallel(c: &mut Criterion) {
    let vocab = build_bench_vocab();
    let tok = Tokenizer::new(vocab);
    let input = b"the quick brown fox jumps over the lazy dog ".repeat(1000);

    let mut group = c.benchmark_group("parallel");
    for chunk_size in [256, 1024, 4096] {
        group.bench_with_input(
            BenchmarkId::from_parameter(chunk_size),
            &chunk_size,
            |b, &cs| {
                b.iter(|| tok.encode_parallel(black_box(&input), cs));
            },
        );
    }
    group.finish();
}

fn bench_roundtrip(c: &mut Criterion) {
    let vocab = build_bench_vocab();
    let tok = Tokenizer::new(vocab);
    let input = b"the quick brown fox jumps over the lazy dog ".repeat(100);

    c.bench_function("roundtrip_4kb", |b| {
        b.iter(|| {
            let ids = tok.encode(black_box(&input));
            tok.decode(black_box(&ids))
        });
    });
}

criterion_group!(
    benches,
    bench_encode,
    bench_decode,
    bench_parallel,
    bench_roundtrip
);
criterion_main!(benches);
