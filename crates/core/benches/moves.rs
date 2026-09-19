use criterion::{black_box, criterion_group, criterion_main, Criterion};
use g2048_core::{apply_move, Direction};

fn bench_apply_move(c: &mut Criterion) {
    // Plateau varié (pas de cases vides pour éviter les cas triviaux).
    let board: u64 = 0x_1234_2341_3412_4123;

    let mut group = c.benchmark_group("apply_move");
    for dir in Direction::ALL {
        group.bench_function(format!("{dir:?}"), |b| {
            b.iter(|| apply_move(black_box(board), black_box(dir)))
        });
    }
    group.finish();
}

criterion_group!(benches, bench_apply_move);
criterion_main!(benches);
