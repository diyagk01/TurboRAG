use anyhow::{Context, Result};
use ndarray::{Array1, Array2};
use ndarray_npy::read_npy;
use std::cmp::Ordering;
use std::path::PathBuf;
use turbo_quant::TurboQuantizer;

fn hit_at_k(top_indices: &[usize], corpus_rows: &Array1<i32>, target_row: i32, k: usize) -> bool {
    top_indices
        .iter()
        .take(k)
        .any(|&idx| corpus_rows[idx] == target_row)
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        anyhow::bail!(
            "usage: cargo run --release -- <data_dir> [bits=4] [projections=32] [seed=42] [top_k=5]"
        );
    }

    let data_dir = PathBuf::from(&args[1]);
    let bits: u8 = args.get(2).and_then(|x| x.parse().ok()).unwrap_or(4);
    let projections: usize = args.get(3).and_then(|x| x.parse().ok()).unwrap_or(32);
    let seed: u64 = args.get(4).and_then(|x| x.parse().ok()).unwrap_or(42);
    let top_k: usize = args.get(5).and_then(|x| x.parse().ok()).unwrap_or(5);

    let corpus_embeddings: Array2<f32> =
        read_npy(data_dir.join("corpus_embeddings.npy")).context("read corpus_embeddings.npy")?;
    let corpus_source_rows: Array1<i32> =
        read_npy(data_dir.join("corpus_source_rows.npy")).context("read corpus_source_rows.npy")?;
    let query_embeddings: Array2<f32> =
        read_npy(data_dir.join("query_embeddings.npy")).context("read query_embeddings.npy")?;
    let query_source_rows: Array1<i32> =
        read_npy(data_dir.join("query_source_rows.npy")).context("read query_source_rows.npy")?;

    let dim = corpus_embeddings.shape()[1];
    let quantizer = TurboQuantizer::new(dim, bits, projections, seed)
        .context("TurboQuantizer::new failed")?;

    let mut codes = Vec::with_capacity(corpus_embeddings.shape()[0]);
    for row in corpus_embeddings.outer_iter() {
        let vec = row.to_vec();
        let code = quantizer.encode(&vec).context("encode failed")?;
        codes.push(code);
    }

    let n_queries = query_embeddings.shape()[0];
    let mut hit1 = 0usize;
    let mut hit3 = 0usize;
    let mut hit5 = 0usize;

    for qi in 0..n_queries {
        let qvec = query_embeddings.row(qi).to_vec();
        let qrow = query_source_rows[qi];

        let mut scored: Vec<(f32, usize)> = Vec::with_capacity(codes.len());
        for (idx, code) in codes.iter().enumerate() {
            let s = quantizer
                .inner_product_estimate(code, &qvec)
                .context("inner_product_estimate failed")?;
            scored.push((s, idx));
        }

        scored.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(Ordering::Equal)
        });

        let top_indices: Vec<usize> = scored.iter().take(top_k).map(|(_, idx)| *idx).collect();

        if hit_at_k(&top_indices, &corpus_source_rows, qrow, 1) {
            hit1 += 1;
        }
        if hit_at_k(&top_indices, &corpus_source_rows, qrow, 3) {
            hit3 += 1;
        }
        if hit_at_k(&top_indices, &corpus_source_rows, qrow, 5) {
            hit5 += 1;
        }
    }

    println!("============================================================");
    println!("Rust turbo_quant benchmark (source_row match proxy)");
    println!("corpus_size={}", codes.len());
    println!("num_queries={}", n_queries);
    println!("dim={dim} bits={bits} projections={projections} seed={seed}");
    println!("Hit@1: {}/{} = {:.3}", hit1, n_queries, hit1 as f64 / n_queries as f64);
    println!("Hit@3: {}/{} = {:.3}", hit3, n_queries, hit3 as f64 / n_queries as f64);
    println!("Hit@5: {}/{} = {:.3}", hit5, n_queries, hit5 as f64 / n_queries as f64);
    println!("============================================================");

    Ok(())
}
