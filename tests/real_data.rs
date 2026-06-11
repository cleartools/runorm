//! Real-data tests on a PBMC 3k subset (genuine 10x counts: 400 cells x 800 genes, see
//! `tests/data/pbmc_subset.mtx`). Validates PFlog1pPF on real single-cell data against an
//! independent dense oracle and checks that overdispersion estimation is sane.

use std::fs;
use std::path::{Path, PathBuf};

use runorm::{estimate_overdispersion, normalize_csr, CsrCounts, NormParams, PfTarget};

fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/data/pbmc_subset.mtx")
}

/// Minimal Matrix Market (coordinate, 1-indexed) reader -> CSR. Tolerates unsorted entries.
fn load_mtx(path: &Path) -> CsrCounts {
    let text = fs::read_to_string(path).unwrap();
    let mut lines = text.lines().filter(|l| !l.starts_with('%'));
    let mut header = lines.next().unwrap().split_whitespace();
    let n_rows: usize = header.next().unwrap().parse().unwrap();
    let n_cols: usize = header.next().unwrap().parse().unwrap();
    let nnz: usize = header.next().unwrap().parse().unwrap();

    let (mut rows, mut cols, mut vals) = (vec![0usize; nnz], vec![0usize; nnz], vec![0f64; nnz]);
    let mut row_ptr = vec![0usize; n_rows + 1];
    for (k, line) in lines.enumerate() {
        let mut p = line.split_whitespace();
        let r = p.next().unwrap().parse::<usize>().unwrap() - 1;
        let c = p.next().unwrap().parse::<usize>().unwrap() - 1;
        let v: f64 = p.next().unwrap().parse().unwrap();
        rows[k] = r;
        cols[k] = c;
        vals[k] = v;
        row_ptr[r + 1] += 1;
    }
    for i in 0..n_rows {
        row_ptr[i + 1] += row_ptr[i];
    }
    let indptr = row_ptr.clone();
    let mut next = row_ptr;
    let (mut data, mut indices) = (vec![0f64; nnz], vec![0usize; nnz]);
    for k in 0..nnz {
        let dst = next[rows[k]];
        data[dst] = vals[k];
        indices[dst] = cols[k];
        next[rows[k]] += 1;
    }
    CsrCounts::new(n_rows, n_cols, data, indices, indptr).unwrap()
}

/// Independent dense PFlog1pPF oracle.
fn dense_oracle(c: &CsrCounts, k: f64) -> Vec<f64> {
    let d = c.n_cols;
    let mut out = vec![0f64; c.n_rows * d];
    for i in 0..c.n_rows {
        let s: f64 = c.data[c.indptr[i]..c.indptr[i + 1]].iter().sum();
        let mut row = vec![0f64; d];
        for p in c.indptr[i]..c.indptr[i + 1] {
            row[c.indices[p]] = (c.data[p] * k / s).ln_1p();
        }
        let m = row.iter().sum::<f64>() / d as f64;
        for j in 0..d {
            out[i * d + j] = row[j] - m;
        }
    }
    out
}

#[test]
fn pflog1ppf_matches_dense_oracle_on_pbmc() {
    let counts = load_mtx(&fixture());
    assert_eq!((counts.n_rows, counts.n_cols), (400, 800));

    for params in [
        NormParams::default(), // PF to mean depth
        NormParams { target: PfTarget::EstimateAlpha, log1p: true, center: true },
    ] {
        let (m, report) = normalize_csr(&counts, &params).unwrap();
        assert!(report.k > 0.0 && report.k.is_finite());

        let oracle = dense_oracle(&counts, report.k);
        let got = m.densify();
        let max_err = got
            .iter()
            .zip(oracle.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f64, f64::max);
        assert!(max_err < 1e-9, "max abs error {max_err} for target {:?}", params.target);

        // CLR property: every cell's transformed values sum to ~0.
        let d = counts.n_cols;
        for i in 0..counts.n_rows {
            let s: f64 = got[i * d..(i + 1) * d].iter().sum();
            assert!(s.abs() < 1e-6, "cell {i} does not sum to zero: {s}");
        }
        // Sparsity is preserved (log1pPF keeps the input nonzeros).
        assert_eq!(m.nnz(), counts.nnz());
    }
}

#[test]
fn overdispersion_is_positive_on_real_counts() {
    let counts = load_mtx(&fixture());
    let od = estimate_overdispersion(&counts).unwrap();
    // Real single-cell counts are overdispersed (negative-binomial), so alpha > 0.
    assert!(od.alpha > 0.0, "expected positive overdispersion, got {}", od.alpha);
    assert!(od.k > 0.0 && od.k.is_finite());

    let total: f64 = counts.data.iter().sum();
    assert!((od.mean_depth - total / counts.n_rows as f64).abs() < 1e-6);
    // K = 4 * alpha * mean_depth.
    assert!((od.k - 4.0 * od.alpha * od.mean_depth).abs() < 1e-6);
}
