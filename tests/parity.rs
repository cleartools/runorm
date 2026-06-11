//! End-to-end parity: runorm's sparse shifted-CLR output fed to rupca's *sparse* PCA must match
//! running rupca's *dense* PCA on the fully materialized (densified) shifted-CLR matrix.

use rupca::{
    pca_scanpy_dense, pca_shifted_clr_sparse_csr, CsrMatrix, DenseMatrix, ScanpyPcaParams,
    ShiftedClrCsrMatrix,
};
use runorm::{normalize_csr, CsrCounts, NormParams, PfTarget};

/// Tiny deterministic PRNG so the test needs no external rng dependency.
struct Lcg(u64);
impl Lcg {
    fn next_u32(&mut self) -> u32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        (self.0 >> 33) as u32
    }
}

/// Build a sparse non-negative integer count matrix with no all-zero rows.
fn random_counts(n_rows: usize, n_cols: usize, seed: u64) -> CsrCounts {
    let mut rng = Lcg(seed);
    let mut data = Vec::new();
    let mut indices = Vec::new();
    let mut indptr = vec![0usize];
    for i in 0..n_rows {
        for j in 0..n_cols {
            // ~60% sparsity; counts in 1..=9.
            let keep = rng.next_u32() % 5 != 0;
            if keep {
                let v = (rng.next_u32() % 9 + 1) as f64;
                data.push(v);
                indices.push(j);
            }
        }
        // Guarantee at least one nonzero per row.
        if *indptr.last().unwrap() == data.len() {
            data.push(((i % 7) + 1) as f64);
            indices.push(i % n_cols);
        }
        indptr.push(data.len());
    }
    CsrCounts::new(n_rows, n_cols, data, indices, indptr).unwrap()
}

fn to_rupca_shifted(m: &runorm::ShiftedClrMatrix) -> ShiftedClrCsrMatrix {
    ShiftedClrCsrMatrix {
        sparse: CsrMatrix {
            n_rows: m.n_rows,
            n_cols: m.n_cols,
            data: m.data.clone(),
            indices: m.indices.clone(),
            indptr: m.indptr.clone(),
        },
        row_center: m.row_center.clone(),
    }
}

/// Compare two score matrices (row-major, n_samples x k) up to a per-component sign flip.
fn scores_match(a: &[f64], b: &[f64], n: usize, k: usize, tol: f64) {
    assert_eq!(a.len(), n * k);
    assert_eq!(b.len(), n * k);
    for comp in 0..k {
        // Pick the sign that best aligns column `comp`.
        let mut dot = 0.0;
        for i in 0..n {
            dot += a[i * k + comp] * b[i * k + comp];
        }
        let sign = if dot < 0.0 { -1.0 } else { 1.0 };
        let mut max_abs = 0.0_f64;
        for i in 0..n {
            let diff = (a[i * k + comp] - sign * b[i * k + comp]).abs();
            max_abs = max_abs.max(diff);
        }
        assert!(max_abs < tol, "component {comp} differs by {max_abs}");
    }
}

fn check_parity(n_rows: usize, n_cols: usize, k: usize, target: PfTarget, seed: u64) {
    let counts = random_counts(n_rows, n_cols, seed);
    let params = NormParams { target, log1p: true, center: true };
    let (m, _) = normalize_csr(&counts, &params).unwrap();

    let pca_params = ScanpyPcaParams { n_components: k, seed: 0, ..Default::default() };

    // Sparse path: runorm -> rupca shifted-CLR sparse PCA.
    let sparse = to_rupca_shifted(&m);
    let r_sparse = pca_shifted_clr_sparse_csr(&sparse, pca_params).unwrap();

    // Dense path: densify the shifted-CLR matrix and run rupca dense PCA.
    let dense = DenseMatrix { n_rows: m.n_rows, n_cols: m.n_cols, data: m.densify() };
    let r_dense = pca_scanpy_dense(&dense, pca_params).unwrap();

    // Explained variance is sign-independent and should match closely.
    for (a, b) in r_sparse.explained_variance.iter().zip(r_dense.explained_variance.iter()) {
        assert!((a - b).abs() < 1e-6, "explained variance {a} vs {b}");
    }
    // Scores match up to a per-component sign.
    scores_match(&r_sparse.scores, &r_dense.scores, n_rows, k, 1e-5);
}

#[test]
fn parity_tall_mean_depth() {
    check_parity(40, 10, 4, PfTarget::MeanDepth, 7);
}

#[test]
fn parity_wide_fixed_k() {
    check_parity(12, 30, 4, PfTarget::Fixed(1e4), 99);
}

#[test]
fn parity_tall_estimate_alpha() {
    check_parity(50, 12, 5, PfTarget::EstimateAlpha, 2024);
}
