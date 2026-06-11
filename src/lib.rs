//! # runorm
//!
//! Sparse **PFlog1pPF** / **shifted-CLR** normalization for single-cell count data.
//!
//! The transform (per cell / row `i`), following the reference in the supplementary note:
//!
//! 1. **Proportional fitting (PF):** scale the row so its total equals a target `K`:
//!    `P[i,j] = C[i,j] * K / s_i`, where `s_i = Σ_j C[i,j]` is the cell depth.
//! 2. **log1p:** `L[i,j] = ln(1 + P[i,j])`. Because `ln(1+0) = 0`, zeros stay zero and the
//!    matrix stays sparse.
//! 3. **Per-cell centering (the CLR step):** `row_center[i] = (Σ_j L[i,j]) / n_cols`, where the
//!    divisor is the number of genes `D = n_cols`, counting the implicit zeros. We do **not**
//!    subtract — we keep the sparse `L` together with the `row_center` vector.
//!
//! The dense value is therefore `L[i,j] - row_center[i]` (and `-row_center[i]` at the implicit
//! zeros). This is exactly the representation consumed by [`rupca::ShiftedClrCsrMatrix`] and
//! serialized by [`ruanndata::MatrixData::ShiftedClrCsr`], so PCA runs on it without densifying.
//!
//! ## Choosing `K`
//!
//! `K` controls variance stabilization. The variance-stabilizing choice from the note is
//! `K = 4·α·s` where `s` is the mean cell depth and `α` is the negative-binomial overdispersion
//! (`Var ≈ μ + α·μ²`). [`estimate_overdispersion`] fits `α` across genes via the closed-form OLS
//! solution of that (linear-in-`α`) model — no optimizer required. See [`PfTarget`].

use rayon::prelude::*;
use thiserror::Error;

#[cfg(feature = "ruanndata")]
mod ruanndata_bridge;
#[cfg(feature = "ruanndata")]
pub use ruanndata_bridge::{counts_from_matrixdata, normalize_anndata, OutTarget};

/// Errors produced while validating input or normalizing.
#[derive(Debug, Error, PartialEq)]
pub enum NormError {
    #[error("data length {data} and indices length {indices} must match")]
    LengthMismatch { data: usize, indices: usize },
    #[error("indptr length {got} must equal n_rows + 1 = {expected}")]
    BadIndptr { got: usize, expected: usize },
    #[error("indptr must start at 0, be nondecreasing, and end at nnz")]
    MalformedIndptr,
    #[error("column index {index} out of bounds for n_cols = {n_cols}")]
    ColIndexOob { index: usize, n_cols: usize },
    #[error("row_center length {got} must equal n_rows = {n_rows}")]
    RowCenterLen { got: usize, n_rows: usize },
    #[error("matrix is empty (n_rows = {n_rows}, n_cols = {n_cols})")]
    EmptyMatrix { n_rows: usize, n_cols: usize },
    #[error("estimated overdispersion is non-positive (alpha = {alpha}); cannot derive K = 4·alpha·s")]
    NonPositiveOverdispersion { alpha: f64 },
    #[error("invalid parameter: {0}")]
    InvalidParam(String),
}

pub type Result<T> = std::result::Result<T, NormError>;

/// Sparse CSR matrix of raw counts (cells × genes). Values are `f64`; callers upcast from
/// integer/`f32` storage at the boundary (e.g. the ruanndata bridge or the Python layer).
#[derive(Debug, Clone)]
pub struct CsrCounts {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
}

impl CsrCounts {
    /// Construct and validate a CSR counts matrix.
    pub fn new(
        n_rows: usize,
        n_cols: usize,
        data: Vec<f64>,
        indices: Vec<usize>,
        indptr: Vec<usize>,
    ) -> Result<Self> {
        let c = Self { n_rows, n_cols, data, indices, indptr };
        c.validate()?;
        Ok(c)
    }

    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn validate(&self) -> Result<()> {
        validate_csr(self.n_rows, self.n_cols, &self.data, &self.indices, &self.indptr)
    }
}

/// Shared CSR structural validation used by [`CsrCounts`] and [`ShiftedClrMatrix`].
fn validate_csr(
    n_rows: usize,
    n_cols: usize,
    data: &[f64],
    indices: &[usize],
    indptr: &[usize],
) -> Result<()> {
    if data.len() != indices.len() {
        return Err(NormError::LengthMismatch { data: data.len(), indices: indices.len() });
    }
    if indptr.len() != n_rows + 1 {
        return Err(NormError::BadIndptr { got: indptr.len(), expected: n_rows + 1 });
    }
    if indptr.first().copied() != Some(0) || indptr.last().copied() != Some(data.len()) {
        return Err(NormError::MalformedIndptr);
    }
    if indptr.windows(2).any(|w| w[0] > w[1]) {
        return Err(NormError::MalformedIndptr);
    }
    if let Some(&bad) = indices.iter().find(|&&j| j >= n_cols) {
        return Err(NormError::ColIndexOob { index: bad, n_cols });
    }
    Ok(())
}

/// Strategy for choosing the PF target `K`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PfTarget {
    /// `K` = mean cell depth across cells. Standard PFlog1pPF.
    MeanDepth,
    /// `K` = median cell depth (robust to depth outliers).
    MedianDepth,
    /// `K` given directly (e.g. `1e4`, Seurat-style). Pseudocount `c = 1/K` on the
    /// depth-normalized scale.
    Fixed(f64),
    /// `K = 4·alpha·s` with a user-supplied overdispersion `alpha` (`s` = mean depth).
    Alpha(f64),
    /// Estimate `alpha` from the data (`Var ≈ μ + alpha·μ²`), then `K = 4·alpha·s`.
    EstimateAlpha,
}

impl Default for PfTarget {
    fn default() -> Self {
        PfTarget::MeanDepth
    }
}

/// Normalization parameters.
#[derive(Debug, Clone)]
pub struct NormParams {
    pub target: PfTarget,
    /// Apply `ln(1+·)` after PF (the "log1p" step). Default `true`.
    pub log1p: bool,
    /// Apply the per-cell CLR centering (produce a nonzero `row_center`). Default `true`.
    /// When `false`, the result is plain log1pPF with an all-zero `row_center`.
    pub center: bool,
}

impl Default for NormParams {
    fn default() -> Self {
        Self { target: PfTarget::MeanDepth, log1p: true, center: true }
    }
}

/// Estimated negative-binomial overdispersion and the derived PF target.
#[derive(Debug, Clone, Copy)]
pub struct Overdispersion {
    /// `alpha` from `Var ≈ μ + alpha·μ²` (per-gene, method of moments + closed-form OLS).
    pub alpha: f64,
    /// Mean cell depth `s`.
    pub mean_depth: f64,
    /// `K = 4·alpha·mean_depth`.
    pub k: f64,
}

/// Provenance for the chosen PF target.
#[derive(Debug, Clone, Copy)]
pub struct NormReport {
    /// The PF target `K` that was applied.
    pub k: f64,
    /// The overdispersion used, when the target derived `K` from `alpha`.
    pub alpha: Option<f64>,
    /// Mean cell depth `s`.
    pub mean_depth: f64,
}

/// Output of normalization: the sparse log1pPF values plus the per-cell mean vector.
///
/// The represented dense value is `data[i,j] - row_center[i]`. Structurally identical to
/// [`rupca::ShiftedClrCsrMatrix`] and [`ruanndata::MatrixData::ShiftedClrCsr`].
#[derive(Debug, Clone)]
pub struct ShiftedClrMatrix {
    pub n_rows: usize,
    pub n_cols: usize,
    pub data: Vec<f64>,
    pub indices: Vec<usize>,
    pub indptr: Vec<usize>,
    /// Per-cell mean over all `n_cols` genes (counting implicit zeros). Length `n_rows`.
    pub row_center: Vec<f64>,
}

impl ShiftedClrMatrix {
    pub fn nnz(&self) -> usize {
        self.data.len()
    }

    pub fn validate(&self) -> Result<()> {
        validate_csr(self.n_rows, self.n_cols, &self.data, &self.indices, &self.indptr)?;
        if self.row_center.len() != self.n_rows {
            return Err(NormError::RowCenterLen { got: self.row_center.len(), n_rows: self.n_rows });
        }
        Ok(())
    }

    /// Materialize the dense `data[i,j] - row_center[i]` matrix (row-major, length
    /// `n_rows * n_cols`). For tests and small data only — this densifies.
    pub fn densify(&self) -> Vec<f64> {
        let mut dense = vec![0.0_f64; self.n_rows * self.n_cols];
        for i in 0..self.n_rows {
            let base = i * self.n_cols;
            let c = self.row_center[i];
            for x in dense[base..base + self.n_cols].iter_mut() {
                *x = -c;
            }
            for p in self.indptr[i]..self.indptr[i + 1] {
                dense[base + self.indices[p]] += self.data[p];
            }
        }
        dense
    }
}

/// Per-row cell depths `s_i = Σ_j C[i,j]` (parallel over rows; each row summed in index order,
/// so the result is independent of thread count).
fn row_depths(c: &CsrCounts) -> Vec<f64> {
    (0..c.n_rows)
        .into_par_iter()
        .map(|i| c.data[c.indptr[i]..c.indptr[i + 1]].iter().sum::<f64>())
        .collect()
}

/// Even split of `[0, n_rows)` into contiguous chunks, one per thread (at least one row each).
fn row_chunks(n_rows: usize) -> Vec<(usize, usize)> {
    if n_rows == 0 {
        return Vec::new();
    }
    let threads = rayon::current_num_threads().max(1);
    let chunk = n_rows.div_ceil(threads).max(1);
    (0..n_rows).step_by(chunk).map(|s| (s, (s + chunk).min(n_rows))).collect()
}

/// Per-gene `(Σ_i C[i,g], Σ_i C[i,g]²)` over cells. Parallel over disjoint row chunks with a
/// deterministic in-order reduction.
fn column_moments(c: &CsrCounts) -> (Vec<f64>, Vec<f64>) {
    let n_cols = c.n_cols;
    let chunks = row_chunks(c.n_rows);
    let partials: Vec<(Vec<f64>, Vec<f64>)> = chunks
        .par_iter()
        .map(|&(s, e)| {
            let mut sum = vec![0.0_f64; n_cols];
            let mut sumsq = vec![0.0_f64; n_cols];
            for i in s..e {
                for p in c.indptr[i]..c.indptr[i + 1] {
                    let j = c.indices[p];
                    let v = c.data[p];
                    sum[j] += v;
                    sumsq[j] += v * v;
                }
            }
            (sum, sumsq)
        })
        .collect();

    let mut sums = vec![0.0_f64; n_cols];
    let mut sumsq = vec![0.0_f64; n_cols];
    for (s, q) in &partials {
        for g in 0..n_cols {
            sums[g] += s[g];
            sumsq[g] += q[g];
        }
    }
    (sums, sumsq)
}

fn median(depths: &[f64]) -> f64 {
    if depths.is_empty() {
        return 0.0;
    }
    let mut v = depths.to_vec();
    v.sort_by(|a, b| a.total_cmp(b));
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        0.5 * (v[n / 2 - 1] + v[n / 2])
    }
}

/// Deterministic mean (sequential sum in index order).
fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return 0.0;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Estimate the negative-binomial overdispersion `alpha` from raw counts.
///
/// Fits `Var_g = μ_g + alpha·μ_g²` across genes, where `μ_g` and `Var_g` are the per-gene mean
/// and variance over cells. The model is linear in `alpha`, so the OLS solution is closed form:
/// `alpha = Σ_g (Var_g − μ_g)·μ_g² / Σ_g μ_g⁴`. Returns the derived `K = 4·alpha·s`.
pub fn estimate_overdispersion(c: &CsrCounts) -> Result<Overdispersion> {
    c.validate()?;
    if c.n_rows == 0 || c.n_cols == 0 {
        return Err(NormError::EmptyMatrix { n_rows: c.n_rows, n_cols: c.n_cols });
    }
    let n = c.n_rows as f64;
    let (sums, sumsq) = column_moments(c);

    let mut num = 0.0_f64;
    let mut den = 0.0_f64;
    for g in 0..c.n_cols {
        let mu = sums[g] / n;
        let ex2 = sumsq[g] / n;
        let var = ex2 - mu * mu;
        let mu2 = mu * mu;
        num += (var - mu) * mu2;
        den += mu2 * mu2;
    }
    if den == 0.0 {
        // All genes are identically zero — overdispersion is undefined.
        return Err(NormError::NonPositiveOverdispersion { alpha: 0.0 });
    }
    let alpha = num / den;
    let total: f64 = sums.iter().sum();
    let mean_depth = total / n;
    let k = 4.0 * alpha * mean_depth;
    if !(alpha > 0.0) || !(k > 0.0) {
        return Err(NormError::NonPositiveOverdispersion { alpha });
    }
    Ok(Overdispersion { alpha, mean_depth, k })
}

/// Resolve the PF target `K` for the given counts/depths. Exposed for CLI reporting and tests.
pub fn resolve_k(c: &CsrCounts, depths: &[f64], target: &PfTarget) -> Result<NormReport> {
    if c.n_rows == 0 {
        return Err(NormError::EmptyMatrix { n_rows: c.n_rows, n_cols: c.n_cols });
    }
    let mean_depth = mean(depths);
    match *target {
        PfTarget::MeanDepth => Ok(NormReport { k: mean_depth, alpha: None, mean_depth }),
        PfTarget::MedianDepth => Ok(NormReport { k: median(depths), alpha: None, mean_depth }),
        PfTarget::Fixed(k) => {
            if !(k > 0.0) {
                return Err(NormError::InvalidParam(format!("fixed target K must be > 0, got {k}")));
            }
            Ok(NormReport { k, alpha: None, mean_depth })
        }
        PfTarget::Alpha(alpha) => {
            if !(alpha > 0.0) {
                return Err(NormError::NonPositiveOverdispersion { alpha });
            }
            let k = 4.0 * alpha * mean_depth;
            Ok(NormReport { k, alpha: Some(alpha), mean_depth })
        }
        PfTarget::EstimateAlpha => {
            let od = estimate_overdispersion(c)?;
            Ok(NormReport { k: od.k, alpha: Some(od.alpha), mean_depth: od.mean_depth })
        }
    }
}

/// Split a mutable buffer into consecutive sub-slices of the given lengths (which must sum to
/// `buf.len()`). Used to hand each row-chunk a disjoint, aliasing-free output slice.
fn split_by_lengths<'a, T>(mut buf: &'a mut [T], lens: &[usize]) -> Vec<&'a mut [T]> {
    let mut out = Vec::with_capacity(lens.len());
    for &l in lens {
        let (head, tail) = buf.split_at_mut(l);
        out.push(head);
        buf = tail;
    }
    out
}

/// Compute the PFlog1pPF / shifted-CLR normalization. Returns the sparse result and a report of
/// the chosen PF target.
pub fn normalize_csr(c: &CsrCounts, p: &NormParams) -> Result<(ShiftedClrMatrix, NormReport)> {
    c.validate()?;
    if c.n_rows == 0 || c.n_cols == 0 {
        return Err(NormError::EmptyMatrix { n_rows: c.n_rows, n_cols: c.n_cols });
    }

    let depths = row_depths(c);
    let report = resolve_k(c, &depths, &p.target)?;
    let k = report.k;
    let n_cols_f = c.n_cols as f64;
    let apply_log = p.log1p;
    let apply_center = p.center;

    let mut out_data = vec![0.0_f64; c.nnz()];
    let mut row_center = vec![0.0_f64; c.n_rows];

    let chunks = row_chunks(c.n_rows);
    let data_lens: Vec<usize> = chunks.iter().map(|&(s, e)| c.indptr[e] - c.indptr[s]).collect();
    let center_lens: Vec<usize> = chunks.iter().map(|&(s, e)| e - s).collect();
    let data_slices = split_by_lengths(&mut out_data, &data_lens);
    let center_slices = split_by_lengths(&mut row_center, &center_lens);

    chunks
        .par_iter()
        .zip(data_slices.into_par_iter())
        .zip(center_slices.into_par_iter())
        .for_each(|((&(s, e), dslice), cslice)| {
            let mut dpos = 0usize;
            for (li, i) in (s..e).enumerate() {
                let start = c.indptr[i];
                let end = c.indptr[i + 1];
                let row_nnz = end - start;
                let dst = &mut dslice[dpos..dpos + row_nnz];
                let depth = depths[i];

                if !(depth > 0.0) {
                    // Empty cell (all-zero counts): emit zeros, center 0. Avoids K/0 = inf.
                    for x in dst.iter_mut() {
                        *x = 0.0;
                    }
                    cslice[li] = 0.0;
                    dpos += row_nnz;
                    continue;
                }

                let scale = k / depth;
                let mut rowsum = 0.0_f64; // f64 accumulation regardless of any future storage dtype
                for (off, p_idx) in (start..end).enumerate() {
                    let scaled = c.data[p_idx] * scale;
                    let val = if apply_log { scaled.ln_1p() } else { scaled };
                    dst[off] = val;
                    rowsum += val;
                }
                cslice[li] = if apply_center { rowsum / n_cols_f } else { 0.0 };
                dpos += row_nnz;
            }
        });

    let out = ShiftedClrMatrix {
        n_rows: c.n_rows,
        n_cols: c.n_cols,
        data: out_data,
        indices: c.indices.clone(),
        indptr: c.indptr.clone(),
        row_center,
    };
    Ok((out, report))
}

/// Convenience: PFlog1pPF / shifted-CLR with default parameters (PF to mean depth, log1p, center).
pub fn pflog1ppf(c: &CsrCounts) -> Result<ShiftedClrMatrix> {
    Ok(normalize_csr(c, &NormParams::default())?.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    // Tiny 3x4 matrix with a non-trivial sparsity pattern (and n_cols != per-row nnz).
    fn sample() -> CsrCounts {
        // row0: [0, 3, 0, 1]; row1: [2, 0, 0, 0]; row2: [0, 0, 5, 4]
        CsrCounts::new(
            3,
            4,
            vec![3.0, 1.0, 2.0, 5.0, 4.0],
            vec![1, 3, 0, 2, 3],
            vec![0, 2, 3, 5],
        )
        .unwrap()
    }

    /// Independent dense PFlog1pPF oracle.
    fn dense_oracle(c: &CsrCounts, k: f64, log1p: bool, center: bool) -> Vec<f64> {
        let d = c.n_cols;
        let mut dense = vec![0.0_f64; c.n_rows * d];
        for i in 0..c.n_rows {
            let depth: f64 = c.data[c.indptr[i]..c.indptr[i + 1]].iter().sum();
            let mut row = vec![0.0_f64; d];
            if depth > 0.0 {
                for p in c.indptr[i]..c.indptr[i + 1] {
                    let scaled = c.data[p] * k / depth;
                    row[c.indices[p]] = if log1p { scaled.ln_1p() } else { scaled };
                }
            }
            let m = if center { row.iter().sum::<f64>() / d as f64 } else { 0.0 };
            for j in 0..d {
                dense[i * d + j] = row[j] - m;
            }
        }
        dense
    }

    #[test]
    fn divisor_is_n_cols_not_nnz() {
        let c = sample();
        let (m, _) = normalize_csr(&c, &NormParams::default()).unwrap();
        // row_center[i] * n_cols must equal the sum of stored log1p values in row i.
        for i in 0..c.n_rows {
            let row_sum: f64 = m.data[m.indptr[i]..m.indptr[i + 1]].iter().sum();
            assert!((m.row_center[i] * c.n_cols as f64 - row_sum).abs() < 1e-12);
        }
    }

    #[test]
    fn matches_dense_oracle_mean_depth() {
        let c = sample();
        let (m, report) = normalize_csr(&c, &NormParams::default()).unwrap();
        let oracle = dense_oracle(&c, report.k, true, true);
        let got = m.densify();
        for (a, b) in got.iter().zip(oracle.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    #[test]
    fn matches_dense_oracle_fixed_and_uncentered() {
        let c = sample();
        for (params, k) in [
            (NormParams { target: PfTarget::Fixed(1e4), log1p: true, center: true }, 1e4),
            (NormParams { target: PfTarget::Fixed(50.0), log1p: true, center: false }, 50.0),
            (NormParams { target: PfTarget::Fixed(50.0), log1p: false, center: true }, 50.0),
        ] {
            let (m, report) = normalize_csr(&c, &params).unwrap();
            assert!((report.k - k).abs() < 1e-9);
            let oracle = dense_oracle(&c, k, params.log1p, params.center);
            for (a, b) in m.densify().iter().zip(oracle.iter()) {
                assert!((a - b).abs() < 1e-12, "{a} vs {b}");
            }
        }
    }

    #[test]
    fn empty_cell_is_handled() {
        // row1 is entirely zero.
        let c = CsrCounts::new(2, 3, vec![1.0, 2.0], vec![0, 2], vec![0, 2, 2]).unwrap();
        let (m, _) = normalize_csr(&c, &NormParams::default()).unwrap();
        assert_eq!(m.row_center[1], 0.0);
        let dense = m.densify();
        // The empty row densifies to all zeros (no NaN/inf).
        assert!(dense[3..6].iter().all(|&x| x == 0.0));
        assert!(dense.iter().all(|x| x.is_finite()));
    }

    #[test]
    fn overdispersion_closed_form_matches_manual() {
        // 3 cells x 2 genes, overdispersed so alpha > 0 (var >> mu).
        // cell0: [1, 0]; cell1: [0, 2]; cell2: [30, 12]
        // gene0 counts: [1, 0, 30]; gene1 counts: [0, 2, 12]
        let c = CsrCounts::new(
            3,
            2,
            vec![1.0, 2.0, 30.0, 12.0],
            vec![0, 1, 0, 1],
            vec![0, 1, 2, 4],
        )
        .unwrap();
        let n = 3.0;
        let mut num = 0.0;
        let mut den = 0.0;
        for g in [vec![1.0, 0.0, 30.0], vec![0.0, 2.0, 12.0]] {
            let mu = g.iter().sum::<f64>() / n;
            let ex2 = g.iter().map(|x| x * x).sum::<f64>() / n;
            let var = ex2 - mu * mu;
            num += (var - mu) * mu * mu;
            den += mu.powi(4);
        }
        let expected_alpha = num / den;
        let od = estimate_overdispersion(&c).unwrap();
        assert!(expected_alpha > 0.0, "test fixture must be overdispersed");
        assert!((od.alpha - expected_alpha).abs() < 1e-12, "{} vs {}", od.alpha, expected_alpha);
        let mean_depth = (1.0 + 2.0 + 30.0 + 12.0) / 3.0;
        assert!((od.mean_depth - mean_depth).abs() < 1e-12);
        assert!((od.k - 4.0 * expected_alpha * mean_depth).abs() < 1e-9);
    }
}
