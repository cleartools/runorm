//! Bridge between runorm's plain CSR types and the [`ruanndata`] AnnData container.
//!
//! Enabled by the `ruanndata` cargo feature. Keeps the core library (`lib.rs`) free of any I/O
//! dependency: this module is the only place that knows about `ruanndata::MatrixData`.

use ruanndata::{ArrayData, ArrayValue, MatrixData, RuAnnData};

use crate::{normalize_csr, CsrCounts, NormError, NormParams, NormReport, Result, ShiftedClrMatrix};

/// Convert a ruanndata `ArrayData` to `Vec<f64>`, upcasting integer/`f32`/bool storage.
fn values_as_f64(a: &ArrayData) -> Result<Vec<f64>> {
    Ok(match &a.values {
        ArrayValue::Float64(v) => v.clone(),
        ArrayValue::Float32(v) => v.iter().map(|&x| x as f64).collect(),
        ArrayValue::Int64(v) => v.iter().map(|&x| x as f64).collect(),
        ArrayValue::Int32(v) => v.iter().map(|&x| x as f64).collect(),
        ArrayValue::UInt64(v) => v.iter().map(|&x| x as f64).collect(),
        ArrayValue::UInt32(v) => v.iter().map(|&x| x as f64).collect(),
        ArrayValue::Bool(v) => v.iter().map(|&x| if x { 1.0 } else { 0.0 }).collect(),
        ArrayValue::String(_) => {
            return Err(NormError::InvalidParam(
                "string-valued matrix cannot be normalized".to_string(),
            ))
        }
    })
}

/// Transpose a CSC `(indices = row idx, indptr over columns)` into CSR `(indices = col idx,
/// indptr over rows)`.
fn csc_to_csr(
    n_rows: usize,
    n_cols: usize,
    data: &[f64],
    row_indices: &[usize],
    col_ptr: &[usize],
) -> (Vec<f64>, Vec<usize>, Vec<usize>) {
    let nnz = data.len();
    // Count nnz per row.
    let mut row_counts = vec![0usize; n_rows + 1];
    for &r in row_indices {
        row_counts[r + 1] += 1;
    }
    for i in 0..n_rows {
        row_counts[i + 1] += row_counts[i];
    }
    let indptr = row_counts.clone(); // prefix sums = CSR indptr
    let mut next = indptr.clone();
    let mut out_data = vec![0.0_f64; nnz];
    let mut out_indices = vec![0usize; nnz];
    for col in 0..n_cols {
        for p in col_ptr[col]..col_ptr[col + 1] {
            let row = row_indices[p];
            let dst = next[row];
            out_data[dst] = data[p];
            out_indices[dst] = col;
            next[row] += 1;
        }
    }
    (out_data, out_indices, indptr)
}

/// Build runorm [`CsrCounts`] from any ruanndata matrix of raw counts (`Csr`, `Csc`, or `Dense`).
/// A `ShiftedClrCsr` input is rejected — it is already normalized.
pub fn counts_from_matrixdata(x: &MatrixData) -> Result<CsrCounts> {
    match x {
        MatrixData::Csr { n_rows, n_cols, data, indices, indptr } => {
            CsrCounts::new(*n_rows, *n_cols, values_as_f64(data)?, indices.clone(), indptr.clone())
        }
        MatrixData::Csc { n_rows, n_cols, data, indices, indptr } => {
            let (d, idx, ptr) =
                csc_to_csr(*n_rows, *n_cols, &values_as_f64(data)?, indices, indptr);
            CsrCounts::new(*n_rows, *n_cols, d, idx, ptr)
        }
        MatrixData::Dense { array } => {
            let shape = array.shape();
            let (n_rows, n_cols) = (shape[0], shape.get(1).copied().unwrap_or(1));
            let dense = values_as_f64(array)?;
            let mut data = Vec::new();
            let mut indices = Vec::new();
            let mut indptr = Vec::with_capacity(n_rows + 1);
            indptr.push(0);
            for i in 0..n_rows {
                for j in 0..n_cols {
                    let v = dense[i * n_cols + j];
                    if v != 0.0 {
                        data.push(v);
                        indices.push(j);
                    }
                }
                indptr.push(data.len());
            }
            CsrCounts::new(n_rows, n_cols, data, indices, indptr)
        }
        MatrixData::ShiftedClrCsr { .. } => Err(NormError::InvalidParam(
            "input is already a shifted-CLR matrix, not raw counts".to_string(),
        )),
    }
}

fn f64_array(shape: Vec<usize>, values: Vec<f64>) -> ArrayData {
    ArrayData { shape, values: ArrayValue::Float64(values) }
}

impl ShiftedClrMatrix {
    /// Convert into a `ruanndata::MatrixData::ShiftedClrCsr` (the sparse log1pPF values plus the
    /// per-cell `row_center`). Round-trips through the native `.rnad` format.
    pub fn into_matrixdata(self) -> MatrixData {
        let nnz = self.data.len();
        MatrixData::ShiftedClrCsr {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data: f64_array(vec![nnz], self.data),
            indices: self.indices,
            indptr: self.indptr,
            row_center: f64_array(vec![self.n_rows], self.row_center),
        }
    }

    /// Convert into a plain `ruanndata::MatrixData::Csr`, dropping `row_center`. Intended for
    /// uncentered output (plain log1pPF), which — unlike `ShiftedClrCsr` — can be written to
    /// h5ad/zarr.
    pub fn into_csr_matrixdata(self) -> MatrixData {
        let nnz = self.data.len();
        MatrixData::Csr {
            n_rows: self.n_rows,
            n_cols: self.n_cols,
            data: f64_array(vec![nnz], self.data),
            indices: self.indices,
            indptr: self.indptr,
        }
    }

    /// Build from a `ruanndata` `ShiftedClrCsr` (or a plain `Csr`, taking `row_center = 0`).
    pub fn from_matrixdata(x: &MatrixData) -> Result<Self> {
        match x {
            MatrixData::ShiftedClrCsr { n_rows, n_cols, data, indices, indptr, row_center } => {
                let m = ShiftedClrMatrix {
                    n_rows: *n_rows,
                    n_cols: *n_cols,
                    data: values_as_f64(data)?,
                    indices: indices.clone(),
                    indptr: indptr.clone(),
                    row_center: values_as_f64(row_center)?,
                };
                m.validate()?;
                Ok(m)
            }
            MatrixData::Csr { n_rows, n_cols, data, indices, indptr } => {
                let m = ShiftedClrMatrix {
                    n_rows: *n_rows,
                    n_cols: *n_cols,
                    data: values_as_f64(data)?,
                    indices: indices.clone(),
                    indptr: indptr.clone(),
                    row_center: vec![0.0; *n_rows],
                };
                m.validate()?;
                Ok(m)
            }
            _ => Err(NormError::InvalidParam(
                "expected a ShiftedClrCsr or Csr matrix".to_string(),
            )),
        }
    }
}

/// Where to write the normalized result inside a [`RuAnnData`].
#[derive(Debug, Clone)]
pub enum OutTarget {
    /// Overwrite `adata.X`.
    ReplaceX,
    /// Insert/overwrite `adata.layers[name]`.
    Layer(String),
}

/// Normalize a [`RuAnnData`] in place: read counts from `X` (or `layer`), normalize, and write the
/// result to `out`. Returns the PF-target report. Centered output is written as `ShiftedClrCsr`;
/// uncentered output (`params.center == false`) is written as a plain `Csr`.
pub fn normalize_anndata(
    adata: &mut RuAnnData,
    layer: Option<&str>,
    out: OutTarget,
    params: &NormParams,
) -> Result<NormReport> {
    let source = match layer {
        Some(name) => adata.layers.get(name).ok_or_else(|| {
            NormError::InvalidParam(format!("layer '{name}' not found"))
        })?,
        None => &adata.x,
    };
    let counts = counts_from_matrixdata(source)?;
    let (matrix, report) = normalize_csr(&counts, params)?;
    let result = if params.center {
        matrix.into_matrixdata()
    } else {
        matrix.into_csr_matrixdata()
    };
    match out {
        OutTarget::ReplaceX => adata.x = result,
        OutTarget::Layer(name) => {
            adata.layers.insert(name, result);
        }
    }
    Ok(report)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::NormParams;

    fn sample_counts() -> CsrCounts {
        // row0: [0,3,0,1]; row1: [2,0,0,0]; row2: [0,0,5,4]
        CsrCounts::new(
            3,
            4,
            vec![3.0, 1.0, 2.0, 5.0, 4.0],
            vec![1, 3, 0, 2, 3],
            vec![0, 2, 3, 5],
        )
        .unwrap()
    }

    /// Feeding runorm output to ruanndata as a `ShiftedClrCsr` must reproduce the dense row sums,
    /// exercising the `Σ(data_row) − row_center·n_cols` contract on the consumer side.
    #[test]
    fn row_sums_match_dense_oracle() {
        let counts = sample_counts();
        let (m, _) = normalize_csr(&counts, &NormParams::default()).unwrap();

        // Dense oracle row sums from runorm's own densify().
        let n_cols = m.n_cols;
        let dense = m.densify();
        let oracle: Vec<f64> = (0..m.n_rows)
            .map(|i| dense[i * n_cols..(i + 1) * n_cols].iter().sum())
            .collect();

        let md = m.into_matrixdata();
        let via_ruanndata = ruanndata::row_sums_par(&md).unwrap();
        assert_eq!(via_ruanndata.len(), oracle.len());
        for (a, b) in via_ruanndata.iter().zip(oracle.iter()) {
            assert!((a - b).abs() < 1e-12, "{a} vs {b}");
        }
    }

    /// Round-trip runorm -> ruanndata MatrixData -> runorm.
    #[test]
    fn matrixdata_roundtrip() {
        let counts = sample_counts();
        let (m, _) = normalize_csr(&counts, &NormParams::default()).unwrap();
        let dense_before = m.densify();
        let md = m.into_matrixdata();
        let back = ShiftedClrMatrix::from_matrixdata(&md).unwrap();
        let dense_after = back.densify();
        assert_eq!(dense_before.len(), dense_after.len());
        for (a, b) in dense_before.iter().zip(dense_after.iter()) {
            assert!((a - b).abs() < 1e-12);
        }
    }

    /// CSC counts convert to the same CSR runorm would normalize.
    #[test]
    fn csc_input_matches_csr() {
        use ruanndata::{ArrayData, ArrayValue, MatrixData};
        // Same 3x4 matrix as CSC: columns 0..3.
        // col0: row1=2 ; col1: row0=3 ; col2: row2=5 ; col3: row0=1,row2=4
        let csc = MatrixData::Csc {
            n_rows: 3,
            n_cols: 4,
            data: ArrayData { shape: vec![5], values: ArrayValue::Float64(vec![2.0, 3.0, 5.0, 1.0, 4.0]) },
            indices: vec![1, 0, 2, 0, 2],
            indptr: vec![0, 1, 2, 3, 5],
        };
        let from_csc = counts_from_matrixdata(&csc).unwrap();
        let direct = sample_counts();
        // Both should densify-normalize identically.
        let a = normalize_csr(&from_csc, &NormParams::default()).unwrap().0.densify();
        let b = normalize_csr(&direct, &NormParams::default()).unwrap().0.densify();
        for (x, y) in a.iter().zip(b.iter()) {
            assert!((x - y).abs() < 1e-12);
        }
    }
}
