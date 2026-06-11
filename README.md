# runorm

Sparse **PFlog1pPF / shifted-CLR** normalization for single-cell count data — a Rust library and
CLI. Part of the cleartools ecosystem ([`ruanndata`](https://github.com/pachterlab/ruanndata),
[`rupca`](https://github.com/pachterlab/rupca), [`scclr`](https://github.com/cleartools/scclr)).

The transform, per cell `i` (see *Depth normalization for single-cell genomics count data*,
Booeshaghi et al.):

1. **Proportional fitting (PF):** scale the row so its total equals a target `K`.
2. **log1p:** `ln(1 + ·)` (keeps the matrix sparse since `ln(1+0)=0`).
3. **Per-cell centering (the CLR step):** subtract the per-cell mean over all genes.

The result is kept **sparse**: the log1pPF values plus a per-cell `row_center` vector, representing
the dense value `data[i,j] − row_center[i]`. This is exactly
[`rupca::ShiftedClrCsrMatrix`](https://github.com/pachterlab/rupca) /
`ruanndata::MatrixData::ShiftedClrCsr`, so sparse PCA runs on it without densifying.

## Choosing K

`K` controls variance stabilization. `runorm` can estimate it from the data: it fits the
negative-binomial overdispersion `Var ≈ μ + α·μ²` across genes (closed-form OLS, no optimizer) and
sets `K = 4·α·s` (`s` = mean depth) — the variance-stabilizing choice from the paper. See
[`PfTarget`]: `MeanDepth` (default), `MedianDepth`, `Fixed(K)`, `Alpha(α)`, `EstimateAlpha`.

## Library

```rust
use runorm::{normalize_csr, CsrCounts, NormParams, PfTarget};

let counts = CsrCounts::new(n_rows, n_cols, data, indices, indptr)?;
let (shifted_clr, report) = normalize_csr(&counts, &NormParams {
    target: PfTarget::EstimateAlpha, log1p: true, center: true,
})?;
// shifted_clr.{data, indices, indptr, row_center}; report.{k, alpha, mean_depth}
```

## CLI (`--features cli`)

```
runorm normalize <in> -o <out> [--target mean|median|auto|<K>] [--alpha <a>]
       [--no-log1p] [--no-center] [--layer NAME] [--out-layer NAME]
runorm overdispersion <in>          # print estimated alpha, mean depth, K = 4*alpha*s
```

Reads/writes via `ruanndata` — native `.rnad` always; `.h5ad`/`.zarr` under the `h5ad`/`io` features
(h5ad needs system libhdf5). Centered (shifted-CLR) output is written as `.rnad` only (the h5ad/zarr
writers cannot represent it); use `--no-center` for a plain log1pPF `Csr` writable to any format.

## License

BSD-2-Clause.
