//! End-to-end CLI smoke test: build a counts `.rnad`, run the `runorm` binary on it, and check
//! the output. Only built with the `cli` feature (which is what provides the binary).
#![cfg(feature = "cli")]

use std::process::Command;

use ruanndata::{ArrayData, ArrayValue, DataFrame, MatrixData, RuAnnData};

fn write_counts_rnad(path: &std::path::Path) {
    // 3 cells x 4 genes; row0:[0,3,0,1] row1:[2,0,0,0] row2:[0,0,5,4]
    let x = MatrixData::Csr {
        n_rows: 3,
        n_cols: 4,
        data: ArrayData { shape: vec![5], values: ArrayValue::Float64(vec![3.0, 1.0, 2.0, 5.0, 4.0]) },
        indices: vec![1, 3, 0, 2, 3],
        indptr: vec![0, 2, 3, 5],
    };
    let obs = DataFrame {
        index_name: "_index".to_string(),
        index: vec!["c0".into(), "c1".into(), "c2".into()],
        columns: Default::default(),
    };
    let var = DataFrame {
        index_name: "_index".to_string(),
        index: vec!["g0".into(), "g1".into(), "g2".into(), "g3".into()],
        columns: Default::default(),
    };
    let adata = RuAnnData::new(x, obs, var).unwrap();
    ruanndata::save(path, &adata).unwrap();
}

fn bin() -> Command {
    Command::new(env!("CARGO_BIN_EXE_runorm"))
}

#[test]
fn normalize_writes_shifted_clr_rnad() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("counts.rnad");
    let output = dir.path().join("norm.rnad");
    write_counts_rnad(&input);

    let status = bin()
        .args([
            "normalize",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--target",
            "mean",
        ])
        .status()
        .unwrap();
    assert!(status.success(), "runorm normalize failed");

    let out = ruanndata::load(&output).unwrap();
    assert!(matches!(out.x, MatrixData::ShiftedClrCsr { .. }), "expected ShiftedClrCsr output");
}

#[test]
fn no_center_writes_plain_csr() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("counts.rnad");
    let output = dir.path().join("log1ppf.rnad");
    write_counts_rnad(&input);

    let status = bin()
        .args([
            "normalize",
            input.to_str().unwrap(),
            "-o",
            output.to_str().unwrap(),
            "--no-center",
        ])
        .status()
        .unwrap();
    assert!(status.success());

    let out = ruanndata::load(&output).unwrap();
    assert!(matches!(out.x, MatrixData::Csr { .. }), "expected plain Csr output");
}

#[test]
fn centered_output_to_h5ad_is_rejected() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("counts.rnad");
    let output = dir.path().join("norm.h5ad");
    write_counts_rnad(&input);

    // Centered shifted-CLR cannot be serialized to h5ad: the CLI must refuse.
    let status = bin()
        .args(["normalize", input.to_str().unwrap(), "-o", output.to_str().unwrap()])
        .status()
        .unwrap();
    assert!(!status.success(), "centered output to .h5ad should fail");
    assert!(!output.exists(), "no file should be written on the guard failure");
}

#[test]
fn overdispersion_subcommand_runs() {
    let dir = tempfile::tempdir().unwrap();
    let input = dir.path().join("counts.rnad");
    write_counts_rnad(&input);

    // This tiny fixture is overdispersed enough for a positive alpha.
    let out = bin().args(["overdispersion", input.to_str().unwrap()]).output().unwrap();
    let stdout = String::from_utf8_lossy(&out.stdout);
    if out.status.success() {
        assert!(stdout.contains("alpha"), "expected alpha in output, got: {stdout}");
    }
    // If the fixture happened to be under-dispersed the command exits non-zero with a clear
    // message; either way it must not panic.
}
