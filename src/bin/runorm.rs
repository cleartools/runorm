//! `runorm` command-line tool: PFlog / shifted-CLR normalization over ruanndata containers.

use std::error::Error;
use std::path::{Path, PathBuf};

use clap::{Args, Parser, Subcommand};
use ruanndata::RuAnnData;
use runorm::{
    counts_from_matrixdata, estimate_overdispersion, normalize_anndata, NormParams, OutTarget,
    PfTarget,
};

#[derive(Parser)]
#[command(
    name = "runorm",
    version,
    about = "Sparse PFlog / shifted-CLR normalization for single-cell count data"
)]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand)]
enum Cmd {
    /// Normalize raw counts and write the shifted-CLR result.
    Normalize(NormalizeArgs),
    /// Estimate the overdispersion alpha and print the variance-stabilizing K = 4·alpha·s.
    Overdispersion(OverdispersionArgs),
}

#[derive(Args)]
struct NormalizeArgs {
    /// Input file (.rnad always; .h5ad/.zarr require the h5ad/io features).
    input: PathBuf,
    /// Output file. Centered (shifted-CLR) output must be .rnad.
    #[arg(short = 'o', long)]
    output: PathBuf,
    /// PF target: mean | median | auto | <float K>. `auto` estimates alpha (K = 4·alpha·s).
    #[arg(long, default_value = "mean")]
    target: String,
    /// Supply the overdispersion alpha directly (sets K = 4·alpha·s); overrides --target.
    #[arg(long)]
    alpha: Option<f64>,
    /// Skip the log1p step (store PF-scaled values).
    #[arg(long = "no-log1p")]
    no_log1p: bool,
    /// Skip the CLR centering (emit plain log1pPF as a Csr; writable to any format).
    #[arg(long = "no-center")]
    no_center: bool,
    /// Read counts from this layer instead of X.
    #[arg(long)]
    layer: Option<String>,
    /// Write the result to this layer instead of replacing X.
    #[arg(long = "out-layer")]
    out_layer: Option<String>,
}

#[derive(Args)]
struct OverdispersionArgs {
    /// Input file (.rnad always; .h5ad/.zarr require the h5ad/io features).
    input: PathBuf,
    /// Read counts from this layer instead of X.
    #[arg(long)]
    layer: Option<String>,
}

#[derive(Clone, Copy, PartialEq)]
enum Format {
    Rnad,
    H5ad,
    Zarr,
}

fn detect_format(path: &Path) -> std::result::Result<Format, String> {
    match path.extension().and_then(|e| e.to_str()) {
        Some("rnad") => Ok(Format::Rnad),
        Some("h5ad") => Ok(Format::H5ad),
        Some("zarr") => Ok(Format::Zarr),
        other => Err(format!(
            "cannot determine format from extension {other:?}; use .rnad, .h5ad, or .zarr"
        )),
    }
}

fn load_anndata(path: &Path) -> std::result::Result<RuAnnData, Box<dyn Error>> {
    match detect_format(path)? {
        Format::Rnad => Ok(ruanndata::load(path)?),
        Format::H5ad => {
            #[cfg(feature = "h5ad")]
            {
                Ok(ruanndata::read_h5ad(path)?)
            }
            #[cfg(not(feature = "h5ad"))]
            {
                Err("reading .h5ad requires building runorm with --features h5ad".into())
            }
        }
        Format::Zarr => {
            #[cfg(feature = "io")]
            {
                Ok(ruanndata::read_zarr(path)?)
            }
            #[cfg(not(feature = "io"))]
            {
                Err("reading .zarr requires building runorm with --features io".into())
            }
        }
    }
}

fn save_anndata(path: &Path, adata: &RuAnnData) -> std::result::Result<(), Box<dyn Error>> {
    match detect_format(path)? {
        Format::Rnad => Ok(ruanndata::save(path, adata)?),
        Format::H5ad => {
            #[cfg(feature = "h5ad")]
            {
                Ok(ruanndata::write_h5ad(path, adata)?)
            }
            #[cfg(not(feature = "h5ad"))]
            {
                Err("writing .h5ad requires building runorm with --features h5ad".into())
            }
        }
        Format::Zarr => {
            #[cfg(feature = "io")]
            {
                Ok(ruanndata::write_zarr(path, adata)?)
            }
            #[cfg(not(feature = "io"))]
            {
                Err("writing .zarr requires building runorm with --features io".into())
            }
        }
    }
}

fn parse_target(s: &str, alpha: Option<f64>) -> std::result::Result<PfTarget, String> {
    if let Some(a) = alpha {
        return Ok(PfTarget::Alpha(a));
    }
    match s {
        "mean" => Ok(PfTarget::MeanDepth),
        "median" => Ok(PfTarget::MedianDepth),
        "auto" => Ok(PfTarget::EstimateAlpha),
        other => other
            .parse::<f64>()
            .map(PfTarget::Fixed)
            .map_err(|_| format!("invalid --target '{other}' (expected mean|median|auto|<float>)")),
    }
}

fn run_normalize(args: NormalizeArgs) -> std::result::Result<(), Box<dyn Error>> {
    let params = NormParams {
        target: parse_target(&args.target, args.alpha)?,
        log1p: !args.no_log1p,
        center: !args.no_center,
    };

    // Guard: a centered (shifted-CLR) matrix can only be serialized to .rnad.
    let out_fmt = detect_format(&args.output)?;
    if params.center && out_fmt != Format::Rnad {
        return Err(
            "centered shifted-CLR output can only be written to .rnad (h5ad/zarr writers reject it); \
             use a .rnad output or pass --no-center for a plain log1pPF Csr"
                .into(),
        );
    }

    let mut adata = load_anndata(&args.input)?;
    let out = match args.out_layer {
        Some(name) => OutTarget::Layer(name),
        None => OutTarget::ReplaceX,
    };
    let report = normalize_anndata(&mut adata, args.layer.as_deref(), out, &params)?;

    save_anndata(&args.output, &adata)?;
    match report.alpha {
        Some(a) => eprintln!(
            "normalized: K = {:.6} (alpha = {:.6}, mean depth = {:.6}) -> {}",
            report.k,
            a,
            report.mean_depth,
            args.output.display()
        ),
        None => eprintln!(
            "normalized: K = {:.6} (mean depth = {:.6}) -> {}",
            report.k,
            report.mean_depth,
            args.output.display()
        ),
    }
    Ok(())
}

fn run_overdispersion(args: OverdispersionArgs) -> std::result::Result<(), Box<dyn Error>> {
    let adata = load_anndata(&args.input)?;
    let source = match args.layer.as_deref() {
        Some(name) => adata
            .layers
            .get(name)
            .ok_or_else(|| format!("layer '{name}' not found"))?,
        None => &adata.x,
    };
    let counts = counts_from_matrixdata(source)?;
    let od = estimate_overdispersion(&counts)?;
    println!("alpha      = {:.6}", od.alpha);
    println!("mean_depth = {:.6}", od.mean_depth);
    println!("K = 4*a*s  = {:.6}", od.k);
    Ok(())
}

fn main() {
    let cli = Cli::parse();
    let result = match cli.cmd {
        Cmd::Normalize(args) => run_normalize(args),
        Cmd::Overdispersion(args) => run_overdispersion(args),
    };
    if let Err(e) = result {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}
