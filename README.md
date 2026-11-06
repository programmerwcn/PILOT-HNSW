# PILOT-HNSW

Code and experimental plotting data accompanying **“Pay As You Search: A Dynamic Query Planner for ANN Search.”** PILOT combines query-difficulty prediction, search-budget selection, and dataset-calibrated distance execution for HNSW.

This project provides two paths:

1. **Generate the paper's figures** from the included aggregate data.
2. **Run a fresh experiment** from vectors: build HNSW, compute ground truth, fit PCA, collect probe features and labels, train a classifier, select budgets/executors on validation queries, and measure held-out queries.

Datasets are downloaded separately. Indexes, PCA assets and trained models are generated locally; no private model files or baseline models are required by the new `pilot_search` entry point.

## Requirements and build

Linux x86-64, Python 3.9+, GCC 9.4+ with C++17/OpenMP, Eigen 3 and jsoncpp. The Python dependencies are NumPy, Matplotlib, LightGBM, scikit-learn and joblib. No GPU is required.

```bash
git clone https://github.com/programmerwcn/PILOT-HNSW.git
cd PILOT-HNSW
sudo apt-get update
sudo apt-get install -y g++ libeigen3-dev libjsoncpp-dev python3-venv
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python build.py
```

`build.py` locates the LightGBM shared library in the active Python environment. For nonstandard installations:

```bash
python build.py --eigen-include /path/to/eigen3 \
  --jsoncpp-include /path/to/jsoncpp \
  --lightgbm-lib /path/to/lib_lightgbm.so
```

Executables are written to `build/`. Use `--target pilot_search` to build only the unified search program, or `--output DIR` to change the destination. `-march=native` targets the build machine; rebuild on a different CPU. `requirements-tested.txt` records the Python package versions used in the local checks.

## Quick start: a complete small experiment

These commands generate synthetic vectors and run the entire pipeline without downloading a large dataset:

```bash
python reproduction/data.py toy --output data/toy
python reproduction/run.py --config configs/toy.json
cat runs/toy/summary.json
```

The output directory contains the index, exact ground truth, PCA parameters, classifier, frozen validation decision, native JSON configurations, logs and test results. A run with existing indexes must resume at a later stage or use a new `work_dir`; it does not silently overwrite the index.

Run the float32 and uint8 integration checks:

```bash
python reproduction/smoke_test.py
python reproduction/test_data.py
```

The checks cover the end-to-end pipeline, disjoint query splits, feature agreement with the retained extractors, exhaustive-distance agreement, partial-distance pruning, and rejection of a model/probe mismatch.

## Datasets: download and references

| Workload | Base vectors | Dimension / type | Source and citation |
|---|---:|---|---|
| GIST1M | 1,000,000 | 960 / float32 | [TexMex ANN datasets](http://corpus-texmex.irisa.fr/); Jégou, Douze and Schmid, *Product Quantization for Nearest Neighbor Search*, TPAMI 2011 ([DOI](https://doi.org/10.1109/TPAMI.2010.57)). |
| SIFT100M | First 100,000,000 BIGANN vectors | 128 / uint8 | [Big ANN dataset release](https://big-ann-benchmarks.com/neurips21.html); Jégou et al., *Searching in One Billion Vectors: Re-rank with Source Coding*, ICASSP 2011 ([DOI](https://doi.org/10.1109/ICASSP.2011.5946540)). |
| Deep100M | First 100,000,000 DEEP vectors | 96 / float32 | [Big ANN / Yandex release](https://big-ann-benchmarks.com/neurips21.html); Babenko and Lempitsky, *Efficient Indexing of Billion-Scale Datasets of Deep Descriptors*, [CVPR 2016](https://openaccess.thecvf.com/content_cvpr_2016/html/Babenko_Efficient_Indexing_of_CVPR_2016_paper.html). |
| LAION-10M | 10,000,000 image embeddings | 512 / float32 | [RoarGraph dataset preparation](https://github.com/matchyc/RoarGraph/blob/main/prepare_data.sh); Chen et al., *RoarGraph: A Projected Bipartite Graph for Efficient Cross-Modal Approximate Nearest Neighbor Search*, PVLDB 2024. |

BibTeX entries are in [references.bib](references.bib). The [Big ANN competition report](https://big-ann-benchmarks.com/neurips21.html) also provides citation and release terms for its dataset distribution. Follow each source's terms.

Download one dataset at a time:

```bash
python reproduction/data.py download --dataset gist  --output data/gist
python reproduction/data.py download --dataset sift  --output data/sift
python reproduction/data.py download --dataset deep  --output data/deep
python reproduction/data.py download --dataset laion --output data/laion
```

- **GIST:** downloads `gist.tar.gz` from the TexMex FTP server (`ftp://ftp.irisa.fr/local/texmex/corpus/gist.tar.gz`) and converts the base and LEARN vectors to `.fbin`. The new workflow splits LEARN queries into training, validation and test sets; the official QUERY vectors are also extracted.
- **SIFT/Deep:** downloads the first 100M vectors from the official 1B binary file, rewrites the count header, and downloads the public 10K query set. HTTP range downloads resume from `.part` files. `--base-count N` supports smaller prefixes; reduce the configured `ef_grid` if necessary and compute new ground truth for that prefix.
- **LAION:** follows RoarGraph's image-shard order `0,1,2,3,4,5,6,7,9,10`, takes the first 10M rows, and obtains text queries from its Zenodo release. Both base and query vectors are normalized for squared L2 search. The workflow recomputes ground truth for these prepared vectors rather than reusing an inner-product ground truth blindly.

Raw base storage alone is about 3.84 GB (GIST), 12.8 GB (SIFT), 38.4 GB (Deep), or 20.48 GB (LAION). Indexes, archives, source shards and PCA projections need additional RAM/disk. Full 100M index building and exhaustive ground-truth computation are substantial offline jobs; start with the toy workflow. No full-scale download or rebuild is performed automatically by the build/smoke-test commands.

Downloads depend on external hosting; GIST also requires outbound FTP access. If a source is unavailable, obtain the same vectors from the source maintainer and convert local `.fvecs`, `.bvecs`, `.npy`, `.fbin` or `.u8bin` files:

```bash
python reproduction/data.py convert --input /data/gist_base.fvecs \
  --output data/gist/base.fbin --dtype float32
```

For official SIFT/Deep **100M-prefix** ground truth, the [benchmark dataset definitions](https://github.com/harsha-simhadri/big-ann-benchmarks/blob/main/benchmark/datasets.py) publish these files:

```bash
curl -fL --retry 3 -o data/sift/gt.download \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/bigann-100M
python reproduction/data.py convert --format bigann-gt \
  --input data/sift/gt.download --output data/sift/gt.ibin

curl -fL --retry 3 -o data/deep/gt.download \
  https://dl.fbaipublicfiles.com/billion-scale-ann-benchmarks/GT_100M/deep-100M
python reproduction/data.py convert --format bigann-gt \
  --input data/deep/gt.download --output data/deep/gt.ibin
```

Add `"ground_truth": "../data/sift/gt.ibin"` (or the Deep path) to the corresponding configuration to skip exhaustive computation. Ground truth must match the exact base subset, query order, normalization and requested `k`. Official ground truth for 100M is not valid for a smaller base prefix.

## Run a dataset experiment

After downloading, select the appropriate preset:

```bash
python reproduction/run.py --config configs/gist.json
# Alternatives: configs/sift.json, configs/deep.json, configs/laion.json
```

All paths in a configuration are resolved relative to that configuration. Edit `base`, `query` and `work_dir` to use another disk. Stages can be run individually:

```bash
python reproduction/run.py --config configs/gist.json --stage prepare
python reproduction/run.py --config configs/gist.json --stage index
python reproduction/run.py --config configs/gist.json --stage collect
python reproduction/run.py --config configs/gist.json --stage train
python reproduction/run.py --config configs/gist.json --stage calibrate
python reproduction/run.py --config configs/gist.json --stage evaluate
```

The dataset presets use `M=16`, `efConstruction=100`, `k=100`, clean51 features and a **50-expansion probe**. The toy presets use 100 expansions for feature-parity checks. Changing the probe budget requires a new run and a newly trained model; the native entry point checks this metadata.

Training uses only training labels. Validation chooses the three `efSearch` budgets, then the fastest feasible executor under the mean-recall target and a 0.01 additional recall-loss budget. Test queries never select the model or policy. A failed validation target stops the workflow; a failed test target is reported without retuning.

Native query timing includes query projection, probing, classification and resumed search; file loading and ground-truth scoring are excluded. Each search evaluation performs an untimed warmup pass. See [REPRODUCTION.md](REPRODUCTION.md) for experiment settings, metrics and generated file formats.

## Build an index or run search directly

Input, output and log paths are explicit; there are no personal path defaults.

```bash
build/index_float --path_data data/gist/base.fbin \
  --path_index runs/manual/gist.hnsw --log-file runs/manual/gist-build.log \
  --M 16 --ef_construction 100 --build_mode hierarchy --random_seed 2025

build/index_uint8 --data data/sift/base.u8bin \
  --index runs/manual/sift.hnsw --log-file runs/manual/sift-build.log \
  --M 16 --ef-construction 100 --threads 1 --seed 2025 --checkpoint-every 0
```

After the toy pipeline, run a small online search using an editable configuration:

```bash
build/pilot_search configs/search_example.json
```

To rerun the calibrated policy exactly as configured by the pipeline:

```bash
build/pilot_search runs/toy/test_pilot.config.json
```

Supported modes are `features`, `grid`, `plain` and `pilot`; `pilot` executes online classification rather than replaying stored tier predictions. It needs the generated classifier and PCA transform. The PCA executor additionally requires base projections/norms. `gt` is optional for search; omitting it suppresses recall scoring. Search output contains neighbor IDs and timing/work measurements.

## Generate the paper figures

```bash
python -m pip install -r plotting/requirements.txt
for script in plotting/plots/fig_*.py; do
    python "$script"
done
```

Figures are saved in `plotting/figures/`. These scripts read the included aggregate data, independently of new runs under `runs/`.

## Layout

- `reproduction/`: portable data preparation, training/validation workflow, online runner and integration checks.
- `configs/`: dataset presets and a search example.
- `src/`: HNSW, index builders and feature extractors.
- `benchmarks/`: shared distance/probe kernels and native runners.
- `python-script/`: SIFT classifier trainer.
- `plotting/`: figure scripts and aggregate data.
