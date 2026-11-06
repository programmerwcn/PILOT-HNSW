# Reproduction protocol

## Experiment workflow

The included aggregate inputs regenerate the experimental plots. The new `reproduction/run.py` workflow independently produces a complete **PILOT versus HNSW experiment** from user-provided or downloaded vectors, including all intermediate assets required by its online runner. It supports float32 and uint8 squared-L2 data with at least 32 dimensions.

The native `pilot_search` runner consumes the portable files produced by `reproduction/run.py`.

## Reproducible choices

- A seed-controlled permutation partitions source queries into disjoint training, validation and test sets. `splits.json` records source row IDs. Query vectors must be independent of the indexed base vectors; do not supply the base itself as a query training pool.
- Index construction uses `M=16`, `efConstruction=100`, seed 2025 and one construction thread by default. The float workflow uses the `hierarchy` builder. Changing threading or insertion order can change results.
- PCA fits a seeded sample of BASE only; queries do not fit the transform. The GIST/LAION presets use 40/256 components; SIFT/Deep retain 32 coordinates for classifier features. SIFT uses exact distances and Deep also screens partial64.
- Probe features use 16 routing statistics, three query statistics and 32 PCA coordinates. The same implementation supplies offline and online features. Model metadata records the probe budget and dimension. Test features are computed online, not used in training.
- Min-EF labels use the configured discrete grid. Queries missing the target at the largest EF are censored and counted in `planner.json`; the labels are not exact continuous minimum search budgets.
- Training sorts min-EF labels into three equal-size groups. Validation chooses a nondecreasing triple of EF budgets. It verifies the resulting online state-reuse path separately; if necessary it tries the maximum grid budget for all tiers, or fails explicitly.
- Executor selection compares exact execution, configured partial-distance execution, and configured PCA candidates. PCA candidates compare all/hard-only scopes with heap-full/plateau activation at the configured prefix.
- The validation decision is frozen in `selection.json` before test execution. Test outcomes are reported without selecting a new winner.

## Timing and metrics

Search runs are single-threaded with one untimed warmup pass followed by the configured number of repetitions. Mean latency is the median of the per-repeat query means. The time includes probe traversal, clean51 feature computation, query PCA projection, classification and resumed search. It excludes asset loading, JSON serialization and recall scoring. There is no default CPU affinity or cache-isolation guarantee; record the hardware/load and optionally use `taskset` when comparing latency.

NDC is full-distance-equivalent work, including probe and hierarchy distance evaluations. PCA and partial-distance work is charged by evaluated dimensions divided by original dimensions. Query projection, feature arithmetic and model inference are included in time but excluded from NDC.

Recall@k uses overlap with the first k ground-truth IDs. The built-in exhaustive generator uses float64 squared L2 and row ID to break ties. The public Big ANN evaluator may give credit for additional tied-distance neighbors, so its reported recall can differ at a tie boundary.

## Data formats and generated files

The portable binary format is little-endian `uint32 n`, `uint32 d`, then a contiguous `n × d` matrix. `.fbin` contains float32, `.u8bin` contains uint8, and the portable `.ibin` contains uint32 IDs only. Big ANN ground-truth downloads can append distances after IDs; use `data.py convert --format bigann-gt` before supplying them to the workflow.

Each run creates:

- `experiment.json`, `splits.json`, `preparation.json`: settings, query IDs and PCA sampling information.
- `index.hnsw`, `queries.*`, `gt.ibin`: graph, selected queries and matching exact neighbors.
- `pca_mean.fbin`, `pca_components.fbin`: the transform; PCA executor runs also produce `pca_scores.fbin` and `pca_norms.fbin`.
- `classifier.txt`, `model_metadata.json`, `planner.json`: trained model and budget-selection results.
- `selection.json`: validation candidates and frozen winner.
- `*.config.json`, `*.log`, `test_*.json`, `summary.json`: runnable native configurations and measurements.

These are generated locally and ignored by Git. They do not change the included plotting data. Run separate configurations in separate work directories; keep the generated assets together when moving a run and update absolute paths in native configurations.

## Validation performed

The original eight C++ targets and the portable `pilot_search` target compile with GCC 9.4. The local tests exercise 96-, 128-, 512- and 960-dimensional fixtures (float32 and uint8), with both 50- and 100-expansion probes, end-to-end, compare clean51 features against retained collectors, check exhaustive nearest distances at a full search budget, check partial-pruning results and reject mismatched probe metadata. All seven plotting scripts were also run during preparation.

Hardware-dependent latency and target attainment must be measured on the selected workload.
