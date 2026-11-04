# PILOT-HNSW

This repository accompanies the paper **“Pay As You Search: A Dynamic Query Planner for ANN Search.”**

It contains the core implementation of PILOT, a dynamic query planner for HNSW-based approximate nearest neighbor search, along with plotting scripts and aggregate experimental data.

## Contents

- `src/`: HNSW index construction and probe feature extraction.
- `benchmarks/`: native search runners, probe-state reuse, and distance execution policies.
- `python-script/`: SIFT query-difficulty classifier training.
- `plotting/`: scripts and aggregate data for generating experimental figures.
- `build.py`: build entry point for the C++ programs.

Full datasets, indexes, trained models, and per-query experimental records are not included. The source snapshot contains implementation variants and is not a complete reproduction package for every configuration reported in the paper.

## Generate Figures

```bash
python -m pip install -r plotting/requirements.txt
for script in plotting/plots/fig_*.py; do
    python "$script"
done
```

Generated figures are saved in `plotting/figures/`.
