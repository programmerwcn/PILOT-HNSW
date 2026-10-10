# PILOT-HNSW

This repository hosts the code and artifacts for **“Pay as You Search: A Dynamic Query Planner for ANN Search.”**

## Overview

Hierarchical Navigable Small World (HNSW) indexes typically expose the
recall–latency trade-off through a single, workload-wide search-width
parameter. However, queries vary substantially in difficulty, and the value of
additional computation also changes as an individual search progresses. A
single static configuration therefore either spends unnecessary work on easy
queries or provides insufficient search capacity for difficult ones.

Our execution analysis identifies two stages in HNSW search:

1. **Rapid approach.** Candidate expansions frequently improve the current
   result as the search frontier moves toward the query's nearest-neighbor
   region. Full-distance evaluation is particularly valuable during this
   high-leverage stage.
2. **Plateau exploration.** For difficult queries, improvements become sparse
   and the search spends much of its time evaluating and dismissing candidates
   while attempting to recover long-tail neighbors.

These stages require different optimizations. Easy queries should stop soon
after satisfying their recall requirement, whereas difficult queries must
continue through the plateau while reducing the cost of each additional
candidate evaluation.

## PILOT

PILOT is a stage-aware query planner for HNSW that treats **search extent**,
**distance fidelity**, and **termination** as separate runtime decisions.
Its query-processing workflow consists of three main steps:

1. **Probe and monitor.** Every query begins with a bounded, full-distance
   probe. The probe preserves high-fidelity early routing and collects search
   trajectory, local-geometry, query, dataset, and index features. Its search
   state is retained and resumed rather than discarded.
2. **Adaptive search extent.** A lightweight monitor assigns the query to a
   calibrated search-complexity tier. Each tier maps to a validated search
   extent, allowing easy queries to terminate early while reserving larger
   budgets for harder queries.
3. **Adaptive distance execution.** When continued search enters the plateau,
   a dataset-calibrated PCA executor screens candidates using progressively
   refined distances. Competitive candidates are evaluated exactly, reducing
   long-tail exploration cost without returning approximate distances.

## Evaluation summary

We evaluate PILOT on four visual, deep, and multimodal workloads containing
between 1 million and 100 million vectors. At operating points targeting 0.90
mean Recall@100, PILOT achieves:

- **2.61–5.68× less full-distance-equivalent search work** than conservatively
  provisioned plain HNSW;
- **1.96–3.17× lower mean end-to-end latency** than plain HNSW; and
- **19–43% lower mean latency** than the faster of two learned-termination
  baselines.

The results show that adapting termination alone is insufficient: jointly
planning how far to search and how candidate distances are evaluated provides
additional efficiency, especially for queries with long plateau stages.

## Artifact status

The implementation, experiment scripts, configurations, and instructions for
reproducing the reported results will be released in this repository.
