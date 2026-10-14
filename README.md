## NetFilterBench

This repository implements a framework for benchmarking network filtering and sparsification techniques. It provides real and simulated networks, a set of common filtering algorithms, and a pipeline that measures how well each filter recovers the original network structure after noise is added.

## Project Structure

- **data/**: Real and simulated networks used as benchmarking input
  - **real_nets/**: Networks extracted from the Index of Complex Networks (ICON), one graph per `.pickle` file under `real_nets/graphs/`, plus `ICON_info.csv` with per-network metadata
  - **simulated_nets/**: Synthetically generated networks (see `generate_simulated_networks.py`)
  - **incident_texts/**: Weighted networks built from incident-report texts, one `.pickle` per product category (case study)
- **src/netfilterbench/**: The installable package — everything below is importable as `netfilterbench` after installing it (see Installation)
  - **net_filtering/**: Implementation of the network filtering/sparsification techniques (`filter.py`)
  - **benchmark/**: Noise-injection and benchmarking logic (`benchmark.py`) and the network comparison metrics (`networkIndicators.py`)
  - **data_loader/**: Utilities for listing graph files on disk (`loader.py`)
  - **experiments/**: Runs a single filter/benchmark/noise-level combination on one graph (`experiment.py`) and orchestrates all combinations in parallel, writing results incrementally to CSV (`runner.py`)
- **src/visualization/**: Generates the paper's result plots (`plots.py`) — project-specific analysis code, not part of the `netfilterbench` package
- **main.py**: Configuration-driven entry point — pick filters, benchmarks, noise levels, and indicators, then run the full pipeline over `data/`
- **results/**: Where benchmark results (CSVs) and generated plots are stored
- **tests/**: Sanity tests for the pipeline (`pytest tests/`)

## Installation

`netfilterbench` (the filters, comparison metrics, and the parallel benchmark runner — everything under `src/netfilterbench/`) can be installed on its own as a library, independently of the rest of this repository.

### As a library, in another project

Not published on PyPI yet. Until it is, install straight from GitHub:

```
pip install git+https://github.com/nathanruf/Benchmarking-Network-Filtering.git
```

or, for local development against a cloned copy (editable install — code changes are picked up without reinstalling):

```
git clone https://github.com/nathanruf/Benchmarking-Network-Filtering.git
cd Benchmarking-Network-Filtering
pip install -e .
```

Either way this pulls in only `netfilterbench`'s own dependencies (`networkx`, `numpy`, `scipy`, `tqdm`) — not the full repository's.

### For working on this repository

Clone the repository, then install everything needed to run `main.py`, regenerate simulated networks, produce plots, and run the test suite:

```
git clone https://github.com/nathanruf/Benchmarking-Network-Filtering.git
cd Benchmarking-Network-Filtering
pip install -r requirements.txt
pip install -e .
```

## Usage

`main.py` is the entry point and doubles as a runnable usage example: edit the configuration section at the top of the file to choose which filters, benchmarks, noise levels, and indicators to run, then execute:

```
python main.py
```

```python
# main.py — configuration section
DATA_PATH = "data/real_nets/graphs"
OUTPUT_PATH = "results/output.csv"

FILTERS = [
    maximum_spanning_tree,
    k_core_decomposition,
    # ... any subset of the filters in src/netfilterbench/net_filtering/filter.py
]

BENCHMARKS = [
    bench_noise_filtering,             # NBF: noise before filtering
    bench_structural_noise_filtering,  # NAF: noise after filtering
]

NOISE_LEVELS = [0.1, 0.2, 0.3]

INDICATOR_FUNCS = [
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    predictive_filtering_metrics,
]

MAX_WORKERS = 4
```

This writes one row per (graph, filter, benchmark, noise_level) combination to `OUTPUT_PATH`, streaming results to disk as they complete rather than holding everything in memory.

To use the pipeline directly instead of going through `main.py` — e.g. from your own project, once `netfilterbench` is installed — call `run_experiment` for a single combination:

```python
from netfilterbench import run_experiment, maximum_spanning_tree, bench_noise_filtering, calculate_jaccard_similarity

result = run_experiment(
    graph_path="data/simulated_nets/watts_strogatz/unweighted/0watts_strogatz_graph100.pickle",
    filter_func=maximum_spanning_tree,
    benchmark_func=bench_noise_filtering,
    noise_level=0.1,
    indicator_funcs=[calculate_jaccard_similarity],
)
print(result)
```

Filters that take parameters (e.g. `threshold`, `disparity_filter`) accept them as keyword arguments; use `functools.partial` to fix them before passing the filter into the pipeline:

```python
from functools import partial
from netfilterbench import disparity_filter

FILTERS = [partial(disparity_filter, alpha=0.1)]
```

## Benchmarking Process

Two benchmark variants are available, both defined in `src/netfilterbench/benchmark/benchmark.py`:

1. **`bench_noise_filtering` (NBF — noise before filtering)**: adds random noise edges to the original network, applies the filter to the noisy network, and compares the result to the *original* network.
2. **`bench_structural_noise_filtering` (NAF — noise after filtering)**: first applies the filter to the original network to obtain a structural baseline, adds noise to that structural network, applies the filter again, and compares the result to the *structural* (once-filtered) network — measuring how well the filter recovers a structure it has already produced, rather than the raw original.

For both variants, the comparison between the two networks is done using the indicator functions configured in `INDICATOR_FUNCS` (see `src/netfilterbench/benchmark/networkIndicators.py`): edge-set similarity (Jaccard), degree-distribution similarity (information retention), edge-level prediction metrics (precision/recall/F1/RMSE), and a set of common structural metrics computed on both networks (degree, clustering, path length, centrality, assortativity, density, etc.). `common_metrics` also reports `triangles_spearman_correlation`: the Spearman correlation between per-node triangle counts in the two networks (over the nodes they share), measuring how well the filter preserves local triangle structure.

> **Note on reproducibility:** noise injection is randomized. `bench_noise_filtering`/`bench_structural_noise_filtering` accept a `seed` parameter (default `42`), but at least one full round of the results currently in `results/` was generated without pinning it explicitly — those results are not guaranteed to be exactly reproducible from a fresh run.

## Available Data

The `/data` directory contains real, simulated, and incident-text network datasets for benchmarking purposes.

### Real Networks

`/data/real_nets/graphs/` contains 550 individual `.pickle` files, one per network, extracted from ICON (Index of Complex Networks). These networks represent various real-world systems and phenomena across different domains: 23% social, 23% economic, 32% biological, 12% technological, 3% information, and 7% transportation graphs — drawn from the corpus used in this [PNAS paper](https://github.com/Aghasemian/OptimalLinkPrediction). Per-network metadata (name, domain, size, etc.) is in `ICON_info.csv`; see the full [ICON network metadata](https://docs.google.com/spreadsheets/d/1DCSPqD3cLDKZ00QC7NjZpjgnE33coCXwigjxTY5NhYc/edit?usp=sharing) for details.

### Simulated Networks

`/data/simulated_nets/` contains artificially generated networks across five graph models:

1. Random Graphs (Erdős-Rényi model)
2. Grid Graphs (periodic)
3. Barabási-Albert Graphs (scale-free networks)
4. LFR Benchmark Graphs (with community structure)
5. Watts-Strogatz Graphs (small-world networks)

For each model, 100 instances are generated at each of 10 sizes (100 to 1,000 nodes, in steps of 100), both as unweighted and edge-weighted graphs — 2,000 graphs per model, 10,000 in total. Files follow the pattern `data/simulated_nets/<model>/<weighted|unweighted>/<instance_index><model>_graph<size>.pickle` (e.g. `data/simulated_nets/watts_strogatz/unweighted/0watts_strogatz_graph100.pickle`).

To regenerate these networks (or generate new ones), run `python data/simulated_nets/generate_simulated_networks.py` (pass `--override` to regenerate files that already exist).

### Incident Texts

`/data/incident_texts/` contains 15 weighted networks built from incident-report texts, one `.pickle` per product category (e.g. `baby.pickle`, `electronics.pickle`, `toys_and_children.pickle`). They are used as a case study (see `case_study_triangles_incident_texts.csv` under Visualizing Results).

> **Note on edge weights:** the real (ICON) networks are unweighted, so only the five filters that don't depend on edge weights (degree-based, random edge, simplified Simmelian, randomized overlapping trees and k-core) give meaningful results on them. The incident-text networks are weighted, so all ten filters can run there.

## Available Network Filtering Techniques

`src/netfilterbench/net_filtering/filter.py` implements the following filtering and sparsification techniques. All of them share a single interface — `func(graph: nx.Graph, **kwargs) -> nx.Graph` — and any parameters are passed as keyword arguments (see the `functools.partial` example above). Directed graphs are automatically converted to undirected via `to_undirected()` before filtering.

1. **Maximum Spanning Tree** — `maximum_spanning_tree(graph)`: keeps the set of edges connecting all nodes with the highest total edge weight. (Named for the maximum, not minimum, variant — despite the classic "MST" abbreviation conventionally denoting the minimum spanning tree.)
2. **Planar Maximally Filtered Graph (PMFG)** — `pmfg(graph)`: builds a planar graph that greedily maximizes total edge weight.
3. **Triangulated Maximally Filtered Graph (TMFG)** — `tmfg(graph)`: builds a 3-planar (triangulated) graph maximizing total edge weight, faster to construct than PMFG.
4. **Global Threshold Filter** — `threshold(graph, quantile=0.5)`: removes edges whose weight falls below a cutoff. By default the cutoff is the `quantile`-th quantile of the graph's own edge-weight distribution (the median), so filtering strength is comparable across graphs with different weight scales (e.g. `quantile=0.7` keeps roughly the strongest 30% of edges). Pass `threshold=<value>` to use a fixed absolute cutoff instead.
5. **Degree-Based Sparsifier** — `degree_based_sparsifier(graph, target_ratio=0.5)`: keeps a target fraction of edges, prioritizing edges between higher-degree nodes.
6. **Random Edge Sparsifier** — `random_edge_sparsifier(graph, target_ratio=0.5, seed=42)`: randomly removes edges down to a target fraction.
7. **Simplified Simmelian Sparsifier** — `simplified_simmelian_sparsifier(graph, max_rank=5)`: keeps, for each node, only its top-ranked neighbors by shared-neighbor overlap. A simplified approximation of Nick et al. (2013)'s Simmelian backbone method.
8. **Disparity Filter** — `disparity_filter(graph, alpha=0.255)`: keeps statistically significant edges per node, as described in Serrano et al. (2009), PNAS. The paper's recommended range for `alpha` is [0.01, 0.5].
9. **Randomized Overlapping Trees** — `randomized_overlapping_trees(graph, num_trees=3, seed=42)`: unions multiple minimum spanning trees, each computed over random edge weights (the original weights are restored afterwards), inspired by Garas & Argyrakis (2009), arXiv:0812.3227.
10. **K-Core Decomposition** — `k_core_decomposition(graph, k=None)`: recursively removes nodes with degree below `k`; defaults to returning the main (largest) core.

### Adding New Filters

To add a new filtering technique:

1. Implement it in `src/netfilterbench/net_filtering/filter.py`, following the `func(graph: nx.Graph, **kwargs) -> nx.Graph` interface. Export it from `src/netfilterbench/net_filtering/__init__.py` and `src/netfilterbench/__init__.py` if it should be part of the package's public API.
2. Add it to the `FILTERS` list in `main.py` (optionally wrapped in `functools.partial` to fix parameters).
3. It will automatically be benchmarked against every configured benchmark, noise level, and indicator function.

## Visualizing Results

`src/visualization/plots.py` generates figures and tables from the result CSVs directly under a given input directory (non-recursive, so it never re-reads its own output):

```
python -m src.visualization.plots --input results/ --output results/graphics
```

It currently produces (NBF benchmark, Real and Simulated nets):

- `lines_canberra_{real,simulated}.png`: structural preservation (RQ2) vs noise level, one panel per network domain/model, one line per filter. Structural preservation is the normalized Canberra distance between the original and filtered structural-metric vectors, `d(o, f) = (1/n) * sum_i |f_i - o_i| / (|f_i| + |o_i|)`. Each term is a symmetric relative change bounded in [0, 1], so no metric dominates because of its scale (as in NetSimile, Berlingerio et al., 2012). Average path length, diameter and degree assortativity are excluded, since they can be undefined.
- `heatmap_filter_metric_{real,simulated}.png`: filter x structural metric heatmaps.
- `scatter_density_jaccard_{real,simulated}.png`: retained density vs Jaccard per filter.
- Triangle-correlation tables for the incident-texts case study and the Real/Simulated appendix, plus information-retention and predictive-metric summary tables (CSV and PNG).

## Testing

Sanity tests live in `tests/` and run small experiments directly through the current pipeline (rather than depending on any specific prior benchmark run). From the repository root:

```
pytest tests/
```

## Releasing New Versions of the Package

1. Bump `version` in `pyproject.toml` (and `__version__` in `src/netfilterbench/__init__.py`) — PyPI (and a plain `pip install` from a git tag) will not accept re-uploading the same version number, so every release needs a new one. Follow [semantic versioning](https://semver.org/): patch (`0.1.1`) for fixes, minor (`0.2.0`) for backwards-compatible additions, major (`1.0.0`) for breaking changes to the public API.
2. Build the distribution:
   ```
   pip install build
   python -m build
   ```
3. Publish it. Options, in order of how immediately available they need to be:
   - **Not on PyPI yet, needed by another project now**: point that project's dependency at this repository directly — `pip install git+https://github.com/nathanruf/Benchmarking-Network-Filtering.git@<tag-or-branch>` — no publishing step required.
   - **PyPI** (once ready to make it public): `pip install twine` then `twine upload dist/*`. Consider a dry run on [TestPyPI](https://test.pypi.org/) first (`twine upload --repository testpypi dist/*`) to make sure the metadata and long description render correctly before the real upload.

Downstream projects then upgrade by bumping the version they depend on and reinstalling — the same as any other PyPI package.
