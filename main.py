"""
Entry point for the network filtering benchmark.

Edit the configuration below to select filters, benchmarks, noise levels,
and indicator functions, then run:

    python main.py

Passing parameters to filters or indicators:
    Use functools.partial to fix parameters before passing to the pipeline.

    Example - disparity filter with custom alpha:
        from functools import partial
        partial(disparity_filter, alpha=0.1)

    Example - common_metrics with approximate betweenness (k=100):
        from functools import partial
        partial(common_metrics, betweenness_k=100)
"""

from functools import partial

from netfilterbench import (
    maximum_spanning_tree,
    pmfg,
    tmfg,
    threshold,
    disparity_filter,
    degree_based_sparsifier,
    random_edge_sparsifier,
    simplified_simmelian_sparsifier,
    randomized_overlapping_trees,
    k_core_decomposition,
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    predictive_filtering_metrics,
    bench_noise_filtering,
    bench_structural_noise_filtering,
    run_all,
)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
#
# incident_texts is weighted (unlike real_nets, which is unweighted and can
# only run the 5 filters that don't depend on edge weight), so all 10
# filters can run here.

DATA_PATH = "data/incident_texts"
OUTPUT_PATH = "results/c4.csv"

FILTERS = [
    maximum_spanning_tree,
    pmfg,
    tmfg,
    threshold,
    disparity_filter,
    degree_based_sparsifier,
    random_edge_sparsifier,
    simplified_simmelian_sparsifier,
    randomized_overlapping_trees,
    k_core_decomposition,
]

BENCHMARKS = [
    bench_noise_filtering,             # NBF: noise before filtering
    bench_structural_noise_filtering,  # NAF: noise after filtering
]

NOISE_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5]

# Choose which indicator functions to compute.
# Use functools.partial to pass optional parameters, e.g.:
# partial(common_metrics, betweenness_k=100)  # approximate betweenness
INDICATOR_FUNCS = [
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    predictive_filtering_metrics,
]

MAX_WORKERS = 4

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    run_all(
        data_path=DATA_PATH,
        output_path=OUTPUT_PATH,
        filters=FILTERS,
        benchmarks=BENCHMARKS,
        noise_levels=NOISE_LEVELS,
        indicator_funcs=INDICATOR_FUNCS,
        max_workers=MAX_WORKERS,
    )
