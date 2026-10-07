"""
Sanity tests for the network filtering benchmark pipeline.

These tests exercise the current pipeline end-to-end on a handful of small
graphs (a few real networks and small n=100 Watts-Strogatz graphs) instead
of relying on pre-generated result CSVs.

The previous version of this file asserted against
results/realNetsResults.csv and results/simulatedNetsResults.csv using a
schema (`filename` column, a `bench_net2net_filtering` benchmark name, and
short indicator column names such as `information_retention`/`jaccard`
straight in the row) that predates the current netfilterbench package.
Today, run_experiment() produces rows keyed by `graph_name`, and scalar
indicators keep the function's own name (e.g.
`calculate_information_retention`) as their column, so those old
assumptions no longer hold. Running small experiments directly, rather
than reading old CSVs, keeps these tests correct as the package evolves
and does not depend on any specific benchmark run having been executed
first.

Run with (from the repository root, since data paths below are relative):
    pytest tests/test_csvresults.py
"""

import functools
import glob

import pytest

from netfilterbench import (
    bench_noise_filtering,
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    load_graph,
    maximum_spanning_tree,
    predictive_filtering_metrics,
    run_experiment,
)

INDICATOR_FUNCS = [
    calculate_information_retention,
    calculate_jaccard_similarity,
    common_metrics,
    predictive_filtering_metrics,
]

# Small, fast fixtures kept deterministic via sorted() + a fixed slice.
REAL_NET_SAMPLE = sorted(glob.glob("data/real_nets/graphs/*.pickle"))[:2]
WS_SMALL_SAMPLE = sorted(
    glob.glob("data/simulated_nets/watts_strogatz/unweighted/*graph100.pickle")
)[:2]

pytestmark = pytest.mark.skipif(
    not REAL_NET_SAMPLE, reason="data/real_nets/graphs is empty or missing"
)


@functools.lru_cache(maxsize=None)
def _run(graph_path, filter_func=maximum_spanning_tree, noise_level=0.1):
    # Cached: several tests below share the same (graph, filter, noise_level)
    # combination and would otherwise redundantly recompute all metrics.
    return run_experiment(
        graph_path, filter_func, bench_noise_filtering, noise_level, INDICATOR_FUNCS
    )


class TestResultSchema:
    """Every experiment result should carry the expected base fields."""

    @pytest.mark.parametrize("graph_path", REAL_NET_SAMPLE)
    def test_base_fields(self, graph_path):
        row = _run(graph_path)
        assert row["filter"] == "maximum_spanning_tree"
        assert row["benchmark"] == "bench_noise_filtering"
        assert row["noise_level"] == 0.1
        assert row["graph_name"]

    @pytest.mark.parametrize("graph_path", REAL_NET_SAMPLE)
    def test_metrics_are_numeric_or_none(self, graph_path):
        row = _run(graph_path)
        for key, value in row.items():
            if key in ("graph_name", "filter", "benchmark"):
                continue
            assert value is None or isinstance(value, (int, float)), (
                f"{key} is not numeric: {value!r}"
            )


class TestDensityInvariants:
    @pytest.mark.parametrize("graph_path", REAL_NET_SAMPLE)
    def test_density_matches_formula(self, graph_path):
        row = _run(graph_path)
        graph = load_graph(graph_path)
        n, m = graph.number_of_nodes(), graph.number_of_edges()
        expected_density = (2 * m) / (n * (n - 1))
        assert row["density_original"] == pytest.approx(expected_density, rel=1e-6)

    @pytest.mark.parametrize("graph_path", REAL_NET_SAMPLE)
    def test_mst_never_increases_density(self, graph_path):
        # A spanning tree only ever removes edges, so density cannot increase
        # regardless of the noise added beforehand.
        row = _run(graph_path)
        assert row["density_filtered"] <= row["density_original"] + 1e-9

    @pytest.mark.parametrize("graph_path", REAL_NET_SAMPLE)
    def test_densities_are_positive(self, graph_path):
        row = _run(graph_path)
        assert row["density_original"] > 0
        assert row["density_filtered"] > 0


class TestWattsStrogatzClustering:
    """Watts-Strogatz networks should keep their signature high clustering
    even after noise + maximum spanning tree filtering is applied to the
    *original* graph (checked here on the unfiltered/original side of the
    result)."""

    @pytest.mark.skipif(
        not WS_SMALL_SAMPLE, reason="no small watts-strogatz graphs found"
    )
    @pytest.mark.parametrize("graph_path", WS_SMALL_SAMPLE)
    def test_clustering_is_high(self, graph_path):
        row = _run(graph_path)
        assert row["average_clustering_original"] > 0.1
