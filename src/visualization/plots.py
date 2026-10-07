"""
Network filtering benchmark — visualization module.

CURRENTLY (work in progress on the RQ1/RQ2 headline figures), generate_graphics only
produces the line plots, per the results/ CSVs:
    - lines_jaccard_real.png       : Jaccard vs noise level, one panel per Real-net domain,
                                      one line per filter, mean +/-1 std error bars (NBF)
    - lines_jaccard_simulated.png  : same, one panel per Simulated-net model
    - lines_canberra_real.png      : normalized Canberra distance (structural-metric vector)
                                      vs noise level, one panel per Real-net domain (NBF)
    - lines_canberra_simulated.png : same, one panel per Simulated-net model

incident_texts is excluded from these -- it's the case-study section, not the general
Real/Simulated comparison. NBF (bench_noise_filtering) is the benchmark used throughout.

The rest of this module (heatmaps, ranking tables, NAF degenerate-case report) is still
defined below, just temporarily not called from generate_graphics -- see its docstring.

Usage:
    python -m src.visualization.plots --input results/ --output results/graphics
"""

import os
import glob
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as patheffects


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

STRUCTURAL_METRICS = [
    'average_degree',
    'average_clustering',
    'average_path_length',
    'diameter',
    'average_betweenness',
    'average_closeness',
    'global_efficiency',
    'degree_assortativity',
    'density',
    'transitivity',
    'degree_variance',
    'maximum_degree',
]

# Metrics excluded from the Canberra distance:
#   - average_path_length, diameter: undefined for disconnected graphs.
#   - degree_assortativity: undefined (NaN, 0/0 in the Pearson correlation) for perfectly
#     regular graphs -- e.g. every periodic grid graph (all degree 4), which would
#     otherwise poison the whole vector and make canberra_distance NaN for 100% of
#     Grid's rows. It's also signed, and Canberra's per-term bound only holds for
#     non-negative values.
_CANBERRA_EXCLUDED = ('average_path_length', 'diameter', 'degree_assortativity')
CANBERRA_METRICS = [m for m in STRUCTURAL_METRICS if m not in _CANBERRA_EXCLUDED]

# Simulated-network model tokens matched against the (lowercased) filename.
# Order matters: check more specific tokens before shorter ones if ever ambiguous.
SIM_MODEL_TOKENS = [
    ('Barabasi_albert', 'barabasi_albert'),
    ('Watts_strogatz', 'watts_strogatz'),
    ('Grid', 'grid'),
    ('Random', 'random'),
    ('LFR', 'lfr'),
]

ICON_INFO_PATH = 'data/real_nets/ICON_info.csv'

GROUP_CLASS_REAL = 'Real nets'
GROUP_CLASS_SIM = 'Simulated nets'
GROUP_CLASS_INC = 'Incident texts'

NBF_BENCHMARK = 'bench_noise_filtering'
NAF_BENCHMARK = 'bench_structural_noise_filtering'

REFERENCE_NOISE_LEVEL = 0.3

# Fixed filter -> color mapping, shared by every figure that plots per-filter
# lines. Keeps a given filter the same color whether it shows up in the Real
# nets figure (5 filters -- weight-independent only, real_nets is unweighted)
# or the Simulated nets one (all 10), so the two are visually comparable.
FILTER_ORDER = [
    'maximum_spanning_tree',
    'pmfg',
    'tmfg',
    'threshold',
    'disparity_filter',
    'degree_based_sparsifier',
    'random_edge_sparsifier',
    'simplified_simmelian_sparsifier',
    'randomized_overlapping_trees',
    'k_core_decomposition',
]
FILTER_COLORS = {f: plt.get_cmap('tab10')(i % 10) for i, f in enumerate(FILTER_ORDER)}

# Display names matching the paper's terminology (some filters are
# simplified/renamed approximations of the cited methods, so the paper uses
# more precise names than the internal function names).
FILTER_LABELS = {
    'maximum_spanning_tree': 'Maximum Spanning Tree',
    'pmfg': 'PMFG',
    'tmfg': 'TMFG',
    'threshold': 'Threshold',
    'disparity_filter': 'Disparity Filter',
    'degree_based_sparsifier': 'Degree-Based Sparsifier',
    'random_edge_sparsifier': 'Random Edge Sparsifier',
    'simplified_simmelian_sparsifier': 'Simplified Simmelian Sparsifier',
    'randomized_overlapping_trees': 'Randomized Overlapping Trees',
    'k_core_decomposition': 'K-Core Decomposition',
}

# Column-name aliases across pipeline versions -> canonical name.
_COLUMN_ALIASES = {
    'calculate_jaccard_similarity': 'jaccard',
    'calculate_information_retention': 'information_retention',
}

# Filter-name aliases across pipeline versions -> canonical name. Filters were
# renamed to match the paper's terminology ('mst' also to remove the
# minimum/maximum ambiguity of the acronym); older results CSVs still carry
# the old names.
_FILTER_ALIASES = {
    'mst': 'maximum_spanning_tree',
    'local_degree_sparsifier': 'degree_based_sparsifier',
    'simmelian_sparsifier': 'simplified_simmelian_sparsifier',
    'overlapping_trees': 'randomized_overlapping_trees',
}


# ---------------------------------------------------------------------------
# Loading and grouping
# ---------------------------------------------------------------------------

def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename older/newer pipeline column names onto one canonical schema."""
    rename = {}
    for src, dst in _COLUMN_ALIASES.items():
        if src in df.columns and dst not in df.columns:
            rename[src] = dst
    if rename:
        df = df.rename(columns=rename)
    # If both aliases somehow ended up present, keep the canonical one.
    for src, dst in _COLUMN_ALIASES.items():
        if src in df.columns and dst in df.columns:
            df = df.drop(columns=[src])
    if 'filter' in df.columns:
        df['filter'] = df['filter'].replace(_FILTER_ALIASES)
    return df


def _match_sim_model(name_lower: str):
    for label, token in SIM_MODEL_TOKENS:
        if token in name_lower:
            return label
    return None


def _assign_group(df: pd.DataFrame, icon_info: pd.DataFrame) -> pd.DataFrame:
    """
    Tag each row with:
        group       — specific network type (e.g. 'Watts_strogatz', 'Biological', 'kitchen')
        group_class — macro category (Real nets / Simulated nets / Incident texts)

    Current pipeline output uses a single 'graph_name' column for every benchmark
    (real nets, simulated nets, incident texts alike) — unlike older runs, which used
    'filename' for real/simulated nets and 'graph_name' only for incident texts. So the
    type is inferred from the name's content, not from which column carries it:
        - ends with '_real_nets'                    -> Real nets (domain via ICON_info.csv)
        - contains a known simulated-model token     -> Simulated nets
        - anything else                              -> Incident texts (name == category)
    Falls back to the legacy 'filename'-only convention when 'graph_name' is absent.
    """
    df = df.copy()

    name_col = 'graph_name' if 'graph_name' in df.columns else ('filename' if 'filename' in df.columns else None)
    if name_col is None:
        df['group'] = None
        df['group_class'] = None
        return df

    name = df[name_col].astype(str)
    name_lower = name.str.lower()
    is_real = name.str.contains('_real_nets', regex=False)
    sim_group = name_lower.apply(_match_sim_model)
    is_sim = sim_group.notna() & ~is_real

    df['group'] = None
    df['group_class'] = None

    df.loc[is_sim, 'group'] = sim_group[is_sim]
    df.loc[is_sim, 'group_class'] = GROUP_CLASS_SIM

    is_incident = ~is_real & ~is_sim
    df.loc[is_incident, 'group'] = name[is_incident]
    df.loc[is_incident, 'group_class'] = GROUP_CLASS_INC

    if is_real.any():
        clean_name = name[is_real].str.replace('_real_nets', '', regex=False)
        domain_lookup = icon_info.set_index('network_name')['networkDomain']
        df.loc[is_real, 'group'] = clean_name.map(domain_lookup).values
        df.loc[is_real, 'group_class'] = GROUP_CLASS_REAL

    return df


def load_all(input_path: str) -> pd.DataFrame:
    """
    Load every results CSV directly under input_path (non-recursive — this deliberately
    does NOT descend into an output/graphics subdirectory, so re-running the script never
    re-ingests its own generated tables as if they were benchmark results).
    """
    icon_info = pd.read_csv(ICON_INFO_PATH)
    frames = []

    for path in sorted(glob.glob(os.path.join(input_path, '*.csv'))):
        df = pd.read_csv(path)
        df = _normalize_columns(df)

        if 'graph_name' not in df.columns and 'filename' not in df.columns:
            print(f"[load] Skipping {os.path.basename(path)}: no graph_name/filename column.")
            continue

        df = _assign_group(df, icon_info)
        n_before = len(df)
        df = df.dropna(subset=['group'])
        n_dropped = n_before - len(df)
        if n_dropped:
            print(f"[load] {os.path.basename(path)}: dropped {n_dropped} row(s) with unrecognized group.")

        frames.append(df)

    if not frames:
        raise ValueError(f"No usable CSV files found directly under {input_path}")

    combined = pd.concat(frames, ignore_index=True, sort=False)
    return combined


def _add_canberra_distance(df: pd.DataFrame) -> pd.DataFrame:
    """Add a 'canberra_distance' column: normalized Canberra distance between the original
    and filtered structural-metric vectors (CANBERRA_METRICS only — see _CANBERRA_EXCLUDED),

        d(o, f) = (1/n) * sum_i |f_i - o_i| / (|f_i| + |o_i|)

    Each term is a symmetric relative change bounded in [0, 1] (halving and doubling a
    metric both give 1/3; a metric dropping to 0 gives 1), so no single metric can
    dominate regardless of its scale — unlike cosine distance on the raw values, where
    large-magnitude metrics (degree_variance, maximum_degree) decide the angle and the
    [0, 1]-ranged ones barely register. Dividing by n keeps the result in [0, 1].
    Canberra-distance comparison of structural-feature vectors follows NetSimile
    (Berlingerio et al., 2012).

    A term where both values are 0 (0/0) counts as 0 -- the metric didn't change. NaN
    where any needed column is missing/NaN.

    Vectorized over the whole dataframe with numpy (not a per-row .apply) — this runs over
    hundreds of thousands of rows (10 filters x ~10k simulated graphs x 5 noise levels x 2
    benchmarks, plus real_nets/incident_texts), and a Python-level row loop there is slow
    enough to blow past a single command's time budget.
    """
    orig_cols = [f'{m}_original' for m in CANBERRA_METRICS]
    filt_cols = [f'{m}_filtered' for m in CANBERRA_METRICS]
    missing = [c for c in orig_cols + filt_cols if c not in df.columns]
    if missing:
        warnings.warn(f"Canberra distance: missing columns {missing}, filling as NaN.")
        for c in missing:
            df[c] = np.nan

    df = df.copy()
    a = df[orig_cols].to_numpy(dtype=float)
    b = df[filt_cols].to_numpy(dtype=float)

    num = np.abs(b - a)
    den = np.abs(a) + np.abs(b)
    with np.errstate(invalid='ignore', divide='ignore'):
        terms = np.where(den > 0, num / den, 0.0)
    dist = terms.mean(axis=1)

    invalid = np.isnan(a).any(axis=1) | np.isnan(b).any(axis=1)
    df['canberra_distance'] = np.where(invalid, np.nan, dist)
    return df


# ---------------------------------------------------------------------------
# Line plots: metric x noise level, one line per filter, faceted by specific
# network type/domain within a macro group (Real nets / Simulated nets).
#
# This is the RQ1/RQ2 headline figure. A single macro-group average (one line
# for "Real nets" as a whole, one for "Simulated nets" as a whole) was
# considered and dropped: it pools topologies/domains that behave too
# differently (e.g. grid vs Barabasi-Albert, or Biological vs Technological)
# into one misleading number. So the facet grid IS the main figure, not an
# appendix detail -- one PNG per (metric, macro group), with one panel per
# specific type/domain inside it.
# ---------------------------------------------------------------------------

def plot_metric_lines(
    df: pd.DataFrame,
    metric: str,
    group_class: str,
    ylabel: str,
    out_file: str,
    benchmark: str = NBF_BENCHMARK,
    ncols: int = 3,
    ylim=None,
    facet: bool = True,
) -> None:
    """
    One panel per specific network type/domain found in `group_class` (e.g.
    group_class=GROUP_CLASS_SIM -> one panel per model: Barabasi_albert, Grid,
    LFR, Random, Watts_strogatz; GROUP_CLASS_REAL -> one panel per ICON domain).

    Within each panel: x = noise level, one line per filter (color fixed via
    FILTER_COLORS so it matches across panels and across the Real/Simulated
    figures). Each point is the mean of `metric` across every graph instance
    contributing to that (type, filter, noise level) cell, with a vertical
    error bar of +/-1 std -- the 100 replicates per size for simulated nets,
    the several real networks per domain for real nets. std is drawn as 0
    (no visible bar) where only one instance contributes.

    All panels in a figure share the same y-axis range so they can be
    compared visually against each other, not just within a panel.

    `facet=False` collapses this into a single panel: every instance in
    `group_class` pooled together regardless of specific type, one line per
    filter, error bar = std across that whole pool. Use only where the types
    within the group are similar enough that pooling doesn't hide anything
    (e.g. incident_texts categories for a case study) -- for Real nets /
    Simulated nets this is deliberately not the default (see module notes).
    """
    d = df[(df['benchmark'] == benchmark) & (df['group_class'] == group_class)]

    if facet:
        group_col = 'group'
        groups = sorted(d['group'].dropna().unique())
    else:
        d = d.assign(_all=group_class)
        group_col = '_all'
        groups = [group_class] if not d.empty else []

    if not groups:
        warnings.warn(f"plot_metric_lines: no data for group_class={group_class!r}, benchmark={benchmark!r}.")
        return

    noise_levels = sorted(d['noise_level'].dropna().unique())
    present = set(d['filter'].dropna().unique())
    filters_present = [f for f in FILTER_ORDER if f in present]

    stats = (
        d.groupby([group_col, 'filter', 'noise_level'])[metric]
        .agg(['mean', 'std', 'count'])
        .reset_index()
    )

    if ylim is None:
        lo = (stats['mean'] - stats['std'].fillna(0)).min()
        hi = (stats['mean'] + stats['std'].fillna(0)).max()
        pad = 0.05 * (hi - lo) if hi > lo else 0.05
        ylim = (max(0.0, lo - pad), hi + pad)

    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    handles_by_filter = {}
    for idx, g in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        gd = stats[stats[group_col] == g]
        for f in filters_present:
            fd = gd[gd['filter'] == f].set_index('noise_level').reindex(noise_levels)
            if fd['mean'].isna().all():
                continue
            handle = ax.errorbar(
                noise_levels, fd['mean'].to_numpy(), yerr=fd['std'].fillna(0).to_numpy(),
                marker='o', markersize=3, linewidth=1.3, capsize=2.5,
                color=FILTER_COLORS[f], label=FILTER_LABELS.get(f, f),
            )
            handles_by_filter[f] = handle
        if facet:
            ax.set_title(g, fontsize=13)
        ax.set_xticks(noise_levels)
        ax.set_xlabel('Noise level', fontsize=12)
        if idx % ncols == 0:
            ax.set_ylabel(ylabel, fontsize=12)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    # Turn off unused trailing panels (grid doesn't divide evenly, e.g. 5 groups / 3 cols)
    for idx in range(len(groups), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    ordered_filters = [f for f in FILTER_ORDER if f in handles_by_filter]
    ordered_handles = [handles_by_filter[f] for f in ordered_filters]
    ordered_labels = [FILTER_LABELS.get(f, f) for f in ordered_filters]
    fig.legend(
        ordered_handles, ordered_labels, loc='lower center',
        ncol=min(5, len(ordered_labels)), bbox_to_anchor=(0.5, -0.02),
        fontsize=12, frameon=False,
    )

    fig.tight_layout(rect=(0, 0.11, 1, 1))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[lines:{metric}] {group_class}: {len(groups)} panel(s), {len(filters_present)} filter(s) -> {out_file}")


# ---------------------------------------------------------------------------
# Scatter: retained density x Jaccard, one panel per specific network type,
# one point per filter -- companion to the Jaccard line plots above, but
# collapsing the noise-level axis into a single point per filter instead of
# tracking it explicitly.
# ---------------------------------------------------------------------------

def plot_density_vs_jaccard_scatter(
    df: pd.DataFrame,
    group_class: str,
    out_file: str,
    benchmark: str = NBF_BENCHMARK,
    ncols: int = 3,
) -> None:
    """
    One panel per specific network type/domain found in `group_class` (same
    faceting as plot_metric_lines). Within each panel: one point per filter,
    x = retained density (density_filtered / density_original), y = Jaccard
    similarity.

    Unlike plot_metric_lines and the heatmap, there is no per-noise-level
    stage and no absolute value: every graph instance at every noise level is
    pooled into one flat set of rows per (type, filter), and both axes are
    plain means over that pool -- noise level is not tracked or averaged
    separately, just ignored as a dimension.

    All panels in a figure share the same x/y ranges so they're visually
    comparable, matching plot_metric_lines.
    """
    d = df[(df['benchmark'] == benchmark) & (df['group_class'] == group_class)].copy()
    d = d[d['density_original'] != 0]

    groups = sorted(d['group'].dropna().unique())
    if not groups:
        warnings.warn(f"plot_density_vs_jaccard_scatter: no data for group_class={group_class!r}, benchmark={benchmark!r}.")
        return

    present = set(d['filter'].dropna().unique())
    filters_present = [f for f in FILTER_ORDER if f in present]

    d['retained_density'] = d['density_filtered'] / d['density_original']
    points = (
        d.groupby(['group', 'filter'])[['retained_density', 'jaccard']]
        .mean()
        .reset_index()
    )

    xlo, xhi = points['retained_density'].min(), points['retained_density'].max()
    ylo, yhi = points['jaccard'].min(), points['jaccard'].max()
    xpad = 0.05 * (xhi - xlo) if xhi > xlo else 0.05
    ypad = 0.05 * (yhi - ylo) if yhi > ylo else 0.05
    xlim = (max(0.0, xlo - xpad), xhi + xpad)
    ylim = (max(0.0, ylo - ypad), yhi + ypad)

    nrows = int(np.ceil(len(groups) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)

    handles_by_filter = {}
    for idx, g in enumerate(groups):
        ax = axes[idx // ncols][idx % ncols]
        gd = points[points['group'] == g]
        for f in filters_present:
            row = gd[gd['filter'] == f]
            if row.empty:
                continue
            handle = ax.scatter(
                row['retained_density'], row['jaccard'],
                s=30, color=FILTER_COLORS[f], label=FILTER_LABELS.get(f, f),
            )
            handles_by_filter[f] = handle
        ax.set_title(g, fontsize=13)
        ax.set_xlabel('Edge-retention ratio', fontsize=12)
        if idx % ncols == 0:
            ax.set_ylabel('Jaccard similarity', fontsize=12)
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)
        ax.tick_params(labelsize=11)
        ax.grid(True, alpha=0.3)

    for idx in range(len(groups), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis('off')

    ordered_filters = [f for f in FILTER_ORDER if f in handles_by_filter]
    ordered_handles = [handles_by_filter[f] for f in ordered_filters]
    ordered_labels = [FILTER_LABELS.get(f, f) for f in ordered_filters]
    fig.legend(
        ordered_handles, ordered_labels, loc='lower center',
        ncol=min(5, len(ordered_labels)), bbox_to_anchor=(0.5, -0.02),
        fontsize=12, frameon=False,
    )

    fig.tight_layout(rect=(0, 0.11, 1, 1))
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[scatter:density-jaccard] {group_class}: {len(groups)} panel(s), {len(filters_present)} filter(s) -> {out_file}")


# ---------------------------------------------------------------------------
# Heatmap: group x noise level, for a single metric (Jaccard, Canberra distance, ...)
# ---------------------------------------------------------------------------

def plot_metric_heatmap(
    df: pd.DataFrame,
    metric: str,
    title: str,
    out_file: str,
    benchmark: str = NBF_BENCHMARK,
    cmap: str = 'viridis',
    vmin=None,
    vmax=None,
    fmt: str = '{:.2f}',
) -> None:
    """
    Rows = network group (every specific type: real-net domains, simulated models,
    incident_texts categories). Columns = noise level. Cell = mean value across all
    filters and all graphs in that group (filter-level detail lives in the ranking
    tables, not here). Annotated as 'mean' when only one sample contributes (e.g.
    incident_texts categories, which are a single graph — no std is defined), or
    'mean (+/-std)' when multiple graphs/instances contribute.
    """
    d = df[df['benchmark'] == benchmark]
    agg = d.groupby(['group', 'noise_level'])[metric].agg(['mean', 'std', 'count']).reset_index()

    groups = sorted(agg['group'].dropna().unique())
    noise_levels = sorted(agg['noise_level'].dropna().unique())

    mean_mat = np.full((len(groups), len(noise_levels)), np.nan)
    annot = np.full((len(groups), len(noise_levels)), '', dtype=object)

    for i, g in enumerate(groups):
        for j, nl in enumerate(noise_levels):
            row = agg[(agg['group'] == g) & (agg['noise_level'] == nl)]
            if row.empty:
                continue
            m, s, n = row.iloc[0][['mean', 'std', 'count']]
            if pd.isna(m):
                continue
            mean_mat[i, j] = m
            if n > 1 and not pd.isna(s):
                annot[i, j] = f"{fmt.format(m)}\n(+/-{fmt.format(s)})"
            else:
                annot[i, j] = fmt.format(m)

    fig_h = max(4.0, 0.32 * len(groups) + 1.5)
    fig, ax = plt.subplots(figsize=(6.5, fig_h))
    im = ax.imshow(mean_mat, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)

    ax.set_xticks(range(len(noise_levels)))
    ax.set_xticklabels([f'{nl:.1f}' for nl in noise_levels])
    ax.set_yticks(range(len(groups)))
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_xlabel('Noise level')

    for i in range(len(groups)):
        for j in range(len(noise_levels)):
            if annot[i, j]:
                ax.text(j, i, annot[i, j], ha='center', va='center', fontsize=6)

    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)

    print(f"[heatmap:{metric}] {len(groups)} groups x {len(noise_levels)} noise levels -> {out_file}")


# ---------------------------------------------------------------------------
# Heatmap: filter x structural metric, mean |% variation|
# ---------------------------------------------------------------------------

HEATMAP_METRICS = list(STRUCTURAL_METRICS)

# degree_assortativity (and to a lesser extent other metrics) oscillates around/
# crosses zero, so a relative ((filtered-original)/original) variation blows up
# to absurd magnitudes whenever the original value is near zero -- observed cell
# values past 10^12 %. The computation is left untouched (uses all the data, no
# rows dropped) -- instead, a fixed display cap kicks in: a cell whose real value
# exceeds TEXT_DISPLAY_CAP is shown as ">=999%" instead of the literal (meaningless)
# number. Fixed, not per-column/percentile -- with only ~10 filters per column, a
# percentile-based cap can itself land on a contaminated value (confirmed: the
# 90th pct of degree_assortativity on simulated nets came out to 1.87e12%, useless
# as a cap). Color normalization (_column_normalize) is unrelated and still
# percentile-based -- that one only ever has to pick a shade, not print a number.
TEXT_DISPLAY_CAP = 999

# Text halo so the black annotation stays legible on both ends of the viridis
# scale (near-black cells at the low end, bright yellow at the high end). Thin --
# just enough for contrast, not a thick outline.
_TEXT_STROKE = [patheffects.withStroke(linewidth=1, foreground='white')]


def _pct_variation(sub: pd.DataFrame, metric: str) -> float:
    """Mean signed % variation ((filtered - original) / original * 100) for one
    metric, across the rows of `sub` (dropping NaN and original==0 rows). NaN if
    none usable."""
    oc, fc = f'{metric}_original', f'{metric}_filtered'
    if oc not in sub.columns or fc not in sub.columns:
        return np.nan
    s = sub[[oc, fc]].dropna()
    s = s[s[oc] != 0]
    if s.empty:
        return np.nan
    return float(((s[fc] - s[oc]) / s[oc]).mean() * 100)


def _column_normalize(mean_mat: pd.DataFrame, cap_quantile: float = 0.9):
    """
    Per-column min-max normalization of `mean_mat` to [0, 1], used only to pick
    each cell's COLOR -- the real values stay in the text annotation, untouched.

    Each metric (column) has its own natural scale of variation (e.g. density
    tops out near 100%, degree_variance can run past 500%), so one shared scale
    across the whole matrix leaves the naturally-small columns looking flat/blue
    regardless of how much they actually vary within themselves. Normalizing per
    column fixes that -- and capping at `cap_quantile` (values at/above it all
    map to 1.0) keeps a single outlier row from compressing the rest of that
    column into a sliver near 0.

    Returns (color_mat, caps): `caps` is the raw (un-normalized, real %) cap
    value per column -- the same threshold used for color, reused by the caller
    to flag cells whose real value sits above it (see plot_filter_metric_heatmap).
    """
    color_mat = pd.DataFrame(index=mean_mat.index, columns=mean_mat.columns, dtype=float)
    caps = pd.Series(index=mean_mat.columns, dtype=float)
    for col in mean_mat.columns:
        vals = mean_mat[col].dropna()
        if vals.empty:
            continue
        lo = vals.min()
        cap = vals.quantile(cap_quantile)
        caps[col] = cap
        span = cap - lo
        if span <= 0:
            color_mat[col] = mean_mat[col].where(mean_mat[col].isna(), 0.5)
        else:
            color_mat[col] = ((mean_mat[col] - lo) / span).clip(lower=0.0, upper=1.0)
    return color_mat, caps


def plot_filter_metric_heatmap(
    df: pd.DataFrame,
    group_class: str,
    out_file: str,
    benchmark: str = NBF_BENCHMARK,
    groups=None,
) -> None:
    """
    Rows = filter, columns = structural metric (HEATMAP_METRICS). Cell = how much
    that filter disturbs that metric in this macro group, summarized without
    letting either the noise-level axis or the network-type axis hide the other:

        1. Within each (network type, filter, metric, noise level): mean signed
           % variation across graph instances (as before).
        2. Across noise levels: take the ABSOLUTE value of each of those numbers,
           then average. Using |x| instead of a signed mean avoids noise-level
           swings that cross zero cancelling out -- confirmed this happens for
           real, e.g. pmfg/average_degree on simulated nets goes -19% -> +6.7%
           across the noise sweep; a signed average reads that as "~0%, no
           effect" when the truth is a large swing in both directions. Gives one
           number per network type.
        3. Across network types within `group_class`: plain (unweighted) mean of
           those per-type numbers -- this is the cell's value/color, so a type
           with many instances (e.g. simulated's 1000-graph types) doesn't drown
           out one with few (e.g. a small real-net domain). The std across types
           is also kept and annotated in the cell, so a reader can tell whether
           the mean reflects a consistent effect or one outlier type dragging it.

    `groups`, if given, restricts step 3 to just those specific types (e.g. a
    single domain) instead of every type found in `group_class` -- for an
    on-demand, single-type heatmap when a specific case needs a closer look,
    without generating one per type by default. Cells are 'N/A' where a filter
    has no usable data for that metric anywhere considered.
    """
    d = df[(df['benchmark'] == benchmark) & (df['group_class'] == group_class)]
    filters = sorted(d['filter'].dropna().unique())
    types = sorted(groups) if groups is not None else sorted(d['group'].dropna().unique())
    noise_levels = sorted(d['noise_level'].dropna().unique())

    mean_mat = pd.DataFrame(index=filters, columns=HEATMAP_METRICS, dtype=float)
    std_mat = pd.DataFrame(index=filters, columns=HEATMAP_METRICS, dtype=float)
    n_mat = pd.DataFrame(index=filters, columns=HEATMAP_METRICS, dtype=float)

    for f in filters:
        fd = d[d['filter'] == f]
        for metric in HEATMAP_METRICS:
            per_type = []
            for t in types:
                td = fd[fd['group'] == t]
                per_noise = []
                for nl in noise_levels:
                    nd = td[np.isclose(td['noise_level'], nl)]
                    v = _pct_variation(nd, metric)
                    if not pd.isna(v):
                        per_noise.append(abs(v))
                if per_noise:
                    per_type.append(np.mean(per_noise))
            if per_type:
                mean_mat.loc[f, metric] = np.mean(per_type)
                std_mat.loc[f, metric] = np.std(per_type, ddof=1) if len(per_type) > 1 else np.nan
                n_mat.loc[f, metric] = len(per_type)

    color_mat, _ = _column_normalize(mean_mat, cap_quantile=0.9)

    fig, ax = plt.subplots(figsize=(10, max(3.0, 0.45 * len(filters) + 1.5)))
    im = ax.imshow(color_mat.values.astype(float), aspect='auto', cmap='coolwarm', vmin=0, vmax=1)

    ax.set_xticks(range(len(mean_mat.columns)))
    ax.set_xticklabels(mean_mat.columns, rotation=45, ha='right', fontsize=8)
    ax.set_yticks(range(len(mean_mat.index)))
    ax.set_yticklabels([FILTER_LABELS.get(f, f) for f in mean_mat.index], fontsize=8)

    for i, f in enumerate(filters):
        for j, metric in enumerate(HEATMAP_METRICS):
            m = mean_mat.loc[f, metric]
            if pd.isna(m):
                ax.text(j, i, 'N/A', ha='center', va='center', fontsize=7, fontweight='bold', path_effects=_TEXT_STROKE)
                continue
            if m > TEXT_DISPLAY_CAP:
                # Real value sits above the fixed display cap -- the exact number is a
                # near-zero-denominator artifact (computation still used all the data;
                # only the text representation is capped), so show ">=999%" instead of
                # a meaningless literal figure. std dropped here too, since it inherits
                # the same distortion.
                txt = f"≥{TEXT_DISPLAY_CAP}%"
            else:
                s, n = std_mat.loc[f, metric], n_mat.loc[f, metric]
                txt = f"{m:.0f}%\n(+/-{s:.0f}%)" if (n > 1 and not pd.isna(s)) else f"{m:.0f}%"
            ax.text(j, i, txt, ha='center', va='center', fontsize=7, fontweight='bold', path_effects=_TEXT_STROKE)

    # No numeric ticks on the colorbar -- color is normalized independently per
    # column (metric), so a single shared numeric scale would misrepresent it
    # (the same shade means different % in different columns). The real values
    # are always in the cell text; color is just a per-column quick-scan aid.
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['Low', 'High'])
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"[heatmap:filter x metric] {group_class}: {len(filters)} filters x {len(HEATMAP_METRICS)} metrics, "
          f"{len(types)} type(s) -> {out_file}")


# ---------------------------------------------------------------------------
# Ranking tables: best filter per group, by Jaccard (reference noise level)
# ---------------------------------------------------------------------------

def build_ranking_table(
    df: pd.DataFrame,
    key: str,
    noise_level: float = REFERENCE_NOISE_LEVEL,
    benchmark: str = NBF_BENCHMARK,
) -> pd.DataFrame:
    """
    One row per value of `key` ('group' for the expanded/per-type table, 'group_class'
    for the grouped/macro-category table): the filter with the highest mean Jaccard at
    the reference noise level, plus that same filter's Jaccard and Canberra-distance values.
    """
    d = df[(df['benchmark'] == benchmark) & (np.isclose(df['noise_level'], noise_level))]

    rows = []
    for g, gd in d.groupby(key):
        by_filter = gd.groupby('filter').agg(
            jaccard=('jaccard', 'mean'),
            canberra_distance=('canberra_distance', 'mean'),
            n=('jaccard', 'count'),
        ).reset_index()
        by_filter = by_filter.dropna(subset=['jaccard'])
        if by_filter.empty:
            continue
        best = by_filter.sort_values('jaccard', ascending=False).iloc[0]
        rows.append({
            key: g,
            'best_filter': best['filter'],
            'jaccard': round(float(best['jaccard']), 3),
            'canberra_distance': (
                round(float(best['canberra_distance']), 3) if not pd.isna(best['canberra_distance']) else None
            ),
            'n_graphs': int(best['n']),
        })

    return pd.DataFrame(rows).sort_values(key).reset_index(drop=True)


def _save_table_png(table: pd.DataFrame, out_file: str, width: float = 9) -> None:
    fig_h = max(1.5, 0.35 * len(table) + 1.0)
    fig, ax = plt.subplots(figsize=(width, fig_h))
    ax.axis('off')
    tbl = ax.table(
        cellText=table.values,
        colLabels=table.columns,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.auto_set_column_width(col=list(range(len(table.columns))))
    tbl.scale(1, 1.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(out_file), exist_ok=True)
    fig.savefig(out_file, dpi=200, bbox_inches='tight')
    plt.close(fig)


# ---------------------------------------------------------------------------
# Information retention / predictive-metrics summary tables (NBF, Real +
# Simulated together, incident_texts excluded -- it's the case-study section)
# ---------------------------------------------------------------------------

def _general_networks(df: pd.DataFrame, benchmark: str) -> pd.DataFrame:
    return df[(df['benchmark'] == benchmark) & (df['group_class'] != GROUP_CLASS_INC)]


def _mean_std_cell(mean: float, std: float, decimals: int = 2) -> str:
    if pd.isna(mean):
        return 'N/A'
    if pd.isna(std):
        return f"{mean:.{decimals}f}"
    return f"{mean:.{decimals}f} (+/-{std:.{decimals}f})"


def build_information_retention_table(df: pd.DataFrame, benchmark: str = NBF_BENCHMARK) -> pd.DataFrame:
    """
    One row per filter, one column per specific network type (simulated models
    first, then real-net domains, all in one table). Cell = mean (+/-std) of
    information retention pooled over every graph instance and every noise
    level of that (filter, type) pair. 'N/A' where the filter never ran for
    that type (weight-dependent filters on unweighted real nets).
    """
    d = _general_networks(df, benchmark)
    filters = [f for f in FILTER_ORDER if f in set(d['filter'].dropna().unique())]
    sim_types = sorted(d.loc[d['group_class'] == GROUP_CLASS_SIM, 'group'].dropna().unique())
    real_types = sorted(d.loc[d['group_class'] == GROUP_CLASS_REAL, 'group'].dropna().unique())
    types = sim_types + real_types

    rows = []
    for f in filters:
        row = {'Filter': FILTER_LABELS.get(f, f)}
        for t in types:
            vals = d[(d['filter'] == f) & (d['group'] == t)]['information_retention'].dropna()
            row[t] = _mean_std_cell(vals.mean(), vals.std(ddof=1) if len(vals) > 1 else np.nan) if len(vals) else 'N/A'
        rows.append(row)
    return pd.DataFrame(rows, columns=['Filter'] + types)


PREDICTIVE_METRICS = [('RMSE', 'RMSE', 3), ('precision', 'Precision', 2), ('recall', 'Recall', 2), ('f1_score', 'F1', 2)]


def build_predictive_metrics_table(df: pd.DataFrame, benchmark: str = NBF_BENCHMARK) -> pd.DataFrame:
    """
    One row per filter, one column per predictive metric. Same three-step
    summarization as plot_filter_metric_heatmap: (1) mean across graph
    instances within each (network type, noise level); (2) mean of the
    absolute value of those across noise levels; (3) plain mean across network
    types (real + simulated together), with the std across types shown.
    """
    d = _general_networks(df, benchmark)
    filters = [f for f in FILTER_ORDER if f in set(d['filter'].dropna().unique())]
    types = sorted(d['group'].dropna().unique())
    noise_levels = sorted(d['noise_level'].dropna().unique())

    rows = []
    for f in filters:
        fd = d[d['filter'] == f]
        row = {'Filter': FILTER_LABELS.get(f, f)}
        for col, label, decimals in PREDICTIVE_METRICS:
            per_type = []
            for t in types:
                td = fd[fd['group'] == t]
                per_noise = []
                for nl in noise_levels:
                    v = td[np.isclose(td['noise_level'], nl)][col].dropna()
                    if len(v):
                        per_noise.append(abs(v.mean()))
                if per_noise:
                    per_type.append(np.mean(per_noise))
            if per_type:
                std = np.std(per_type, ddof=1) if len(per_type) > 1 else np.nan
                row[label] = _mean_std_cell(np.mean(per_type), std, decimals)
            else:
                row[label] = 'N/A'
        rows.append(row)
    return pd.DataFrame(rows, columns=['Filter'] + [label for _, label, _ in PREDICTIVE_METRICS])


# ---------------------------------------------------------------------------
# NAF: self-consistency footnote table (no heatmap, no noise-level line -- see
# module docstring / plan: NAF isn't a headline result, just a footnote check)
# ---------------------------------------------------------------------------

def build_naf_consistency_table(df: pd.DataFrame, benchmark: str = NAF_BENCHMARK) -> pd.DataFrame:
    """
    Self-consistency check for NAF (bench_structural_noise_filtering): does
    reapplying the same filter -- after noise is injected on top of an already-
    filtered network -- reliably recover that same already-filtered network?
    This is NOT about preserving the original raw network (that's NBF's job,
    covered by the Jaccard/Canberra line plots) -- it's about the filter's own
    robustness to being reapplied.

    One row per filter, one column per specific network type (group) -- open,
    NOT grouped into Real/Simulated macro categories. incident_texts excluded,
    same as everywhere else in the main body (it's the case-study section).
    Cell = 'mean (+/-std)' NAF Jaccard, pooled across every graph instance and
    noise level for that (filter, type) pair -- a plain pool, not the
    hierarchical per-type averaging used in the main heatmap: this is a
    footnote, not a headline metric. 'N/A' where the filter never ran for that
    type (e.g. the 5 weight-dependent filters, for any Real-net domain --
    unweighted).
    """
    d = df[(df['benchmark'] == benchmark) & (df['group_class'] != GROUP_CLASS_INC)]
    filters = [f for f in FILTER_ORDER if f in set(d['filter'].dropna().unique())]
    types = sorted(d['group'].dropna().unique())

    rows = []
    for f in filters:
        row = {'filter': f}
        for t in types:
            sub = d[(d['filter'] == f) & (d['group'] == t)]['jaccard'].dropna()
            if sub.empty:
                row[t] = 'N/A'
            elif len(sub) > 1:
                row[t] = f"{sub.mean():.2f} (+/-{sub.std(ddof=1):.2f})"
            else:
                row[t] = f"{sub.mean():.2f}"
        rows.append(row)

    return pd.DataFrame(rows, columns=['filter'] + types)


# ---------------------------------------------------------------------------
# Case study: per-type metric table (one column per specific type, + a final
# Average column) -- e.g. triangles_spearman_correlation on incident_texts.
# ---------------------------------------------------------------------------

def _type_averaged_cell(d: pd.DataFrame, metric: str, f: str, types) -> str:
    """
    Per-type mean of `metric` over noise levels (for filter `f`), then mean
    +/-std across those per-type values -- 'N/A' if none. Shared by the main
    metric's Average column and by any extra_columns in build_case_study_table,
    so every aggregate column in that table is computed the same way.
    """
    per_type_values = []
    for t in types:
        sub = d[(d['filter'] == f) & (d['group'] == t)][metric].dropna()
        if not sub.empty:
            per_type_values.append(float(sub.mean()))
    if not per_type_values:
        return 'N/A'
    avg = float(np.mean(per_type_values))
    if len(per_type_values) > 1:
        std = float(np.std(per_type_values, ddof=1))
        return f"{avg:.2f} (+/-{std:.2f})"
    return f"{avg:.2f}"


def build_case_study_table(
    df: pd.DataFrame,
    metric: str,
    group_class: str,
    benchmark: str = NBF_BENCHMARK,
    extra_columns: dict = None,
) -> pd.DataFrame:
    """
    One row per filter, one column per specific network/type found in
    `group_class` (e.g. each incident_texts category), plus a final 'Average'
    column.

    Per-type cell: mean of `metric` across noise levels for that (filter, type)
    pair -- no std shown here. With just the 5 ordered noise-level points per
    type (not independent replicates), a std across them isn't a meaningful
    dispersion measure, so it's left out to keep the table simple. 'N/A' where
    the filter has no data for that type.

    'Average' column: mean of the per-type cell values for that filter,
    +/-std across types -- this one DOES carry a std, since here it summarizes
    across genuinely different types/instances, not across noise levels.

    `extra_columns`, if given, is {label: metric_name} for additional aggregate-
    only columns appended after 'Average' -- e.g. {'Jaccard': 'jaccard'}, to sit
    next to the main metric as a sanity check (a filter with a high average on
    `metric` but a Jaccard near the noise-level's do-nothing baseline, ~1-noise,
    likely isn't doing much real filtering). Same per-type-then-aggregate
    computation as the 'Average' column, just without per-type columns of its
    own -- one number per filter.
    """
    d = df[(df['benchmark'] == benchmark) & (df['group_class'] == group_class)]
    filters = [f for f in FILTER_ORDER if f in set(d['filter'].dropna().unique())]
    types = sorted(d['group'].dropna().unique())

    rows = []
    for f in filters:
        row = {'filter': f}
        per_type_values = []
        for t in types:
            sub = d[(d['filter'] == f) & (d['group'] == t)][metric].dropna()
            if sub.empty:
                row[t] = 'N/A'
            else:
                v = float(sub.mean())
                row[t] = f"{v:.2f}"
                per_type_values.append(v)
        if per_type_values:
            avg = float(np.mean(per_type_values))
            if len(per_type_values) > 1:
                std = float(np.std(per_type_values, ddof=1))
                row['Average'] = f"{avg:.2f} (+/-{std:.2f})"
            else:
                row['Average'] = f"{avg:.2f}"
        else:
            row['Average'] = 'N/A'
        for label, extra_metric in (extra_columns or {}).items():
            row[label] = _type_averaged_cell(d, extra_metric, f, types)
        rows.append(row)

    return pd.DataFrame(rows, columns=['filter'] + types + ['Average'] + list((extra_columns or {}).keys()))


# ---------------------------------------------------------------------------
# NAF: degenerate-case report only (no chart — cited as footnote/text per plan)
# ---------------------------------------------------------------------------

def naf_degenerate_report(df: pd.DataFrame) -> pd.DataFrame:
    """
    (group, filter, noise_level) combinations where NAF (bench_structural_noise_filtering)
    produced a filtered graph with zero edges (true_positive + false_positive == 0) — e.g.
    the disparity filter emptying the graph. Meant to back a footnote in the ranking table,
    not a chart.
    """
    d = df[df['benchmark'] == NAF_BENCHMARK].copy()
    if 'true_positive' not in d.columns or 'false_positive' not in d.columns:
        return pd.DataFrame(columns=['group', 'group_class', 'filter', 'noise_level'])

    d['filtered_edges'] = d['true_positive'].fillna(0) + d['false_positive'].fillna(0)
    degenerate = d[d['filtered_edges'] == 0]
    cols = ['group', 'group_class', 'filter', 'noise_level']
    return degenerate[cols].drop_duplicates().sort_values(cols).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------

def generate_graphics(input_path: str, output_path: str) -> None:
    """
    Generate Results visuals from the raw results CSVs.

    Each figure is its own function above (plot_metric_lines, plot_filter_metric_heatmap,
    ...); this just loads the data once and calls the ones currently wanted, writing
    each output file directly. It does NOT touch anything already in `output_path`
    other than the specific files it writes -- no wipe/recreate of the directory, so
    re-running this after adding/enabling one more figure never deletes the others.
    """
    os.makedirs(output_path, exist_ok=True)

    df = load_all(input_path)
    df = _add_canberra_distance(df)

    print(f"[load] {len(df)} rows | groups: {sorted(df['group'].dropna().unique())}")

    # Line plots — normalized Canberra distance vs noise level, RQ2 headline figures.
    plot_metric_lines(
        df, 'canberra_distance', GROUP_CLASS_REAL, 'Canberra distance',
        os.path.join(output_path, 'lines_canberra_real.png'),
    )
    plot_metric_lines(
        df, 'canberra_distance', GROUP_CLASS_SIM, 'Canberra distance',
        os.path.join(output_path, 'lines_canberra_simulated.png'),
    )

    # NOTE: build_naf_consistency_table call still temporarily removed from here
    # (function unchanged above, just not invoked).

    # Heatmap — filter x structural metric, RQ2 diagnostic detail.
    plot_filter_metric_heatmap(
        df, GROUP_CLASS_REAL,
        os.path.join(output_path, 'heatmap_filter_metric_real.png'),
    )
    plot_filter_metric_heatmap(
        df, GROUP_CLASS_SIM,
        os.path.join(output_path, 'heatmap_filter_metric_simulated.png'),
    )

    # Scatter — retained density x Jaccard, companion to the Jaccard line plots.
    plot_density_vs_jaccard_scatter(
        df, GROUP_CLASS_REAL,
        os.path.join(output_path, 'scatter_density_jaccard_real.png'),
    )
    plot_density_vs_jaccard_scatter(
        df, GROUP_CLASS_SIM,
        os.path.join(output_path, 'scatter_density_jaccard_simulated.png'),
    )

    # Case study — triangles_spearman_correlation, restricted to incident_texts (NBF only,
    # NAF doesn't apply to this question -- see build_naf_consistency_table's docstring).
    triangles_table = build_case_study_table(
        df, 'triangles_spearman_correlation', GROUP_CLASS_INC,
        extra_columns={'Jaccard': 'jaccard'},
    )
    triangles_table.to_csv(os.path.join(output_path, 'case_study_triangles_incident_texts.csv'), index=False)

    # Appendix — same metric/table shape, other macro groups, to support generalization
    # without taking space from the case study above.
    real_triangles_table = build_case_study_table(
        df, 'triangles_spearman_correlation', GROUP_CLASS_REAL,
        extra_columns={'Jaccard': 'jaccard'},
    )
    real_triangles_table.to_csv(os.path.join(output_path, 'appendix_triangles_real.csv'), index=False)

    sim_triangles_table = build_case_study_table(
        df, 'triangles_spearman_correlation', GROUP_CLASS_SIM,
        extra_columns={'Jaccard': 'jaccard'},
    )
    sim_triangles_table.to_csv(os.path.join(output_path, 'appendix_triangles_simulated.csv'), index=False)

    # Summary tables — Real + Simulated together, NBF.
    ir_table = build_information_retention_table(df)
    ir_table.to_csv(os.path.join(output_path, 'table_information_retention.csv'), index=False)
    _save_table_png(ir_table, os.path.join(output_path, 'table_information_retention.png'), width=16)

    pred_table = build_predictive_metrics_table(df)
    pred_table.to_csv(os.path.join(output_path, 'table_predictive_metrics.csv'), index=False)
    _save_table_png(pred_table, os.path.join(output_path, 'table_predictive_metrics.png'), width=7)

    print("[done] filter x metric heatmaps + incident_texts case-study table + Real/Simulated appendix tables generated.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Generate benchmark visualization plots.')
    parser.add_argument('--input', required=True, help='Directory containing results CSVs (top level only)')
    parser.add_argument('--output', required=True, help="Directory to save plots (must be named 'graphics')")
    args = parser.parse_args()

    generate_graphics(input_path=args.input, output_path=args.output)
