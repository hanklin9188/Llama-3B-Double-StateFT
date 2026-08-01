#!/usr/bin/env python3
import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

from scipy.stats import spearmanr


def read_rows(path):
    if not path.is_file():
        return []
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def correlation(name, pairs):
    pairs = [(left, right) for left, right in pairs if left is not None and right is not None]
    if (
        len(pairs) < 3
        or len({pair[0] for pair in pairs}) < 2
        or len({pair[1] for pair in pairs}) < 2
    ):
        return {"analysis": name, "n": len(pairs), "spearman_rho": "", "p_value": ""}
    result = spearmanr([pair[0] for pair in pairs], [pair[1] for pair in pairs])
    return {
        "analysis": name,
        "n": len(pairs),
        "spearman_rho": float(result.statistic),
        "p_value": float(result.pvalue),
    }


def geometry_lookup(rows):
    by_exact_step = {}
    by_branch = defaultdict(list)
    for row in rows:
        key = (int(row["layer"]), row["branch"])
        step = int(row["step"])
        by_exact_step[(step, *key)] = row
        by_branch[key].append((step, row))
    for values in by_branch.values():
        values.sort(key=lambda item: item[0])
    return by_exact_step, by_branch


def nearest_geometry(step, layer, branch, exact, by_branch):
    if (step, layer, branch) in exact:
        return exact[(step, layer, branch)]
    candidates = [item for item in by_branch[(layer, branch)] if item[0] <= step]
    return candidates[-1][1] if candidates else None


def write_rows(path, rows):
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="Analyze ID-to-rank causality and rank stability")
    parser.add_argument("--adapter-dir", required=True)
    args = parser.parse_args()
    adapter = Path(args.adapter_dir).expanduser().resolve()
    metrics = adapter / "metrics"
    geometry_rows = read_rows(metrics / "branch_geometry_all.csv")
    probe_rows = read_rows(metrics / "rank_probe_all.csv")
    rank_rows = read_rows(metrics / "rank_all.csv")
    if not geometry_rows:
        raise SystemExit(f"No geometry metrics found in {metrics}")

    exact, by_branch = geometry_lookup(geometry_rows)
    joined_probes = []
    for probe in probe_rows:
        geometry = nearest_geometry(
            int(probe["step"]), int(probe["layer"]), probe["branch"], exact, by_branch
        )
        if geometry is not None:
            joined_probes.append((probe, geometry))
    latest_rank = {}
    for row in rank_rows:
        key = (int(row["layer"]), row["branch"])
        if key not in latest_rank or int(row["step"]) >= int(latest_rank[key]["step"]):
            latest_rank[key] = row
    latest_geometry = {key: values[-1][1] for key, values in by_branch.items()}

    summaries = []
    for probe_type in ("add", "remove"):
        filtered = [(probe, geo) for probe, geo in joined_probes if probe["probe_type"] == probe_type]
        summaries.append(
            correlation(
                f"input_id_lcb_vs_{probe_type}_gain",
                [(finite(geo["id_input_lcb"]), finite(probe["marginal_gain"])) for probe, geo in filtered],
            )
        )
        summaries.append(
            correlation(
                f"output_saturation_vs_{probe_type}_gain",
                [
                    (finite(geo["output_id_saturation"]), finite(probe["marginal_gain"]))
                    for probe, geo in filtered
                ],
            )
        )
    summaries.append(
        correlation(
            "input_id_lcb_vs_final_rank",
            [
                (
                    finite(geometry["id_input_lcb"]),
                    finite(latest_rank.get(key, {}).get("rank_after", geometry["active_rank"])),
                )
                for key, geometry in latest_geometry.items()
            ],
        )
    )
    write_rows(metrics / "id_rank_gain_correlation.csv", summaries)

    trajectories = defaultdict(list)
    for row in rank_rows:
        trajectories[(int(row["layer"]), row["branch"])].append(
            (int(row["step"]), int(row["rank_after"]))
        )
    stability = []
    for (layer, branch), values in sorted(trajectories.items()):
        values.sort()
        changes = [abs(values[index][1] - values[index - 1][1]) for index in range(1, len(values))]
        stability.append(
            {
                "layer": layer,
                "branch": branch,
                "events": len(values),
                "rank_changes": sum(change > 0 for change in changes),
                "total_rank_variation": sum(changes),
                "final_rank": values[-1][1],
            }
        )
    write_rows(metrics / "rank_stability.csv", stability)
    for row in summaries:
        print(
            f"{row['analysis']}: n={row['n']}, rho={row['spearman_rho']}, "
            f"p={row['p_value']}"
        )
    print(f"Correlation: {metrics / 'id_rank_gain_correlation.csv'}")
    print(f"Stability:   {metrics / 'rank_stability.csv'}")


if __name__ == "__main__":
    main()
