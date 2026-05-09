"""Report formatting helpers for Optical Pipeline Lab output."""

from __future__ import annotations

from collections.abc import Iterable


def format_summary_rows(rows: Iterable[dict[str, float | str]]) -> list[str]:
    """Format timing summary rows in the style used by existing examples."""
    lines: list[str] = []
    for row in rows:
        if row["count"] > 1:
            lines.append(
                f"{row['phase']}: repeat={int(row['count'])}, "
                f"p50={row['p50_ms']:.3f}, p90={row['p90_ms']:.3f}, "
                f"mean={row['mean_ms']:.3f}"
            )
        else:
            lines.append(f"{row['phase']}: {row['mean_ms']:.3f}")
    return lines
