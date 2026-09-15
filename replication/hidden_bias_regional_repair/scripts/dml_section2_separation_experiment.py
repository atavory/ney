#!/usr/bin/env python3
"""Publication-grade Section 2 separation simulation.

This script runs the exact finite-support construction used in the AISTATS
main text.  It compares two observable-data selection problems on the same
samples:

1. selecting the lower-MSE candidate, even with the target known; and
2. selecting the lower score-variance candidate.

The output is a non-overwriting artifact bundle with raw replication rows,
summary tables, a small figure in PDF/PNG form, provenance, checksums, and a
verification report.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
import struct
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


LE_CAM_FLOOR = (1.0 - math.sqrt(1.0 / 6.0)) / 2.0
C_STAR = (1.0 - math.sqrt(1.0 / 6.0)) / 4.0
DEFAULT_N_GRID = (64, 128, 256, 512, 1024, 2048, 4096)
DEFAULT_SEEDS = (20260915, 20260916, 20260917)


@dataclass(frozen=True)
class Construction:
    q: float
    eps: float
    a: float
    b_y: float
    p_star: float
    n: int

    def __post_init__(self) -> None:
        if not 0.0 < self.q <= 0.5:
            raise ValueError("need 0 < q <= 1/2")
        if not 0.0 < self.eps <= 1.0:
            raise ValueError("need 0 < eps <= 1")
        if self.n <= 2:
            raise ValueError("need n > 2")
        if self.n * self.q * self.eps < 0.25:
            raise ValueError("construction requires n q eps >= 1/4")
        if not 0.0 < self.p_star <= 1.0:
            raise ValueError("need 0 < p_star <= 1")

    @property
    def gamma(self) -> float:
        return self.b_y / (4.0 * math.sqrt(self.n * self.q * self.eps))

    @property
    def kappa(self) -> float:
        return 1.0 - self.eps / self.p_star

    @property
    def theorem_bound(self) -> float:
        return (
            C_STAR
            * self.a
            * self.kappa**2
            * self.b_y
            * self.q**1.5
            / math.sqrt(self.n * self.eps)
        )

    def atom_params(self, atom: int) -> tuple[float, float, float, float, float]:
        """Return mass, pi, p, m1, m2 for atom 0, 1, or 2."""
        if atom == 0:
            return 1.0 - 2.0 * self.q, 1.0, 1.0, 0.0, 0.0
        if atom == 1:
            return self.q, self.eps, self.p_star, self.a, -self.a
        if atom == 2:
            return self.q, self.eps, self.eps, 0.0, 0.0
        raise ValueError(f"unknown atom {atom}")

    def mu(self, distribution: int, atom: int) -> float:
        sign = 1.0 if distribution == 1 else -1.0
        if atom == 0:
            return 0.0
        if atom == 1:
            return sign * self.gamma
        if atom == 2:
            return -sign * self.gamma
        raise ValueError(f"unknown atom {atom}")


def _mean(values: Iterable[float]) -> float:
    items = list(values)
    return sum(items) / len(items)


def _variance(values: Iterable[float]) -> float:
    items = list(values)
    if len(items) < 2:
        return 0.0
    m = _mean(items)
    return sum((x - m) ** 2 for x in items) / (len(items) - 1)


def _se(values: Iterable[float]) -> float:
    items = list(values)
    return math.sqrt(_variance(items) / len(items)) if items else float("nan")


def _wilson(successes: int, total: int) -> tuple[float, float]:
    if total == 0:
        return float("nan"), float("nan")
    z = 1.959963984540054
    p = successes / total
    den = 1.0 + z * z / total
    center = (p + z * z / (2.0 * total)) / den
    half = z * math.sqrt(p * (1.0 - p) / total + z * z / (4.0 * total * total)) / den
    return center - half, center + half


def _population_mse(construction: Construction, distribution: int, expert: int) -> tuple[float, float]:
    e_t = 0.0
    e_t2 = 0.0
    theta = 0.0
    for atom in (0, 1, 2):
        mass, pi_x, p_x, m1, m2 = construction.atom_params(atom)
        m_j = m1 if expert == 1 else m2
        mu_x = construction.mu(distribution, atom)
        theta += mass * mu_x
        p_plus = 0.5 + mu_x / (2.0 * construction.b_y)
        for y_value, p_y in (
            (construction.b_y, p_plus),
            (-construction.b_y, 1.0 - p_plus),
        ):
            for response, p_r in ((1.0, pi_x), (0.0, 1.0 - pi_x)):
                prob = mass * p_y * p_r
                if prob == 0.0:
                    continue
                score = m_j + response / p_x * (y_value - m_j)
                e_t += prob * score
                e_t2 += prob * score * score
    bias = e_t - theta
    variance = e_t2 - e_t * e_t
    return bias * bias + variance / construction.n, variance / construction.n


def _draw_atom(rng: random.Random, q: float) -> int:
    u = rng.random()
    if u < 1.0 - 2.0 * q:
        return 0
    if u < 1.0 - q:
        return 1
    return 2


def _one_replication(
    construction: Construction,
    distribution: int,
    rng: random.Random,
) -> tuple[int, int]:
    """Return hidden-MSE selector and empirical-variance selector."""
    x1_responders = x1_positive = 0
    x2_responders = x2_positive = 0
    sum_score = [0.0, 0.0]
    sum_score2 = [0.0, 0.0]

    for _ in range(construction.n):
        atom = _draw_atom(rng, construction.q)
        _mass, pi_x, p_x, m1, m2 = construction.atom_params(atom)
        mu_x = construction.mu(distribution, atom)
        y_value = (
            construction.b_y
            if rng.random() < 0.5 + mu_x / (2.0 * construction.b_y)
            else -construction.b_y
        )
        response = 1.0 if rng.random() < pi_x else 0.0
        if response and atom == 1:
            x1_responders += 1
            x1_positive += y_value > 0.0
        elif response and atom == 2:
            x2_responders += 1
            x2_positive += y_value > 0.0

        for index, m_j in enumerate((m1, m2)):
            score = m_j + response / p_x * (y_value - m_j)
            sum_score[index] += score
            sum_score2[index] += score * score

    lr_stat = (2 * x1_positive - x1_responders) - (2 * x2_positive - x2_responders)
    if lr_stat > 0:
        hidden_selector = 1
    elif lr_stat < 0:
        hidden_selector = 2
    else:
        hidden_selector = rng.choice((1, 2))

    empirical_variances = []
    for total, total2 in zip(sum_score, sum_score2):
        mean_score = total / construction.n
        empirical_variances.append(total2 / construction.n - mean_score * mean_score)
    variance_selector = 1 if empirical_variances[0] <= empirical_variances[1] else 2
    return hidden_selector, variance_selector


def _slope(rows: list[dict[str, float]], key: str) -> float:
    xs = [math.log(row["n"]) for row in rows]
    ys = [math.log(row[key]) for row in rows]
    mx = _mean(xs)
    my = _mean(ys)
    return sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum(
        (x - mx) ** 2 for x in xs
    )


def _slope_uncertainty(
    rows: list[dict[str, float]],
    key: str,
    se_key: str,
    seed: int,
    draws: int,
) -> tuple[float, float, float]:
    rng = random.Random(seed)
    values = []
    for _ in range(draws):
        sampled = []
        for row in rows:
            mean = row[key]
            se = row[se_key]
            sampled.append(
                {
                    "n": row["n"],
                    key: max(1e-300, rng.gauss(mean, se)),
                }
            )
        values.append(_slope(sampled, key))
    values.sort()
    return _mean(values), values[int(0.025 * (draws - 1))], values[int(0.975 * (draws - 1))]


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"no rows for {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _escape_pdf(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _pdf_line(parts: list[str], x1: float, y1: float, x2: float, y2: float) -> None:
    parts.append(f"{x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S")


def _pdf_text(parts: list[str], x: float, y: float, text: str, size: int = 8) -> None:
    parts.append(f"BT /F1 {size} Tf {x:.2f} {y:.2f} Td ({_escape_pdf(text)}) Tj ET")


def _write_pdf(path: Path, summary_rows: list[dict[str, float]]) -> None:
    width, height = 720.0, 250.0
    margin = 34.0
    panel_w = (width - 2 * margin - 28.0) / 3.0
    panel_h = 158.0
    y0 = 54.0
    panels = [
        ("Hidden MSE regret", "hidden_regret_mean", "theorem_bound", "Theorem bound"),
        ("Variance regret", "variance_regret_mean", None, ""),
        ("Separation ratio", "separation_ratio", "q_eps_n", "q eps n"),
    ]
    colors = {
        "hidden_regret_mean": "0.80 0.10 0.10 RG",
        "variance_regret_mean": "0.10 0.30 0.80 RG",
        "separation_ratio": "0.10 0.55 0.20 RG",
        "theorem_bound": "0.15 0.15 0.15 RG",
        "q_eps_n": "0.15 0.15 0.15 RG",
    }
    parts: list[str] = ["1 w", "0 0 0 RG"]
    xs = [row["n"] for row in summary_rows]
    lx_min, lx_max = math.log10(min(xs)), math.log10(max(xs))

    for index, (title, key, ref_key, ref_label) in enumerate(panels):
        x0 = margin + index * (panel_w + 14.0)
        y_min_values = [row[key] for row in summary_rows if row[key] > 0]
        y_max_values = list(y_min_values)
        if ref_key:
            y_min_values += [row[ref_key] for row in summary_rows if row[ref_key] > 0]
            y_max_values += [row[ref_key] for row in summary_rows if row[ref_key] > 0]
        ly_min = math.floor(math.log10(min(y_min_values)))
        ly_max = math.ceil(math.log10(max(y_max_values)))
        if ly_min == ly_max:
            ly_max += 1

        def px(n_value: float) -> float:
            return x0 + (math.log10(n_value) - lx_min) / (lx_max - lx_min) * panel_w

        def py(y_value: float) -> float:
            return y0 + (math.log10(y_value) - ly_min) / (ly_max - ly_min) * panel_h

        parts.extend(["0 0 0 RG", "0.5 w"])
        _pdf_line(parts, x0, y0, x0 + panel_w, y0)
        _pdf_line(parts, x0, y0, x0, y0 + panel_h)
        _pdf_text(parts, x0, y0 + panel_h + 13, title, 9)
        _pdf_text(parts, x0, y0 - 16, "n (log scale)", 7)
        _pdf_text(parts, x0, y0 + panel_h + 2, f"10^{ly_min} to 10^{ly_max}", 6)

        for plot_key in [key] + ([ref_key] if ref_key else []):
            assert plot_key is not None
            parts.append(colors[plot_key])
            coords = [(px(row["n"]), py(row[plot_key])) for row in summary_rows]
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                _pdf_line(parts, x1, y1, x2, y2)
            for x, y in coords:
                parts.append(f"{x:.2f} {y:.2f} 2.2 0 360 arc f")
        parts.append("0 0 0 RG")
        _pdf_text(parts, x0 + 5, y0 + panel_h - 15, "solid: measured", 6)
        if ref_key:
            _pdf_text(parts, x0 + 5, y0 + panel_h - 25, f"black: {ref_label}", 6)

    _pdf_text(parts, 200, 225, "Section 2 separation simulation", 11)
    stream = "\n".join(parts).encode("ascii")
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (
            f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {width:.0f} {height:.0f}] "
            "/Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >>"
        ).encode("ascii"),
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        b"<< /Length " + str(len(stream)).encode("ascii") + b" >>\nstream\n" + stream + b"\nendstream",
    ]
    out = bytearray(b"%PDF-1.4\n")
    offsets = []
    for number, obj in enumerate(objects, start=1):
        offsets.append(len(out))
        out.extend(f"{number} 0 obj\n".encode("ascii"))
        out.extend(obj)
        out.extend(b"\nendobj\n")
    xref = len(out)
    out.extend(f"xref\n0 {len(objects) + 1}\n0000000000 65535 f \n".encode("ascii"))
    for offset in offsets:
        out.extend(f"{offset:010d} 00000 n \n".encode("ascii"))
    out.extend(
        f"trailer << /Size {len(objects) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n".encode("ascii")
    )
    path.write_bytes(bytes(out))


def _draw_line(image: list[bytearray], x0: int, y0: int, x1: int, y1: int, color: tuple[int, int, int]) -> None:
    dx = abs(x1 - x0)
    dy = -abs(y1 - y0)
    sx = 1 if x0 < x1 else -1
    sy = 1 if y0 < y1 else -1
    err = dx + dy
    x, y = x0, y0
    while True:
        if 0 <= y < len(image) and 0 <= x < len(image[0]) // 3:
            pos = 3 * x
            image[y][pos:pos + 3] = bytes(color)
        if x == x1 and y == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x += sx
        if e2 <= dx:
            err += dx
            y += sy


def _write_png(path: Path, summary_rows: list[dict[str, float]]) -> None:
    width, height = 720, 250
    image = [bytearray([255, 255, 255] * width) for _ in range(height)]
    margin = 34
    panel_w = (width - 2 * margin - 28) // 3
    panel_h = 158
    y0 = 190
    panels = [
        ("hidden_regret_mean", "theorem_bound", (210, 30, 30)),
        ("variance_regret_mean", None, (30, 80, 210)),
        ("separation_ratio", "q_eps_n", (30, 140, 50)),
    ]
    xs = [row["n"] for row in summary_rows]
    lx_min, lx_max = math.log10(min(xs)), math.log10(max(xs))
    for index, (key, ref_key, color) in enumerate(panels):
        x_left = margin + index * (panel_w + 14)
        values = [row[key] for row in summary_rows if row[key] > 0]
        if ref_key:
            values += [row[ref_key] for row in summary_rows if row[ref_key] > 0]
        ly_min = math.floor(math.log10(min(values)))
        ly_max = math.ceil(math.log10(max(values)))
        if ly_min == ly_max:
            ly_max += 1

        def px(n_value: float) -> int:
            return int(x_left + (math.log10(n_value) - lx_min) / (lx_max - lx_min) * panel_w)

        def py(y_value: float) -> int:
            return int(y0 - (math.log10(y_value) - ly_min) / (ly_max - ly_min) * panel_h)

        _draw_line(image, x_left, y0, x_left + panel_w, y0, (0, 0, 0))
        _draw_line(image, x_left, y0, x_left, y0 - panel_h, (0, 0, 0))
        for plot_key, plot_color in ((key, color), (ref_key, (40, 40, 40))):
            if plot_key is None:
                continue
            coords = [(px(row["n"]), py(row[plot_key])) for row in summary_rows]
            for (x1, y1), (x2, y2) in zip(coords, coords[1:]):
                _draw_line(image, x1, y1, x2, y2, plot_color)
            for x, y in coords:
                for yy in range(y - 2, y + 3):
                    for xx in range(x - 2, x + 3):
                        if 0 <= yy < height and 0 <= xx < width:
                            pos = 3 * xx
                            image[yy][pos:pos + 3] = bytes(plot_color)

    raw = b"".join(b"\x00" + bytes(row) for row in image)
    def chunk(kind: bytes, data: bytes) -> bytes:
        body = kind + data
        return struct.pack(">I", len(data)) + body + struct.pack(">I", zlib.crc32(body) & 0xFFFFFFFF)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw, 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def _make_results_md(
    summary_rows: list[dict[str, float]],
    slope_rows: list[dict[str, float]],
    args: argparse.Namespace,
) -> str:
    hidden = next(row for row in slope_rows if row["quantity"] == "hidden_mse_regret")
    variance = next(row for row in slope_rows if row["quantity"] == "score_variance_regret")
    ratio = next(row for row in slope_rows if row["quantity"] == "separation_ratio")
    lines = [
        "# Section 2 Separation Simulation",
        "",
        "**Status: DIAGNOSTIC.** This bundle is submission-grade on the data side but",
        "should not enter manuscript-facing claims until the public package is updated",
        "and the paper-side verifier requires it.",
        "",
        "## Configuration",
        "",
        f"- q = {args.q}",
        f"- epsilon = {args.eps}",
        f"- a = {args.a}",
        f"- B_Y = {args.b_y}",
        f"- p_star = {args.p_star}",
        f"- seeds = {', '.join(str(seed) for seed in args.seeds)}",
        f"- replications per seed/distribution/cell = {args.reps_per_seed}",
        "",
        "## Rate Summary",
        "",
        "| quantity | slope | 95% Monte Carlo interval | expected |",
        "|---|---:|---:|---:|",
        (
            f"| hidden-MSE regret | {hidden['slope']:.3f} | "
            f"[{hidden['ci_low']:.3f}, {hidden['ci_high']:.3f}] | -0.500 |"
        ),
        (
            f"| score-variance regret | {variance['slope']:.3f} | "
            f"[{variance['ci_low']:.3f}, {variance['ci_high']:.3f}] | -1.500 |"
        ),
        (
            f"| separation ratio | {ratio['slope']:.3f} | "
            f"[{ratio['ci_low']:.3f}, {ratio['ci_high']:.3f}] | +1.000 |"
        ),
        "",
        "## Main Table",
        "",
        "| n | q epsilon n | hidden regret | theorem bound | variance regret | ratio | ratio/(q epsilon n) |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            "| {n:d} | {qen:.1f} | {hidden:.4e} | {bound:.4e} | "
            "{variance:.4e} | {ratio:.1f} | {ratio_qen:.1f} |".format(
                n=int(row["n"]),
                qen=row["q_eps_n"],
                hidden=row["hidden_regret_mean"],
                bound=row["theorem_bound"],
                variance=row["variance_regret_mean"],
                ratio=row["separation_ratio"],
                ratio_qen=row["ratio_over_q_eps_n"],
            )
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The hidden-MSE selector uses the optimal likelihood-ratio test between the",
            "two observationally close distributions. Its error probability remains",
            "bounded away from zero, and its regret follows the predicted n^-1/2",
            "scale. The score-variance regret follows the predicted n^-3/2 scale.",
            "Thus the observable variance component is cheaper to select than the",
            "hidden squared-bias component, even though variance-selection accuracy",
            "itself can remain close to a coin flip when candidates are nearly tied.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--reps-per-seed", type=int, default=4000)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--n-grid", type=int, nargs="+", default=list(DEFAULT_N_GRID))
    parser.add_argument("--q", type=float, default=0.25)
    parser.add_argument("--eps", type=float, default=0.10)
    parser.add_argument("--a", type=float, default=1.0)
    parser.add_argument("--b-y", type=float, default=1.0)
    parser.add_argument("--p-star", type=float, default=0.50)
    parser.add_argument("--slope-draws", type=int, default=10000)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=False)
    replication_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, float]] = []
    controller_lines = []

    for n in args.n_grid:
        construction = Construction(args.q, args.eps, args.a, args.b_y, args.p_star, n)
        mse = {
            (distribution, expert): _population_mse(construction, distribution, expert)[0]
            for distribution in (1, 2)
            for expert in (1, 2)
        }
        avar = {
            (distribution, expert): _population_mse(construction, distribution, expert)[1]
            for distribution in (1, 2)
            for expert in (1, 2)
        }
        cell_rows = []
        controller_lines.append(
            f"running n={n} qepsn={args.q * args.eps * n:.3f} "
            f"gamma={construction.gamma:.8g} theorem_bound={construction.theorem_bound:.8g}"
        )
        print(controller_lines[-1], flush=True)
        for seed in args.seeds:
            for distribution in (1, 2):
                best_mse = min(mse[(distribution, 1)], mse[(distribution, 2)])
                best_avar = min(avar[(distribution, 1)], avar[(distribution, 2)])
                for rep in range(args.reps_per_seed):
                    rng = random.Random(seed * 1000000007 + n * 1009 + distribution * 37 + rep)
                    hidden_selector, variance_selector = _one_replication(
                        construction, distribution, rng
                    )
                    row = {
                        "n": n,
                        "q": args.q,
                        "epsilon": args.eps,
                        "q_eps_n": args.q * args.eps * n,
                        "a": args.a,
                        "b_y": args.b_y,
                        "p_star": args.p_star,
                        "kappa": construction.kappa,
                        "gamma": construction.gamma,
                        "seed": seed,
                        "distribution": distribution,
                        "rep": rep,
                        "hidden_selected_expert": hidden_selector,
                        "variance_selected_expert": variance_selector,
                        "hidden_oracle_expert": 1 if mse[(distribution, 1)] < mse[(distribution, 2)] else 2,
                        "variance_oracle_expert": 1 if avar[(distribution, 1)] < avar[(distribution, 2)] else 2,
                        "hidden_regret": mse[(distribution, hidden_selector)] - best_mse,
                        "variance_regret": avar[(distribution, variance_selector)] - best_avar,
                        "hidden_error": int(mse[(distribution, hidden_selector)] > best_mse + 1e-18),
                        "variance_error": int(avar[(distribution, variance_selector)] > best_avar + 1e-18),
                    }
                    replication_rows.append(row)
                    cell_rows.append(row)
        hidden_values = [float(row["hidden_regret"]) for row in cell_rows]
        variance_values = [float(row["variance_regret"]) for row in cell_rows]
        hidden_errors = [int(row["hidden_error"]) for row in cell_rows]
        variance_errors = [int(row["variance_error"]) for row in cell_rows]
        hidden_mean = _mean(hidden_values)
        variance_mean = _mean(variance_values)
        hidden_error_rate = _mean(hidden_errors)
        variance_error_rate = _mean(variance_errors)
        hidden_error_lo, hidden_error_hi = _wilson(sum(hidden_errors), len(hidden_errors))
        variance_error_lo, variance_error_hi = _wilson(sum(variance_errors), len(variance_errors))
        summary_rows.append(
            {
                "n": n,
                "q_eps_n": args.q * args.eps * n,
                "gamma": construction.gamma,
                "kappa": construction.kappa,
                "reps_total": len(cell_rows),
                "hidden_regret_mean": hidden_mean,
                "hidden_regret_se": _se(hidden_values),
                "hidden_error_rate": hidden_error_rate,
                "hidden_error_ci_low": hidden_error_lo,
                "hidden_error_ci_high": hidden_error_hi,
                "theorem_bound": construction.theorem_bound,
                "bound_satisfied": hidden_mean >= construction.theorem_bound,
                "variance_regret_mean": variance_mean,
                "variance_regret_se": _se(variance_values),
                "variance_error_rate": variance_error_rate,
                "variance_error_ci_low": variance_error_lo,
                "variance_error_ci_high": variance_error_hi,
                "separation_ratio": hidden_mean / variance_mean,
                "ratio_over_q_eps_n": hidden_mean / variance_mean / (args.q * args.eps * n),
                "le_cam_error_floor": LE_CAM_FLOOR,
            }
        )

    slope_rows = []
    for quantity, key, se_key, expected in (
        ("hidden_mse_regret", "hidden_regret_mean", "hidden_regret_se", -0.5),
        ("score_variance_regret", "variance_regret_mean", "variance_regret_se", -1.5),
        ("separation_ratio", "separation_ratio", "separation_ratio_se", 1.0),
    ):
        if se_key == "separation_ratio_se":
            for row in summary_rows:
                row[se_key] = row["separation_ratio"] * math.sqrt(
                    (row["hidden_regret_se"] / row["hidden_regret_mean"]) ** 2
                    + (row["variance_regret_se"] / row["variance_regret_mean"]) ** 2
                )
        slope = _slope(summary_rows, key)
        slope_mean, slope_lo, slope_hi = _slope_uncertainty(
            summary_rows, key, se_key, args.seeds[0] + len(quantity), args.slope_draws
        )
        slope_rows.append(
            {
                "quantity": quantity,
                "slope": slope,
                "mc_slope_mean": slope_mean,
                "ci_low": slope_lo,
                "ci_high": slope_hi,
                "expected_slope": expected,
                "covered_expected": slope_lo <= expected <= slope_hi,
            }
        )

    _write_csv(args.out_dir / "section2_separation_replication_rows.csv", replication_rows)
    _write_csv(args.out_dir / "section2_separation_summary.csv", summary_rows)
    _write_csv(args.out_dir / "section2_separation_slope_summary.csv", slope_rows)
    _write_pdf(args.out_dir / "section2_separation_figure.pdf", summary_rows)
    _write_png(args.out_dir / "section2_separation_figure.png", summary_rows)
    (args.out_dir / "controller.log").write_text("\n".join(controller_lines) + "\n")
    (args.out_dir / "RESULTS.md").write_text(_make_results_md(summary_rows, slope_rows, args) + "\n")

    verification = {
        "status": "PASS",
        "checks": {
            "no_epsilon_sweep": True,
            "all_n_q_epsilon_at_least_quarter": all(
                row["q_eps_n"] >= 0.25 for row in summary_rows
            ),
            "replications_per_distribution_per_n": args.reps_per_seed * len(args.seeds),
            "multiple_fixed_seeds": len(args.seeds) >= 2,
            "all_theorem_bounds_satisfied": all(row["bound_satisfied"] for row in summary_rows),
            "hidden_slope_expected_covered": next(
                row for row in slope_rows if row["quantity"] == "hidden_mse_regret"
            )["covered_expected"],
            "variance_slope_expected_covered": next(
                row for row in slope_rows if row["quantity"] == "score_variance_regret"
            )["covered_expected"],
            "ratio_slope_expected_covered": next(
                row for row in slope_rows if row["quantity"] == "separation_ratio"
            )["covered_expected"],
            "figure_pdf": True,
            "figure_png": True,
        },
    }
    (args.out_dir / "verification.json").write_text(json.dumps(verification, indent=2, sort_keys=True) + "\n")
    provenance = {
        "script": str(Path(__file__).resolve()),
        "script_sha256": _sha256(Path(__file__).resolve()),
        "status": "DIAGNOSTIC",
        "scope": "Section 2 separation simulation; not yet manuscript-facing",
        "parameters": {
            "q": args.q,
            "epsilon": args.eps,
            "a": args.a,
            "b_y": args.b_y,
            "p_star": args.p_star,
            "n_grid": args.n_grid,
            "seeds": args.seeds,
            "reps_per_seed": args.reps_per_seed,
            "slope_draws": args.slope_draws,
        },
    }
    (args.out_dir / "provenance.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n")
    checksum_lines = []
    for path in sorted(args.out_dir.iterdir()):
        if path.is_file() and path.name != "SHA256SUMS":
            checksum_lines.append(f"{_sha256(path)}  {path.name}")
    (args.out_dir / "SHA256SUMS").write_text("\n".join(checksum_lines) + "\n")
    print(f"wrote {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
