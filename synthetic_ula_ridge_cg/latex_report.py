from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from .methods import METHOD_LABELS
from .public_artifact import (
    DISPLAY_METRICS,
    K_VALUES,
    METHODS,
    selection_slopes,
    theory_diagnostics,
)


def _format_pair(mean: float, half_width: float, metric: str) -> str:
    if metric == "hyperobjective_gap" and mean < 0.01:
        return (
            f"$({mean * 1000:.2f}\\mathbin{{\\pm}}"
            f"{half_width * 1000:.2f})\\!\\times\\!10^{{-3}}$"
        )
    if metric == "hyperobjective_gap" and mean >= 1.0:
        return f"${mean:.2f}\\mathbin{{\\pm}}{half_width:.2f}$"
    return f"${mean:.3f}\\mathbin{{\\pm}}{half_width:.3f}$"


def render_end_to_end_table(
    final_summary: Sequence[Mapping[str, Any]],
) -> str:
    indexed = {
        (int(row["k"]), row["method"]): row for row in final_summary
    }
    lines = [
        r"\begin{table*}[t]",
        r"  \centering",
        r"  \caption{\textbf{Fixed-wall-clock synthetic comparison.} "
        r"Each method receives 60 single-thread CPU seconds for each of five "
        r"held-out seeds. Entries are mean $\pm$ symmetric bootstrap 95\% CI "
        r"half-width (5,000 resamples): objective gap / stationarity / "
        r"optimistic-solution distance. Lower is better.}",
        r"  \label{tab:synthetic-end-to-end}",
        r"  \resizebox{\textwidth}{!}{%",
        r"  \begin{tabular}{lcccc}",
        r"    \toprule",
        r"    Method & $k=1$ & $k=2$ & $k=4$ & $k=10$ \\",
        r"    \midrule",
    ]
    for method in METHODS:
        cells = []
        for k_value in K_VALUES:
            row = indexed[(k_value, method)]
            cells.append(
                " / ".join(
                    _format_pair(
                        float(row[f"{metric}_mean"]),
                        float(row[f"{metric}_ci95_half_width"]),
                        metric,
                    )
                    for metric in DISPLAY_METRICS
                )
            )
        label = METHOD_LABELS[method]
        if method == "hg_ms":
            label = rf"\textbf{{{label}}}"
            cells = [rf"\textbf{{{cell}}}" for cell in cells]
        lines.append(f"    {label} & " + " & ".join(cells) + r" \\")
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}%",
            r"  }",
            r"\end{table*}",
            "",
        ]
    )
    return "\n".join(lines)


def render_selection_table(
    selection: Sequence[Mapping[str, Any]],
) -> str:
    slopes = selection_slopes(selection)
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{Best-of-$N$ selection scaling.} Slopes are "
        r"least-squares log--log slopes over the six largest tested values "
        r"of $N$.}",
        r"  \label{tab:synthetic-selection-scaling}",
        r"  \begin{tabular}{rrrrr}",
        r"    \toprule",
        r"    $k$ & Observed & Quadrature & $-2/k$ & $-1/k$ \\",
        r"    \midrule",
    ]
    for k_value in K_VALUES:
        row = slopes[k_value]
        lines.append(
            f"    {k_value} & {row['observed']:.3f} & "
            f"{row['quadrature']:.3f} & "
            f"{row['sharp_asymptotic']:.3f} & "
            f"{row['theorem_envelope']:.3f} \\\\"
        )
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def render_mechanism_table(
    selection: Sequence[Mapping[str, Any]],
    ridge: Sequence[Mapping[str, Any]],
    ula: Sequence[Mapping[str, Any]],
) -> str:
    diagnostics = theory_diagnostics(selection, ridge, ula)
    ks = diagnostics["marginal_ks"]
    rows = [
        (
            "Pseudoinverse formula error (maximum)",
            f"{diagnostics['pseudoinverse_error_max']:.2e}",
        ),
        (
            "Ridge-bias log--log slope (prediction: 1)",
            f"{diagnostics['ridge_bias_slope']:.3f}",
        ),
        (
            r"CG solution error at $\eta=10^{-8}$ (mean)",
            f"{diagnostics['cg_solution_error_mean_at_1e-8']:.2e}",
        ),
        (
            "ULA-selected hyper-gradient MSE",
            f"{diagnostics['ula_hypergradient_mse']:.3f}",
        ),
        (
            "Exact-Gibbs-selected hyper-gradient MSE",
            f"{diagnostics['exact_gibbs_hypergradient_mse']:.3f}",
        ),
        (
            "Relative ULA/Gibbs MSE difference",
            f"{100 * diagnostics['relative_mse_difference']:.1f}\\%",
        ),
        (
            "Radial / angular / normal marginal KS",
            f"{ks['active_radial']:.3f} / {ks['angular']:.3f} / "
            f"{ks['normal']:.3f}",
        ),
    ]
    lines = [
        r"\begin{table}[t]",
        r"  \centering",
        r"  \caption{\textbf{ULA/Gibbs and ridge-CG mechanism diagnostics.}}",
        r"  \label{tab:synthetic-mechanisms}",
        r"  \begin{tabular}{lr}",
        r"    \toprule",
        r"    Diagnostic & Value \\",
        r"    \midrule",
    ]
    lines.extend(f"    {label} & {value} \\\\" for label, value in rows)
    lines.extend(
        [
            r"    \bottomrule",
            r"  \end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    return "\n".join(lines)


def write_latex_tables(
    output_directory: Path,
    final_summary: Sequence[Mapping[str, Any]],
    selection: Sequence[Mapping[str, Any]],
    ridge: Sequence[Mapping[str, Any]],
    ula: Sequence[Mapping[str, Any]],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    (output_directory / "end_to_end_table.tex").write_text(
        render_end_to_end_table(final_summary),
        encoding="utf-8",
    )
    (output_directory / "selection_scaling_table.tex").write_text(
        render_selection_table(selection),
        encoding="utf-8",
    )
    (output_directory / "mechanism_diagnostics_table.tex").write_text(
        render_mechanism_table(selection, ridge, ula),
        encoding="utf-8",
    )
