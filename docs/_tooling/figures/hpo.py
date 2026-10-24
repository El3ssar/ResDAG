"""Draw standard Optuna diagnostics from the recorded Lorenz validation search.

Rebuilding documentation does not run trials, integrate an ODE or fit a model.
The source trial states and parameter distributions are retained in the study.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from _style import PAPER, save
from forecast_data import ROOT, load_showcase


def recorded_study(system: str = "lorenz", showcase: dict | None = None):
    import optuna
    from optuna.distributions import CategoricalDistribution, FloatDistribution
    from optuna.trial import TrialState, create_trial

    showcase = load_showcase() if showcase is None else showcase
    path = ROOT / "docs/assets/data" / Path(showcase["source_study"]).name
    if hashlib.sha256(path.read_bytes()).hexdigest() != showcase["source_study_sha256"]:
        raise ValueError("Recorded study changed after the forecast showcase was exported")
    record = json.loads(path.read_text())
    plan = record["preregistration"]["protocol"]
    validation = record["results"]["validation"][system]
    ranges = plan["other_ranges"]
    distributions = {
        "reservoir_size": CategoricalDistribution(plan["widths"]),
        "mean_degree": CategoricalDistribution(ranges["mean_degree"]),
        "input_to_readout": CategoricalDistribution(plan["input_to_readout_choices"]),
    }
    for key in ("spectral_radius", "leak_rate"):
        distributions[key] = FloatDistribution(*ranges[key])
    for key in ("input_scaling", "bias_scaling", "alpha"):
        distributions[key] = FloatDistribution(*ranges[f"{key}_log"], log=True)
    study = optuna.create_study(direction="maximize")
    trials = sorted(validation["trials"], key=lambda trial: trial["trial"])
    for expected, trial in enumerate(trials):
        if trial["trial"] != expected:
            raise ValueError("Recorded trial numbers must be consecutive")
        status = validation["states"][str(expected)]
        if status not in ("COMPLETE", "FAIL"):
            raise ValueError(f"Unfinished trial cannot be used as a completed diagnostic: {status}")
        value = (
            trial["evaluation"]["summary"]["0.3"]["median_of_trajectory_medians"]
            if status == "COMPLETE"
            else None
        )
        study.add_trial(
            create_trial(
                state=TrialState[status],
                params=trial["params"],
                distributions={key: distributions[key] for key in trial["params"]},
                value=value,
            )
        )
    return study


def main(showcase: dict | None = None) -> None:
    from optuna.importance import PedAnovaImportanceEvaluator
    from optuna.visualization.matplotlib import (
        plot_contour,
        plot_optimization_history,
        plot_param_importances,
        plot_slice,
    )

    study = recorded_study(showcase=showcase)

    def style(fig, width, height, name):
        for axis in fig.axes:
            axis.set_facecolor(PAPER)
            axis.tick_params(labelsize=9)
            axis.xaxis.label.set_fontsize(10)
            axis.yaxis.label.set_fontsize(10)
            axis.title.set_fontsize(11)
            for text in axis.texts:
                text.set_fontsize(8)
            legend = axis.get_legend()
            if legend is not None:
                for text in legend.get_texts():
                    text.set_fontsize(8)
        fig.set_facecolor(PAPER)
        fig.set_size_inches(width, height)
        save(fig, name)

    ax = plot_optimization_history(study, target_name="Validation score")
    ax.set_title("Lorenz validation: recorded trials")
    ax.set_ylabel("Valid prefix (Lyapunov times)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=2, frameon=False)
    style(ax.figure, 8.2, 3.4, "hpo_history.png")
    ax = plot_param_importances(study, evaluator=PedAnovaImportanceEvaluator())
    style(ax.figure, 8.2, 3.4, "hpo_importances.png")
    axes = plot_slice(study, params=["spectral_radius", "leak_rate", "input_scaling", "alpha"])
    style(axes[0].figure, 11.0, 3.0, "hpo_slice.png")
    ax = plot_contour(study, params=["spectral_radius", "leak_rate"])
    ax.figure.axes[-1].set_ylabel("Valid prefix (Lyapunov times)")
    style(ax.figure, 6.0, 4.4, "hpo_contour.png")


if __name__ == "__main__":
    main()
