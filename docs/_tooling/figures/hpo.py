"""Optuna study figures: a real 70-trial valid-horizon study on Lorenz."""

import torch

from _style import PAPER, lorenz_series, plt, save


def main() -> None:
    import optuna
    import resdag as rd
    from optuna.importance import PedAnovaImportanceEvaluator
    from optuna.visualization.matplotlib import (
        plot_contour, plot_optimization_history, plot_param_importances, plot_slice)
    from resdag.training import ESNTrainer

    optuna.logging.set_verbosity(optuna.logging.WARNING)

    data = lorenz_series(300 + 2500 + 400 + 1)
    data = (data - data.mean(dim=1, keepdim=True)) / data.std(dim=1, keepdim=True)
    warmup, train, target, f_warmup, val = rd.utils.prepare_esn_data(
        data, warmup_steps=300, train_steps=2500, val_steps=400)

    def objective(trial):
        torch.manual_seed(trial.number)
        model = rd.models.ott_esn(
            reservoir_size=trial.suggest_int("reservoir_size", 100, 700, step=50),
            feedback_size=3, output_size=3,
            spectral_radius=trial.suggest_float("spectral_radius", 0.5, 1.4),
            leak_rate=trial.suggest_float("leak_rate", 0.3, 1.0),
            readout_alpha=trial.suggest_float("alpha", 1e-9, 1e-3, log=True),
        )
        ESNTrainer(model).fit((warmup,), (train,), targets={"output": target})
        pred = model.forecast(f_warmup, horizon=400)
        err = (pred - val).norm(dim=-1).squeeze(0)
        vh = int((err > 0.9).nonzero()[0]) if (err > 0.9).any() else 400
        return -vh

    sampler = optuna.samplers.TPESampler(seed=11, multivariate=True)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=70, show_progress_bar=False)
    print(f"  best valid horizon: {-study.best_value:.0f} steps, params {study.best_params}")

    def style(fig, w, h, name):
        for a in fig.axes:
            a.set_facecolor(PAPER)
        fig.set_facecolor(PAPER)
        fig.set_size_inches(w, h)
        save(fig, name)

    ax = plot_optimization_history(study)
    style(ax.figure, 8.2, 3.4, "hpo_history.png")

    ax = plot_param_importances(study, evaluator=PedAnovaImportanceEvaluator())
    style(ax.figure, 8.2, 3.4, "hpo_importances.png")

    axes = plot_slice(study)
    fig = axes[0].figure if hasattr(axes, "__len__") else axes.figure
    style(fig, 9.0, 2.8, "hpo_slice.png")

    ax = plot_contour(study, params=["spectral_radius", "leak_rate"])
    style(ax.figure, 5.4, 4.2, "hpo_contour.png")


if __name__ == "__main__":
    main()
