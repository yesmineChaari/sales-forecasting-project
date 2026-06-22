"""Run real-data Prophet daily-total diagnostics for Rossmann sales."""

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIR = PROJECT_ROOT / "include"
sys.path.insert(0, str(INCLUDE_DIR))

from ml_models.prophet_experiments import (  # noqa: E402
    run_prophet_experiments,
    select_best_variant,
)
from utils.rossmann_loader import RossmannDataLoader  # noqa: E402


REGRESSOR_GROUP_VARIANTS = {
    "promo": "prophet_promo_only",
    "open_store": "prophet_open_store_only",
    "holiday": "prophet_holiday_only",
}


def _load_config(config_path: Path) -> Dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def _metric_for(
    metrics_df: pd.DataFrame,
    variant: str,
    split: str,
    metric: str = "rmse",
) -> float:
    rows = metrics_df[
        (metrics_df["variant"] == variant) & (metrics_df["split"] == split)
    ]
    if rows.empty:
        return float("nan")
    return float(rows.iloc[0][metric])


def _bool_text(value: bool) -> str:
    return "yes" if value else "no"


def _format_float(value: Optional[float]) -> str:
    if value is None or pd.isna(value):
        return "n/a"
    return f"{float(value):,.4f}"


def _build_prophet_params(config: Dict[str, Any]) -> Dict[str, Any]:
    prophet_params = dict(config.get("models", {}).get("prophet", {}).get("params", {}))
    prophet_params.update(
        {
            "mcmc_samples": 0,
            "uncertainty_samples": 0,
        }
    )
    return prophet_params


def _load_full_row_rossmann_data(train_path: Path, store_path: Path) -> pd.DataFrame:
    loader = RossmannDataLoader(
        train_path=str(train_path),
        store_path=str(store_path),
    )
    return loader._load_and_prepare(include_closed_zero_sales=True)


def _prepare_prophet_splits(
    full_row_df: pd.DataFrame,
    config: Dict[str, Any],
):
    from ml_models.train_models import ModelTrainer

    trainer = object.__new__(ModelTrainer)
    trainer.training_config = config["training"]
    return trainer.prepare_prophet_data(
        full_row_df,
        target_col="sales",
        date_col="date",
    )


def _best_regressor_group(metrics_df: pd.DataFrame, split: str = "test") -> Dict[str, Any]:
    univariate_rmse = _metric_for(metrics_df, "prophet_univariate", split)
    improvements = {}

    for group_name, variant in REGRESSOR_GROUP_VARIANTS.items():
        variant_rmse = _metric_for(metrics_df, variant, split)
        if pd.isna(univariate_rmse) or pd.isna(variant_rmse):
            improvements[group_name] = {
                "variant": variant,
                "rmse": variant_rmse,
                "rmse_improvement": float("nan"),
            }
        else:
            improvements[group_name] = {
                "variant": variant,
                "rmse": variant_rmse,
                "rmse_improvement": univariate_rmse - variant_rmse,
            }

    valid_groups = {
        group_name: details
        for group_name, details in improvements.items()
        if not pd.isna(details["rmse_improvement"])
    }
    if not valid_groups:
        return {
            "group": None,
            "variant": None,
            "rmse": float("nan"),
            "rmse_improvement": float("nan"),
            "all_groups": improvements,
        }

    best_group, best_details = max(
        valid_groups.items(),
        key=lambda item: item[1]["rmse_improvement"],
    )
    return {
        "group": best_group,
        **best_details,
        "all_groups": improvements,
    }


def _prophet_usefulness(metrics_df: pd.DataFrame, verdict: Dict[str, Any]) -> str:
    best_prophet_rmse = verdict.get("best_prophet_test_rmse")
    seasonal_rmse = verdict.get("seasonal_naive_7_test_rmse")
    best_variant = verdict.get("best_variant")
    best_prophet_variant = verdict.get("best_prophet_variant")

    if (
        best_prophet_variant == best_variant
        and best_prophet_rmse is not None
        and seasonal_rmse is not None
        and not pd.isna(best_prophet_rmse)
        and not pd.isna(seasonal_rmse)
        and best_prophet_rmse < seasonal_rmse
    ):
        return "keep Prophet as a useful daily-total model"

    prophet_rows = metrics_df[
        (metrics_df["split"] == "test")
        & (metrics_df["variant"].str.startswith("prophet_"))
    ]
    if not prophet_rows.empty and bool(prophet_rows["beats_seasonal_naive_7"].any()):
        return "keep Prophet as a useful daily-total model"

    return "treat Prophet only as a weak daily-total baseline"


def _build_verdict(metrics_df: pd.DataFrame) -> Dict[str, Any]:
    verdict = dict(metrics_df.attrs.get("verdict", {}))

    best_validation_variant = select_best_variant(
        metrics_df,
        split="validation",
        metric="rmse",
    )
    best_test_variant = select_best_variant(metrics_df, split="test", metric="rmse")

    all_regressors_rmse = _metric_for(metrics_df, "prophet_all_regressors", "test")
    univariate_rmse = _metric_for(metrics_df, "prophet_univariate", "test")
    all_regressors_improve = bool(
        not pd.isna(all_regressors_rmse)
        and not pd.isna(univariate_rmse)
        and all_regressors_rmse < univariate_rmse
    )

    best_group = _best_regressor_group(metrics_df, split="test")

    verdict.update(
        {
            "best_validation_variant": best_validation_variant,
            "best_validation_rmse": _metric_for(
                metrics_df,
                best_validation_variant,
                "validation",
            )
            if best_validation_variant
            else float("nan"),
            "best_test_variant": best_test_variant,
            "best_test_rmse": _metric_for(metrics_df, best_test_variant, "test")
            if best_test_variant
            else float("nan"),
            "all_regressors_improve_over_univariate": all_regressors_improve,
            "prophet_all_regressors_test_rmse": all_regressors_rmse,
            "prophet_univariate_test_rmse": univariate_rmse,
            "best_regressor_group": best_group,
        }
    )
    verdict["recommendation"] = _prophet_usefulness(metrics_df, verdict)
    return verdict


def _verdict_lines(verdict: Dict[str, Any]) -> str:
    best_group = verdict["best_regressor_group"]
    group_name = best_group.get("group") or "none"
    group_improvement = best_group.get("rmse_improvement")

    lines = [
        "Prophet Diagnostics Verdict",
        "===========================",
        "",
        (
            "Best validation RMSE variant: "
            f"{verdict.get('best_validation_variant')} "
            f"({_format_float(verdict.get('best_validation_rmse'))})"
        ),
        (
            "Best test RMSE variant: "
            f"{verdict.get('best_test_variant')} "
            f"({_format_float(verdict.get('best_test_rmse'))})"
        ),
        (
            "Prophet beats seasonal_naive_7 on test RMSE: "
            f"{_bool_text(verdict.get('prophet_beats_seasonal_naive_7', False))}"
        ),
        (
            "All regressors improve over univariate Prophet on test RMSE: "
            f"{_bool_text(verdict.get('all_regressors_improve_over_univariate', False))}"
        ),
        (
            "Best individual regressor group: "
            f"{group_name} "
            f"(RMSE improvement vs univariate: {_format_float(group_improvement)})"
        ),
        f"Recommendation: {verdict.get('recommendation')}",
    ]
    return "\n".join(lines) + "\n"


def _print_summary_table(metrics_df: pd.DataFrame) -> None:
    summary_df = (
        metrics_df[metrics_df["split"].isin(["validation", "test"])]
        .sort_values(["split", "rmse"])
        .loc[
            :,
            [
                "split",
                "variant",
                "rmse",
                "mae",
                "mape",
                "r2",
                "beats_seasonal_naive_7",
            ],
        ]
    )

    print("\nProphet diagnostic summary")
    print(summary_df.to_string(index=False, float_format=lambda value: f"{value:,.4f}"))


def _json_safe(value):
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if pd.isna(value) else float(value)
    return value


def _write_outputs(
    metrics_df: pd.DataFrame,
    verdict: Dict[str, Any],
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / "prophet_experiment_results.csv"
    json_path = output_dir / "prophet_experiment_results.json"
    verdict_path = output_dir / "prophet_verdict.txt"

    metrics_df.to_csv(csv_path, index=False)

    payload = {
        "results": metrics_df.to_dict(orient="records"),
        "verdict": _json_safe(verdict),
    }
    with open(json_path, "w", encoding="utf-8") as json_file:
        json.dump(payload, json_file, indent=2)

    with open(verdict_path, "w", encoding="utf-8") as verdict_file:
        verdict_file.write(_verdict_lines(verdict))

    print(f"\nSaved CSV: {csv_path}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved verdict: {verdict_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real-data Prophet diagnostics on Rossmann daily totals.",
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PROJECT_ROOT / "include" / "data" / "rossmann" / "train.csv",
    )
    parser.add_argument(
        "--store-path",
        type=Path,
        default=PROJECT_ROOT / "include" / "data" / "rossmann" / "store.csv",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=PROJECT_ROOT / "include" / "config" / "ml_config.yaml",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "reports" / "prophet_diagnostics",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config_path)

    print("Loading full-row Rossmann data for Prophet diagnostics...")
    full_row_df = _load_full_row_rossmann_data(args.train_path, args.store_path)
    print(
        "Loaded rows: "
        f"{len(full_row_df):,}; dates: "
        f"{full_row_df['date'].min().date()} to {full_row_df['date'].max().date()}"
    )
    print(
        "Closed/zero-sales rows preserved: "
        f"{len(full_row_df[(full_row_df['is_open'] == 0) | (full_row_df['sales'] == 0)]):,}"
    )

    train_df, val_df, test_df = _prepare_prophet_splits(full_row_df, config)
    print(
        "Prophet split rows: "
        f"train={len(train_df):,}, validation={len(val_df):,}, test={len(test_df):,}"
    )

    print("Running Prophet daily-total experiments...")
    metrics_df = run_prophet_experiments(
        train_df,
        val_df,
        test_df,
        prophet_params=_build_prophet_params(config),
    )
    verdict = _build_verdict(metrics_df)
    metrics_df.attrs["verdict"] = verdict

    _print_summary_table(metrics_df)
    print()
    print(_verdict_lines(verdict))
    _write_outputs(metrics_df, verdict, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
