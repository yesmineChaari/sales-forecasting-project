import os


class ModelVisualizer:
    def create_comprehensive_report(
        self,
        metrics_dict,
        predictions_dict,
        actual_data,
        feature_importance_dict=None,
        save_dir="/tmp",
    ):
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        os.makedirs(save_dir, exist_ok=True)
        saved_files = {}

        if metrics_dict:
            metric_names = sorted({key for metrics in metrics_dict.values() for key in metrics})
            model_names = list(metrics_dict)

            fig, ax = plt.subplots(figsize=(10, 6))
            for metric_name in metric_names:
                values = [metrics_dict[model].get(metric_name, 0) for model in model_names]
                ax.plot(model_names, values, marker="o", label=metric_name)
            ax.set_title("Model Metrics")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = os.path.join(save_dir, "metrics_comparison.png")
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            saved_files["metrics_comparison"] = path

        if predictions_dict and "sales" in actual_data.columns:
            fig, ax = plt.subplots(figsize=(12, 6))
            actual = actual_data["sales"].reset_index(drop=True)
            ax.plot(actual.index, actual.values, label="actual", linewidth=1)
            for model_name, pred_df in predictions_dict.items():
                ax.plot(pred_df["prediction"].reset_index(drop=True), label=model_name, linewidth=1)
            ax.set_title("Predictions Comparison")
            ax.legend()
            ax.grid(True, alpha=0.3)
            path = os.path.join(save_dir, "predictions_comparison.png")
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            saved_files["predictions_comparison"] = path

        if feature_importance_dict:
            first_name, first_importance = next(iter(feature_importance_dict.items()))
            fig, ax = plt.subplots(figsize=(10, 6))
            top_features = first_importance.head(20)
            ax.barh(top_features["feature"], top_features["importance"])
            ax.set_title(f"Feature Importance: {first_name}")
            path = os.path.join(save_dir, "feature_importance.png")
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            saved_files["feature_importance"] = path

        return saved_files
