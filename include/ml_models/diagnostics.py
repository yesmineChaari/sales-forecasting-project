import numpy as np


def diagnose_model_performance(train_df, val_df, test_df, test_predictions, target_col="sales"):
    recommendations = []

    if train_df.empty or val_df.empty or test_df.empty:
        recommendations.append("One or more train/validation/test splits are empty.")

    if target_col in test_df.columns:
        actual = test_df[target_col].to_numpy()
        if np.any(actual == 0):
            recommendations.append("Test target contains zeros; MAPE can be unstable.")

    for model_name, predictions in test_predictions.items():
        if predictions is None:
            continue
        predictions = np.asarray(predictions)
        if not np.all(np.isfinite(predictions)):
            recommendations.append(f"{model_name} produced non-finite predictions.")

    return {"recommendations": recommendations}
