import numpy as np


class EnsembleModel:
    def __init__(self, models, weights):
        self.models = models
        self.weights = weights

    def predict(self, X):
        weighted_predictions = []

        for model_name, model in self.models.items():
            if not hasattr(model, "predict"):
                continue

            weight = self.weights.get(model_name, 0)
            prediction = np.asarray(model.predict(X), dtype=float)
            weighted_predictions.append(weight * prediction)

        if not weighted_predictions:
            raise ValueError("No ensemble models can produce predictions")

        return np.sum(weighted_predictions, axis=0)
