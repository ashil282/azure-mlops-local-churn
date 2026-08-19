import pathlib


from src.features.feature_engineering import get_feature_target, load_test_data
from src.models.model_definitions import evaluate_model

MODELS_DIR = pathlib.Path("src/models")


def run_evaluate_pipeline():
    data = load_test_data()
    X_test, y_test = get_feature_target(data)

    model = {"model_type": "dummy", "trained": True}
    metrics = evaluate_model(model, X_test, y_test)
    print(f"Evaluation metrics: {metrics}")


if __name__ == "__main__":
    run_evaluate_pipeline()
