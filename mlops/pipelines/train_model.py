import pathlib
from datetime import datetime, timezone


from src.features.feature_engineering import get_feature_target, load_train_data
from src.models.model_definitions import log_experiment, save_model, train_model

MODELS_DIR = pathlib.Path("src/models")


def run_train_pipeline():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    data = load_train_data()
    X_train, y_train = get_feature_target(data)

    print("Training model...")
    version = datetime.now(timezone.utc).strftime("v_%Y%m%d_%H%M%S")
    model = train_model(X_train, y_train)

    model_path = save_model(model, version)
    print(f"Model saved to {model_path}")

    metrics = {"accuracy": None, "f1": None, "roc_auc": None}
    log_experiment(version, metrics, model_path)
    print("Experiment logged.")


if __name__ == "__main__":
    run_train_pipeline()
