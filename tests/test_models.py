from src.features.feature_engineering import get_feature_target, load_train_data
from src.models.model_definitions import evaluate_model, train_model


def test_train_and_evaluate():
    data = load_train_data()
    X, y = get_feature_target(data)
    model = train_model(X, y)
    assert model["trained"] is True

    metrics = evaluate_model(model, X, y)
    assert "accuracy" in metrics
