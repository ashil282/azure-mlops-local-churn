def train_model(X, y):
    return {"model_type": "dummy", "trained": True}

def evaluate_model(model, X_test, y_test):
    return {"accuracy": 0.95, "f1": 0.94}

def save_model(model, version):
    return f"src/models/model_{version}.joblib"

def log_experiment(version, metrics, model_path):
    pass
