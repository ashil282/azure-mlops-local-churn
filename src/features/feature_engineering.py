def load_train_data():
    return [[1.0, 2.0], [3.0, 4.0]], [0, 1]

def load_test_data():
    return [[1.5, 2.5]], [0]

def get_feature_target(data):
    features, target = data
    return features, target
