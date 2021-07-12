from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import mlflow


def prepare_data():
    iris = load_iris()
    data = iris.data
    labels = iris.target
    target_names = iris.target_names

    X_train, X_test, y_train, y_test = train_test_split(
        data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels
    )

    return X_train, X_test, y_train, y_test, target_names


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


if __name__ == "__main__":

    X_train, X_test, y_train, y_test, target_names = prepare_data()

    max_depth = 3
    n_estimators = 50

    model = RandomForestClassifier(max_depth=3,
                                   n_estimators=20)

    model = train_model(model, X_train, y_train)

    with mlflow.start_run():

        y_predicted = model.predict(X_test)
        y_predicted_proba = model.predict_proba(X_test)[:, 1]
        # log params
        params_dict = {'n_estimators': n_estimators,
                       'max_depth': max_depth}
        mlflow.log_params(params_dict)

        # log metrics
        accuracy = accuracy_score(y_test, y_predicted)
        #auc_score = roc_auc_score(y_test, y_predicted_proba)

        metrics_dict = {'accuracy': accuracy}
        mlflow.log_metrics(metrics_dict)

        # log model
        mlflow.sklearn.log_model(model, 'model')
        model_path = mlflow.get_artifact_uri('model')
        print(model_path)


# ------------------------------------------------------------

# iris = load_iris()
# data = iris.data
# labels = iris.target
# target_names = iris.target_names

# X_train, X_test, y_train, y_test = train_test_split(
#     data, labels, test_size=0.2, random_state=42, shuffle=True, stratify=labels)

# max_depth = 3
# n_estimators = 50

# model = RandomForestClassifier(max_depth=3, n_estimators=20)

# model.fit(X_train, y_train)

# with mlflow.start_run():

#     y_predicted = model.predict(X_test)
#     y_predicted_proba = model.predict_proba(X_test)[:, 1]
#     # log params
#     params_dict = {'n_estimators': n_estimators,
#                    'max_depth': max_depth}
#     mlflow.log_params(params_dict)

#     # log metrics
#     accuracy = accuracy_score(y_test, y_predicted)
#     #auc_score = roc_auc_score(y_test, y_predicted_proba)

#     metrics_dict = {'accuracy': accuracy}
#     mlflow.log_metrics(metrics_dict)

#     # log model
#     mlflow.sklearn.log_model(model, 'model')
#     model_path = mlflow.get_artifact_uri('model')
#     print(model_path)
