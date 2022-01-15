import argparse
from urllib.parse import urlparse

import mlflow
import pandas as pd
import yaml
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


def read_params(config_path):
    """
    read parameters from the params.yaml file
    input: params.yaml location
    output: parameters as dictionary
    """
    with open(config_path) as yaml_file:
        config = yaml.safe_load(yaml_file)
    return config


def accuracymeasures(y_test, predictions):
    mse = mean_squared_error(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    # target_names = ['0', '1']

    print("Accuracy Measures")
    print("---------------------", "\n")
    print("MAE: ", mae)
    print("MSE: ", mse)
    print("R2_Score: ", r2)

    return mae, mse, r2


def get_feat_and_target(df, target):
    """
    Get features and target variables separately from given dataframe and target
    input: dataframe and target column
    output: two dataframes for x and y
    """
    x = df.drop(target, axis=1)
    y = df[[target]]
    return x, y


def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config["processed_data_config"]["train_data_csv"]
    test_data_path = config["processed_data_config"]["test_data_csv"]
    target = config["raw_data_config"]["target"]
    max_depth = config["random_forest"]["max_depth"]
    n_estimators = config["random_forest"]["n_estimators"]

    train = pd.read_csv(train_data_path, sep=",")
    test = pd.read_csv(test_data_path, sep=",")
    train_x, train_y = get_feat_and_target(train, target)
    test_x, test_y = get_feat_and_target(test, target)

    ################### MLFLOW ###############################
    mlflow_config = config["mlflow_config"]
    remote_server_uri = mlflow_config["remote_server_uri"]

    mlflow.set_tracking_uri(remote_server_uri)
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name=mlflow_config["run_name"]) as mlops_run:
        model = RandomForestRegressor(max_depth=max_depth, n_estimators=n_estimators)
        model.fit(train_x, train_y)
        y_pred = model.predict(test_x)
        mae, mse, r2 = accuracymeasures(test_y, y_pred)

        mlflow.log_param("max_depth", max_depth)
        mlflow.log_param("n_estimators", n_estimators)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("mse : ", mse)
        mlflow.log_metric("r2_Score : ", r2)

        tracking_url_type_store = urlparse(mlflow.get_artifact_uri()).scheme

        if tracking_url_type_store != "file":
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=mlflow_config["registered_model_name"])
        else:
            mlflow.sklearn.load_model(model, "model")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)
