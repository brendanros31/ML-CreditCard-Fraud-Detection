# Libraries
from src import data_loader, features, models, evaluate, utils
import yaml


def main():
    plots = False

    # Load dataset
    config = yaml.safe_load(open("config/config.yaml"))
    df = data_loader.load_data(config['data']['raw_path'])

    # Remove Null
    df = data_loader.clean_data(df)

    # Heatmap
    utils.heatmap(df.corr(), _annot=False, show=False)

    # Scale Amount, Time
    features.scale_features(df, "Time", "Amount")

    # Train-Test split
    X = df.drop(columns=["Class"])
    y = df["Class"]     # Target
    X_train, X_test, y_train, y_test = features.split(X,y, show=False)


    # Models: Decision Tree
    decisionTree = models.build_model("DecisionTree")
    models.train_model(decisionTree, X_train, y_train)
    evaluate.evaluate(decisionTree,y_test, X_test)
    if plots:
        evaluate.plots.ConfusionMatrix(decisionTree, X_test, y_test)
        evaluate.plots.auc(decisionTree, X_test, y_test)

    # Models: Random Forest
    randomForest = models.build_model("RandomForest")
    models.train_model(randomForest, X_train, y_train)
    evaluate.evaluate(randomForest,y_test, X_test)
    if plots:
        evaluate.plots.ConfusionMatrix(randomForest, X_test, y_test)
        evaluate.plots.auc(randomForest, X_test, y_test)

    # Models: XGBoost
    xgb = models.build_model("XGBoost")
    models.train_model(xgb, X_train, y_train)
    evaluate.evaluate(xgb,y_test, X_test)
    if plots:
        evaluate.plots.ConfusionMatrix(xgb, X_test, y_test)
        evaluate.plots.auc(xgb, X_test, y_test)
    print('\n')



if __name__ == "__main__":
    main()