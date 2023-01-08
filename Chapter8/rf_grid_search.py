import pandas as pd
from sklearn import ensemble, model_selection

if __name__ == "__main__":
    df = pd.read_csv(".\input\mobile_train.csv")
    # 特徴量に price_range 以外のすべての列を使用
    X = df.drop("price_range", axis=1).values
    y = df["price_range"].values

    classifier = ensemble.RandomForestClassifier(n_jobs=-1)
    # パラメタの探索範囲
    param_grid = {
        "n_estimators": [100, 200, 250, 300, 400, 500],
        "max_depth": [1, 2, 5, 7, 11, 15],
        "criterion": ["gini", "entropy"],
    }

    # グリッドサーチ
    # 評価指標は正答率
    # verbose で大きい値だとより詳細に出力
    # cv=5 で 5分割して交差検証
    # 層化抽出 stratified k-fold ではない
    model = model_selection.GridSearchCV(
        estimator=classifier,
        param_grid=param_grid,
        scoring="accuracy",
        verbose=10,
        n_jobs=1,
        cv=5,
    )

    model.fit(X, y)
    print(f"Best score: {model.best_score_}")
    print(f"Best parameters set:")
    best_parameters = model.best_estimator_.get_params()
    for param_name in sorted(param_grid.keys()):
        print(f"\t{param_name}: {best_parameters[param_name]}")
