from functools import partial

import numpy as np
import pandas as pd
from sklearn import ensemble, metrics, model_selection
from skopt import gp_minimize, space


def optimize(params, param_names, x, y):
    """最適化関数

    探索範囲、特徴量、目的変数を受け取る
    選ばれたパラメタでモデルを学習し、交差検証し、正答率にマイナスをかけた値を算出
    正答率に -1 をかけることで最小化の問題に変換している

    :param params: gp_minimize 用のパラメタ
    :param param_names:
    :param x: 特徴量
    :param y: 目的変数
    :return: 正答率にマイナスをかけた値
    """
    # パラメタを辞書に
    params = dict(zip(param_names, params))
    model = ensemble.RandomForestClassifier(**params)
    kf = model_selection.StratifiedKFold(n_splits=5)
    accuracies = []

    # 各分割のループ
    for idx in kf.split(X=x, y=y):
        train_idx, test_idx = idx[0], idx[1]
        xtrain = x[train_idx]
        ytrain = y[train_idx]
        xtest = x[test_idx]
        ytest = y[test_idx]

        model.fit(xtrain, ytrain)
        preds = model.predict(xtest)
        fold_acc = metrics.accuracy_score(ytest, preds)
        accuracies.append(fold_acc)

    return -1 * np.mean(accuracies)


if __name__ == "__main__":
    df = pd.read_csv(".\input\mobile_train.csv")
    # 特徴量に price_range 以外のすべての列を使用
    X = df.drop("price_range", axis=1).values
    y = df["price_range"].values

    # 探索範囲の定義
    param_space = [
        space.Integer(3, 15, name="max_depth"),
        space.Integer(100, 1500, name="n_estimators"),
        space.Categorical(["gini", "entropy"], name="criterion"),
        # 分布を指定した real 型
        space.Real(0.01, 1, prior="uniform", name="max_features"),
    ]

    param_names = ["max_depth", "n_estimators", "criterion", "max_features"]

    # params 以外の引数を埋めた関数を作成
    optimization_function = partial(optimize, param_names=param_names, x=X, y=y)

    # gp_minimize では関数の最小化のためにベイズ最適化を使う
    # 参考: https://www.slideshare.net/hoxo_m/ss-77421091
    # パラメタの探索範囲、最小化する関数、反復回数が必要
    result = gp_minimize(
        optimization_function,
        dimensions=param_space,
        n_calls=15,
        n_random_starts=10,
        verbose=10,
    )
    # 最良のパラメタ
    best_params = dict(zip(param_names, result.x))
    print(best_params)
