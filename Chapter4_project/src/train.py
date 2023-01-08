import argparse
from ast import parse
import os
from pyexpat import model

import joblib
import pandas as pd
from sklearn import metrics
from sklearn.metrics import accuracy_score

import config
import model_dispatcher


def run(fold, model):
    # 学習用データセットの読み込み
    df = pd.read_csv('../input/mnist_train_folds.csv')

    # 引数のfold番号と一致しないデータを学習に利用
    # 合わせて、indexをリセット
    df_train = df[df.kfold != fold].reset_index(drop=True)

    # 引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)

    # 目的変数の列を削除し、.valueを用いてnumpy配列に変換
    # 目的変数の列はy_trainとして利用
    x_train = df_train.drop('label', axis=1).values
    y_train = df_train.label.values

    # 検証用も同様に処理
    x_valid = df_valid.drop('label', axis=1).values
    y_valid = df_valid.label.values
    
    # model_dispatcherからモデルを取り出す
    clf = model_dispatcher.models[model]

    # モデルの学習
    clf.fit(x_train, y_train)

    # 検証用データセットに対する予測
    preds = clf.predict(x_valid)

    # 正答率を計算し、表示
    accuracy = metrics.accuracy_score(y_valid, preds)
    print(f'Fold={fold}, Accuracy={accuracy}')

    # モデルを保存
    joblib.dump(
        clf,
        os.path.join(config.MODEL_OUTPUT, f'{model}_fold{fold}.bin')
    )

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--fold', type=int)
    parser.add_argument('--model', type=str)

    args = parser.parse_args()

    run(fold=args.fold, model=args.model)