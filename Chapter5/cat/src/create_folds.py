# ライブラリの読み込み
import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':

    # 学習データセットの読み込み
    df = pd.read_csv('../input/train.csv')

    # kfoldという新しい列を作り、-1で初期化
    df['kfold'] = -1

    # サンプルをシャッフル
    df = df.sample(frac=1).reset_index(drop=True)

    # 目的変数の取り出し
    y = df.target.values

    # 初期化
    kf = model_selection.StratifiedKFold(n_splits=5)

    # kfold列を埋める
    for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
        df.loc[v_, 'kfold'] = f

    # データセットを新しい列と共に保存
    df.to_csv('../input/cat_train_folds.csv', index=False)