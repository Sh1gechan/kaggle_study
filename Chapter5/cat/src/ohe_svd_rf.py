import pandas as pd

from scipy import sparse
from sklearn import decomposition
from sklearn import ensemble
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    
    # 学習用データセットの読み込み
    df = pd.read_csv('../input/cat_train_folds.csv')
    
    # インデックスと目的変数との列を除き、特徴量とする
    features = [
        f for f in df.columns if f not in ('id', 'target', 'kfold')        
    ]
    
    # すべての欠損値をNONEで補完、文字列型に変換
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
    
    # 引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # 引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # 初期化
    ohe = preprocessing.OneHotEncoder()
    
    # 学習用と検証用のデータセットを結合し、One Hotエンコーダを学習
    full_data = pd.concat(
        [df_train[features], df_valid[features]],
        axis=0
    )
    ohe.fit(full_data[features])
    
    # 学習用データセットを変換
    x_train = ohe.transform(df_train[features])
    
    # 検証用データセットを変換
    x_valid = ohe.transform(df_valid[features])
    
    # 初期化
    # 120次元に圧縮
    svd = decomposition.TruncatedSVD(n_components=120)
    
    # 学習用と検証用のデータセットを結合し、学習
    full_sparse = sparse.vstack((x_train, x_valid))
    
    # 初期化
    model = ensemble.RandomForestClassifier(n_jobs=-1)
    
    # モデルの学習
    model.fit(x_train, df_train.target.values)
    
    # 検証用データセットに対する予測
    # AUCを計算するために予測値が必要
    # 1である予測値を利用
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # AUCを計算
    auc = metrics.roc_auc_score(df_valid.target.values, valid_preds)
    
    # AUCを表示
    print(f'Fold = {fold}, AUC = {auc}')


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)