import pandas as pd
import xgboost as xgb

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
        
    # 特徴量のラベルエンコーディング
    for col in features:
        
        # 初期化
        lbl = preprocessing.LabelEncoder()
        
        # ラベルエンコーダーの学習
        lbl.fit(df[col])
        
        # データセットの変換
        df.loc[:, col] = lbl.transform(df[col])
        
    # 引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # 引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # 学習用データセットの準備
    x_train = df_train[features].values
    
    # 検証用データセットの準備
    x_valid = df_valid[features].values
    
    # 初期化
    model = xgb.XGBClassifier(
        n_job=-1,
        max_depth=7,
        n_estimators=200
    )
    
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