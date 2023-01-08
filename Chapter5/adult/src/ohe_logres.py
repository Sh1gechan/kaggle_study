import pandas as pd

from sklearn import linear_model
from sklearn import metrics
from sklearn import preprocessing


def run(fold):
    
    # 学習用データセットの読み込み
    df = pd.read_csv('../input/adult_skfolds.csv')
    
    # 数値を含む列
    num_cols = [
        'fnlwgt',
        'age',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
    ]
    
    # 数値を含む列の削除
    df = df.drop(num_cols, axis=1)
    
    # 目的変数を0と1に置換
    target_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    df.loc[:, 'income'] = df.income.map(target_mapping)
    
    # 目的変数とfold番号の列を除き、特徴量とする
    features = [
        f for f in df.columns if f not in ('skfold', 'income')
    ]
    
    # すべての欠損値をNONEで補完、文字列型に変換
    for col in features:
        df.loc[:, col] = df[col].astype(str).fillna('NONE')
        
    # 引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.skfold != fold].reset_index(drop=True)
    
    # 引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.skfold == fold].reset_index(drop=True)
    
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
    model = linear_model.LogisticRegression()
    
    # モデルの学習
    model.fit(x_train, df_train.income.values)
    
    # 検証用データセットに対する予測
    # AUCを計算するために予測値が必要
    # 1である予測値を利用
    valid_preds = model.predict_proba(x_valid)[:, 1]
    
    # AUCを計算
    auc = metrics.roc_auc_score(df_valid.income.values, valid_preds)
    
    # AUCを表示
    print(f'Fold = {fold}, AUC = {auc}')


if __name__ == '__main__':
    for fold_ in range(5):
        run(fold_)
    