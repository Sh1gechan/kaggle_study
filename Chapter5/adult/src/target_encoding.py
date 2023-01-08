import copy
import pandas as pd
import xgboost as xgb
from sklearn import metrics
from sklearn import preprocessing


def mean_target_encoding(data):
    
    # データセットのコピー
    df = copy.deepcopy(data)
    
    # 数値を含む列
    num_cols = [
        'fnlwgt',
        'age',
        'capital-gain',
        'capital-loss',
        'hours-per-week',
    ]
    
    # 目的変数を0と1に置換
    target_mapping = {
        '<=50K': 0,
        '>50K': 1
    }
    
    df.loc[:, 'income'] = df.income.map(target_mapping)
    
    # 目的変数とfold番号の列を除き、特徴量とする
    features = [
        f for f in df.columns if f not in ('skfold', 'income')
        and f not in num_cols
    ]
    
    # すべての欠損値をNONEで補完、文字列型に変換
    for col in features:
        # 数値を含む列の場合、変換しない
        if col not in num_cols:
            df.loc[:, col] = df[col].astype(str).fillna('NONE')
        
    # 特徴量のラベルエンコーディング
    for col in features:
        if col not in num_cols:
            # 初期化
            lbl = preprocessing.LabelEncoder()
            
            # ラベルエンコーダーの学習
            lbl.fit(df[col])
            
            # データセットの変換
            df.loc[:, col] = lbl.transform(df[col])
    
    # 検証用データセットを格納するリスト
    encoded_dfs = []
    
    # すべての分割についてのループ
    for fold in range(5):
        # 学習用と検証用データセットの準備
        df_train = df[df.skfold != fold].reset_index(drop=True)
        df_valid = df[df.skfold == fold].reset_index(drop=True)
        # すべての特徴量についてのループ
        for column in features:
            # カテゴリごとの目的変数の平均についての辞書を作成
            mapping_dict = dict(
                df_train.groupby(column)['income'].mean()    
            )
            # 元の列名の末尾に'enc'を加えた名前で、新しい列を作成
            df_valid.loc[:, column + '_enc'] = df_valid[column].map(mapping_dict)
        
        # リストに格納
        encoded_dfs.append(df_valid)
    #統合したデータセットを返す
    encoded_df = pd.concat(encoded_dfs, axis=0)
    return encoded_df


def run(df, fold):
    
    # 引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.skfold != fold].reset_index(drop=True)
    
    # 引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.skfold == fold].reset_index(drop=True)
    
    # 目的変数とfold番号の列を除き、特徴量とする
    features = [
        f for f in df.columns if f not in ('skfold', 'income')
    ]
    
    # 学習用データセットの準備
    x_train = df_train[features].values
    
    # 検証用データセットの準備
    x_valid = df_valid[features].values
    
    # 初期化
    model = xgb.XGBClassifier(
        n_job=-1,
        max_depth=7,
    )
    
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
    # 学習用データセットの読み込み
    df = pd.read_csv('../input/adult_skfolds.csv')
    
    # mean targetエンコーディングの実行
    df = mean_target_encoding(df)
    
    # 各分割で実行
    for fold_ in range(5):
        run(df, fold_)