import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.datasets import fetch_california_housing


# 回帰問題用のデータセットの読み込み
data = fetch_california_housing()
X = data['data']
col_names = data['feature_names']
y = data['target']

# 初期化
model = LinearRegression()
# 再帰的特徴量削減用のクラスの初期化
rfe = RFE(
    estimator=model,
    n_features_to_select=3
)

# モデルの学習
rfe.fit(X, y)

# データセットの変換
X_transformed = rfe.transform(X)