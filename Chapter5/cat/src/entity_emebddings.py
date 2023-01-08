import os
import gc
import joblib
import pandas as pd
import numpy as np
from sklearn import metrics, preprocessing
from tensorflow.keras import layers, optimizers, callbacks, utils
from tensorflow.keras.models import Model, load_model
from tensorflow.keras import backend as K


def create_model(data, catcols):
    """
    エンティティエンベッティング用のtf.kerasモデルを返す関数
    :param data: pandasデータフレーム
    :param catcols: 質的変数の列のリスト
    :return: tf.kerasモデル
    """
    
    # 入力用リストの初期化
    inputs = []
    
    # 出力用のリストの初期化
    outputs = []
    
    # 質的変数についてのループ
    for c in catcols:
        # 列内のカテゴリ数
        num_unique_values = int(data[c].nunique())
        # 埋め込みの次元数の計算
        # カテゴリ数の半分か、50の小さい方を次元数として採用
        # 大抵50は大きすぎるがカテゴリ数が十分に大きい場合は、ある程度の次元数が必要になる
        embed_dim = int(min(np.ceil((num_unique_values)/2), 50))
        
        # kerasのサイズ1の入力層
        inp = layers.Input(shape=(1,))
        
        # 埋め込み層
        # 入力のサイズは常に入力のカテゴリ数+1
        out = layers.Embedding(num_unique_values, embed_dim, name=c)(inp)
        
        # 1-d spatial dropoutは埋め込み層でよく使われる
        out = layers.SpatialDropout1D(0.3)(out)
        
        # 出力のために変形
        out = layers.Reshape(target_shape=(embed_dim, ))(out)
        
        # 入力をリストに格納
        inputs.append(inp)
        
        # 出力をリストに格納
        outputs.append(out)
        
    # リストを結合
    x = layers.Concatenate()(outputs)
    
    # batchnorm層の追加
    # ここからは自由に構造を決められる
    # 量的変数を含む場合には、ここで結合すると良い
    x = layers.BatchNormalization()(x)
    
    # ドロップアウト付きの全結合層を何層か重ねる
    # 1層か2層辺りから始めるのが良い
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    x = layers.Dense(300, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)
    
    # ソフトマックス関数を追加し、二値分類問題を解く
    # シグモイド関数を追加し、出力を1次元にする選択肢もある
    y = layers.Dense(2, activation='softmax')(x)
    
    # 最終的なモデル
    model = Model(inputs=inputs, outputs=y)
    
    # モデルの作成
    # オプティマイザはAdam, 損失は二値交差エントロピー（binary cross entropy）
    # 自由に切り替えて、挙動の違いを確認してほしい
    model.compile(loss='binary_crossentropy', optimizer='adam')
    return model


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
    
    for feat in features:
        lbl_enc = preprocessing.LabelEncoder()
        df.loc[:, feat] = lbl_enc.fit_transform(df[feat].values)
    
    # 引数のfold番号と一致しないデータを学習に利用
    df_train = df[df.kfold != fold].reset_index(drop=True)
    
    # 引数のfold番号と一致するデータを検証に利用
    df_valid = df[df.kfold == fold].reset_index(drop=True)
    
    # tf.kerasモデルの作成
    model = create_model(df, features)
    
    # 学習用と検証用データセットの準備
    xtrain = [
        df_train[features].values[:, k] for k in range(len(features))
    ]
    
    xvalid = [
        df_valid[features].values[:, k] for k in range(len(features))
    ]
    
    # 目的変数の取り出し
    ytrain = df_train.target.values
    yvalid = df_valid.target.values
    
    # 目的変数の二値化
    ytrain_cat = utils.to_categorical(ytrain)
    yvalid_cat = utils.to_categorical(yvalid)
    
    # モデルの学習
    model.fit(xtrain,
              ytrain_cat,
              validation_data=(xvalid, yvalid_cat),
              verbose=1,
              batch_size=1024,
              epochs=3
              )
    
    # 検証用データセットに対する予測
    # 1である予測値を利用
    valid_preds = model.predict(xvalid)[:, 1]
    
    # AUCを計算
    print(metrics.roc_auc_score(yvalid, valid_preds))
    
    # GPUメモリを開放するためセッションを終了
    K.clear_session()
    
if __name__ == '__main__':
    run(0)
    run(1)
    run(2)
    run(3)
    run(4)