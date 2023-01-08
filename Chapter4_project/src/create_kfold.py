# ライブラリのインポート
import pandas as pd
from sklearn import model_selection


if __name__ == '__main__':
    
    df = pd.read_csv('../input/mnist_train.csv')
    
    # kfoldという新しい列を作り、-1で初期化
    df['kfold'] = -1
    
    '''
    サンプルをシャッフル
    引数frac=1とすると、すべての行数分のランダムサンプリングをすることになり、
    全体をランダムに並び替える（シャッフルする）ことに等しい。
    '''
    df = df.sample(frac=1).reset_index(drop=True)
    
    # KFoldクラスの初期化
    kf = model_selection.KFold(n_splits=5)
    
    # kfold列を埋める
    for fold, (train_, val_) in enumerate(kf.split(X=df)):
        df.loc[val_, 'kfold'] = fold
    
    # データセットを新しい列と共に保存
    df.to_csv('../input/mnist_train_folds.csv', index=False)