import pandas as pd
from sklearn import linear_model
from sklearn import metrics
from sklearn.datasets import make_classification


class GreedyFeaturesSelection:
    """
    貪欲法による特徴量選択のクラス
    対象のデータセットに適用するためには微修正が必要
    """
    def evaluate_score(self, X, y):
        """
        モデルを学習し、AUCを計算する関数
        学習とAUCの計算に同じデータセットを使っているのに注意
        過学習しているが、貪欲法の1つの実装方法である
        交差検証とすると、分割数倍の時間がかかる
        
        もし、真に正しい方法で実装したい場合は、交差検証でAUCを計算する必要がある
        既に本書で何度か示している方法で実装できる
        
        X: 学習用データセット
        y: 目的変数
        return AUC
        """
        # ロジスティック回帰モデルを学習し、同じデータセットに対するAUCを計算
        # データセットに適したモデルに変更可能
        model = linear_model.LogisticRegression()
        model.fit(X, y)
        predictions = model.predict_proba(X)[:, 1]
        auc = metrics.roc_auc_score(y, predictions)
        return auc
    
    def _feature_selection(self, X, y):
        """
        貪欲法による特徴量選択のための関数
        X: numpy配列の特徴量
        y: numpy配列の目的変数
        return （最も良いスコア、選ばれた特徴量）
        """
        # スコアと選ばれた特徴量を格納するリストの初期化
        good_features = []
        best_scores = []
        
        # 特徴量の数
        num_features = X.shape[1]
        
        # ループの初期化
        while True:
            # 最も良い特徴量とスコアの初期化
            this_feature = None
            best_score = 0
            
            # 各特徴量についてのループ
            for feature in range(num_features):
                # 既に選ばれた特徴量のリストに含まれている場合は処理しない
                if feature in good_features:
                    continue
                # 既存のリストに新しい特徴量を追加
                selected_features = good_features + [feature]
                # 対象としない特徴量を削除
                xtrain = X[:, selected_features]
                # スコアを計算（今回はAUC）
                score = self.evaluate_score(xtrain, y)
                # これまでのスコアより良い場合は、暫定のスコアと特徴量のリストを更新
                if score > best_score:
                    this_feature = feature
                    best_score = score
            
            # スコアと特徴量をリストに追加
            if this_feature != None:
                good_features.append(this_feature)
                best_scores.append(best_score)
                
            # 直前の反復で改善しなかった場合には、ループ処理を終了
            if len(best_scores) > 2:
                if best_scores[-1] < best_scores[-2]:
                    break
        # 最も良いスコアと選ばれた特徴量を返す
        # なぜリストの最後の値を除いているか
        return best_scores[:-1], good_features[:-1]
    
    def __call__(self, X, y):
        """
        引数を与えて関数を呼び出した際の処理
        """
        # 特徴量選択
        scores, features = self._feature_selection(X, y)
        # 選ばれた特徴量とスコアを返す
        return X[:, features], scores
    
if __name__ == '__main__':
    # 二値分類用のデータセットの生成
    X, y = make_classification(n_samples=1000, n_features=100)
    
    # 貪欲法による特徴量選択
    X_transformed, scores = GreedyFeaturesSelection()(X, y)