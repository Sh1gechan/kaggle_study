from sklearn.feature_selection import (
    chi2, f_classif, f_regression, mutual_info_classif, 
    mutual_info_regression, SelectKBest, SelectPercentile
)


class UnivariateFeatureSelection:
    
    def __init__(self, n_features, problem_type, scoring):
        
        """
        scikit-learnの複数の手法に対応した
        単変量特徴量選択のためのラッパークラス
        n_features: float型の場合は SelectPercentile、それ以外のときは SelectKBestを利用
        problem_type: 分類か回帰か
        scoring: 単変量特徴量の手法名、文字列型
        """
        # 指定された問題の種類に対応している手法
        # 自由に拡張できる
        if problem_type == 'classification':
            valid_scoring = {
                'f_classif': f_classif, # ANOVA F値
                'chi2': chi2, # カイ二乗値
                'mutual_info_classif': mutual_info_classif # 相互情報量
            }
        else:
            valid_scoring = {
                'f_regression': f_regression, # 回帰分析のF値
                'mutual_info_regression': mutual_info_regression # 相互情報量
            }
        
        # 手法が対応していない場合の例外の発生
        if scoring not in valid_scoring:
            raise Exception('Invalid scoring function')
        
        """
        n_featuresがint型の場合はSelectKBest、
        float型の場合はSelectPercentileを利用
        float型の場合もint型に変換
        """
        if isinstance(n_features, int):
            self.selection = SelectKBest(
                valid_scoring[scoring],
                k=n_features
            )
        elif isinstance(n_features, float):
            self.selection = SelectPercentile(
                valid_scoring[scoring],
                percentile=int(n_features * 100)
            )
        else:
            raise Exception('Invalid type of feature')

    # fit関数
    def fit(self, X, y):
        return self.selection.fit(X, y)
    
    # transform関数
    def transform(self, X):
        return self.selection.transform(X)
    
    # fit_transform関数
    def fit_transform(self, X, y):
        return self.selection.fit_transform(X, y)