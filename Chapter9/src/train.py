import os

import pandas as pd
import numpy as np

import albumentations

import torch

from sklearn import metrics
from sklearn.model_selection import train_test_split

import dataset
import engine
from model import get_model

if __name__ == "__main__":
    # train.csvとpng形式の画像を格納したtrain_pngディレクトリを配置
    data_path = "../input/"
    # cuda/cpu device
    device = "cuda"
    # 10 epochs
    epochs = 10
    # load the dataframe
    df = pd.read_csv(os.path.join(data_path, "train.csv"))
    # 画像インデックス
    images = df.ImageId.values.tolist()
    # 画像のパスのリスト
    images = [
            os.path.join(data_path, "train_png", i + ".png") for i in images
            ]
    # 二値の目的変数のnumpy配列
    targets = df.target.values
    # モデルの取得
    # 事前学習済みの重みの有無の両社を試す
    model = get_model(pretrained=True)
    # モデルをデバイスに転送
    model.to(device)
    # imagenetの書くチャンネルの平均と標準偏差
    # 学習済みモデルの重みを使う場合、事前に計算した値を利用
    # 使わない場合、対象のデータセットで別途計算した値を使う
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    
    # albumentations は様々な画像のデータ拡張が利用できるライブラリ
    # ここでは正規化のみを利用
    # notice always_apply=Trueにして、正規化を常に適用
    aug = albumentations.Compose(
                        [
                            albumentations.Normalize(
                                mean, std, max_pixel_value=255.0, always_apply=True
                                )
                            ]
                        )
    
    # kfoldの代わりにホールドアウト検証
    # 分割の乱数を固定
    train_images, valid_images, train_targets, valid_targets = train_test_split(
                                images, targets, stratify=targets, random_state=42
                                )
                                
    # ClassificationDataset class
    train_dataset = dataset.ClassificationDataset(
                                image_paths=train_images,
                                203
                                targets=train_targets,
                                resize=(227, 227),
                                augmentations=aug,
                                )
    
    # データローダ
    train_loader = torch.utils.data.DataLoader(
                        train_dataset, batch_size=16, shuffle=True, num_workers=4
                        )

    valid_dataset = dataset.ClassificationDataset(
                                    image_paths=valid_images,
                                    targets=valid_targets,
                                    resize=(227, 227),
                                    augmentations=aug,
                                    )
    
    valid_loader = torch.utils.data.DataLoader(
                        valid_dataset, batch_size=16, shuffle=False, num_workers=4
                        )
    
    # Adam optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    
    # すべてのエポックについて学習し、AUCを出力
    for epoch in range(epochs):
        engine.train(train_loader, model, optimizer, device=device)
        predictions, valid_targets = engine.evaluate(
                                            valid_loader, model, device=device
                                            )
        
        roc_auc = metrics.roc_auc_score(valid_targets, predictions)
        
        print(
                f"Epoch={epoch}, Valid ROC AUC={roc_auc}"
                )
        
        