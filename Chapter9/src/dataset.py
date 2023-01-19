import torch

import numpy as np
from PIL import Image
from PIL import ImageFile

#終了を示すビットを持たない(破損破損した)画像に対応するための処理
ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClassificationDataset:
    """
    一般的な画像分類問題のためのクラス
    二値分類・多クラス分類・多ラベル問題など
    """
    def __init___(
        self,
        image_paths,
        targets,
        resize=None,
        augumentations=None
    ):
        """
        :param image_paths: list of path to images
        :param targets: numpy array
        :param resize: tuple, e.g. (256, 256), Noneの場合は変形しない
        :param augmentations: albumentationによるデータ拡張
        """
    self.image_paths = image_paths
    self.targets = targets
    self.resize = resize
    self.augmentations = augmentations

    def __len__(self):
        """
        データセットないのサンプル数を返す
        """
        return len(self.image_paths)

    def __getitem__(self, item):
        """
        指定されたインデックスに対して、モデルの学習や評価に必要なすべての要素を返す
        """
        # PILを使って画像を開く
        image = Image.open(self.image_paths[item])
        # グレースケールをRGBに変換
        image = image.convert("RGB")
        # 目的変数の準備
        targets = self.targets[item]

        # 画像の変形
        if self.resize is not None:
            image = image.resize(
                (self.resize[1], resize[0]),
                resample=Image.BILINEAR
            )
        # numpy配列に変換
        image = np.array(image)

        # albumentationによるデータ拡張
        if self.augmentations is not None:
            augmented = self.augmentations(image=image)
            image = augmented["image"]
        
        # Pytorchで期待される形式に変換
        # (高さ, 幅, チャンネル) -> (チャンネル, 高さ, 幅)
        image = np.transpose(image, (2,0,1)).astype(np.float32)

        # 画像と目的変数のテンソルを返す
        # 型に注目
        # 回帰の場合は目的変数の型がtorch.float
        return {
        "image": torch.tensor(image, dtype=torch.float),
        "targets": torch.tensor(targets, dtype=torch.long),
        }