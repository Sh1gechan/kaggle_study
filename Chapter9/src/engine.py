import torch
import torch.nn as nn

from tqdm import tqdm

def train(data_loader, model, optimizer, device):
    """
    1エポック学習する関数
    :param data_loader: pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer, for e.g. adam, sgd, etc
    :param device: cuda/cpu
    """
    # trainモード
    model.train()
    # データローダ内のバッチについてループ
    for data in data_loader:
        # 画像と目的変数
        inputs = data["image"]
        targets = data["targets"]
        
        # deviceに転送
        inputs = inputs.to(device, dtype=torch.float)
        targets = targets.to(device, dtype=torch.float)
        
        # optimizerを0で初期化
        optimizer.zero_grad()
        
        # モデルの学習
        outputs = model(inputs)
        
        # calculate loss
        loss = nn.BCEWithLogitsLoss()(outputs, targets.view(-1, 1))
        
        # backward step the loss
        loss.backward()
        
        # step optimizer
        optimizer.step()
 

def evaluate(data_loader, model, device):
    """
    1エポック評価する関数
    :param data_loader: pytorch dataloader
    :param model: pytorch model
    :param optimizer: optimizer, for e.g. adam, sgd, etc
    :param device: cuda/cpu
    """
    # evaluation mode
    model.eval()
    # 目的変数と予測を格納するリスト
    final_targets = []
    final_outputs = []
    # 勾配は計算しない
    with torch.no_grad():
        for data in data_loader:
            inputs = data["image"]
            targets = data["targets"]
            inputs = inputs.to(device, dtype=torch.float)
            targets = targets.to(device, dtype=torch.float)
            # モデルの予測
            output = model(inputs)
            # 目的変数と予測をリストに変換
            targets = targets.detach().cpu().numpy().tolist()
            output = output.detach().cpu().numpy().tolist()
            # リストに格納
            final_targets.extend(targets)
            final_outputs.extend(output)
            
    
    # return final output and final targets
    return final_outputs, final_targets