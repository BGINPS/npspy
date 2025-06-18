#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: trainer.py
@Description: description of this file
@Datatime: 2025/02/26 20:26:59
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

import numpy as np
import pandas as pd
from typing import Dict, Optional, Tuple, Union, Literal, List
import torch
from torch import nn
import torch.nn.functional as F
import torchvision.ops as ops
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
import random
import os
from torch.utils.data import DataLoader
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['font.sans-serif'] = "Arial"
mpl.rcParams['font.family'] = "sans-serif"
from matplotlib.patches import Rectangle

from .. import plot as pl




def set_device(
    device: Optional[Literal['cpu', 'cuda', 'mps']] = 'mps',
) -> torch.device:
    if device == 'cpu':
        return torch.device("cpu")
    elif device == 'cuda':
        if torch.cuda.is_available():
            return torch.device("cuda")
        else:
            raise ValueError("CUDA is not available.")
    elif device == 'mps':
        if torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            raise ValueError("MPS is not available.")


DEFAULT_RANDOM_SEED = 42

def seedBasic(seed=DEFAULT_RANDOM_SEED):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    
# torch random seed
def seedTorch(seed=DEFAULT_RANDOM_SEED):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
      
# basic + tensorflow + torch 
def seed_everything(seed=DEFAULT_RANDOM_SEED):
    seedBasic(seed)
    seedTorch(seed)

class SqueezeExcitation1d(torch.nn.Module):
    """
    This block implements the Squeeze-and-Excitation block from https://arxiv.org/abs/1709.01507 (see Fig. 1).
    Parameters ``activation``, and ``scale_activation`` correspond to ``delta`` and ``sigma`` in eq. 3.

    Args:
        input_channels (int): Number of channels in the input image
        squeeze_channels (int): Number of squeeze channels
        activation (Callable[..., torch.nn.Module], optional): ``delta`` activation. Default: ``torch.nn.ReLU``
        scale_activation (Callable[..., torch.nn.Module]): ``sigma`` activation. Default: ``torch.nn.Sigmoid``
    """

    def __init__(
        self,
        input_channels: int,
        squeeze_channels: int,
        activation = torch.nn.ReLU,
        scale_activation = torch.nn.Sigmoid,
    ) -> None:
        super().__init__()
        self.avgpool = torch.nn.AdaptiveAvgPool1d(1)
        self.fc1 = torch.nn.Conv1d(input_channels, squeeze_channels, 1)
        self.fc2 = torch.nn.Conv1d(squeeze_channels, input_channels, 1)
        self.activation = activation()
        self.scale_activation = scale_activation()

    def _scale(self, input):
        scale = self.avgpool(input)
        scale = self.fc1(scale)
        scale = self.activation(scale)
        scale = self.fc2(scale)
        return self.scale_activation(scale)

    def forward(self, input):
        scale = self._scale(input)
        return scale * input


class SpatialAttention1d(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        assert kernel_size % 2 == 1, "Kernel size must be odd."
        self.conv = nn.Conv1d(2, 1, kernel_size=kernel_size, padding=kernel_size // 2)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        out = torch.cat([max_out, avg_out], dim=1)
        out = self.conv(out)
        x = self.sigmoid(out) * x
        return x
    

class CNN1DL000(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()

        self.output_class = num_classes

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        # 添加 Squeeze-and-Excitation 模块
        self.se1 = SqueezeExcitation1d(
            input_channels= 32, 
            squeeze_channels= 32 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa1 = SpatialAttention1d(kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.se2 = SqueezeExcitation1d(
            input_channels= 64, 
            squeeze_channels= 64 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa2 = SpatialAttention1d(kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        # 添加 Squeeze-and-Excitation 模块
        self.se3 = SqueezeExcitation1d(
            input_channels= 128, 
            squeeze_channels= 128 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa3 = SpatialAttention1d(kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.se4 = SqueezeExcitation1d(
            input_channels= 256, 
            squeeze_channels= 256 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa4 = SpatialAttention1d(kernel_size=3)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.se5 = SqueezeExcitation1d(
            input_channels= 512, 
            squeeze_channels= 512 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa5 = SpatialAttention1d(kernel_size=3)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.se6 = SqueezeExcitation1d(
            input_channels= 1024, 
            squeeze_channels= 1024 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa6 = SpatialAttention1d(kernel_size=3)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(16384, 256)
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)
        

    def forward(self, x):
        x.unsqueeze_(1) # N * L -> N * 1 * L
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.se1(x)
        x = self.sa1(x)
        x = self.pool1(x)
        # print('x', x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.se2(x)
        x = self.sa2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.se3(x)
        x = self.sa3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.se4(x)
        x = self.sa4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.se5(x)
        x = self.sa5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.se6(x)
        x = self.sa6(x)
        x = self.pool6(x)

        out = torch.flatten(x, start_dim=1)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.fc2(out)

        return out

class CNN1D(nn.Module):
    def __init__(self, num_classes=None):
        super().__init__()

        self.output_class = num_classes

        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        # 添加 Squeeze-and-Excitation 模块
        self.se1 = SqueezeExcitation1d(
            input_channels= 32, 
            squeeze_channels= 32 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa1 = SpatialAttention1d(kernel_size=3)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.se2 = SqueezeExcitation1d(
            input_channels= 64, 
            squeeze_channels= 64 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa2 = SpatialAttention1d(kernel_size=3)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn3 = nn.BatchNorm1d(128)
        self.relu3 = nn.ReLU(inplace=True)
        # 添加 Squeeze-and-Excitation 模块
        self.se3 = SqueezeExcitation1d(
            input_channels= 128, 
            squeeze_channels= 128 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa3 = SpatialAttention1d(kernel_size=3)
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv1d(128, 256, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn4 = nn.BatchNorm1d(256)
        self.relu4 = nn.ReLU(inplace=True)
        self.se4 = SqueezeExcitation1d(
            input_channels= 256, 
            squeeze_channels= 256 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa4 = SpatialAttention1d(kernel_size=3)
        self.pool4 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv1d(256, 512, kernel_size=1, stride=1, padding=1, bias=False)
        self.bn5 = nn.BatchNorm1d(512)
        self.relu5 = nn.ReLU(inplace=True)
        self.se5 = SqueezeExcitation1d(
            input_channels= 512, 
            squeeze_channels= 512 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa5 = SpatialAttention1d(kernel_size=3)
        self.pool5 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv6 = nn.Conv1d(512, 1024, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn6 = nn.BatchNorm1d(1024)
        self.relu6 = nn.ReLU(inplace=True)
        self.se6 = SqueezeExcitation1d(
            input_channels= 1024, 
            squeeze_channels= 1024 // 16,  # 压缩通道数通常是输入通道数的 1/16
        )
        self.sa6 = SpatialAttention1d(kernel_size=3)
        self.pool6 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.drop = nn.Dropout(p=0.5)
        self.fc1 = nn.Linear(2048, 256) 
        self.fc1_relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Linear(256, num_classes)
        

    def forward(self, x):
        x.unsqueeze_(1) # N * L -> N * 1 * L
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.se1(x)
        x = self.sa1(x)
        x = self.pool1(x)
        # print('x', x.shape)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.se2(x)
        x = self.sa2(x)
        x = self.pool2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.se3(x)
        x = self.sa3(x)
        x = self.pool3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.relu4(x)
        x = self.se4(x)
        x = self.sa4(x)
        x = self.pool4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.relu5(x)
        x = self.se5(x)
        x = self.sa5(x)
        x = self.pool5(x)

        x = self.conv6(x)
        x = self.bn6(x)
        x = self.relu6(x)
        x = self.se6(x)
        x = self.sa6(x)
        x = self.pool6(x)

        out = torch.flatten(x, start_dim=1)
        out = self.drop(out)
        out = self.fc1(out)
        out = self.fc1_relu(out)
        out = self.fc2(out)

        return out
    


class PositionalEncoding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :]
        return x


class CNNTransformer(nn.Module):
    def __init__(
        self, 
        num_classes: int, 
        # transformer_dim: int = 128, 
        num_heads: int = 4, 
        num_layers: int = 1,
    ):
        super().__init__()
        

        # CNN 部分
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2)

        self.positional_encoding = PositionalEncoding(max_len=100, d_model=64)
        
        # Transformer Encoder 部分
        self.transformer_encoder = TransformerEncoder(
            TransformerEncoderLayer(d_model=64, nhead=num_heads, dropout=0.1, batch_first=True),
            num_layers=num_layers
        )
        
        # 输出层
        self.fc = nn.Linear(64, num_classes)
    
    def forward(self, x):
        # CNN 提取局部特征
        x.unsqueeze_(1) # N * L -> N * 1 * L
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.pool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        # print('x', x.shape)

        
        
        # 重新排列为序列
        x = x.permute(0, 2, 1)  # batch_size, channels, length -> (batch_size, seq_len, channels)
        x = self.positional_encoding(x)
        
        # print('x before trans', x.shape)
        # Transformer Encoder 提取全局特征
        x = self.transformer_encoder(x)
        # print('x after trans', x.shape)
        
        # 取最后一个时间步的输出
        x = x[:, -1, :]
        
        # 输出层
        x = self.fc(x)
        return x



class Package():
    def __init__(self,) -> None:
        self.package = []
    def add_one_element(self, element):
        self.package.append(element)
    
    def get_package_ave(self,):
        return sum(self.package) / len(self.package)

class Trainer():
    def __init__(
        self,
        num_classes: int,
        model_name: Literal['CNN1D', 'CNNTransformer', 'CNN1DL1000'] = 'CNN1D',
        device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
        lr: float = 0.005,
        epochs: int = 200,
        lr_scheduler_patience: int = 10,
        weight_decay: float = 0.0,
        label_smoothing: float = 0.0,

    ) -> None:
        self.model_name = model_name
        self.device = set_device(device)
        self.num_classes = num_classes
        self.lr = lr
        self.epochs = epochs
        self.lr_scheduler_patience = lr_scheduler_patience
        self.weight_decay = weight_decay
        self.label_smoothing = label_smoothing

        if self.model_name == 'CNN1D':
            self.model = CNN1D(self.num_classes)
        elif self.model_name == 'CNNTransformer':
            self.model = CNNTransformer(self.num_classes)
        elif self.model_name == 'CNN1DL1000':
            self.model = CNN1DL000(self.num_classes)
        else:
            raise ValueError(f'Unknown model: {self.model}')
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.loss_fn = nn.CrossEntropyLoss(reduction='mean', label_smoothing=self.label_smoothing)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='max', factor=0.5, 
            min_lr=0.01*self.lr,  patience=self.lr_scheduler_patience,
        )
        # self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=20, eta_min=0)

        print(f'Model {self.model_name} has total parameter number: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.2f} M')

        self.history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [],
                        'best_val_acc': 0, 'best_val_epoch': 0}

    def fit(
        self, 
        train_loader, 
        val_loader, 
        name: str = 'train_for_something',
        early_stopping_patience: int = 30,
        save_model: bool = True,
    ):
        # 早停参数
        early_stopping_best_acc = 0.0
        early_stopping_counter = 0

        if save_model:
            if os.path.exists(f'{name}_best_model.pth'):
                raise FileExistsError(f"{name}_best_model.pth already exists")

        for epoch in range(self.epochs):
            self.model.train()
            losses_in_an_epoch = Package()
            accs_in_an_epoch = Package()
            for indx, (read_ids, X, y) in enumerate(train_loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_in_an_epoch.add_one_element(loss.item())
                accs_in_an_epoch.add_one_element(
                    torch.sum(self.output_to_class(outputs) == y).item() / len(y)
                )
            print(f'Epoch {epoch:>3} / {self.epochs} train_loss: {losses_in_an_epoch.get_package_ave():.4f} train_acc: {accs_in_an_epoch.get_package_ave():.4f}', end='')
            self.history['train_loss'].append(losses_in_an_epoch.get_package_ave())
            self.history['train_acc'].append(accs_in_an_epoch.get_package_ave())

            self.model.eval()
            with torch.no_grad():
                val_losses_in_an_epoch = Package()
                val_accs_in_an_epoch = Package()
                for indx, (read_ids, X, y) in enumerate(val_loader):
                    X, y = X.to(self.device), y.to(self.device)
                    outputs = self.model(X)
                    loss = self.loss_fn(outputs, y)
                    val_losses_in_an_epoch.add_one_element(loss.item())
                    val_accs_in_an_epoch.add_one_element(
                        torch.sum(self.output_to_class(outputs) == y).item() / len(y)
                    )
                print(f' val_loss: {val_losses_in_an_epoch.get_package_ave():.4f} val_acc: {val_accs_in_an_epoch.get_package_ave():.4f}', end='')
                self.scheduler.step(val_accs_in_an_epoch.get_package_ave())
                # self.scheduler.step()
                print(f' lr: {self.scheduler.get_last_lr()[0]}')
                self.history['val_loss'].append(val_losses_in_an_epoch.get_package_ave())
                self.history['val_acc'].append(val_accs_in_an_epoch.get_package_ave())
            
            # 早停机制
            if val_accs_in_an_epoch.get_package_ave() > early_stopping_best_acc:
                early_stopping_best_acc = val_accs_in_an_epoch.get_package_ave()
                early_stopping_counter = 0
                # 保存最佳模型
                if save_model:
                    torch.save(self.model.state_dict(), f'{name}_best_model.pth')
                self.history['best_val_acc'] = early_stopping_best_acc
                self.history['best_val_epoch'] = epoch
                
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= early_stopping_patience:
                    print("Early stopping triggered.")
                    break


    def output_to_class(self, outputs):
        class_labels = torch.argmax(outputs, dim=1)
        return class_labels

    def predict_proba(
        self,
        test_loader,
        name,
    ):
        self.model.load_state_dict(torch.load(f'{name}_best_model.pth', weights_only=True))
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            all_read_ids, all_outputs, all_ys = [], [], []
            for indx, (read_ids, X, y) in enumerate(test_loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                m = nn.Softmax(dim=1)
                outputs = m(outputs)
                all_read_ids.extend(read_ids)
                all_outputs.append(outputs)
                all_ys.append(y)
            all_outputs = torch.cat(all_outputs, dim=0)
            all_ys = torch.cat(all_ys, dim=0)
        acc = torch.sum(self.output_to_class(all_outputs) == all_ys).item() / len(all_ys)
        print(f' test_acc: {acc:.4f}')
        pred_df = pd.DataFrame(all_outputs.cpu().numpy())
        pred_df['y'] = all_ys.cpu().numpy()
        pred_df.index = all_read_ids
        return pred_df

    def predict(
        self,
        test_loader: DataLoader,
        name: str,
        y_to_label_dict: dict = None
    ):
        pred_pro_df = self.predict_proba(test_loader, name)
        pred_df = pd.DataFrame(pred_pro_df.iloc[:,0:-1].idxmax(axis=1), columns=['pred'])
        pred_df['true'] = pred_pro_df['y']
        if y_to_label_dict is not None:
            pred_df['pred'] = pred_df['pred'].map(y_to_label_dict)
            pred_df['true'] = pred_df['true'].map(y_to_label_dict)
        return pred_df
    


    
def get_cm(
    pred_df: pd.DataFrame,
    label_order: List[str],
):
    cm_df = pred_df.copy()
    cm_df['count'] = 1
    cm_df = cm_df.groupby(["pred", 'true']).sum().reset_index()
    cm_df = cm_df.pivot(index='true', columns='pred', values='count').fillna(0)
    cm_df = cm_df.reindex(label_order, axis=0)
    cm_df = cm_df.reindex(label_order, axis=1)
    cm_df = cm_df.fillna(0).astype(int)
    return cm_df

def plot_cm(
        cm_df,
        nor_to_percent_for_each_pred: bool = True, 
        figsize: tuple = (10,10), 
        ax = None,
        annot: bool = True,
        annot_size: float = 8.0,
        lw_of_rectangle: float = 1.0,
        mark_diagonal_line: bool = True,
        save_fig: bool = False,
        save_dir: str = './',
        save_name: str = 'cm.pdf',
    ) -> Union[None, plt.Axes]:
        if nor_to_percent_for_each_pred:
            cm_df = cm_df.div(cm_df.sum(axis=1), axis=0)*100
        if not ax:
            fig, ax = plt.subplots(figsize=figsize)
        cmap = sns.cubehelix_palette(20, light=0.95, dark=0.15)
        if nor_to_percent_for_each_pred:
            vmax = 100
        else:
            vmax = None
        sns.heatmap(cm_df, annot=annot, annot_kws={"size": annot_size}, cmap=cmap, fmt=".1f", ax=ax, vmin=0, vmax=vmax)

        if mark_diagonal_line:
            for i in range(len(cm_df)):
                ax.add_patch(Rectangle((i, i), 1, 1, fill=False, edgecolor='red', lw=lw_of_rectangle, clip_on=False))
        plt.yticks(rotation=0)
        plt.xticks(rotation=90)
        if save_fig:
            pl.create_dir_if_not_exist(save_dir)
            plt.savefig(f'{save_dir}/{save_name}', bbox_inches='tight')
            return None
        return ax




