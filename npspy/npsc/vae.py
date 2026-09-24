#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: vae.py
@Description: description of this file
@Datatime: 2025/07/17 09:23:26
@Author: Hailin Pan
@Email: panhailin@genomics.cn, hailinpan1988@163.com
@Version: v1.0
'''

from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
from scipy.spatial.distance import mahalanobis
import re


from .. import machine_learning as ml
from .. import io
from .. import tools as tl
from .. import machine_learning as ml
from . import tools as thistl
from .. import plot as pl
from . import plot as thispl

class LSTMAutoencoder(nn.Module):
    def __init__(self, seq_len=1000, input_dim=1, hidden_dim=128, latent_dim=64, num_layers=1):
        super().__init__()
        self.seq_len = seq_len
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # 编码器
        self.encoder_lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 瓶颈层
        self.bottleneck = nn.Linear(hidden_dim, latent_dim)
        
        # 解码器
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True
        )
        
        self.decoder_fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def forward(self, x):
        # 输入维度: (batch_size, seq_len=1000)
        x = x.unsqueeze(-1)  # (batch_size, seq_len, input_dim=1)
        
        # 编码过程
        _, (hidden, _) = self.encoder_lstm(x)
        # 取最后一层的隐藏状态
        encoded = hidden[-1]  # (batch_size, hidden_dim)
        latent = self.bottleneck(encoded)  # (batch_size, latent_dim)
        
        # 解码过程
        # 将潜在向量重复seq_len次作为解码器输入
        decoder_input = latent.unsqueeze(1).repeat(1, self.seq_len, 1)
        lstm_out, _ = self.decoder_lstm(decoder_input)
        reconstructed = self.decoder_fc(lstm_out).squeeze(-1)
        
        return reconstructed
    


class CNNAutoencoder(nn.Module):
    """增强型时间序列自编码器"""
    def __init__(self, input_size=1000, encoding_dim=16):
        super().__init__()
        
        # 编码器：使用1D卷积
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=5, stride=2, padding=2),  # [batch, 16, seq_len/2]
            nn.ReLU(),
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),  # [batch, 32, seq_len/4]
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * (input_size // 4), 128),
            nn.ReLU(),
            nn.Linear(128, encoding_dim)
        )
        
        # 解码器：使用转置卷积
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 32 * (input_size // 4)),
            nn.Unflatten(1, (32, input_size // 4)),
            nn.ConvTranspose1d(32, 16, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(16, 1, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.Sigmoid()  # 确保输出在合理范围
        )
    
    def forward(self, x):
        # 添加通道维度 [batch, 1, seq_len]
        x = x.unsqueeze(1) if x.dim() == 2 else x
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded.squeeze(1) # 移除通道维度
    
class ShapeAwareAutoencoder(nn.Module):
    def __init__(self, seq_len=1000, input_dim=1, 
                 cnn_channels=64, lstm_units=128, latent_dim=32): # atent_dim=32
        super().__init__()
        
        # CNN形状特征提取器
        self.cnn_encoder = nn.Sequential(
            nn.Conv1d(input_dim, 32, kernel_size=25, padding=12),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(4),  # 250
            
            nn.Conv1d(32, cnn_channels, kernel_size=15, padding=7),
            nn.BatchNorm1d(cnn_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool1d(5)   # 50
        )
        
        # LSTM时序建模
        self.lstm_encoder = nn.LSTM(
            input_size=cnn_channels,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # 形状特征瓶颈层
        self.bottleneck = nn.Sequential(
            nn.Linear(2*lstm_units, latent_dim),
            nn.Tanh()
        )
        
        # 解码器LSTM
        self.lstm_decoder = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_units,
            num_layers=2,
            batch_first=True
        )
        
        # CNN上采样重建
        self.cnn_decoder = nn.Sequential(
            nn.ConvTranspose1d(lstm_units, 64, kernel_size=15, stride=5, padding=5),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            nn.ConvTranspose1d(64, 32, kernel_size=25, stride=4, padding=12, output_padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            nn.Conv1d(32, input_dim, kernel_size=3, padding=1)
        )
        
        self.seq_len = seq_len
    
    def forward(self, x):
        # 输入: (batch, seq_len=1000)
        x = x.unsqueeze(1)  # (batch, 1, 1000)
        
        # CNN形状特征提取
        cnn_features = self.cnn_encoder(x)  # (batch, cnn_channels, 50)
        cnn_features = cnn_features.permute(0, 2, 1)  # (batch, 50, cnn_channels)
        
        # LSTM时序编码
        lstm_out, _ = self.lstm_encoder(cnn_features)
        encoded = lstm_out[:, -1, :]  # 取最终状态
        latent = self.bottleneck(encoded)
        
        # 解码
        decoder_input = latent.unsqueeze(1).repeat(1, 50, 1)
        lstm_dec_out, _ = self.lstm_decoder(decoder_input)
        lstm_dec_out = lstm_dec_out.permute(0, 2, 1)  # (batch, lstm_units, 50)
        
        # CNN上采样重建
        recon = self.cnn_decoder(lstm_dec_out).squeeze(1)  # (batch, 1000)
        
        return recon, latent

class CNNLSTMVAE(nn.Module):
    def __init__(self, input_dim=1, seq_len=1000, latent_dim=64, 
                 cnn_channels=[32, 64, 128, 256, 256], lstm_hidden=128, num_layers=2):
        """
        CNN-LSTM变分自编码器
        
        参数:
        input_dim: 输入特征维度 (时间序列为1)
        seq_len: 输入序列长度 (1000)
        latent_dim: 潜在空间维度
        cnn_channels: CNN各层通道数
        lstm_hidden: LSTM隐藏层大小
        num_layers: LSTM层数
        """
        super(CNNLSTMVAE, self).__init__()
        
        # 存储参数
        self.seq_len = seq_len
        self.latent_dim = latent_dim
        self.cnn_channels = cnn_channels
        self.lstm_hidden = lstm_hidden
        self.num_layers = num_layers
        
        # 计算CNN压缩后的序列长度
        self.compressed_len = self.calculate_compressed_len()
        
        # ====================== 编码器 ======================
        # CNN部分 - 压缩序列并提取特征
        self.encoder_cnn = nn.Sequential(
            nn.Conv1d(input_dim, cnn_channels[0], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(cnn_channels[0], cnn_channels[1], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.LeakyReLU(0.2),
            
            nn.Conv1d(cnn_channels[1], cnn_channels[2], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.LeakyReLU(0.2),

            nn.Conv1d(cnn_channels[2], cnn_channels[3], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels[3]),
            nn.LeakyReLU(0.2),

            nn.Conv1d(cnn_channels[3], cnn_channels[4], kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(cnn_channels[4]),
            nn.LeakyReLU(0.2)
        )
        
        # LSTM部分 - 处理压缩后的序列
        self.encoder_lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=False
        )
        
        # 潜在空间映射层
        self.fc_mu = nn.Linear(lstm_hidden, latent_dim)
        self.fc_logvar = nn.Linear(lstm_hidden, latent_dim)
        
        # ====================== 解码器 ======================
        # 初始化LSTM状态
        self.decoder_init = nn.Linear(latent_dim, num_layers * lstm_hidden)
        
        # LSTM解码器
        self.decoder_lstm = nn.LSTM(
            input_size=latent_dim,
            hidden_size=lstm_hidden,
            num_layers=num_layers,
            batch_first=True
        )
        
        # 转置卷积部分 - 上采样序列
        self.decoder_deconv = nn.Sequential(
            nn.ConvTranspose1d(lstm_hidden, cnn_channels[3], kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(cnn_channels[3]),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(cnn_channels[3], cnn_channels[2], kernel_size=5, stride=2, padding=2, output_padding=0),
            nn.BatchNorm1d(cnn_channels[2]),
            nn.LeakyReLU(0.2),

            nn.ConvTranspose1d(cnn_channels[2], cnn_channels[1], kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(cnn_channels[1]),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(cnn_channels[1], cnn_channels[0], kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.BatchNorm1d(cnn_channels[0]),
            nn.LeakyReLU(0.2),
            
            nn.ConvTranspose1d(cnn_channels[0], input_dim, kernel_size=3, stride=2, padding=2, output_padding=1),
            nn.Tanh()
        )
        
        # 最终调整层 (确保输出长度精确匹配)
        self.final_adjust = nn.Conv1d(input_dim, input_dim, kernel_size=3, padding=0)

    def calculate_compressed_len(self):
        """计算经过CNN压缩后的序列长度"""
        length = self.seq_len
        for _ in range(len(self.cnn_channels)):  # 对应3个卷积层
            length = (length + 2 * 2 - 5) // 2 + 1  # 公式: (W - F + 2P)/S + 1
        return length

    def encode(self, x):
        """编码器前向传播"""
        # 输入形状: (batch, 1, seq_len)
        
        # CNN编码
        x = self.encoder_cnn(x)  # 输出: (batch, cnn_channels[2], compressed_len)
        x = x.permute(0, 2, 1)   # 转置: (batch, compressed_len, cnn_channels[2])
        
        # LSTM编码
        _, (h_n, _) = self.encoder_lstm(x)  # h_n形状: (num_layers, batch, hidden_size)
        h_n = h_n[-1]  # 取最后一层隐藏状态: (batch, hidden_size)
        
        # 计算潜在空间参数
        mu = self.fc_mu(h_n)
        logvar = self.fc_logvar(h_n)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """重参数化技巧"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        """解码器前向传播"""
        batch_size = z.size(0)
        
        # 准备初始隐藏状态
        init_hidden = self.decoder_init(z)  # (batch, num_layers * hidden_size)
        init_hidden = init_hidden.view(self.num_layers, batch_size, self.lstm_hidden)
        init_cell = torch.zeros_like(init_hidden)
        
        # 创建输入序列 (重复潜在向量)
        z_seq = z.unsqueeze(1).repeat(1, self.compressed_len, 1)  # (batch, compressed_len, latent_dim)
        
        # LSTM解码
        lstm_out, _ = self.decoder_lstm(z_seq, (init_hidden, init_cell))  # (batch, compressed_len, hidden_size)
        lstm_out = lstm_out.permute(0, 2, 1)  # (batch, hidden_size, compressed_len)
        
        # 转置卷积上采样
        x_hat = self.decoder_deconv(lstm_out)  # (batch, 1, ~seq_len)

        # 精确调整输出长度
        if x_hat.size(2) != self.seq_len:
            x_hat = self.final_adjust(x_hat)  # (batch, 1, seq_len)

        x_hat = (x_hat + 1) / 2 # 0 - 1
        
        return x_hat

    def forward(self, x, sample: bool = True):
        """完整前向传播"""
        # 输入形状: (batch, seq_len) -> 添加通道维度
        x = x.unsqueeze(1)  # (batch, 1, seq_len)
        
        # 编码
        mu, logvar = self.encode(x)
        
        # 重参数化
        z = self.reparameterize(mu, logvar)
        
        # 解码
        if sample:
            recon_x = self.decode(z)
        else:
            recon_x = self.decode(mu)
        
        # 移除通道维度以匹配输入
        return recon_x.squeeze(1), mu, logvar


def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    VAE损失函数
    
    参数:
    recon_x: 重建样本
    x: 原始样本
    mu: 潜在均值
    logvar: 潜在对数方差
    beta: KL散度权重
    """
    # 重建损失 (MSE)
    recon_loss = F.mse_loss(recon_x, x, reduction='sum')
    
    # KL散度 (正则化项)
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_loss

def get_X_from_an_obj(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
) -> pd.DataFrame:
    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    X = tl.get_signals_for_reads_in_an_obj(obj, att=att, down_sample_to=down_sample_to)
    X = X.astype(np.float32)
    read_ids = list(obj.keys())
    df = pd.DataFrame(X, index=read_ids)
    df.columns = [i for i in range(df.shape[1])]
    return df

class Cus_Dataset(Dataset):
    def __init__(self, data_df,):
        self.all_read_ids = data_df.index.to_numpy()
        self.X = data_df.to_numpy(dtype=np.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        one_read_id = self.all_read_ids[idx]
        one_X = self.X[idx]

        return one_read_id, one_X
    
def construct_dataloader_from_data_df(
    data_df: pd.DataFrame,
    batch_size: int = 64,
    shuffle: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    dataset = Cus_Dataset(data_df)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=drop_last)
    return dataloader


class Trainer():
    def __init__(
        self,
        model_name: Literal['CNNLSTMVAE'] = 'CNNLSTMVAE',
        device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
        lr: float = 0.005,
        epochs: int = 200,
        beta: float = 1.0,
    ) -> None:
        self.model_name = model_name
        self.device = ml.set_device(device)
        self.lr = lr
        self.epochs = epochs
        self.beta = beta

        if self.model_name == 'CNNLSTMVAE':
            self.model = CNNLSTMVAE()
        else:
            raise ValueError(f'Unknown model: {self.model}')
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = vae_loss
   

        print(f'Model {self.model_name} has total parameter number: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.2f} M')

        self.history = {'train_loss': []}

    def fit(
        self, 
        train_loader, 
        name: str = 'train_for_something',
        save_model: bool = True,
    ):

        if save_model:
            if os.path.exists(f'{name}_best_model.pth'):
                raise FileExistsError(f"{name}_best_model.pth already exists")

        for epoch in range(self.epochs):
            self.model.train()
            losses_in_an_epoch = ml.Package()
            for indx, (read_ids, X) in enumerate(train_loader):
                X = X.to(self.device)
                recon_x, mu, logvar = self.model(X)
                loss = self.loss_fn(recon_x, X, mu, logvar, beta=self.beta)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                losses_in_an_epoch.add_one_element(loss.item())
                
            print(f'Epoch {epoch:>3} / {self.epochs} train_loss: {losses_in_an_epoch.get_package_ave():.6f}')
            self.history['train_loss'].append(losses_in_an_epoch.get_package_ave())
        if save_model:
            torch.save(self.model.state_dict(), f'{name}_best_model.pth')
            print(f'Model saved to {name}_best_model.pth')
        
    def save_model(
        self,
        save_dir: str = '.',
        name: str = 'train_for_something',
    ):
        pl.create_dir_if_not_exist(save_dir)
        if os.path.exists(f'{save_dir}/{name}_best_model.pth'):
            raise FileExistsError(f"{save_dir}/{name}_best_model.pth already exists")
        torch.save(self.model.state_dict(), f'{save_dir}/{name}_best_model.pth')
        print(f'Model saved to {name}_best_model.pth'  
    )

    def get_reconstructed_signals(
        self,
        dataloader: DataLoader,
    ) -> pd.DataFrame:
        self.model.eval()
        reconstructed_signals = []
        reconstruction_errors = []
        mus = []
        read_ids = []
        with torch.no_grad():
            for indx, (read_id, X) in enumerate(dataloader):
                X = X.to(self.device)
                # outputs, latent = self.model(X)
                outputs, mu, logvar = self.model(X, sample=False)
                reconstructed_signals.append(outputs.cpu().numpy())
                read_ids.extend(read_id)
                mus.append(mu.cpu().numpy())

                # 计算每个序列的MSE
                mse = nn.functional.mse_loss(outputs, X, reduction='none')
                batch_errors = mse.mean(dim=1).cpu().numpy()
                reconstruction_errors.extend(batch_errors)

        reconstructed_signals = np.concatenate(reconstructed_signals, axis=0)
        df = pd.DataFrame(reconstructed_signals, index=read_ids)
        df.columns = [i for i in range(df.shape[1])]
        error_df = pd.DataFrame(reconstruction_errors, index=read_ids, columns=['reconstruction_error'])
        mus = np.concatenate(mus, axis=0)
        mu_df = pd.DataFrame(mus, index=read_ids)
        mu_df.columns = [f'mu_{i}' for i in range(mus.shape[1])]
        return df, error_df, mu_df
    

def train_ae_iteratively(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    max_iteration: int = 3,
    min_read_num: int = 10000,
    seed: int = 41,
    model_name: Literal['CNNLSTMVAE'] = 'CNNLSTMVAE',
    save_dir: str = '.',
    fit_name: str = 'train_for_something',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    lr: float = 0.001,
    epochs: int = 50,
    beta: float = 1.0
) -> None:
    """ Train an autoencoder iteratively to clean the object.
    Args:
        obj (Union[dict, str]): The input object containing sequences, can be a dictionary or a file path.
        att (str): The attribute to use for the sequences, default is 'signal'.
        down_sample_to (int): The length to downsample the sequences to, default is 1000.
        max_iteration (int): Maximum number of iterations for training, default is 3.
        min_read_num (int): Minimum number of reads required to continue training, default is 10000.
        seed (int): Random seed for reproducibility, default is 41.
        model_name (Literal): The name of the autoencoder model to use, default is 'ShapeAwareAutoencoder'.
        save_dir (str): Directory where the model will be saved, default is '.'.
        fit_name (str): The name of the fit, used for saving the model, default is 'train_for_something'.
        device (Literal): Device to run the model on, default is 'cuda'.
        lr (float): Learning rate for the optimizer, default is 0.001.
        epochs (int): Number of epochs for training in each iteration, default is 50.
    """
    ml.seed_everything(seed)
    if os.path.exists(f'{save_dir}/{fit_name}_best_model.pth'):
        raise FileExistsError(f"{save_dir}/{fit_name}_best_model.pth already exists")

    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    
    if len(obj) < min_read_num:
        raise ValueError(f'The number of reads ({len(obj)}) is less than the minimum required ({min_read_num}).')

    for i in range(max_iteration):
        print(f'Iteration {i+1}/{max_iteration} for training {model_name}...')
        train_df = get_X_from_an_obj(obj, att=att, down_sample_to=down_sample_to)
        dataloader = construct_dataloader_from_data_df(train_df, batch_size=64, shuffle=True, drop_last=False)
        trainer = Trainer(
            model_name=model_name,
            device=device,
            lr=lr,
            epochs=epochs,
            beta = beta
        )
        trainer.fit(dataloader, name=f'{fit_name}_iteration{i+1}', save_model=False)
        reconstructed_df, error_df, latent_df = trainer.get_reconstructed_signals(dataloader)
        # cut = np.quantile(error_df['reconstruction_error'], 0.8)
        error_cut = thistl.get_mad_cutoff(error_df['reconstruction_error'])
       
        # cut0, cut1 = np.quantile(error_df['reconstruction_error'], 0.1), np.quantile(error_df['reconstruction_error'], 0.9)
        mahalanobis_distances = get_mahalanobis_distances_from_lattent_df(latent_df)
        mahalanobis_cut = thistl.get_mad_cutoff(mahalanobis_distances)
        obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] <= error_cut) & (mahalanobis_distances <= mahalanobis_cut)].index)
        # high_quality_reads = np.intersect1d(error_df[error_df['reconstruction_error'] <= error_cut].index.to_list(), lzc_df[lzc_df['lzc'] <= lzc_cut].index.to_list())
        # obj = tl.extract_reads_as_an_obj(obj, error_df[error_df['reconstruction_error'] <= error_cut].index)
        # obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] >= cut0) & (error_df['reconstruction_error'] <= cut1)].index)
        # obj = tl.extract_reads_as_an_obj(obj, lzc_df[lzc_df['lzc'] <= lzc_cut].index)
        # tmp = [re.search(r'.*_(\S+)', i).group(1) for i in high_quality_reads]
        # lzc_df.to_csv(f'/home/panhailin/project/sd0456_20250317/npsc/03.results/simulation/sim4/lzc_df{i}.csv')
        # print(pd.Series(tmp).value_counts())
        # obj = tl.extract_reads_as_an_obj(obj, high_quality_reads)
        if len(obj) < min_read_num:
            print(f'Number of reads after iteration {i+1}: {len(obj)}. Stopping training as it is less than {min_read_num}.')
            break
    trainer.save_model(name=f'{fit_name}', save_dir=save_dir)



def get_mahalanobis_distances_from_lattent_df(
    latent_df: pd.DataFrame,
) -> np.ndarray:
    """ Calculate Mahalanobis distances from the latent representations.
    Args:
        latent_df (pd.DataFrame): DataFrame containing the latent representations.
    Returns:
        np.ndarray: Array of Mahalanobis distances for each sample.
    """
    # 计算嵌入空间的均值和协方差
    embeddings = np.array(latent_df)
    mean_embed = np.mean(embeddings, axis=0)
    cov_matrix = np.cov(embeddings, rowvar=False)
    cov_inv = np.linalg.inv(cov_matrix)
    # 计算每个样本的马氏距离
    mahalanobis_dists = np.array([mahalanobis(embed, mean_embed, cov_inv) 
                        for embed in embeddings])
    return mahalanobis_dists

def detect_inconsistent_sequences(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['CNNLSTMVAE'] = 'CNNLSTMVAE',
    fit_name: str = 'train_for_something',
    model_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    beta: float = 1.0
) -> Tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]:
    """ Detect inconsistent sequences in an object using a pre-trained autoencoder model.
    Args:
        obj (Union[dict, str]): The input object containing sequences, can be a dictionary or a file path.
        att (str): The attribute to use for the sequences, default is 'signal'.
        down_sample_to (int): The length to downsample the sequences to, default is 1000.
        seed (int): Random seed for reproducibility, default is 41.
        model_name (Literal): The name of the autoencoder model to use, default is 'ShapeAwareAutoencoder'.
        fit_name (str): The name of the fit, used for saving the model, default is 'train_for_something'.
        model_dir (str): Directory where the model is saved, default is '.'.
        device (Literal): Device to run the model on, default is 'cuda'.
    Returns:
        Tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame, float]: 
            - inconsistent_obj: Object containing inconsistent sequences.
            - clean_obj: Object containing clean sequences.
            - reconstructed_df: DataFrame of reconstructed signals.
            - error_df: DataFrame of reconstruction errors.
            - latent_df: DataFrame of latent representations.
            - threshold: MAD cutoff for reconstruction error.
    """
    ml.seed_everything(seed)
    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    train_df = get_X_from_an_obj(obj, att=att, down_sample_to=down_sample_to)
    dataloader = construct_dataloader_from_data_df(train_df, batch_size=64, shuffle=True, drop_last=False)
    trainer = Trainer(
            model_name=model_name,
            device=device,
            beta=beta,
    )
    trainer.model.load_state_dict(torch.load(f'{model_dir}/{fit_name}_best_model.pth', weights_only=True))
    trainer.model = trainer.model.to(trainer.device)
    reconstructed_df, error_df, latent_df = trainer.get_reconstructed_signals(dataloader)
    error_threshold =  thistl.get_mad_cutoff(error_df['reconstruction_error'])
    print(f'MAD cutoff for reconstruction error: {error_threshold}')
    # lzc_df = thistl.get_lzc_for_each_line_in_a_dataframe(train_df)
    # lzc_threshold = thistl.get_mad_cutoff(lzc_df['lzc'])
    # print(f'MAD cutoff for Lzc: {lzc_threshold}')

    mahalanobis_dists = get_mahalanobis_distances_from_lattent_df(latent_df)
    mahalanobis_dist_threshold = thistl.get_mad_cutoff(mahalanobis_dists)
    print(f'MAD cutoff for Mahalanobis distance: {mahalanobis_dist_threshold}')
    mahalanobis_dist_df = pd.DataFrame(mahalanobis_dists, index=latent_df.index, columns=['mahalanobis_distance'])

    # inconsistent_reads = error_df[(error_df['reconstruction_error'] > error_threshold) | (mahalanobis_dists > mahalanobis_dist_threshold)].index.tolist()
    inconsistent_reads_from_error = error_df[error_df['reconstruction_error'] > error_threshold].index.tolist()
    inconsistent_reads_from_mah = mahalanobis_dist_df[mahalanobis_dist_df['mahalanobis_distance'] > mahalanobis_dist_threshold].index.tolist()
    inconsistent_reads = list(set(inconsistent_reads_from_error + inconsistent_reads_from_mah))
    inconsistent_obj = tl.extract_reads_as_an_obj(obj, inconsistent_reads)
    print(f'Number of inconsistent reads: {len(inconsistent_obj)}')
    # clean_obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] <= error_threshold) & (mahalanobis_dists <= mahalanobis_dist_threshold)].index.tolist())
    # clean_obj = tl.extract_reads_as_an_obj(obj, error_df[error_df['reconstruction_error'] <= error_threshold].index.tolist())
    clean_reads = list(obj.keys() - inconsistent_reads)
    clean_obj = tl.extract_reads_as_an_obj(obj, clean_reads)
    print(f'Number of clean reads: {len(clean_obj)}')
    # return inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold
    return inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, error_threshold, mahalanobis_dist_df, mahalanobis_dist_threshold
        

def clean_obj_by_ae(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['CNNLSTMVAE'] = 'CNNLSTMVAE',
    fit_name: str = 'train_for_something',
    save_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    epochs: int = 20,
    lr: float = 0.001,
    max_iteration: int = 3,
    min_read_num: int = 10000,
    beta: float = 1.0
):
    """ Clean an object by training an autoencoder iteratively.
    Args:
        obj (Union[dict, str]): The input object containing sequences, can be a dictionary or a file path.
        att (str): The attribute to use for the sequences, default is 'signal'.
        down_sample_to (int): The length to downsample the sequences to, default is 1000.
        seed (int): Random seed for reproducibility, default is 41.
        model_name (Literal): The name of the autoencoder model to use, default is 'ShapeAwareAutoencoder'.
        fit_name (str): The name of the fit, used for saving the model, default is 'train_for_something'.
        save_dir (str): Directory where the model will be saved, default is '.'.
        device (Literal): Device to run the model on, default is 'cuda'.
        epochs (int): Number of epochs for training in each iteration, default is 20.
        lr (float): Learning rate for the optimizer, default is 0.001.
        max_iteration (int): Maximum number of iterations for training, default is 3.
        min_read_num (int): Minimum number of reads required to continue training, default is 10000.
    Returns:
        Tuple[dict, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
            - clean_obj: Object containing clean sequences.
            - inconsistent_obj: Object containing inconsistent sequences.
            - reconstructed_df: DataFrame of reconstructed signals.
            - error_df: DataFrame of reconstruction errors.
            - latent_df: DataFrame of latent representations.
    """
    train_ae_iteratively(
        obj, 
        fit_name=fit_name, 
        save_dir=save_dir,
        epochs=epochs, 
        lr=lr,
        max_iteration=max_iteration,
        att=att,
        down_sample_to=down_sample_to,
        seed=seed,
        model_name=model_name,
        device=device,
        min_read_num=min_read_num,
        beta=beta
    )

    # inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold = detect_inconsistent_sequences(
    inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, error_threshold, mahalanobis_dist_df, mahalanobis_dist_threshold = detect_inconsistent_sequences(
        obj,
        att=att,
        down_sample_to=down_sample_to,
        seed=seed,
        model_name=model_name,
        fit_name=fit_name,
        model_dir=save_dir,
        device=device,
        beta=beta
    )
    # return clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold
    return clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, error_threshold, mahalanobis_dist_df, mahalanobis_dist_threshold

def clean_ref_obj_by_ae_and_save(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['CNNLSTMVAE'] = 'CNNLSTMVAE',
    fit_name: str = 'train_for_something',
    model_save_dir: str = '.',
    output_save_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    epochs: int = 20,
    lr: float = 0.001,
    max_iteration: int = 3,
    min_read_num: int = 10000,
    save_reconstruction: bool = False,
    beta: float = 1.0
):
    # clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold = clean_obj_by_ae(
    clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, error_threshold, mahalanobis_dist_df, mahalanobis_dist_threshold = clean_obj_by_ae(
        obj=obj,
        att=att,
        down_sample_to=down_sample_to,
        seed=seed,
        model_name=model_name,
        fit_name=fit_name,
        save_dir=model_save_dir,
        device=device,
        epochs=epochs,
        lr=lr,
        max_iteration=max_iteration,
        min_read_num=min_read_num,
        beta=beta,
    )

    io.save_pickle(clean_obj, f'{output_save_dir}/{fit_name}_clean_obj.pkl')
    io.save_pickle(inconsistent_obj, f'{output_save_dir}/{fit_name}_rejected_obj.pkl')
    error_threshold_df = pd.DataFrame({'threshold': [error_threshold]})
    error_threshold_df.to_csv(f'{output_save_dir}/{fit_name}_error_threshold.csv')
    error_df.to_csv(f'{output_save_dir}/{fit_name}_error.csv')
    mahalanobis_dist_df.to_csv(f'{output_save_dir}/{fit_name}_mahalanobis_dist.csv')
    mahalanobis_dist_threshold_df = pd.DataFrame({'mahalanobis_dist_threshold': [mahalanobis_dist_threshold]})
    mahalanobis_dist_threshold_df.to_csv(f'{output_save_dir}/{fit_name}_mahalanobis_dist_threshold.csv')

    latent_df.to_csv(f'{output_save_dir}/{fit_name}_latent.csv')
    # mahalanobis_dist_df.to_csv(f'{output_save_dir}/{fit_name}_mahalanobis_distance.csv')
    # mahalanobis_dist_threshold_df = pd.DataFrame({'threshold': [mahalanobis_dist_threshold]})
    # mahalanobis_dist_threshold_df.to_csv(f'{output_save_dir}/{fit_name}_mahalanobis_distance_threshold.csv')
    thispl.draw_reconstruction_error_distribution(
        error_df=error_df,
        threshold=error_threshold,
        save_dir=output_save_dir,
        save_name=f'{fit_name}_reconstruction_error_distribution.pdf'
    )
    thispl.draw_mahalanobis_distance_distribution(
        mahalanobis_dist_df=mahalanobis_dist_df,
        threshold=mahalanobis_dist_threshold,
        save_dir=output_save_dir,
        save_name=f'{fit_name}_mahalanobis_distance_distribution.pdf'
    )

    # thispl.draw_lzc_distribution(
    #     lzc_df=mahalanobis_dist_df,
    #     threshold=mahalanobis_dist_threshold,
    #     save_dir=output_save_dir,
    #     save_name=f'{fit_name}_lzc_distribution.pdf'
    # )

    if save_reconstruction:
        reconstructed_df.to_csv(f'{output_save_dir}/{fit_name}_reconstructed_signals.csv')


def clean_an_obj_with_trained_model(
    obj: Union[dict, str],
    save_name: str,
    save_dir: str = '.',
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['CNNLSTMVAE'] = 'CNNLSTMVAE',
    fit_name: str = 'train_for_something',
    model_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    error_threshold_file: str = None,
    lzc_threshold_file: str = None,
) -> None:
    """Clean an object using a trained autoencoder model.

    Args:
        obj (Union[dict, str]): The object to be cleaned, can be a dictionary or a file path.
        save_name (str): The name of the saved object.
        save_dir (str): The directory where the saved object will be stored, default is '.'.
        att (str): The attribute of the object to be cleaned, default is 'signal'.
        down_sample_to (int): The length to downsample the sequences to, default is 1000.
        seed (int): Random seed for reproducibility, default is 41.
        model_name (Literal): The name of the autoencoder model to use, default is 'ShapeAwareAutoencoder'.
        fit_name (str): The name of the fit, used for saving the model, default is 'train_for_something'.
        model_dir (str): The directory where the model is saved, default is '.'.
        device (Literal): Device to run the model on, default is 'cuda'.
        threshold_file (str): Path to the file containing the threshold for reconstruction error, default is None.
    """
    ml.seed_everything(seed)
    if isinstance(obj, str):
        obj = io.read_pickle(obj)
    data_df = get_X_from_an_obj(obj, att=att, down_sample_to=down_sample_to)
    dataloader = construct_dataloader_from_data_df(data_df, batch_size=64, shuffle=False, drop_last=False)
    trainer = Trainer(
            model_name=model_name,
            device=device,
    )
    trainer.model.load_state_dict(torch.load(f'{model_dir}/{fit_name}_best_model.pth', weights_only=True))
    trainer.model = trainer.model.to(trainer.device)
    reconstructed_df, error_df, latent_df = trainer.get_reconstructed_signals(dataloader)
    # mahalanobis_distances = get_mahalanobis_distances_from_lattent_df(latent_df)

    error_threshold = pd.read_csv(error_threshold_file, index_col=0).iloc[0, 0]
    print(f'error threshold: {error_threshold}')
    lzc_threshold = pd.read_csv(lzc_threshold_file, index_col=0).iloc[0, 0]
    print(f'lzc threshold: {lzc_threshold}')

    # mahalanobis_dist_threshold = pd.read_csv(mahalanobis_dist_threshold_file, index_col=0).iloc[0, 0]
    # print(f'mahalanobis distance threshold: {mahalanobis_dist_threshold}')

    # inconsistent_reads = error_df[(error_df['reconstruction_error'] > error_threshold) | (mahalanobis_distances > mahalanobis_dist_threshold)].index.tolist()
    # inconsistent_reads = error_df[error_df['reconstruction_error'] > error_threshold].index.tolist()

    lzc_df = thistl.get_lzc_for_each_line_in_a_dataframe(data_df)
    inconsistent_reads_from_error = error_df[error_df['reconstruction_error'] > error_threshold].index.tolist()
    inconsistent_reads_from_lzc = lzc_df[lzc_df['lzc'] > lzc_threshold].index.tolist()
    inconsistent_reads = list(set(inconsistent_reads_from_error + inconsistent_reads_from_lzc))
    inconsistent_obj = tl.extract_reads_as_an_obj(obj, inconsistent_reads)
    print(f'Number of inconsistent reads: {len(inconsistent_obj)}')
    # clean_obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] <= error_threshold) & (mahalanobis_distances <= mahalanobis_dist_threshold)].index)

    clean_reads = list(obj.keys() - inconsistent_reads)
    clean_obj = tl.extract_reads_as_an_obj(obj, clean_reads)
    print(f'Number of clean reads: {len(clean_obj)}')
    io.save_pickle(clean_obj, f'{save_dir}/{save_name}_clean_obj.pkl')
    io.save_pickle(inconsistent_obj, f'{save_dir}/{save_name}_rejected_obj.pkl')
    error_df.to_csv(f'{save_dir}/{save_name}_error.csv')
    latent_df.to_csv(f'{save_dir}/{save_name}_latent.csv')
