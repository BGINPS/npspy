#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Filename: ae.py
@Description: description of this file
@Datatime: 2025/07/02 16:34:59
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
        model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'CNNAutoencoder',
        device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
        lr: float = 0.005,
        epochs: int = 200,

    ) -> None:
        self.model_name = model_name
        self.device = ml.set_device(device)
        self.lr = lr
        self.epochs = epochs

        if self.model_name == 'LSTMAutoencoder':
            self.model = LSTMAutoencoder()
        elif self.model_name == 'CNNAutoencoder':
            self.model = CNNAutoencoder()
        elif self.model_name == 'ShapeAwareAutoencoder':
            self.model = ShapeAwareAutoencoder()
        else:
            raise ValueError(f'Unknown model: {self.model}')
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss(reduction='mean')
   

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
                outputs, _ = self.model(X)
                loss = self.loss_fn(outputs, X)
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
        latents = []
        read_ids = []
        with torch.no_grad():
            for indx, (read_id, X) in enumerate(dataloader):
                X = X.to(self.device)
                outputs, latent = self.model(X)
                reconstructed_signals.append(outputs.cpu().numpy())
                read_ids.extend(read_id)
                latents.append(latent.cpu().numpy())

                # 计算每个序列的MSE
                mse = nn.functional.mse_loss(outputs, X, reduction='none')
                batch_errors = mse.mean(dim=1).cpu().numpy()
                reconstruction_errors.extend(batch_errors)

        reconstructed_signals = np.concatenate(reconstructed_signals, axis=0)
        df = pd.DataFrame(reconstructed_signals, index=read_ids)
        df.columns = [i for i in range(df.shape[1])]
        error_df = pd.DataFrame(reconstruction_errors, index=read_ids, columns=['reconstruction_error'])
        latents = np.concatenate(latents, axis=0)
        latent_df = pd.DataFrame(latents, index=read_ids)
        latent_df.columns = [f'latent_{i}' for i in range(latents.shape[1])]
        return df, error_df, latent_df
    

# def train_ae_iteratively(
#     obj: Union[dict, str],
#     att: str = 'signal',
#     down_sample_to: int = 1000,
#     max_iteration: int = 3,
#     min_read_num: int = 10000,
#     seed: int = 41,
#     model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'ShapeAwareAutoencoder',
#     save_dir: str = '.',
#     fit_name: str = 'train_for_something',
#     device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
#     lr: float = 0.001,
#     epochs: int = 50,
# ) -> None:
#     """ Train an autoencoder iteratively to clean the object.
#     Args:
#         obj (Union[dict, str]): The input object containing sequences, can be a dictionary or a file path.
#         att (str): The attribute to use for the sequences, default is 'signal'.
#         down_sample_to (int): The length to downsample the sequences to, default is 1000.
#         max_iteration (int): Maximum number of iterations for training, default is 3.
#         min_read_num (int): Minimum number of reads required to continue training, default is 10000.
#         seed (int): Random seed for reproducibility, default is 41.
#         model_name (Literal): The name of the autoencoder model to use, default is 'ShapeAwareAutoencoder'.
#         save_dir (str): Directory where the model will be saved, default is '.'.
#         fit_name (str): The name of the fit, used for saving the model, default is 'train_for_something'.
#         device (Literal): Device to run the model on, default is 'cuda'.
#         lr (float): Learning rate for the optimizer, default is 0.001.
#         epochs (int): Number of epochs for training in each iteration, default is 50.
#     """
#     ml.seed_everything(seed)
#     if os.path.exists(f'{save_dir}/{fit_name}_best_model.pth'):
#         raise FileExistsError(f"{save_dir}/{fit_name}_best_model.pth already exists")

#     if isinstance(obj, str):
#         obj = io.read_pickle(obj)
    
#     if len(obj) < min_read_num:
#         raise ValueError(f'The number of reads ({len(obj)}) is less than the minimum required ({min_read_num}).')


#     for i in range(max_iteration):
#         print(f'Iteration {i+1}/{max_iteration} for training {model_name}...')
#         train_df = get_X_from_an_obj(obj, att=att, down_sample_to=down_sample_to)
#         dataloader = construct_dataloader_from_data_df(train_df, batch_size=64, shuffle=True, drop_last=False)
#         trainer = Trainer(
#             model_name=model_name,
#             device=device,
#             lr=lr,
#             epochs=epochs,
#         )
#         trainer.fit(dataloader, name=f'{fit_name}_iteration{i+1}', save_model=False)
#         reconstructed_df, error_df, latent_df = trainer.get_reconstructed_signals(dataloader)
#         # cut = np.quantile(error_df['reconstruction_error'], 0.8)
#         error_cut = thistl.get_mad_cutoff(error_df['reconstruction_error'])
#         if i == 0:
#             lzc_df = thistl.get_lzc_for_each_line_in_a_dataframe(train_df)
#         else:
#             lzc_df = lzc_df.loc[train_df.index,:]
#         lzc_cut = thistl.get_mad_cutoff(lzc_df['lzc'])
#         # cut0, cut1 = np.quantile(error_df['reconstruction_error'], 0.1), np.quantile(error_df['reconstruction_error'], 0.9)
#         # mahalanobis_distances = get_mahalanobis_distances_from_lattent_df(latent_df)
#         # mahalanobis_cut = np.quantile(mahalanobis_distances, 0.8)
#         # obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] <= cut) & (mahalanobis_distances <= mahalanobis_cut)].index)
#         high_quality_reads = np.intersect1d(error_df[error_df['reconstruction_error'] <= error_cut].index.to_list(), lzc_df[lzc_df['lzc'] <= lzc_cut].index.to_list())
#         # obj = tl.extract_reads_as_an_obj(obj, error_df[error_df['reconstruction_error'] <= error_cut].index)
#         # obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] >= cut0) & (error_df['reconstruction_error'] <= cut1)].index)
#         # obj = tl.extract_reads_as_an_obj(obj, lzc_df[lzc_df['lzc'] <= lzc_cut].index)
#         # tmp = [re.search(r'.*_(\S+)', i).group(1) for i in high_quality_reads]
#         # lzc_df.to_csv(f'/home/panhailin/project/sd0456_20250317/npsc/03.results/simulation/sim4/lzc_df{i}.csv')
#         # print(pd.Series(tmp).value_counts())
#         obj = tl.extract_reads_as_an_obj(obj, high_quality_reads)
#         if len(obj) < min_read_num:
#             print(f'Number of reads after iteration {i+1}: {len(obj)}. Stopping training as it is less than {min_read_num}.')
#             break
#     trainer.save_model(name=f'{fit_name}', save_dir=save_dir)


def train_ae_iteratively(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    max_iteration: int = 3,
    min_read_num: int = 10000,
    seed: int = 41,
    model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'ShapeAwareAutoencoder',
    save_dir: str = '.',
    fit_name: str = 'train_for_something',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    lr: float = 0.001,
    epochs: int = 50,
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

    pre_trainer = None
    for i in range(max_iteration):
        print(f'Iteration {i+1}/{max_iteration} for training {model_name}...')
        train_df = get_X_from_an_obj(obj, att=att, down_sample_to=down_sample_to)
        dataloader = construct_dataloader_from_data_df(train_df, batch_size=64, shuffle=True, drop_last=False)
        trainer = Trainer(
            model_name=model_name,
            device=device,
            lr=lr,
            epochs=epochs,
        )
        if i >= 1:
            trainer.model.load_state_dict(pre_trainer.model.state_dict())
        trainer.fit(dataloader, name=f'{fit_name}_iteration{i+1}', save_model=False)
        reconstructed_df, error_df, latent_df = trainer.get_reconstructed_signals(dataloader)
        error_cut = thistl.get_mad_cutoff(error_df['reconstruction_error'])
        if i == 0:
            lzc_df = thistl.get_lzc_for_each_line_in_a_dataframe(train_df)
        else:
            lzc_df = lzc_df.loc[train_df.index,:]
        lzc_cut = thistl.get_mad_cutoff(lzc_df['lzc'])
        high_quality_reads = np.intersect1d(error_df[error_df['reconstruction_error'] <= error_cut].index.to_list(), lzc_df[lzc_df['lzc'] <= lzc_cut].index.to_list())
        obj = tl.extract_reads_as_an_obj(obj, high_quality_reads)
        if len(obj) < min_read_num:
            print(f'Number of reads after iteration {i+1}: {len(obj)}. Stopping training as it is less than {min_read_num}.')
            break
        pre_trainer = trainer
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
    model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'ShapeAwareAutoencoder',
    fit_name: str = 'train_for_something',
    model_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
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
    )
    trainer.model.load_state_dict(torch.load(f'{model_dir}/{fit_name}_best_model.pth', weights_only=True))
    trainer.model = trainer.model.to(trainer.device)
    reconstructed_df, error_df, latent_df = trainer.get_reconstructed_signals(dataloader)
    error_threshold =  thistl.get_mad_cutoff(error_df['reconstruction_error'])
    print(f'MAD cutoff for reconstruction error: {error_threshold}')
    lzc_df = thistl.get_lzc_for_each_line_in_a_dataframe(train_df)
    lzc_threshold = thistl.get_mad_cutoff(lzc_df['lzc'])
    print(f'MAD cutoff for Lzc: {lzc_threshold}')

    # # 计算嵌入空间的均值和协方差
    # embeddings = np.array(latent_df)
    # mean_embed = np.mean(embeddings, axis=0)
    # cov_matrix = np.cov(embeddings, rowvar=False)
    # cov_inv = np.linalg.inv(cov_matrix)
    # # 计算每个样本的马氏距离
    # mahalanobis_dists = np.array([mahalanobis(embed, mean_embed, cov_inv) 
    #                     for embed in embeddings])
    # mahalanobis_dist_threshold = thistl.get_mad_cutoff(mahalanobis_dists)
    # print(f'MAD cutoff for Mahalanobis distance: {mahalanobis_dist_threshold}')
    # mahalanobis_dist_df = pd.DataFrame(mahalanobis_dists, index=latent_df.index, columns=['mahalanobis_distance'])

    # inconsistent_reads = error_df[(error_df['reconstruction_error'] > error_threshold) | (mahalanobis_dists > mahalanobis_dist_threshold)].index.tolist()
    inconsistent_reads_from_error = error_df[error_df['reconstruction_error'] > error_threshold].index.tolist()
    inconsistent_reads_from_lzc = lzc_df[lzc_df['lzc'] > lzc_threshold].index.tolist()
    inconsistent_reads = list(set(inconsistent_reads_from_error + inconsistent_reads_from_lzc))
    inconsistent_obj = tl.extract_reads_as_an_obj(obj, inconsistent_reads)
    print(f'Number of inconsistent reads: {len(inconsistent_obj)}')
    # clean_obj = tl.extract_reads_as_an_obj(obj, error_df[(error_df['reconstruction_error'] <= error_threshold) & (mahalanobis_dists <= mahalanobis_dist_threshold)].index.tolist())
    # clean_obj = tl.extract_reads_as_an_obj(obj, error_df[error_df['reconstruction_error'] <= error_threshold].index.tolist())
    clean_reads = list(obj.keys() - inconsistent_reads)
    clean_obj = tl.extract_reads_as_an_obj(obj, clean_reads)
    print(f'Number of clean reads: {len(clean_obj)}')
    # return inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold
    return inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, error_threshold, lzc_df, lzc_threshold
        

def clean_obj_by_ae(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'ShapeAwareAutoencoder',
    fit_name: str = 'train_for_something',
    save_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    epochs: int = 20,
    lr: float = 0.001,
    max_iteration: int = 3,
    min_read_num: int = 10000,
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
    )

    # inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold = detect_inconsistent_sequences(
    inconsistent_obj, clean_obj, reconstructed_df, error_df, latent_df, error_threshold, lzc_df, lzc_threshold = detect_inconsistent_sequences(
        obj,
        att=att,
        down_sample_to=down_sample_to,
        seed=seed,
        model_name=model_name,
        fit_name=fit_name,
        model_dir=save_dir,
        device=device,
    )
    # return clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold
    return clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, error_threshold, lzc_df, lzc_threshold

def clean_ref_obj_by_ae_and_save(
    obj: Union[dict, str],
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'ShapeAwareAutoencoder',
    fit_name: str = 'train_for_something',
    model_save_dir: str = '.',
    output_save_dir: str = '.',
    device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
    epochs: int = 20,
    lr: float = 0.001,
    max_iteration: int = 3,
    min_read_num: int = 10000,
    save_reconstruction: bool = False,
):
    # clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, mahalanobis_dist_df, error_threshold, mahalanobis_dist_threshold = clean_obj_by_ae(
    clean_obj, inconsistent_obj, reconstructed_df, error_df, latent_df, error_threshold, lzc_df, lzc_threshold = clean_obj_by_ae(
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
    )

    io.save_pickle(clean_obj, f'{output_save_dir}/{fit_name}_clean_obj.pkl')
    io.save_pickle(inconsistent_obj, f'{output_save_dir}/{fit_name}_rejected_obj.pkl')
    error_threshold_df = pd.DataFrame({'threshold': [error_threshold]})
    error_threshold_df.to_csv(f'{output_save_dir}/{fit_name}_error_threshold.csv')
    error_df.to_csv(f'{output_save_dir}/{fit_name}_error.csv')
    lzc_df.to_csv(f'{output_save_dir}/{fit_name}_lzc.csv')
    lzc_threshold_df = pd.DataFrame({'lzc_threshold': [lzc_threshold]})
    lzc_threshold_df.to_csv(f'{output_save_dir}/{fit_name}_lzc_threshold.csv')

    # latent_df.to_csv(f'{output_save_dir}/{fit_name}_latent.csv')
    # mahalanobis_dist_df.to_csv(f'{output_save_dir}/{fit_name}_mahalanobis_distance.csv')
    # mahalanobis_dist_threshold_df = pd.DataFrame({'threshold': [mahalanobis_dist_threshold]})
    # mahalanobis_dist_threshold_df.to_csv(f'{output_save_dir}/{fit_name}_mahalanobis_distance_threshold.csv')
    thispl.draw_reconstruction_error_distribution(
        error_df=error_df,
        threshold=error_threshold,
        save_dir=output_save_dir,
        save_name=f'{fit_name}_reconstruction_error_distribution.pdf'
    )
    # thispl.draw_mahalanobis_distance_distribution(
    #     mahalanobis_dist_df=mahalanobis_dist_df,
    #     threshold=mahalanobis_dist_threshold,
    #     save_dir=output_save_dir,
    #     save_name=f'{fit_name}_mahalanobis_distance_distribution.pdf'
    # )

    thispl.draw_lzc_distribution(
        lzc_df=lzc_df,
        threshold=lzc_threshold,
        save_dir=output_save_dir,
        save_name=f'{fit_name}_lzc_distribution.pdf'
    )

    if save_reconstruction:
        reconstructed_df.to_csv(f'{output_save_dir}/{fit_name}_reconstructed_signals.csv')


def clean_an_obj_with_trained_model(
    obj: Union[dict, str],
    save_name: str,
    save_dir: str = '.',
    att: str = 'signal',
    down_sample_to: int = 1000,
    seed: int = 41,
    model_name: Literal['LSTMAutoencoder', 'CNNAutoencoder', 'ShapeAwareAutoencoder'] = 'ShapeAwareAutoencoder',
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
