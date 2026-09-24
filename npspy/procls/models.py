from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.nn.functional as F

from .tools import time_warp_augmentation, signal_shift_augmentation, add_noise_augmentation

class TimeSeriesDataset(Dataset):
    def __init__(self, data_df):
        """
        初始化数据集
        :param data_df: pandas DataFrame，最后一列是 label，其余是时间序列数据
        """
        # 提取特征数据（除去最后一列），并转换为 float32 的 numpy 数组
        self.data = data_df.iloc[:, :-1].values.astype(np.float32)
        # 提取标签（最后一列）
        self.labels = data_df.iloc[:, -1].values
    
    def __len__(self):
        # 返回样本的总数量
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. 根据索引获取当前这一行的 1D 时间序列和标签
        series = self.data[idx]
        label = self.labels[idx]
        
        # 2. 对该时间序列调用两次 time_warp_augmentation，得到两个不同的增强样本
        # 注意：每次调用都会因为随机性产生不同的扭曲结果
        aug_series_1 = time_warp_augmentation(series)
        aug_series_1 = signal_shift_augmentation(aug_series_1)
        aug_series_1 = add_noise_augmentation(aug_series_1)
        
        aug_series_2 = time_warp_augmentation(series)
        aug_series_2 = signal_shift_augmentation(aug_series_2)
        aug_series_2 = add_noise_augmentation(aug_series_2)
        
        # 3. 将 numpy 数组转换为 PyTorch 的 Tensor
        # 通常时间序列模型期望的输入维度是 (序列长度,) 或 (通道数, 序列长度)
        # 这里保持 (序列长度,)，模型 forward 时可以根据需要 unsqueeze
        tensor_1 = torch.from_numpy(aug_series_1)
        tensor_2 = torch.from_numpy(aug_series_2)
        
        # 4. 返回两个增强后的输出以及对应的标签
        return tensor_1, tensor_2, label

class TimeSeriesTestDataset(Dataset):
    def __init__(self, data_df, with_label: bool = True):
        """
        初始化数据集
        :param data_df: pandas DataFrame，最后一列是 label，其余是时间序列数据
        """
        self.with_label = with_label
        if with_label:
            # 提取特征数据（除去最后一列），并转换为 float32 的 numpy 数组
            self.data = data_df.iloc[:, :-1].values.astype(np.float32)
            # 提取标签（最后一列）
            self.labels = data_df.iloc[:, -1].values
        else:
            self.data = data_df.iloc[:, :].values.astype(np.float32)
        self.read_ids = list(data_df.index)
    
    def __len__(self):
        # 返回样本的总数量
        return len(self.data)
    
    def __getitem__(self, idx):
        # 1. 根据索引获取当前这一行的 1D 时间序列和标签
        series = self.data[idx]
        if self.with_label:
            label = self.labels[idx]
        else:
            label = None
        read_id = self.read_ids[idx]
        series = torch.from_numpy(series)
        return series, label, read_id



class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5):
        super().__init__()
        self.temperature = temperature
        self.cosine_sim = nn.CosineSimilarity(dim=-1)

    def forward(self, z_i, z_j):
        """
        计算对比损失
        :param z_i: 第一个增强视图的 embedding, shape: [batch_size, hidden_dim]
        :param z_j: 第二个增强视图的 embedding, shape: [batch_size, hidden_dim]
        """
        batch_size = z_i.size(0)
        # 1. 将两个视图拼接到一起，形成一个 2N 的批次
        representations = torch.cat([z_i, z_j], dim=0)  # [2*batch_size, hidden_dim]
        
        # 2. 计算相似度矩阵 (2N x 2N)
        # 利用矩阵乘法计算余弦相似度：sim(a,b) = (a*b) / (||a||*||b||)
        norm_repr = F.normalize(representations, dim=1)
        similarity_matrix = torch.matmul(norm_repr, norm_repr.T) / self.temperature  # [2*batch_size, 2*batch_size]
        
        # 3. 构建掩码，找到正样本对的位置
        # 对角线上的元素是样本与自身的相似度，需要排除
        mask = torch.eye(2 * batch_size, dtype=torch.bool).to(z_i.device)
        # 正样本对的索引：(i, i+batch_size) 和 (i+batch_size, i)
        pos_mask = torch.zeros(2 * batch_size, 2 * batch_size, dtype=torch.bool).to(z_i.device)
        pos_mask[:batch_size, batch_size:] = torch.eye(batch_size, dtype=torch.bool)
        pos_mask[batch_size:, :batch_size] = torch.eye(batch_size, dtype=torch.bool)
        
        # 4. 提取正样本相似度 (分子) 和 负样本相似度 (分母)
        pos_sim = similarity_matrix[pos_mask].view(2 * batch_size, 1)  # [2*batch_size, 1]
        # 排除自身和正样本，剩下的都是负样本
        neg_sim = similarity_matrix[~mask & ~pos_mask].view(2 * batch_size, -1) # [2*batch_size, 2*batch_size-2]
        
        # 5. 计算 InfoNCE 损失
        logits = torch.cat([pos_sim, neg_sim], dim=1)  # [2*batch_size, 2*batch_size-1]
        labels = torch.zeros(2 * batch_size, dtype=torch.long).to(z_i.device)
        
        loss = F.cross_entropy(logits, labels)
        return loss
        

class CNNLSTMEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=1, 
        hidden_dim=128, 
        projection_dim=64,
        num_layers=2
    ):
        super().__init__()
        
        self.cnn = nn.Sequential(
            # Layer 1: 1000 -> 500
            nn.Conv1d(input_dim, 32, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            
            # Layer 2: 500 -> 250
            nn.Conv1d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            
            # Layer 3: 250 -> 125
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            
            # Layer 4 (修复版): 125 -> 62
            nn.Conv1d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
        )
        
        # 增加一个 1x1 卷积，专门用于将 CNN 的固定通道(128) 映射到 LSTM 的 hidden_dim
        self.cnn_to_lstm_conv = nn.Conv1d(128, hidden_dim, kernel_size=1)
        
        # LSTM 层
        self.lstm = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, 
                            num_layers=num_layers, batch_first=True, bidirectional=False)
        
        # 投影头
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        # x: [batch_size, seq_len] -> [16, 1000]
        
        x = x.unsqueeze(1)  # [16, 1, 1000]
        
        # 经过 CNN 提取局部特征并下采样
        x = self.cnn(x)     # [16, 128, 62] (通道数固定为128，长度≈62)
        
        # 通过 1x1 卷积将通道数映射为 hidden_dim
        x = self.cnn_to_lstm_conv(x) # [16, hidden_dim, 62]
        
        # 转换为 LSTM 需要的 [batch, seq_len, channels]
        x = x.transpose(1, 2)  # [16, 62, hidden_dim]
        
        # 经过 LSTM 提取全局时序依赖
        lstm_out, (h_n, c_n) = self.lstm(x) 
        # lstm_out: [16, 62, hidden_dim]
        
        # # 取 LSTM 最后一个时间步的输出作为序列的 Embedding
        # embedding = lstm_out[:, -1, :]  # [16, hidden_dim]
        # 使用LSTM所有时间步的平均
        embedding = lstm_out.mean(dim=1) 
        
        # 经过投影头得到最终的对比特征
        z = self.projection_head(embedding)  # [16, projection_dim]
        
        return z, embedding


class CNNTransformerEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=1, 
        hidden_dim=256, 
        projection_dim=128,
        num_transformer_layers=6,   # Transformer 编码器的层数
        nhead=8,                    # 多头注意力的头数 (需能被 hidden_dim 整除)
        dim_feedforward=1024,        # FFN 隐藏层维度
        dropout=0.1                 # Dropout 概率
    ):
        super().__init__()
        
        # ==========================
        # 1. CNN 局部特征提取模块
        # ==========================
        self.cnn = nn.Sequential(
            # Layer 1: 1000 -> 500
            nn.Conv1d(input_dim, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            
            # Layer 2: 500 -> 250
            nn.Conv1d(64, 128, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            
            # Layer 3: 250 -> 125
            nn.Conv1d(128, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
            
            # Layer 4: 125 -> 62
            nn.Conv1d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2, stride=2), 
        )
        
        # 使用 1x1 卷积将 CNN 输出的通道数映射到 Transformer 的 embed_dim
        self.cnn_to_trans_conv = nn.Conv1d(256, hidden_dim, kernel_size=1)

        # ==========================
        # 可学习的位置编码 (Learned Positional Encoding)
        # ==========================
        # 假设经过 CNN 下采样后最大序列长度为 100（实际为62，留点余量即可）
        max_seq_len = 100 
        self.position_embedding = nn.Parameter(torch.zeros(max_seq_len, 1,hidden_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)  # 初始化以加速收敛
        
        # ==========================
        # 2. Transformer 编码器模块 (替换 LSTM)
        # ==========================
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout, 
            activation='gelu',      # GELU 激活函数在时序任务中表现更好
            batch_first=False       # 保持默认 False，即 [seq_len, batch, dim]
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_transformer_layers
        )
        
        # ==========================
        # 3. 投影头模块
        # ==========================
        # self.projection_head = nn.Sequential(
        #     nn.Linear(hidden_dim, hidden_dim),
        #     nn.ReLU(),
        #     nn.Linear(hidden_dim, projection_dim)
        # )

        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),  # 加入 BN 稳定训练
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        """
        :param x: 原始时间序列 [batch_size, seq_len]，例如 [16, 1000]
        :return z: 对比学习用的投影特征 [batch_size, projection_dim]
        :return embedding: 下游任务用的通用表示 [batch_size, hidden_dim]
        """
        # 增加通道维度: [batch_size, seq_len] -> [batch_size, 1, seq_len]
        x = x.unsqueeze(1) 
        
        # 经过 CNN 提取局部特征并下采样: [batch_size, 128, L']
        x = self.cnn(x) 
        
        # 通过 1x1 卷积调整通道数: [batch_size, hidden_dim, L']
        x = self.cnn_to_trans_conv(x) 
        
        # 转换为 Transformer 需要的格式: [L', batch_size, hidden_dim]
        x = x.permute(2, 0, 1)      
        
        # 注入位置编码
        # x 的形状现在是 [seq_len, batch_size, hidden_dim]
        # position_embedding 的形状是 [max_seq_len, 1, hidden_dim]
        # PyTorch 广播机制会自动将位置编码加到每一个 batch 样本上
        seq_len = x.size(0)
        x = x + self.position_embedding[:seq_len, :, :]

        # 经过 Transformer Encoder 建立全局上下文依赖: [L', batch_size, hidden_dim]
        trans_out = self.transformer_encoder(x)  
        
        # 转换回 [batch_size, L', hidden_dim] 并进行时间步维度的平均池化
        trans_out = trans_out.permute(1, 0, 2)    # [batch_size, L', hidden_dim]
        embedding = trans_out.mean(dim=1)         # [batch_size, hidden_dim]
        
        # 经过投影头得到最终的对比特征: [batch_size, projection_dim]
        z = self.projection_head(embedding)
        
        return z, embedding

        

class HierarchicalCNNTransformerEncoder(nn.Module):
    def __init__(
        self, 
        input_dim=1, 
        hidden_dim=256, 
        projection_dim=128,
        num_transformer_layers=6,   
        nhead=8,                    
        dim_feedforward=1024,       
        dropout=0.1                 
    ):
        super().__init__()
        
        # ==========================
        # 1. 共享的 CNN 基础层 (逐级递进)
        # ==========================
        # 所有卷积 stride=1，长度减半完全依赖后续的 MaxPool1d
        layer_configs = [
            (input_dim, 64, 7, 1, 3),   # Layer 1: L -> L -> L//2
            (64, 128, 5, 1, 2),         # Layer 2: L//2 -> L//2 -> L//4
            (128, 256, 5, 1, 2),        # Layer 3: L//4 -> L//4 -> L//8
            (256, 256, 3, 1, 1),        # Layer 4: L//8 -> L//8 -> L//16
        ]
        
        self.conv_layers = nn.ModuleList()
        for in_ch, out_ch, k, s, p in layer_configs:
            self.conv_layers.append(nn.Sequential(
                nn.Conv1d(in_ch, out_ch, kernel_size=k, stride=s, padding=p),
                nn.BatchNorm1d(out_ch),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2, stride=2)  
            ))
            
        # 每个尺度需要一个 1x1 卷积将通道映射到统一的 hidden_dim
        self.scale_projections = nn.ModuleList([
            nn.Conv1d(cfg[1], hidden_dim, kernel_size=1) for cfg in layer_configs
        ])
        
        # 每个尺度分配一个独立的 Transformer Encoder
        self.transformer_encoders = nn.ModuleList()
        for _ in range(4):
        # for _ in range(1):
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, 
                nhead=nhead, 
                dim_feedforward=dim_feedforward, 
                dropout=dropout, 
                activation='gelu', 
                batch_first=False       # [seq_len, batch, hidden_dim]
            )
            self.transformer_encoders.append(
                nn.TransformerEncoder(encoder_layer, num_layers=num_transformer_layers)
            )

        # 为每个尺度分配独立的位置编码 (最大长度预留 600 足够覆盖第一层的 500)
        for i in range(4):
            pos_emb = nn.Parameter(torch.zeros(600, 1, hidden_dim))
            nn.init.trunc_normal_(pos_emb, std=0.02)
            setattr(self, f'position_embedding_scale_{i+1}', pos_emb)

        # ==========================
        # 2. 投影头模块
        # ==========================
        self.projection_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, projection_dim)
        )

    def forward(self, x):
        """
        :param x: 原始时间序列 [batch_size, seq_len]，例如 [16, 1000]
        :return z: 对比学习用的投影特征 [batch_size, projection_dim]
        :return embedding: 下游任务用的通用表示 [batch_size, hidden_dim]
        """
        # 增加通道维度: [batch_size, seq_len] -> [batch_size, 1, seq_len]
        x = x.unsqueeze(1) 
        
        scale_embeddings = []
        
        # 遍历 4 个共享的 CNN 层，实现层级递进的特征提取
        for i in range(4):
            # 1. 经过第 i 层共享 CNN (局部特征提取与 2x 下采样)
            x = self.conv_layers[i](x) 
            
            # 2. 通道维度映射到 hidden_dim: [B, C_out, L'] -> [B, hidden_dim, L']
            x_proj = self.scale_projections[i](x)
            
            # 3. 转换为 Transformer 需要的格式: [B, hidden_dim, L'] -> [L', B, hidden_dim]
            x_trans = x_proj.permute(2, 0, 1)      
            
            # 4. 注入对应尺度的位置编码
            seq_len = x_trans.size(0)
            pos_emb = getattr(self, f'position_embedding_scale_{i+1}')
            x_trans = x_trans + pos_emb[:seq_len, :, :]
            
            # 5. 经过当前尺度的 Transformer Encoder: [L', B, hidden_dim] -> [L', B, hidden_dim]
            trans_out = self.transformer_encoders[i](x_trans)  
            # trans_out = self.transformer_encoders[0](x_trans)  
            
            # 6. 转换回 Batch First 并进行 Mean Pooling: [L', B, hidden_dim] -> [B, hidden_dim]
            trans_out = trans_out.permute(1, 0, 2)    
            scale_emb = trans_out.mean(dim=1)         
            scale_embeddings.append(scale_emb)
            
        # ==========================
        # 基于特征均值的自适应尺度加权融合
        # ==========================
        # 将 4 个 [B, hidden_dim] 堆叠为 [4, B, hidden_dim]
        stacked_embeddings = torch.stack(scale_embeddings, dim=0)
        
        # 计算每个尺度特征的绝对值均值作为重要性指标: [4, B, hidden_dim] -> [4, B]
        scale_importance = stacked_embeddings.abs().mean(dim=-1) 
        
        # 沿尺度维度进行 Softmax 归一化，生成动态权重: [4, B]
        dynamic_weights = torch.softmax(scale_importance, dim=0)
        
        # 扩展权重维度以匹配特征: [4, B] -> [4, B, 1]
        dynamic_weights = dynamic_weights.unsqueeze(-1)
        
        # 加权求和: [4, B, hidden_dim] * [4, B, 1] -> sum(dim=0) -> [B, hidden_dim]
        embedding = (stacked_embeddings * dynamic_weights).sum(dim=0)
        
        # 经过投影头得到最终的对比特征: [B, hidden_dim] -> [B, projection_dim]
        z = self.projection_head(embedding)
        
        return z, embedding



class MoCo(nn.Module):
    def __init__(
        self, 
        base_encoder: nn.Module = HierarchicalCNNTransformerEncoder, 
        dim: int = 128, 
        K:int = 65536,
        m: float =0.999, 
        T: float = 0.07
    ):
        super(MoCo, self).__init__()
        
        self.K = K
        self.m = m
        self.T = T
        
        # 1. 创建 Query 编码器 (在线网络)
        self.encoder_q = base_encoder(projection_dim=dim)
        
        # 2. 创建 Key 编码器 (目标网络)
        self.encoder_k = base_encoder(projection_dim=dim)
        
        # 初始化：将 Key 编码器的参数完全复制自 Query 编码器
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False  # Key 编码器不需要梯度
            
        # 3. 创建负样本队列 (FIFO Queue)
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = nn.functional.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """动量更新 Key 编码器参数"""
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data = param_k.data * self.m + param_q.data * (1. - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        """更新队列：入队新 keys，出队最旧的 keys"""
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr)
        
        if ptr + batch_size > self.K:
            remaining = self.K - ptr
            self.queue[:, ptr:] = keys[:remaining].T
            self.queue[:, :batch_size - remaining] = keys[remaining:].T
            ptr = batch_size - remaining
        else:
            self.queue[:, ptr:ptr + batch_size] = keys.T
            ptr = (ptr + batch_size) % self.K
            
        self.queue_ptr[0] = ptr

    def forward(self, im_q, im_k):
        """MoCo 对比学习的前向传播与损失计算"""
        # 1. 计算 Query 特征
        q, _ = self.encoder_q(im_q)  
        q = nn.functional.normalize(q, dim=1)
        
        # 2. 计算 Key 特征 (不计算梯度)
        with torch.no_grad():
            self._momentum_update_key_encoder()  
            k, _ = self.encoder_k(im_k)
            k = nn.functional.normalize(k, dim=1)
            
        # 3. 计算正负样本相似度
        l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
        l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
        
        # 4. 拼接 logits 并计算 InfoNCE Loss
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(im_q.device)
        loss = F.cross_entropy(logits, labels)
        
        # 5. 更新队列
        self._dequeue_and_enqueue(k)
        
        return loss

    @torch.no_grad()
    def predict_embedding(self, x):
        """
        提取用于下游任务的通用特征表示 (Embedding)。
        此函数会跳过投影头，直接返回 Mean Pooling 后的特征。
        
        :param x: 输入的时间序列数据 [batch_size, seq_len]
        :return embedding: 通用特征表示 [batch_size, hidden_dim]
        """
        # 确保模型处于评估模式
        self.eval()
        
        # 通过 Query 编码器前向传播
        # 你的 base_encoder 返回的是 (z, embedding)
        _, embedding = self.encoder_q(x)
        
        return embedding