from types import new_class
from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
import joblib
import torch
import torch.cuda
import torch.nn.functional as F
from numba import jit, prange
from torch.autograd import Function
from numba import cuda
import os

from .soft_dtw_cuda import SoftDTW
from .. import plot as pl
from .tools import get_signals_for_dtw
from .. import io
from .. import machine_learning as ml

def get_cluster_data_from_an_obj(
    obj: Union[dict, str],
    down_sample_to: int = 500,
    cut_flank: int = 20,
    norm_by_polyT: bool = True,
):
    obj = io.read_obj_or_pickle(obj)
    data = get_signals_for_dtw(obj, down_sample_to=down_sample_to, norm_by_polyT=norm_by_polyT)
    data = data[:,cut_flank:down_sample_to-cut_flank]
    data = data[:,:,None].astype(np.float32)
    return data # N, T, 1



def compute_soft_dtw_barycenter(
    series_list: torch.Tensor, 
    init_center: torch.Tensor, 
    sdtw_module: SoftDTW, 
    n_iterations: int = 10, 
    lr: float = 0.01, 
    batch_size: int = 1000,
):
    """
    计算一个簇内时间序列的 Soft-DTW Barycenter
    series_list: [N, T, D] 的 torch.Tensor
    init_center: [1, T, D] 的初始中心
    batch_size: 每批处理的序列数量
    """
    center = init_center.clone().detach().requires_grad_(True)
    optimizer = torch.optim.Adam([center], lr=lr)

    for _ in range(n_iterations):
        N = series_list.size(0)  # 总序列数

        indices = torch.randperm(N)
        shuffled_data = series_list[indices]

        # 分批处理
        for i in range(0, N, batch_size):
            # total_loss = 0.0

            batch_end = min(i + batch_size, N)
            batch_series = shuffled_data[i:batch_end]  # [B, T, D] where B <= batch_size
            if batch_series.size(0) < batch_size:
                continue
            
            # Expand center to match batch size
            expanded_center = center.expand(batch_series.size(0), -1, -1)  # [B, T, D]
            
            # Calculate Soft-DTW distances for this batch
            batch_losses = sdtw_module(batch_series, expanded_center)  # [B]
            
            # Sum up the losses for this batch
            batch_loss = batch_losses.sum()  # scalar tensor
            # total_loss += batch_loss
        
            # Perform backward pass on the total accumulated loss
            optimizer.zero_grad()
            batch_loss.backward()
            optimizer.step()

    return center.detach()


# ----------------------------------------------------------------------------------------------------------------------
# GPU 加速的 Soft-DTW K-Means 实现
# ----------------------------------------------------------------------------------------------------------------------
class SoftDTWKMeans:
    def __init__(
        self, 
        n_clusters: int = 2, 
        gamma: float = 0.05, 
        max_iter: int = 100, 
        tol: float = 1e-4, 
        use_cuda = True, 
        verbose = True, 
        barycenter_lr: float = 0.01, 
        barycenter_iter: int = 10,
        seed: int = 42,
    ):
        self.n_clusters = n_clusters
        self.gamma = gamma
        self.max_iter = max_iter
        self.tol = tol
        self.use_cuda = use_cuda and torch.cuda.is_available()
        self.verbose = verbose
        self.barycenter_lr = barycenter_lr
        self.barycenter_iter = barycenter_iter
        self.seed = seed
        
        self.sdtw = SoftDTW(use_cuda=self.use_cuda, gamma=gamma, normalize=False)
        
    def _init_centroids(self, X):
        """
        使用欧氏距离的 K-means 初始化聚类中心
        X: [N, T, D] 的 torch.Tensor
        """
        # 将tensor转为numpy数组，形状为 [N, T*D] 以便于K-means处理
        X_np = X.cpu().numpy()  # [N, T, D]
        N, T, D = X_np.shape
        X_flat = X_np.reshape(N, T * D)  # [N, T*D]
        
        # 使用sklearn的K-means进行聚类
        kmeans_euclidean = KMeans(n_clusters=self.n_clusters, random_state=self.seed, n_init=10)
        labels = kmeans_euclidean.fit_predict(X_flat)  # [N]
        
        # 获取每个簇的中心点
        euclidean_centroids = kmeans_euclidean.cluster_centers_  # [K, T*D]
        euclidean_centroids = euclidean_centroids.reshape(self.n_clusters, T, D)  # [K, T, D]
        
        # 对于每个簇，从该簇中随机选择一个序列作为初始Soft-DTW中心
        centroids = torch.zeros(self.n_clusters, T, D, dtype=X.dtype).cuda()
        
        np.random.seed(self.seed)
        for k in range(self.n_clusters):
            # 找到属于第k个簇的所有样本索引
            cluster_mask = (labels == k)
            cluster_indices = np.where(cluster_mask)[0]
            
            if len(cluster_indices) > 0:
                # 从该簇中随机选择一个样本
                selected_idx = np.random.choice(cluster_indices)
                centroids[k] = X[selected_idx]
            else:
                # 如果某个簇为空，随机选择一个序列
                print(f"Warning: Cluster {k} is empty during initialization. Selecting a random sequence.")
                random_idx = np.random.randint(0, N)
                centroids[k] = X[random_idx]
        
        return centroids

    def fit_predict(self, X: Union[torch.Tensor, np.ndarray], batch_size: int = 1000):
        ml.seed_everything(42)
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()

        N, T, D = X.shape
        centroids = self._init_centroids(X)  # [K, T, D]
        
        for iteration in range(self.max_iter):
            # Step 1: 计算距离矩阵
            dist_matrix = torch.zeros(N, self.n_clusters, device=X.device)
            for i in range(0, N, batch_size):
                sub_X = X[i:i+batch_size, :, :]
                if self.use_cuda:
                    sub_X = sub_X.cuda()
                for k in range(self.n_clusters):
                    C_k = centroids[k:k+1].expand(len(sub_X), -1, -1)  # [N, T, D]
                    dists = self.sdtw(sub_X, C_k)  # [N]
                    dist_matrix[i:i+batch_size, k] = dists

            # Step 2: 分配标签
            labels = torch.argmin(dist_matrix, dim=1)  # [N]

            # Step 3: 更新聚类中心（使用真正的 Soft-DTW Barycenter）
            new_centroids = torch.zeros_like(centroids)
            for k in range(self.n_clusters):
                mask = (labels == k)
                if mask.sum() > 0:
                    cluster_series = X[mask]  # [N_k, T, D]
                    if self.use_cuda:
                        cluster_series = cluster_series.cuda()
                    init_c = centroids[k:k+1]  # [1, T, D]
                    new_centroid = compute_soft_dtw_barycenter(
                        cluster_series, init_c, self.sdtw,
                        n_iterations=self.barycenter_iter, lr=self.barycenter_lr
                    ).squeeze(0)  # [T, D]
                    new_centroids[k] = new_centroid
                else:
                    # 空簇：保持原中心不变
                    new_centroids[k] = centroids[k]

            # Step 4: 检查收敛
            centroid_shift = torch.norm(centroids - new_centroids, dim=(1,2)).max().item()
            centroids = new_centroids

            if self.verbose:
                print(f"Iter {iteration+1}: max centroid shift = {centroid_shift:.6f}")

            if centroid_shift < self.tol:
                if self.verbose:
                    print("Converged!")
                break

        # 最终分配标签
        final_dist_matrix = torch.zeros(N, self.n_clusters, device=X.device)
        for k in range(self.n_clusters):
            for i in range(0, N, batch_size):
                sub_X = X[i:i+batch_size, :, :]
                if self.use_cuda:
                    sub_X = sub_X.cuda()
                C_k = centroids[k:k+1].expand(len(sub_X), -1, -1)
                final_dist_matrix[i:i+batch_size, k] = self.sdtw(sub_X, C_k)
        final_labels = torch.argmin(final_dist_matrix, dim=1)

        self.centroids_ = centroids.cpu().numpy()
        return final_labels.cpu().numpy()
    
    def set_label_name(self, label_cluster_map: dict):
        self.label_cluster_map = label_cluster_map
    
    def predict(self, X, return_dist_matrix=False, batch_size: int = 1000):
        """
        预测新数据点的簇标签
        X: [N, T, D] 的 torch.Tensor 或 numpy array
        返回: [N] 的簇标签数组
        """
        if not hasattr(self, 'centroids_'):
            raise ValueError("模型尚未训练，请先调用 fit_predict 方法")
        
        if isinstance(X, np.ndarray):
            X = torch.from_numpy(X).float()
  
        
        # 将聚类中心转换为tensor
        centroids_tensor = torch.from_numpy(self.centroids_).float()
        if self.use_cuda:
            centroids_tensor = centroids_tensor.cuda()
        
        N = X.shape[0]
        dist_matrix = torch.zeros(N, self.n_clusters, device=X.device)
        # 计算新数据点到各个聚类中心的距离
        for k in range(self.n_clusters):
            for i in range(0, N, batch_size):
                sub_X = X[i:i+batch_size, :, :]
                if self.use_cuda:
                    sub_X = sub_X.cuda()
                C_k = centroids_tensor[k:k+1].expand(len(sub_X), -1, -1)  # [N, T, D]
                dists = self.sdtw(sub_X, C_k)  # [N]
                dist_matrix[i:i+batch_size, k] = dists
        
        # 找到最近的聚类中心
        labels = torch.argmin(dist_matrix, dim=1)  # [N]
        if hasattr(self, 'label_cluster_map'):
            label_cluster_map = self.label_cluster_map
        else:
            label_cluster_map = None

        if return_dist_matrix:
            if label_cluster_map is None:
                dist_matrix = pd.DataFrame(dist_matrix.cpu().numpy(), columns=[f'cluster_{k}' for k in range(self.n_clusters)])
            else:
                dist_matrix = pd.DataFrame(dist_matrix.cpu().numpy(), columns=[label_cluster_map[k] for k in range(self.n_clusters)])
            return dist_matrix

        labels = labels.cpu().numpy()
        if label_cluster_map is not None:
            labels = [label_cluster_map[label] for label in labels]
            labels = np.array(labels)
        
        return labels

    def predict_proba(self, X, batch_size: int = 1000, temp: float = 1.0, return_dist_matrix=False,):
        """
        预测新数据点属于每个类别的概率
        X: [N, T, D] 的 torch.Tensor 或 numpy array
        返回: [N, K] 的概率矩阵，其中第i行第j列表示第i个样本属于第j类的概率
        """
        dist_matrix = self.predict(X, return_dist_matrix=True, batch_size=batch_size,)

        # 使用负距离作为logits，通过softmax计算概率
        # 负距离：距离越远，值越小；距离越近，值越大
        dist_matrix_torch = torch.from_numpy(dist_matrix.values).float()
        self.temp = temp
        logits = -dist_matrix_torch / self.temp  # 除以温度参数
        probabilities = torch.softmax(logits, dim=1)  # 对每个样本在所有类别上应用softmax
        if return_dist_matrix:
            return dist_matrix
        
        probabilities = probabilities.cpu().numpy()
        probabilities = pd.DataFrame(probabilities, columns=dist_matrix.columns)
        probabilities['state'] = probabilities.idxmax(axis=1)
        return probabilities
    
    # def fit_predict_proba(self, X, batch_size: int = 1000, temp: float = 1.0):
    #     """
    #     训练模型并预测训练数据属于每个类别的概率
    #     X: [N, T, D] 的 torch.Tensor 或 numpy array
    #     返回: [N, K] 的概率矩阵
    #     """
    #     # 首先运行完整的训练过程
    #     _ = self.fit_predict(X, batch_size=batch_size)
        
    #     # 返回训练数据的概率分布
    #     probabilities = self.predict_proba(X, batch_size=batch_size, temp=temp)
        
    #     return probabilities
    
    def save_model(self, save_dir: str = './', model_file_name: str = 'kmeans_model.pkl'):
        pl.create_dir_if_not_exist(save_dir)
        joblib.dump(self, os.path.join(save_dir, model_file_name))

def load_model_by_pkl_file(
    pkl_file: str,
):
    model = joblib.load(pkl_file)
    return model

