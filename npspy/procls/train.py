from typing import Dict, Optional, Tuple, Union, Literal, List
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import math
from torch.amp import autocast

from .models import NTXentLoss, CNNTransformerEncoder, HierarchicalCNNTransformerEncoder, MoCo
from .. import machine_learning as ml

def adjust_learning_rate(optimizer, step, total_steps, base_lr, warmup_steps):
    if step < warmup_steps:
        lr = base_lr * (step + 1) / warmup_steps
    else:
        # 余弦退火公式
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class Trainer():
    def __init__(
        self, 
        device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
        lr: float = 0.005,
        epochs: int = 200,
        temperature: float = 0.5,
    ) -> None:
        self.device = ml.set_device(device)
        self.model = CNNTransformerEncoder().to(self.device)
        #self.model = torch.compile(self.model, mode="reduce-overhead")
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = NTXentLoss(temperature=temperature)
        self.epochs = epochs
        self.lr = lr

        print(f'Model has total parameter number: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.6f} M')
    
    def fit(
        self,
        train_loader: DataLoader,
        name: str = 'train_for_something',
    ):
        import wandb
        if os.path.exists(f'{name}_best_model.pth'):
            raise ValueError(f'Best model {name}_best_model.pth already exists')
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hailinpan1988-bgi-group",
            # Set the wandb project where this run will be logged.
            project="proclr",
        )

        # 计算每个 epoch 的步数
        steps_per_epoch = len(train_loader)

        # 计算总步数
        total_steps = steps_per_epoch * self.epochs

        # 计算 Warmup 步数 (例如前 10% 的步数用于预热)
        warmup_ratio = 0.1
        warmup_steps = int(total_steps * warmup_ratio)

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        min_epoch_loss = np.inf
        self.model.train()
        log_file = open(f'{name}_log.txt', 'w')
        step_now = 0
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, (aug1, aug2, _)  in enumerate(train_loader):
                step_now += 1
                aug1 = aug1.to(self.device)
                aug2 = aug2.to(self.device)
                self.optimizer.zero_grad()
                #with autocast(device_type='cuda', dtype=torch.bfloat16):
                z1, embedding1 = self.model(aug1)
                z2, embedding2 = self.model(aug2)
                loss = self.criterion(z1.float(), z2.float())
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                adjust_learning_rate(self.optimizer, step_now, total_steps, self.lr, warmup_steps)
                total_loss += loss.item()
                log_file.write(f'step [{step_now}], step_loss: {loss.item():.4f}\n')
                log_file.flush()
                run.log({'steps': step_now, "step_loss": loss.item(), "lr": self.optimizer.param_groups[0]['lr']})
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.epochs}], loss: {avg_loss:.4f}')
            log_file.write(f'Epoch [{epoch+1}/{self.epochs}], loss: {avg_loss:.4f}\n')
            log_file.flush()
            run.log({'epoch': epoch+1, "loss": avg_loss})
            if avg_loss < min_epoch_loss:
                min_epoch_loss = avg_loss
                torch.save(self.model.state_dict(), f'{name}_best_model.pth')
                print(f'Best model saved at epoch [{epoch+1}/{self.epochs}], loss: {avg_loss:.4f}')
        run.finish()
        log_file.close()
    
    def predict_embedding(
        self,
        test_loader: DataLoader,
        model_path: str = 'train_for_somethihg_best_model.pth',
    ):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            embeddings, labels, read_ids = [], [], []
            for batch_idx, (x, label, read_id) in enumerate(test_loader):
                x = x.to(self.device)
                z, embedding = self.model(x)
                embeddings.append(embedding)
                labels.extend(label)
                read_ids.extend(read_id)
        embeddings = torch.cat(embeddings, dim=0)
        emb_df = pd.DataFrame(embeddings.cpu().numpy(), columns=[f'emb{i}' for i in range(embeddings.shape[1])])
        emb_df['label'] = labels
        emb_df.index = read_ids
        return emb_df



class TrainerMoCo():
    def __init__(
        self, 
        device: Literal['cpu', 'cuda', 'mps'] = 'cuda',
        lr: float = 0.005,
        epochs: int = 200,
        K: int = 65536,
        m: float = 0.999,
        temperature: float = 0.07,
    ) -> None:
        self.device = ml.set_device(device)
        self.model = MoCo(T=temperature, K=K, m=m).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.encoder_q.parameters(), lr=lr)
        self.epochs = epochs
        self.lr = lr

        print(f'Model has total parameter number: {sum(p.numel() for p in self.model.parameters() if p.requires_grad)/1e6:.6f} M')
    
    def fit(
        self,
        train_loader: DataLoader,
        name: str = 'train_for_something',
    ):
        import wandb
        if os.path.exists(f'{name}_best_model.pth'):
            raise ValueError(f'Best model {name}_best_model.pth already exists')
        run = wandb.init(
            # Set the wandb entity where your project will be logged (generally your team name).
            entity="hailinpan1988-bgi-group",
            # Set the wandb project where this run will be logged.
            project="proclr",
        )

        # 计算每个 epoch 的步数
        steps_per_epoch = len(train_loader)

        # 计算总步数
        total_steps = steps_per_epoch * self.epochs

        # 计算 Warmup 步数 (例如前 10% 的步数用于预热)
        warmup_ratio = 0.1
        warmup_steps = int(total_steps * warmup_ratio)

        print(f"Steps per epoch: {steps_per_epoch}")
        print(f"Total training steps: {total_steps}")
        print(f"Warmup steps: {warmup_steps}")

        min_epoch_loss = np.inf
        self.model.train()
        log_file = open(f'{name}_log.txt', 'w')
        step_now = 0
        for epoch in range(self.epochs):
            total_loss = 0.0
            for batch_idx, (aug1, aug2, _)  in enumerate(train_loader):
                step_now += 1
                aug1 = aug1.to(self.device)
                aug2 = aug2.to(self.device)
                loss = self.model(aug1, aug2)
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.encoder_q.parameters(), max_norm=1.0)
                self.optimizer.step()
                adjust_learning_rate(self.optimizer, step_now, total_steps, self.lr, warmup_steps)
                total_loss += loss.item()
                log_file.write(f'step [{step_now}], step_loss: {loss.item():.4f}\n')
                log_file.flush()
                run.log({'steps': step_now, "step_loss": loss.item(), "lr": self.optimizer.param_groups[0]['lr']})
            avg_loss = total_loss / len(train_loader)
            print(f'Epoch [{epoch+1}/{self.epochs}], loss: {avg_loss:.4f}')
            log_file.write(f'Epoch [{epoch+1}/{self.epochs}], loss: {avg_loss:.4f}\n')
            log_file.flush()
            run.log({'epoch': epoch+1, "loss": avg_loss})
            if avg_loss < min_epoch_loss:
                min_epoch_loss = avg_loss
                torch.save(self.model.state_dict(), f'{name}_best_model.pth')
                print(f'Best model saved at epoch [{epoch+1}/{self.epochs}], loss: {avg_loss:.4f}')
        run.finish()
        log_file.close()
    
    def predict_embedding(
        self,
        test_loader: DataLoader,
        model_path: str = 'train_for_somethihg_best_model.pth',
    ):
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        with torch.no_grad():
            embeddings, labels, read_ids = [], [], []
            for batch_idx, (x, label, read_id) in enumerate(test_loader):
                x = x.to(self.device)
                embedding = self.model.predict_embedding(x)
                embeddings.append(embedding)
                labels.extend(label)
                read_ids.extend(read_id)
        embeddings = torch.cat(embeddings, dim=0)
        emb_df = pd.DataFrame(embeddings.cpu().numpy(), columns=[f'emb{i}' for i in range(embeddings.shape[1])])
        emb_df['label'] = labels
        emb_df.index = read_ids
        return emb_df