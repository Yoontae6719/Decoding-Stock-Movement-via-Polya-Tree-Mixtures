from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from models import treeMoE

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import os
import time

import warnings
import matplotlib.pyplot as plt
import numpy as np

from utils.tools import EarlyStopping, adjust_learning_rate, TemperatureScheduler
from sklearn.metrics import matthews_corrcoef, confusion_matrix

class Exp_Main(Exp_Basic):
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)
        self.history = {
            'epoch': [],
            'train_loss': [],
            'train_acc': []}
        

    def _build_model(self):
        model_dict = {
            'treeMoE': treeMoE,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def fit(self, setting):
        sched = TemperatureScheduler(  initial_temp=1.0,
                                        final_temp=2.0,
                                        anneal_epochs=5,
                                        schedule_type="linear",
                                    )

        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')        
        
        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)


        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        model_optim = self._select_optimizer()
        
        for epoch in range(self.args.train_epochs):
            self.model.train()
            total_loss = 0.0
            total_correct = 0
            total_samples = 0

            if self.args.temperature_scheduler is not None:
                current_temp = sched.get_temp(epoch)
            else:
                current_temp = 0.5

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                model_optim.zero_grad()
                logits = self.model(x_batch, temperature=current_temp)

                ce_loss = F.cross_entropy(logits, y_batch)
                reg_loss = self.model.regularization_loss()
                loss = ce_loss + reg_loss

                loss.backward()
                if self.args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

                total_loss += loss.item() * x_batch.size(0)
                preds = torch.argmax(logits, dim=1)
                correct = (preds == y_batch).sum().item()
                total_correct += correct
                total_samples += x_batch.size(0)

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples

            self.history['epoch'].append(epoch+1)
            self.history['train_loss'].append(avg_loss)
            self.history['train_acc'].append(avg_acc)
    
            vali_loss, valid_acc = self.evaluate(vali_loader)
            test_loss, test_acc = self.evaluate(test_loader)

            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "  f"Temp={current_temp:.3f} train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}")
            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "  f"valid_loss={vali_loss:.4f}, valid_acc={valid_acc:.4f}")
            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}")
            
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, epoch + 1, self.args)
        
        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model


    def evaluate(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        test_temp = 0.1

        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)
                logits = self.model(x_batch, temperature=test_temp)
                ce = F.cross_entropy(logits, y_batch, reduction='sum').item()
                preds = torch.argmax(logits, dim=1)
                correct = (preds == y_batch).sum().item()

                total_loss += ce
                total_correct += correct
                total_samples += x_batch.size(0)

        return total_loss/total_samples, total_correct/total_samples

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x, temperature=0.1)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()






















