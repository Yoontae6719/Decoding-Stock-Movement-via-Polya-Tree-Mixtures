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
import shutil
import optuna
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

        if self.args.use_gpu:
            model = model.to(self.device)
        
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
        sched = TemperatureScheduler( initial_temp=self.args.initial_temp,
                                        final_temp=self.args.final_temp,
                                        anneal_epochs=self.args.anneal_epochs,
                                        schedule_type=self.args.schedule_type,
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
            
            train_preds = []
            train_labels = []
            
            current_temp = sched.get_temp(epoch) if self.args.temperature_scheduler else 0.5

            for x_batch, y_batch in train_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                model_optim.zero_grad()
                
                probs = self.model(x_batch, temperature=current_temp)
                
                # Get NLL
                # probs.shape = (batch_size, num_classes)
                # y_batch.shape = (batch_size,)
                nll_loss = -torch.log(probs[torch.arange(x_batch.size(0)), y_batch] + 1e-8).mean()
                
                reg_loss = self.model.regularization_loss()
                loss = nll_loss# + reg_loss
                
                loss.backward()
                if self.args.max_grad_norm > 0:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                model_optim.step()

                # loss accumulate
                total_loss += loss.item() * x_batch.size(0)
                
                # for acc -> probs -> argmax
                preds = torch.argmax(probs, dim=1)
                correct = (preds == y_batch).sum().item()
                total_correct += correct
                total_samples += x_batch.size(0)
                
                train_preds.extend(preds.cpu().numpy())
                train_labels.extend(y_batch.cpu().numpy())

            avg_loss = total_loss / total_samples
            avg_acc = total_correct / total_samples
            train_mcc = matthews_corrcoef(train_labels, train_preds)

            # validation
            vali_loss, vali_acc, vali_mcc = self.evaluate(vali_loader)
            test_loss, test_acc, test_mcc = self.evaluate(test_loader)

            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "
                  f"Temp={current_temp:.3f} "
                  f"train_loss={avg_loss:.4f}, train_acc={avg_acc:.4f}, train_mcc={train_mcc:.4f}")
            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "
                  f"valid_loss={vali_loss:.4f}, valid_acc={vali_acc:.4f}, valid_mcc={vali_mcc:.4f}")
            print(f"[Epoch {epoch+1}/{self.args.train_epochs}] "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f}, test_mcc={test_mcc:.4f}")

            
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
        preds_list = []
        labels_list = []

        test_temp = 0.1  # (가령 inference시 낮은 온도)
        with torch.no_grad():
            for x_batch, y_batch in data_loader:
                x_batch = x_batch.to(self.device)
                y_batch = y_batch.to(self.device).squeeze(-1)

                # 확률 리턴
                probs = self.model(x_batch, temperature=test_temp)
                # NLL
                nll = -torch.log(probs[torch.arange(x_batch.size(0)), y_batch] + 1e-8).sum().item()

                preds = torch.argmax(probs, dim=1)
                correct = (preds == y_batch).sum().item()
                
                reg_loss = self.model.regularization_loss()

                total_loss += nll# + reg_loss
                total_correct += correct
                total_samples += x_batch.size(0)

                preds_list.extend(preds.cpu().numpy())
                labels_list.extend(y_batch.cpu().numpy())

        avg_loss = total_loss / total_samples
        avg_acc = total_correct / total_samples
        mcc = matthews_corrcoef(labels_list, preds_list)
        return avg_loss, avg_acc, mcc
        

    def predict(self, x):
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device)
            logits = self.model(x, temperature=0.1)
            preds = torch.argmax(logits, dim=1)
        return preds.cpu()
    
       
    def objective(self, trial):
        alpha_fs = trial.suggest_float("alpha", 0.1, 2.0, log=True)
        beta_fs  = trial.suggest_float("beta", 0.1, 2.0, log=True)

        max_depth  = trial.suggest_int("max_depth", 1, 5)

        use_gating_mlp = trial.suggest_categorical("use_gating_mlp", [0, 1])
        if use_gating_mlp:
            gating_mlp_hidden = trial.suggest_categorical("gating_mlp_hidden", [8, 16, 32])
        else:
            gating_mlp_hidden = 0

        hidden_dim_expert = trial.suggest_categorical("hidden_dim_expert", [16, 32, 64])
        initial_temp  = trial.suggest_categorical("initial_temp", [1.0, 1.5, 2.0])
        final_temp    = trial.suggest_categorical("final_temp", [0.5, 1.0, 0.2])
        anneal_epochs = trial.suggest_categorical("anneal_epochs", [5, 10, 20, 30])
        schedule_type = trial.suggest_categorical("schedule_type", ["linear"])
        learning_rate = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 1e-3, 3e-3])

        # update argument
        self.args.alpha_fs = alpha_fs
        self.args.beta_fs  = beta_fs
        self.args.max_depth  = max_depth
        self.args.use_gating_mlp = use_gating_mlp
        self.args.gating_mlp_hidden = gating_mlp_hidden
        self.args.hidden_dim_expert = hidden_dim_expert
        self.args.initial_temp  = initial_temp
        self.args.final_temp    = final_temp
        self.args.anneal_epochs = anneal_epochs
        self.args.learning_rate = learning_rate


        setting = (
            f"trial_{trial.number}_"
            f"alpha{alpha_fs:.3f}_beta{beta_fs:.3f}_lr{learning_rate:.3f}"
            f"depth{max_depth}_mlp{use_gating_mlp}_hdim{hidden_dim_expert}"
        )

        self.model = self._build_model()
        self.fit(setting=setting)

        checkpoint_dir = os.path.join(self.args.checkpoints, setting)
        if os.path.exists(checkpoint_dir):
            shutil.rmtree(checkpoint_dir)


        _, vali_loader = self._get_data(flag='val')
        vali_loss, vali_acc, vali_mcc = self.evaluate(vali_loader)

        if self.args.optuna_metric == "mcc":
            trial.report(vali_mcc, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return vali_mcc

        elif self.args.optuna_metric == "loss":
            trial.report(vali_loss, step=0)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
            return vali_loss

        else:
            raise ValueError("self.args.optuna_metric must be either 'mcc' or 'loss'.")
