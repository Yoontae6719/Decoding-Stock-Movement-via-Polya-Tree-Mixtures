# ./data_provider/data_loader.py

import os
import numpy as np
import pandas as pd
import glob
import re
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer, PowerTransformer
import warnings
warnings.filterwarnings('ignore')

class Dataset_SNP(Dataset):
    def __init__(self, args, root_path, flag='train', data_path='SNP.csv', scale=True, stop_loss = 0):
        
        self.args = args

        assert flag in ['train', 'test', 'val']
        type_map = {'train': 0, 'val': 1, 'test': 2}
        self.set_type = type_map[flag]

        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.stop_loss = stop_loss
        self.__read_data__()

    def __read_data__(self):
        # Step 1. Get dataset
        df_raw = pd.read_feather(os.path.join(self.root_path, self.data_path))
        df_raw['Date'] = pd.to_datetime(df_raw['Date'])

        df_Close = df_raw['Close']
        self.data_Close = df_Close.values
        df_raw = df_raw.drop(["Close"], axis = 1)

        
        df_raw = df_raw.drop(["Stock"], axis=1)
        df_raw = df_raw.dropna()

        df_raw[['Y', 'Y_2', 'Y_3', 'Y_4',"Y_5"]] = df_raw[['Y', 'Y_2', 'Y_3', 'Y_4',"Y_5"]].apply(lambda x: x.map({'SELL': 0, 'BUY': 1}))
        
        # Step 2. Train // valid // test
        num_train = df_raw[(df_raw['Date'] >= '2020-01-01') & (df_raw['Date'] <= '2022-12-31')].shape[0]
        num_vali =  df_raw[(df_raw['Date'] >= '2023-01-01') & (df_raw['Date'] <= '2023-12-31')].shape[0]
        num_test =  df_raw[(df_raw['Date'] >= '2024-01-01')].shape[0] 
        
        border1s = [0,
                    num_train,
                    len(df_raw) - num_test]
        
        border2s = [num_train,
                    num_train + num_vali,
                    len(df_raw)]
        
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        # Step 3. Scaling
        df_x = df_raw.drop(['Y', 'Y_2', 'Y_3', 'Y_4',"Y_5"], axis=1).drop(["Date"], axis=1)
        if self.scale:
            train_data = df_x[border1s[0]:border2s[0]]
            quantile_train = np.copy(train_data.values).astype(np.float64)
            stds = np.std(quantile_train, axis=0, keepdims=True)
            noise_std = 1e-3 / np.maximum(stds, 1e-3)
            quantile_train += noise_std * np.random.randn(*quantile_train.shape)
            self.scaler = QuantileTransformer(output_distribution='normal', random_state=1004)
            self.scaler.fit(quantile_train)  
            data_all = self.scaler.transform(df_x.values)  
        else:
            data_all = df_x.values


        self.data_x = data_all[border1:border2]  
        if self.stop_loss == 0:
            df_y = df_raw[['Y']].values
            self.data_y = df_y[border1:border2]
        elif self.stop_loss == 2:
            df_y = df_raw[['Y_2']].values
            self.data_y = df_y[border1:border2]
        elif self.stop_loss == 3:
            df_y = df_raw[['Y_3']].values
            self.data_y = df_y[border1:border2]
        elif self.stop_loss == 4:
            df_y = df_raw[['Y_4']].values
            self.data_y = df_y[border1:border2]
        elif self.stop_loss == 5:
            df_y = df_raw[['Y_5']].values
            self.data_y = df_y[border1:border2]
        else:
            raise ValueError('You should choose stop_loss as 0, 2, 3, or 4.')
            
        self.stock_Close = self.data_Close[border1:border2]

    def __getitem__(self, index):
        stock_x = self.data_x[index]
        stock_y = self.data_y[index]
        stock_Close = self.data_Close[index]
        return torch.tensor(stock_x, dtype=torch.float32), torch.tensor(stock_y, dtype=torch.long), torch.tensor(stock_Close, dtype=torch.float32)


    def __len__(self):
        return len(self.data_x)
