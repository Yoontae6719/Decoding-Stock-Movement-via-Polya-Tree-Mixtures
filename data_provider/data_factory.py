import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data_provider.data_loader import Dataset_SNP

# 데이터셋 등록
data_dict = {
    'SNP': Dataset_SNP,
}

def data_provider(args, flag):
    if args.data not in data_dict:
        raise ValueError(f"Unknown data type: {args.data}")
    
    shuffle_flag = not (flag.lower() == 'test')
    
    drop_last = False

    DataClass = data_dict[args.data]
    data_set = DataClass(
        args=args,
        root_path=args.root_path,
        data_path=args.data_path,
        flag=flag,
        scale=args.scale,
        stop_loss=args.stop_loss
    )

    print(f"{flag} dataset size:", len(data_set))
    
    data_loader = DataLoader(
        data_set,
        batch_size=args.batch_size,
        shuffle=shuffle_flag,
        num_workers=args.num_workers,
        drop_last=drop_last )
    
    return data_set, data_loader
