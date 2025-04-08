# run.py
import argparse
import os
import torch
import torch.backends
import random
import numpy as np
import optuna  
from exp.exp_main import Exp_Main
from exp.exp_trading import Exp_Trading

import multiprocessing
import csv
import sys

if __name__ == '__main__':
    fix_seed = 2025
    random.seed(fix_seed)
    torch.manual_seed(fix_seed)
    np.random.seed(fix_seed)

    parser = argparse.ArgumentParser(description='treeMoE')
    
    # basic config
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, required=True, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='MOE', help='model name')

    # data loader
    parser.add_argument('--data', type=str, required=True, default='SNP', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./dataset', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='SNP.csv', help='data file')
    parser.add_argument('--stop_loss', type=float, default=0, help='stop loss ratio')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scale', type=bool, default=True, help='scale param')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=5, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0003, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='cosine', help='adjust learning rate')
    parser.add_argument('--temperature_scheduler', type=bool, default=True, help='temperature scheduler')

    # GPU
    parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')

    # Models param
    parser.add_argument('--dim_input', type=int, default=437, help='input dimension')
    parser.add_argument('--num_classes', type=int, default=2, help='output dimension ')
    parser.add_argument('--max_depth', type=int, default=5, help='tree max depth')
    parser.add_argument('--hidden_dim_expert', type=int, default=32, help='hidden dim for expert')
    parser.add_argument('--alpha_fs', type=float, default=1.0, help='Beta-Bernoulli alpha')
    parser.add_argument('--beta_fs', type=float, default=1.0, help='Beta-Bernoulli beta')
    parser.add_argument('--use_gating_mlp', type=int, default=0, help='use_gating_mlp')
    parser.add_argument('--gating_mlp_hidden', type=int, default=32, help='Gating mlp hidden dim')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='max_grad_norm')

    parser.add_argument('--initial_temp', type=float, default=1.0, help='initial_temp')
    parser.add_argument('--final_temp', type=float, default=0.1, help='final_temp')
    parser.add_argument('--anneal_epochs', type=int, default=5, help='anneal_epochs')
    parser.add_argument('--schedule_type', type=str, default="linear", help='schedule_type')

    parser.add_argument('--use_feature_selection', action='store_true', help='use multiple gpus', default=True)
    parser.add_argument('--use_leaf_feature_selector_only', type=int, help='use multiple gpus', default=1)
    parser.add_argument('--lambda_KL', type=float, default=0.01, help='lambda_KL')

    # Optuna settings
    parser.add_argument('--top_k', type=int, default=20, help='number of tuning trials') 

    args = parser.parse_args()

    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device('cuda:{}'.format(args.gpu))
        print('Using GPU')
    else:
        if hasattr(torch.backends, "mps"):
            args.device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
        else:
            args.device = torch.device("cpu")
        print('Using cpu or mps')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '')
        device_ids = args.devices.split(',')
        args.device_ids = [int(id_) for id_ in device_ids]
        args.gpu = args.device_ids[0]

    print('Args in experiment:')
    print(args)

    # (A) Training (is_training = 1)
    if args.is_training == 1:
        Exp = Exp_Main
        for ii in range(args.itr):
            setting = '{}_{}_md{}_hde{}_alp{}_use{}_mlp{}_gde_{}_lr_{}_des{}'.format(
                args.model,
                args.data,
                args.max_depth,
                args.hidden_dim_expert,
                int(args.alpha_fs * 10),
                args.use_leaf_feature_selector_only,
                args.use_gating_mlp,
                args.gating_mlp_hidden,
                args.lradj,
                args.des
            ) + f"_{ii}"

            exp = Exp(args)  
            print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
            exp.fit(setting)

            torch.cuda.empty_cache()

        
    else:
        Exp = Exp_Main        
        setting = '{}_{}_md{}_hde{}_alp{}_use{}_mlp{}_gde_{}_lr_{}_des{}_0'.format(
            args.model,
            args.data,
            args.max_depth,
            args.hidden_dim_expert,
            int(args.alpha_fs * 10),
            args.use_leaf_feature_selector_only,
            args.use_gating_mlp,
            args.gating_mlp_hidden,
            args.lradj,
            args.des
        )
        
        exp = Exp(args)  # Create experiment
        
        checkpoint_path = os.path.join(args.checkpoints, setting, 'checkpoint.pth')
        print("Loading model from", checkpoint_path)
        
        if os.path.exists(checkpoint_path):
            exp.model.load_state_dict(torch.load(checkpoint_path, map_location=args.device))
            print("Checkpoint loaded successfully")
        else:
            print(f"Checkpoint not found at {checkpoint_path}")
            print("Searching for similar checkpoints...")
            for root, dirs, files in os.walk(args.checkpoints):
                for dir in dirs:
                    if args.data in dir and args.model in dir:
                        check_file = os.path.join(args.checkpoints, dir, 'checkpoint.pth')
                        if os.path.exists(check_file):
                            print(f"  Found: {dir}")
            sys.exit(1)
        
        # Run only XAI analysis
        print(f"Running standalone XAI analysis for {args.data}")
        results = exp.xai(setting=setting)
        print("XAI analysis completed successfully!")
