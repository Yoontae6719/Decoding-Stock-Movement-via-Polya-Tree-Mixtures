import argparse
import os
import torch
import torch.backends
import random
import numpy as np
from exp.exp_main import Exp_Main

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
    parser.add_argument('--stop_loss', type=float, default=0, help='stop loss ratio [0,2,3,4,5]')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    parser.add_argument('--scale', type=bool, default=True, help='scale param')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--batch_size', type=int, default=512, help='batch size of train input data')
    parser.add_argument('--patience', type=int, default=10, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
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
    parser.add_argument('--use_gating_mlp', type=bool, default=False, help='use_gating_mlp')
    parser.add_argument('--max_grad_norm', type=int, default=5.0, help='max_grad_norm')

    
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

    Exp = Exp_Main
    
    for ii in range(args.itr):
        setting = '{}_{}_md{}_hde{}_al{}_be{}_mlp{}_{}'.format(
                args.model,
                args.data,
                args.max_depth,
                args.hidden_dim_expert,
                int(args.alpha_fs * 10),
                int(args.beta_fs*10),
                args.use_gating_mlp,
                args.des, ii)
    
        exp = Exp(args)  # set experiments
        print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
        exp.fit(setting)

        torch.cuda.empty_cache()



