import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import math

plt.switch_backend('agg')

class TemperatureScheduler:
    def __init__(self,
                 initial_temp: float = 1.0,
                 final_temp: float = 0.1,
                 anneal_epochs: int = 10,
                 schedule_type: str = "linear"):
        self.initial_temp = initial_temp
        self.final_temp = final_temp
        self.anneal_epochs = anneal_epochs
        self.schedule_type = schedule_type.lower()

    def get_temp(self, current_epoch: int) -> float:
        if current_epoch >= self.anneal_epochs:
            return self.final_temp

        progress = float(current_epoch) / float(self.anneal_epochs)
        if self.schedule_type == "linear":
            return self.initial_temp + (self.final_temp - self.initial_temp) * progress
        elif self.schedule_type == "exp":
            ratio = (self.final_temp / self.initial_temp) ** progress
            return self.initial_temp * ratio
        else:
            # 기본은 linear
            return self.initial_temp + (self.final_temp - self.initial_temp) * progress


def adjust_learning_rate(optimizer, epoch, args):
    # lr = args.learning_rate * (0.2 ** (epoch // 2))
    if args.lradj == 'type1':
        lr_adjust = {epoch: args.learning_rate * (0.5 ** ((epoch - 1) // 1))}
    
    elif args.lradj == 'type2':
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }

    elif args.lradj == 'type3':
        lr_adjust = {
            6: 5e-5, 9: 1e-5, 12: 5e-6, 15: 1e-6,
            20: 5e-7, 25: 1e-7, 30: 5e-8
        }
        
    elif args.lradj == 'cosine':
        T_max = args.train_epochs 
        #lr = 0.5 * args.learning_rate * (1 + math.cos(math.pi * (epoch - 1) / (T_max - 1)))
        lr = 1e-5 + (args.learning_rate - 1e-5) * (1 + math.cos(math.pi * epoch / T_max)) / 2
        
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f'[Cosine] Updating learning rate to {lr}')
        return 

    elif args.lradj == '3':
        lr_adjust = {epoch: args.learning_rate if epoch < 10 else args.learning_rate*0.1}
    elif args.lradj == '4':
        lr_adjust = {epoch: args.learning_rate if epoch < 15 else args.learning_rate*0.1}
    elif args.lradj == '5':
        lr_adjust = {epoch: args.learning_rate if epoch < 25 else args.learning_rate*0.1}
    elif args.lradj == '6':
        lr_adjust = {epoch: args.learning_rate if epoch < 5 else args.learning_rate*0.1}  
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print('Updating learning rate to {}'.format(lr))


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), path + '/' + 'checkpoint.pth')
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class StandardScaler():
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, data):
        return (data * self.std) + self.mean


def visual(true, preds=None, name='./pic/test.pdf'):
    """
    Results visualization
    """
    plt.figure()
    plt.plot(true, label='GroundTruth', linewidth=2)
    if preds is not None:
        plt.plot(preds, label='Prediction', linewidth=2)
    plt.legend()
    plt.savefig(name, bbox_inches='tight')

def test_params_flop(model,x_shape):
    """
    If you want to thest former's flop, you need to give default value to inputs in model.forward(), the following code can only pass one argument to forward()
    """
    model_params = 0
    for parameter in model.parameters():
        model_params += parameter.numel()
        print('INFO: Trainable parameter count: {:.2f}M'.format(model_params / 1000000.0))
    from ptflops import get_model_complexity_info    
    with torch.cuda.device(0):
        macs, params = get_model_complexity_info(model.cuda(), x_shape, as_strings=True, print_per_layer_stat=True)
        # print('Flops:' + flops)
        # print('Params:' + params)
        print('{:<30}  {:<8}'.format('Computational complexity: ', macs))
        print('{:<30}  {:<8}'.format('Number of parameters: ', params))