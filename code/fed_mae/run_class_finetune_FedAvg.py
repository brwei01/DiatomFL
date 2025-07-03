# --------------------------------------------------------
# Based on MAE code bases
# Integrate MAE for Federated Learning
# Reference: https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------
def set_seed(seed: int):

     random.seed(seed)  # Python random module
     np.random.seed(seed)  # NumPy
     torch.manual_seed(seed)  # PyTorch
     torch.cuda.manual_seed(seed)  # CUDA
     torch.cuda.manual_seed_all(seed)  # Multi-GPU
     torch.backends.cudnn.deterministic = True  # Ensure deterministic CUDA
     torch.backends.cudnn.benchmark = False  # Disable benchmark mode for reproducibility
     os.environ['PYTHONHASHSEED'] = str(seed)  # 确保哈希种子固
import os
os.environ["PYTHONHASHSEED"] = "0"
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import torch
import random
import numpy as np

set_seed(0)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)  # 强制启用确定性，遇到不支持的操作直接报错
import argparse
import datetime
import json

import time
from pathlib import Path

import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
import timm
assert timm.__version__ == "0.3.2" # version check
from copy import deepcopy

import os
import sys
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)

import models_vit
#import fed_mae.models_vit as models_vit
from engine_for_finetuning import train_one_epoch
#from fed_mae.engine_for_finetuning import train_one_epoch
import util.misc as misc
from util.FedAvg_utils import Partial_Client_Selection, valid, average_model, weighted_average_model
from util.data_utils import DatasetFLFinetune, create_dataset_and_evalmetrix
from util.start_config import print_options
############################################
import seaborn as sn
import pandas as pd
import matplotlib.pyplot as plt

import warnings
# Suppress all warnings
warnings.filterwarnings("ignore")
###############################################
os.environ['NCCL_BLOCKING_WAIT'] = '1'
os.environ['NCCL_ASYNC_ERROR_HANDLING'] = '1'

def get_args():
    parser = argparse.ArgumentParser('Fed-MAE fine-tuning for image classification', add_help=False)
    parser.add_argument('--batch_size', default=16, type=int,
                        help='Batch size per GPU (effective batch size is batch_size * accum_iter * # gpus')
    parser.add_argument('--save_ckpt_freq', default=100, type=int)
    parser.add_argument('--accum_iter', default=1, type=int,
                        help='Accumulate gradient iterations (for increasing the effective batch size under memory constraints)')
    parser.add_argument('--patience', default=3, type=int)
    # Model parameters
    parser.add_argument('--model_name', default='mae', type=str)
    parser.add_argument('--model', default='vit_large_patch16', type=str, metavar='MODEL',
                        help='Name of model to train')
    parser.add_argument('--input_size', default=224, type=int,
                        help='images input size')
    parser.add_argument('--drop_path', type=float, default=0.1, metavar='PCT',
                        help='Drop path rate (default: 0.1)')
    parser.add_argument('--disable_eval_during_finetuning', action='store_true', default=False)

    # Optimizer parameters
    parser.add_argument('--clip_grad', type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (absolute lr)')
    parser.add_argument('--blr', type=float, default=1e-3, metavar='LR',
                        help='base learning rate: absolute_lr = base_lr * total_batch_size / 256')
    parser.add_argument('--layer_decay', type=float, default=0.75,
                        help='layer-wise lr decay from ELECTRA/BEiT')

    parser.add_argument('--min_lr', type=float, default=1e-6, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0')

    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR')

    # Augmentation parameters
    parser.add_argument('--color_jitter', type=float, default=None, metavar='PCT',
                        help='Color jitter factor (enabled only when not using Auto/RandAug)')
    parser.add_argument('--aa', type=str, default='rand-m9-mstd0.5-inc1', metavar='NAME',
                        help='Use AutoAugment policy. "v0" or "original". " + "(default: rand-m9-mstd0.5-inc1)'),
    parser.add_argument('--smoothing', type=float, default=0.1,
                        help='Label smoothing (default: 0.1)')

    # * Random Erase params
    parser.add_argument('--reprob', type=float, default=0.25, metavar='PCT',
                        help='Random erase prob (default: 0.25)')
    parser.add_argument('--remode', type=str, default='pixel',
                        help='Random erase mode (default: "pixel")')
    parser.add_argument('--recount', type=int, default=1,
                        help='Random erase count (default: 1)')
    parser.add_argument('--resplit', action='store_true', default=False,
                        help='Do not random erase first (clean) augmentation split')

    # * Mixup params
    parser.add_argument('--mixup', type=float, default=0,
                        help='mixup alpha, mixup enabled if > 0.')
    parser.add_argument('--cutmix', type=float, default=0,
                        help='cutmix alpha, cutmix enabled if > 0.')
    parser.add_argument('--cutmix_minmax', type=float, nargs='+', default=None,
                        help='cutmix min/max ratio, overrides alpha and enables cutmix if set (default: None)')
    parser.add_argument('--mixup_prob', type=float, default=1.0,
                        help='Probability of performing mixup or cutmix when either/both is enabled')
    parser.add_argument('--mixup_switch_prob', type=float, default=0.5,
                        help='Probability of switching to cutmix when both mixup and cutmix enabled')
    parser.add_argument('--mixup_mode', type=str, default='batch',
                        help='How to apply mixup/cutmix params. Per "batch", "pair", or "elem"')

    # * Finetuning params
    parser.add_argument('--finetune', default='',
                        help='finetune from checkpoint')
    parser.add_argument('--global_pool', action='store_true')
    parser.set_defaults(global_pool=True)
    parser.add_argument('--cls_token', action='store_false', dest='global_pool',
                        help='Use class token instead of global pool for classification')

    # Dataset parameters\
    parser.add_argument('--data_set', default='Diatom', type=str,
                        help='ImageNet dataset path') # choices=['Retina', 'Derm', 'COVIDfl']
    parser.add_argument('--data_path', default='/../../data/Retina', type=str, 
                        help='dataset path')
    parser.add_argument('--nb_classes', default=2, type=int,
                        help='number of the classification types')
    parser.add_argument('--output_dir', default='',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default=None,
                        help='path where to tensorboard log')
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--resume', default='',
                        help='resume from checkpoint')
    
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--dist_eval', action='store_true', default=False,
                        help='Enabling distributed evaluation (recommended during training for faster monitor')
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--pin_mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no_pin_mem', action='store_false', dest='pin_mem')
    parser.set_defaults(pin_mem=True)

    # distributed training parameters
    parser.add_argument('--world_size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--local_rank', default=0, type=int)
    parser.add_argument('--sync_bn', default=False, action='store_true')
    parser.add_argument('--dist_on_itp', action='store_true')
    parser.add_argument('--dist_url', default='env://',
                        help='url used to set up distributed training')
    
    # FL related parameters
    parser.add_argument("--n_clients", default=4, type=int, help="Number of clients")
    parser.add_argument("--E_epoch", default=1, type=int, help="Local training epoch in FL")
    parser.add_argument("--max_communication_rounds", default=100, type=int,
                        help="Total communication rounds.")
    parser.add_argument("--num_local_clients", default=-1, choices=[4, -1], type=int,
                        help="Num of local clients joined in each FL train. -1 indicates all clients")
    parser.add_argument("--split_type", type=str,default="central", help="Which data partitions to use")


    return parser.parse_args()


def worker_init_fn(worker_id):
    # 计算唯一种子：全局种子 + 进程号 + Worker编号
    worker_seed = args.seed + misc.get_rank() * 1000 + worker_id
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)
'''
def worker_init_fn(worker_id):
    seed = torch.initial_seed() % (2**32)  # 获取当前线程的初始种子，并将其限定在 32 位整数范围内。
    np.random.seed(seed)                   # 使用该种子设置 NumPy 的随机数生成器。
    random.seed(seed)                      # 使用该种子设置 Python 内置的随机数生成器。
'''
def collect_client_losses(train_stats):
    """
    This function extracts and returns the training loss from the training statistics.
    """
    return train_stats.get('loss', 0.0)
def main(args, model):
    set_seed(args.seed)
    misc.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    misc.fix_random_seeds(args)

    cudnn.benchmark = True

    # prepare dataset
    create_dataset_and_evalmetrix(args, mode='finetune')

    if args.disable_eval_during_finetuning:
        dataset_val = None
    else:
        dataset_val = DatasetFLFinetune(args=args, phase='test')

    if args.eval:
        dataset_test = DatasetFLFinetune(args=args, phase='test')
    else:
        dataset_test = None

    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()

    if args.dist_eval:
        if len(dataset_val) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
            sampler_val = torch.utils.data.DistributedSampler(
                dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=True)  # shuffle=True to reduce monitor bias
    else:
        sampler_val = torch.utils.data.SequentialSampler(dataset_val)
    sampler_test = torch.utils.data.SequentialSampler(dataset_test)

    if dataset_val is not None:
        data_loader_val = torch.utils.data.DataLoader(
            dataset_val, sampler=sampler_val,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )
    else:
        data_loader_val = None

    if dataset_test is not None:
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, sampler=sampler_test,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=args.pin_mem,
            drop_last=False,
            worker_init_fn=worker_init_fn
        )
    else:
        data_loader_test = None

    # configuration for FedAVG, prepare model, optimizer, scheduler 
    model_all, optimizer_all, criterion_all, loss_scaler_all, mixup_fn_all = Partial_Client_Selection(args, model, mode='finetune')
    model_avg = deepcopy(model).cpu()

    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None


# ---------- Train! (use different clients)    
    print("=============== Running fine-tuning ===============")
    tot_clients = args.dis_cvs_files
    print('total_clients: ', tot_clients)
    epoch = -1

    start_time = time.time()
    max_accuracy = 0.0
    best_val_loss = float('inf')

    while True:
        print('epoch: ', epoch)
        epoch += 1
#################  losses collecting  ############################
        client_losses = {}  # 初始化一个字典用于存储每个客户端的训练损失
###############################################################
        print("=============== Phase 1: Client-Side KD ===============")
        print(f"args.num_local_clients: {args.num_local_clients}")
        print(f"tot_clients: {tot_clients}")
        print(f"args.proxy_clients: {args.proxy_clients}")
        print(f"len(args.dis_cvs_files): {len(args.dis_cvs_files)}")
        # randomly select partial clients


        # 在main函数中的客户端选择部分：
        if args.num_local_clients != len(args.dis_cvs_files):
            # 只在主进程生成客户端列表，然后广播
            if global_rank == 0:
                cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()
            else:
                cur_selected_clients = [None] * args.num_local_clients
            # 广播客户端列表到所有进程
            dist.broadcast_object_list(cur_selected_clients, src=0)
        else:
            cur_selected_clients = args.proxy_clients
            print('cur_selected_clients: ', cur_selected_clients)

        '''
        if args.num_local_clients == len(args.dis_cvs_files):
            # just use all the local clients
            cur_selected_clients = args.proxy_clients
            print('cur_selected_clients: ', cur_selected_clients)
        else:
            cur_selected_clients = np.random.choice(tot_clients, args.num_local_clients, replace=False).tolist()
'''
        # Get the quantity of clients joined in the FL train for updating the clients weights
        cur_tot_client_Lens = 0
        for client in cur_selected_clients:
            cur_tot_client_Lens += args.clients_with_len[client]
            print('cur_tot_client_Lens: ', cur_tot_client_Lens)

        for cur_single_client, proxy_single_client in zip(cur_selected_clients, args.proxy_clients):
            print('cur_single_client: ', cur_single_client)
            print('proxy_single_client: ', proxy_single_client)

            args.single_client = cur_single_client
            args.clients_weightes[proxy_single_client] = args.clients_with_len[cur_single_client] / cur_tot_client_Lens

            # ---- get dataset for each client for pretraining finetuning 
            dataset_train = DatasetFLFinetune(args=args, phase='train')

            num_tasks = misc.get_world_size()
            global_rank = misc.get_rank()

            print(f'=========client: {proxy_single_client} ==============')
            if args.distributed:
                sampler_train = torch.utils.data.DistributedSampler(
                    dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True, seed=args.seed
                )
            else:    
                sampler_train = torch.utils.data.RandomSampler(dataset_train)
                    
            print("Sampler_train = %s" % str(sampler_train))

            data_loader_train = torch.utils.data.DataLoader(
                dataset_train, sampler=sampler_train,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                pin_memory=args.pin_mem,
                drop_last=True,
                worker_init_fn=worker_init_fn
            )

            # ---- prepare model for a client
            model = model_all[proxy_single_client]
            optimizer = optimizer_all[proxy_single_client]
            criterion = criterion_all[proxy_single_client]
            loss_scaler = loss_scaler_all[proxy_single_client]
            mixup_fn = mixup_fn_all[proxy_single_client]

            if args.distributed:
                model_without_ddp = model.module
            else:
                model_without_ddp = model

            n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)

            total_batch_size = args.batch_size * args.accum_iter * misc.get_world_size()
            num_training_steps_per_inner_epoch = len(dataset_train) // total_batch_size

            if args.lr is None:  # only base_lr is specified
                args.lr = args.blr * total_batch_size / 256

            print("base lr: %.2e" % (args.lr * 256 / total_batch_size))
            print("actual lr: %.2e" % args.lr)

            print("Batch size = %d" % total_batch_size)
            print("Number of training examples = %d" % len(dataset_train))
            print("Number of training training per epoch = %d" % num_training_steps_per_inner_epoch)

            if args.distributed:
                data_loader_train.sampler.set_epoch(epoch)
            if log_writer is not None:
                log_writer.set_step(epoch)

            if args.eval:
                ########################################################################
                misc.load_model(args=args, model_without_ddp=model_without_ddp,
                                optimizer=optimizer, loss_scaler=loss_scaler, model_ema=None)
                ########################################################################
                test_stats, cf_matrix, macro_avg_accuracy = valid(args, model, data_loader_test)
                # print(f"Accuracy of the network on the {len(dataset_test)} test images: {test_stats['acc1']:.1f}%")
                print(f"Micro Average Classification Accuracy: {test_stats['micro_acc1']:.2f}%")
                class_accuracy = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
                macro_accuracy = np.mean(class_accuracy)

                # print("\nClass-wise Classification Accuracy:")
                # for i, acc in enumerate(class_accuracy):
                # print(f"Class {i}: {acc:.4f}")
                print(f"Macro Average Classification Accuracy: {macro_accuracy * 100:.2f}%")

                ##############################################################################
                # print("diagonal", cf_matrix.diagonal())
                cf = np.around(cf_matrix / cf_matrix.sum(1)[:, np.newaxis], 2)
                df_cm = pd.DataFrame(cf, index=[i for i in range(args.nb_classes)],
                                     columns=[i for i in range(args.nb_classes)])
                # plt.figure(figsize=(15, 7))
                plt.figure(figsize=(30, 15))
                sn.heatmap(df_cm, cmap="viridis", annot=True, fmt='4g')
                plt.savefig(os.path.join(args.output_dir, 'cf-matrix.png'))
                model.cpu()

                exit(0)

            for inner_epoch in range(args.E_epoch):
                # ============ training one epoch of BEiT  ============
                train_stats = train_one_epoch(
                        model, criterion, data_loader_train,
                        optimizer, device, epoch, loss_scaler,
                        args.clip_grad, proxy_single_client,
                        mixup_fn,
                        log_writer=log_writer,
                        args=args
                        )
                ##########################################
                # 收集每个客户端的损失
                client_loss = collect_client_losses(train_stats)
                if proxy_single_client not in client_losses:
                    client_losses[proxy_single_client] = []
                client_losses[proxy_single_client].append(client_loss)
                ##########################################

                # ============ writing logs ============
                log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                             'client': cur_single_client,
                             'epoch': epoch,
                             'inner_epoch': inner_epoch,
                             'n_parameters': n_parameters}

                if args.output_dir and misc.is_main_process():
                    if log_writer is not None:
                        log_writer.flush()
                    with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                        f.write(json.dumps(log_stats) + "\n")
       ########################################################################################
        # 计算每个客户端的平均损失
        avg_client_losses = [np.mean(losses) for losses in client_losses.values()]
        ########################################################################################
        # =========== model average and eval ============ 
        # average model
        average_model(args, model_avg, model_all)
        ########################################################################################
        #weighted_average_model(args, model_avg, model_all, avg_client_losses)
        ########################################################################################
        # save the global model
        # TO CHECK: global model is the same for each client?
        #if args.output_dir:
        #    if (epoch + 1) % args.save_ckpt_freq == 0 or epoch + 1 == args.max_communication_rounds:
        #        misc.save_model(
        #            args=args, model=model_avg, model_without_ddp=model_avg,
        #            optimizer=optimizer, loss_scaler=loss_scaler, epoch=epoch)
       #
        if data_loader_val is not None:
            model_avg.to(args.device)
            ########################################################################
            test_stats, cf_matrix, macro_avg_accuracy = valid(args, model_avg, data_loader_val)
            #print(f"Accuracy of the network on the {len(dataset_val)} validation images: {test_stats['acc1']:.1f}%")
            print(
                f"Micro Average Classification Accuracy: {test_stats['micro_acc1']:.2f}%")
            class_accuracy = cf_matrix.diagonal() / cf_matrix.sum(axis=1)
            macro_accuracy = np.mean(class_accuracy)
            print(f"Macro Average Classification Accuracy: {macro_accuracy * 100:.2f}%")
            ############################################################################### Check for improvement

            # print("diagonal", cf_matrix.diagonal())
            cf = np.around(cf_matrix / cf_matrix.sum(1)[:, np.newaxis], 2)
            df_cm = pd.DataFrame(cf, index=[i for i in range(args.nb_classes)],
                                 columns=[i for i in range(args.nb_classes)])
            # plt.figure(figsize=(15, 7))
            plt.figure(figsize=(30, 15))
            sn.heatmap(df_cm, cmap="viridis", annot=True, fmt='4g')
            plt.savefig(os.path.join(args.output_dir, 'cf-matrix.png'))

            if max_accuracy < macro_avg_accuracy:
                max_accuracy = macro_avg_accuracy
                #if args.output_dir:
                #    misc.save_model(
                #        args=args, model=model_avg,
                #        model_without_ddp=model_without_ddp, optimizer=optimizer,
                #        loss_scaler=loss_scaler, epoch="macro_best", model_ema=None)

            print(f'Max macro accuracy: {max_accuracy*100:.2f}%')

            #################################
            if log_writer is not None:
                log_writer.update(test_acc1=test_stats['micro_acc1'], head="perf", step=epoch)
                log_writer.update(test_acc5=test_stats['acc5'], head="perf", step=epoch)
                log_writer.update(test_loss=test_stats['loss'], head="perf", step=epoch)
            
            log_stats = {**{f'test_{k}': v for k, v in test_stats.items()},
                         'epoch': epoch,
                         'n_parameters': n_parameters}
            
        if args.output_dir and misc.is_main_process():
                if log_writer is not None:
                    log_writer.flush()
                with open(os.path.join(args.output_dir, "log.txt"), mode="a", encoding="utf-8") as f:
                    f.write(json.dumps(log_stats) + "\n")
        
        model_avg.to('cpu')
        
        print('global_step_per_client: ', args.global_step_per_client[proxy_single_client])
        print('t_total: ', args.t_total[proxy_single_client])
        
        if args.global_step_per_client[proxy_single_client] >= args.t_total[proxy_single_client]:
            total_time = time.time() - start_time
            total_time_str = str(datetime.timedelta(seconds=int(total_time)))
            print('Training time {}'.format(total_time_str))
            break


if __name__ == '__main__':
    args = get_args()

    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    
    model = models_vit.__dict__[args.model](
        num_classes=args.nb_classes,
        drop_path_rate=args.drop_path,
        global_pool=args.global_pool,
        )

    print_options(args, model)
    
    # set train val related paramteres
    args.best_acc = {}
    args.current_acc = {}
    args.current_test_acc = {}
    
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
        
    # run finetuning
    main(args, model)
