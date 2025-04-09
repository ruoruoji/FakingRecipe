import argparse
import os
import random
import warnings
import numpy as np
import torch
from run import Run

# 核心功能：作为项目入口脚本，负责：
# 1. 命令行参数解析
# 2. 运行环境配置
# 3. 全局配置初始化
# 4. 主流程调度

# 参数解析器配置
parser = argparse.ArgumentParser()
parser.add_argument('--dataset', default='fakett', help='fakett/fakesv')
parser.add_argument('--mode', default='inference_test', help='train/inference_test')
parser.add_argument('--epoches', type=int, default=30)
parser.add_argument('--batch_size', type = int, default=128)
parser.add_argument('--early_stop', type=int, default=5)
parser.add_argument('--seed', type=int, default=2023)
parser.add_argument('--gpu', default='0')
parser.add_argument('--lr', type=float)
parser.add_argument('--alpha',type=float)
parser.add_argument('--beta',type=float)
parser.add_argument('--inference_ckp', help='input path of inference checkpoint when mode is inference')
parser.add_argument('--path_ckp', default= './checkpoints/')
parser.add_argument('--path_tb', default= './tensorboard/')
args = parser.parse_args()

# 环境初始化（确保实验可复现性）
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

print (args)

# 配置字典构建（参数桥接）
config={
    'dataset':args.dataset,
    'mode':args.mode,
    'epoches':args.epoches,
    'batch_size':args.batch_size,
    'early_stop':args.early_stop,
    'device':args.gpu,
    'lr':args.lr,
    'alpha':args.alpha,
    'beta':args.beta,
    'inference_ckp':args.inference_ckp,
    'path_ckp':args.path_ckp,
    'path_tb':args.path_tb
}

# 主流程启动
Run(config = config).main()