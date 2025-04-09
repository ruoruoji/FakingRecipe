import collections
import json
import os
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils.dataloader import *
from utils.Trainer import *
from model.FakingRecipe import *

class Run():
    def __init__(self,config):
        # 初始化训练/推理配置参数
        self.dataset = config['dataset']       # 数据集类型 (fakesv/fakett)
        self.mode = config['mode']             # 运行模式 (train/inference_test)
        self.epoches = config['epoches']       # 训练总轮次
        self.batch_size = config['batch_size'] # 批处理大小
        self.early_stop = config['early_stop']  # 早停机制阈值
        self.device = config['device']         # 计算设备 (cuda/cpu)
        self.lr = config['lr']                 # 学习率
        # 多任务损失权重系数
        self.alpha = config['alpha']           # 内容一致性损失权重
        self.beta = config['beta']             # 语义匹配损失权重
        # 系统路径配置
        self.path_ckp=config['path_ckp']       # 模型检查点保存路径
        self.path_tb=config['path_tb']          # TensorBoard日志路径
        self.inference_ckp=config['inference_ckp'] # 推理时模型加载路径

    def get_dataloader(self,data_path):
        """构建数据加载器
        Args:
            data_path: 数据集划分文件路径
        Returns:
            DataLoader: 配置好的数据加载器
        """
        dataset=FakingRecipe_Dataset(data_path,self.dataset)  # 实例化自定义数据集
        collate_fn=collate_fn_FakeingRecipe  # 使用自定义批处理函数
        dataloader = DataLoader(dataset, batch_size=self.batch_size, 
                               shuffle=True, num_workers=0, collate_fn=collate_fn)
        return dataloader

    def main(self):
        """主执行函数，根据模式选择训练或推理"""
        self.model = FakingRecipe_Model(self.dataset)  # 初始化多模态模型
        
        if self.mode=='train':
            # 训练模式下的数据路径配置
            if self.dataset=='fakesv':
                data_split_dir='./data/FakeSV/data-split/'  # FakeSV数据集划分目录
                save_predict_result_path='./predict_result/FakeSV/'  # 预测结果保存路径
            elif self.dataset=='fakett':
                data_split_dir='./data/FakeTT/data-split/'  # FakeTT数据集划分目录
                save_predict_result_path='./predict_result/FakeTT/'
            
            # 加载训练/验证/测试集
            train_data_path=data_split_dir+'vid_time3_train.txt'
            test_data_path=data_split_dir+'vid_time3_test.txt'
            val_data_path=data_split_dir+'vid_time3_val.txt'

            # 初始化数据加载器
            data_load_time_start = time.time()
            train_dataloader=self.get_dataloader(train_data_path)
            test_dataloader=self.get_dataloader(test_data_path)
            val_dataloader=self.get_dataloader(val_data_path)
            dataloaders=dict(zip(['train','test','val'],[train_dataloader,test_dataloader,val_dataloader]))
            
            # 初始化训练器并启动训练流程
            print ('data load time: %.2f' % (time.time() - data_load_time_start))
            trainer=Trainer(
                model=self.model,
                device=self.device,
                lr=self.lr,
                dataloaders=dataloaders,
                epoches=self.epoches,
                model_name='FakingRecipe',
                save_predict_result_path=save_predict_result_path,
                beta_c=self.alpha,  # 内容一致性损失系数
                beta_n=self.beta,   # 语义匹配损失系数
                early_stop=self.early_stop,
                save_param_path=self.path_ckp+self.dataset+"/",
                writer=SummaryWriter(self.path_tb+self.dataset+"/")  # TensorBoard记录器
            )
            ckp_path=trainer.train()  # 执行训练
            result=trainer.test(ckp_path)  # 最终测试
            
        elif self.mode=='inference_test':
            # 推理模式配置
            if self.dataset=='fakesv':
                test_file='./data/FakeSV/data-split/vid_time3_test.txt'
                save_predict_result_path='./predict_result/FakeSV/'
            elif self.dataset=='fakett':
                test_file='./data/FakeTT/data-split/vid_time3_test.txt'
                save_predict_result_path='./predict_result/FakeTT/'
                
            # 初始化推理器并执行预测
            dataloader=self.get_dataloader(test_file)
            inferncer=Inferencer(
                model=self.model,
                device=self.device,
                model_name='FakingRecipe',
                dataset=self.dataset,
                dataloader=dataloader,
                save_predict_result_path=save_predict_result_path
            )
            result=inferncer.inference(self.inference_ckp)  # 加载预训练模型进行推理
