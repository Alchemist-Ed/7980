from torch.utils.tensorboard import SummaryWriter
import torch
from torch.nn import functional as F
import re
import os
import json
import numpy as np
import random
import pickle
from tqdm import tqdm
from Evaluation import Evaluation
from MetaLearner_new import MetaLearner
from Task_specific import Task_specific
from collections import namedtuple
from util import *


class Learner(torch.nn.Module):
    def __init__(self, config):
        super(Learner, self).__init__()
        self.output_dir = config['output_dir']
        self.writer = SummaryWriter(log_dir=os.path.join(self.output_dir, 'tensorboard'))
        self.config = config
        self.use_cuda = config['use_cuda']
        self.device = torch.device("cuda" if config['use_cuda'] else "cpu")

        if self.config['dataset'] == 'yelp':
            from EmbeddingInitializer import UserEmbeddingYelp, ItemEmbeddingYelp
            self.item_emb = ItemEmbeddingYelp(config)
            self.user_emb = UserEmbeddingYelp(config)
        elif self.config['dataset'] == 'dbook':
            from EmbeddingInitializer import UserEmbeddingDB, ItemEmbeddingDB
            self.item_emb = ItemEmbeddingDB(config)
            self.user_emb = UserEmbeddingDB(config)

        self.meta_learner = MetaLearner(config)

        self.local_lr = config['local_lr']
        self.outler_lr = config['outler_lr']
        self.cal_metrics = Evaluation()

        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.ml_weight_size = self.meta_learner.get_parameter_size()
        self.task_specific = Task_specific(config, self.ml_weight_size)
        # global update
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config['outler_lr'], weight_decay=config['weight_decay'])    
        
        self.time_spend = {}

    def reset_time(self):
        for k in self.time_spend.keys():
            self.time_spend[k] = 0

    def add_time(self, key, value):
        if key not in self.time_spend:
            self.time_spend[key] = 0
        self.time_spend[key] = self.time_spend[key] + value

    def print_time(self):
        for k, v in self.time_spend.items():
            print(k, v)

    def get_user_item_embedding(self, x):
        shape = list(x.size())
        if shape[0] != 0:
            x = torch.reshape(x, (-1, shape[-1]))
            users_embedding = self.user_emb(x[:, self.config['num_fea_item']:])
            users_embedding = torch.reshape(users_embedding, shape[:-1] + [-1])
            items_embedding = self.item_emb(x[:, 0:self.config['num_fea_item']])
            items_embedding = torch.reshape(items_embedding, shape[:-1] + [-1])
            return users_embedding, items_embedding
        else:
            return None, None
        ### Adding implicit gradient calculation function

    # local update
    def local_update(self, task_data):
        """
        Mete-update the parameters of MetaPathLearner, AggLearner and MetaLearner.
        """
        support_x = task_data['supp_x'].long()
        support_y = task_data['supp_y'].float()
        query_x = task_data['query_x'].long()
        query_y = task_data['query_y'].float()

        # start_time = time.time()
        support_user_emb = self.user_emb(support_x[:, self.config['num_fea_item']:])
        support_item_emb = self.item_emb(support_x[:, 0:self.config['num_fea_item']])
        query_user_emb = self.user_emb(query_x[:, self.config['num_fea_item']:])
        query_item_emb = self.item_emb(query_x[:, 0:self.config['num_fea_item']])
        # self.add_time('user_item_embd_lookup', time.time() - start_time)

        # start_time = time.time()
        task_embd_dict = {}

        task_self_feat, task_self_mask = task_data['task_self']
        task_self_feat = task_self_feat.long()
        task_embd_dict['self_users_embd'], task_embd_dict['self_items_embd'] = self.get_user_item_embedding(task_self_feat)
        task_embd_dict['self_mask'] = task_self_mask

        task_social_feat, task_social_mask = task_data['task_social']
        task_social_feat = task_social_feat.long()
        task_embd_dict['social_users_embd'], task_embd_dict['social_items_embd'] = self.get_user_item_embedding(task_social_feat)
        task_embd_dict['social_mask'] = task_social_mask

        task_implicit_feat, task_implicit_mask = task_data['task_implicit']
        task_implicit_feat = task_implicit_feat.long()
        task_embd_dict['implicit_users_embd'], task_embd_dict['implicit_items_embd'] = self.get_user_item_embedding(task_implicit_feat)
        task_embd_dict['implicit_mask'] = task_implicit_mask

        if self.config['use_coclick']:
            task_coclick_feat, task_coclick_mask = task_data['task_coclick']
            task_coclick_feat = task_coclick_feat.long()
            task_embd_dict['coclick_users_embd'], task_embd_dict['coclick_items_embd'] = self.get_user_item_embedding(task_coclick_feat)
            task_embd_dict['coclick_mask'] = task_coclick_mask

        #Modified LPZ
        def loss_function(task_weights, support_item_emb, support_user_emb, support_y, task_embd_dict):
            task_weights_dict = unflatten_task_weights(task_weights)
            support_pred = self.meta_learner(support_item_emb, support_user_emb, vars_dict=task_weights_dict, **task_embd_dict)
            return F.mse_loss(support_pred, support_y)
        
        # 使用隐式梯度更新参数
        def update_weights_with_implicit_gradient(task_weights, support_item_emb, support_user_emb, support_y, task_embd_dict, local_lr):
            flattened_task_weights = flatten_task_weights(task_weights)
            grad = implicit_gradient(flattened_task_weights, support_item_emb, support_user_emb, support_y, task_embd_dict)
            grad_dict = unflatten_task_weights(grad) 
            #for idx, weight_name in enumerate(self.ml_weight_name):
                #task_weights[weight_name] = task_weights[weight_name] - local_lr * grad[idx]
            for weight_name in task_weights:
                task_weights[weight_name] = task_weights[weight_name] - local_lr * grad_dict[weight_name]
            return task_weights

        # 计算隐式梯度
        def implicit_gradient(task_weights, support_item_emb, support_user_emb, support_y, task_embd_dict):
            # 计算 Hessian 矩阵
            hessian = torch.autograd.functional.hessian(lambda x: loss_function(x, support_item_emb, support_user_emb, support_y, task_embd_dict), task_weights)          
            # 计算 Jacobian 矩阵
            jacobian = torch.autograd.functional.jacobian(lambda x: loss_function(x, support_item_emb, support_user_emb, support_y, task_embd_dict), task_weights)         
            # 计算隐式梯度
            lambda_reg = 1e-6  # 正则化系数
            hessian_reg = hessian + lambda_reg * torch.eye(hessian.shape[0], device=hessian.device)
            implicit_grad = -torch.inverse(hessian_reg) @ jacobian
            return implicit_grad
        
        def flatten_task_weights(task_weights):
            # 将 task_weights 展平并拼接成一个 1 维向量
            flattened_tensors = [tensor.flatten() for tensor in task_weights.values()]
            return torch.cat(flattened_tensors)

        def unflatten_task_weights(flattened_tensor):
            # 将 1 维向量恢复为原始的 task_weights 结构
            task_weights_dict = {}
            start_idx = 0
            for weight_name, original_tensor in task_weights.items():
                num_elements = original_tensor.numel()
                task_weights_dict[weight_name] = flattened_tensor[start_idx:start_idx + num_elements].view(original_tensor.shape)
                start_idx += num_elements
            return task_weights_dict
        

        ml_weights = self.meta_learner.update_parameters()
        task_emb = self.meta_learner.get_task_embd(**task_embd_dict)
        task_weights = self.task_specific(task_emb, ml_weights)

        # 使用隐式梯度更新参数
        fast_ml_weights = update_weights_with_implicit_gradient(task_weights, support_item_emb, support_user_emb, support_y, task_embd_dict, self.local_lr)

        # 多次更新参数
        for _ in range(1, self.config['local_update']):
            fast_ml_weights = update_weights_with_implicit_gradient(fast_ml_weights, support_item_emb, support_user_emb, support_y, task_embd_dict, self.local_lr)

        # 预测 query 集
        query_pred = self.meta_learner(query_item_emb, query_user_emb, vars_dict=fast_ml_weights, **task_embd_dict)
        loss = F.mse_loss(query_pred, query_y)

        self.query_y_real = query_y.data.cpu().numpy()
        self.query_y_pred = query_pred.data.cpu().numpy()
        return loss
                        

    def global_update(self, global_step, batch_data):
        """
        全局更新函数，使用隐式梯度更新本地参数后，计算全局损失并更新全局模型。
        """
        loss_s = []
        for i in range(len(batch_data['supp_xs'])):  # 遍历每个任务
            task_data = {
                'supp_x': batch_data['supp_xs'][i],
                'supp_y': batch_data['supp_ys'][i],
                'query_x': batch_data['query_xs'][i],
                'query_y': batch_data['query_ys'][i],
                'task_self': batch_data['task_self_s'][i],
                'task_social': batch_data['task_social_s'][i],
                'task_implicit': batch_data['task_implicit_s'][i],
                'task_coclick': batch_data['task_coclick_s'][i]
            }

            # 本地更新（假设 local_update 已经使用了隐式梯度）
            _loss = self.local_update(task_data)
            loss_s.append(_loss)

        # 全局更新
        start_time = time.time()
        loss = torch.stack(loss_s).mean(0)  # 对所有任务的损失取平均

        # 反向传播更新全局模型参数
        self.meta_optimizer.zero_grad()
        loss.backward()
        self.meta_optimizer.step()
        self.add_time('global gradient', time.time() - start_time)

        return loss.cpu().data.numpy()

    # meta_test
    def evaluation(self, task_data):
        """
        """
        # local_update
        loss = self.local_update(task_data)
        mae, rmse = self.cal_metrics.prediction(self.query_y_real, self.query_y_pred) #query_y_real => real score, query_y_pred => pred score
        ndcg_5 = self.cal_metrics.ranking(self.query_y_real, self.query_y_pred, k=5)
        return loss, mae, rmse, (ndcg_5, self.query_y_real, self.query_y_pred) #modify

    def get_user_sim_scores(self, user_x, user_mask, other_x, other_mask):
        """
        :param user_x:
        :param user_mask:
        :param other_x:
        :param other_mask:
        :return:
        """
        user_emb, item_emb = self.get_user_item_embedding(user_x)
        user_preference = self.meta_learner.get_user_preference_embedding(item_emb, user_emb, user_mask)

        other_user_emb, other_item_emb = self.get_user_item_embedding(other_x)
        other_preference = self.meta_learner.get_user_preference_embedding(other_item_emb, other_user_emb, other_mask)

        sim_scores = torch.mm(user_preference, other_preference.transpose(0, 1))
        return sim_scores.data.cpu().numpy()
    
    
