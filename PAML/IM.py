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
from util import *
import torch.utils.bottleneck as bottleneck
from torch.cuda.amp import autocast, GradScaler

class IM(torch.nn.Module):
    def __init__(self, config):
        super(IM, self).__init__()
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
        self.cal_metrics = Evaluation()

        self.ml_weight_len = len(self.meta_learner.update_parameters())
        self.ml_weight_name = list(self.meta_learner.update_parameters().keys())
        self.ml_weight_size = self.meta_learner.get_parameter_size()
        self.task_specific = Task_specific(config, self.ml_weight_size)
        # global update
        
        self.meta_optimizer = torch.optim.Adam(self.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])    
        
        
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
        # self.add_time('social_embd_lookup', time.time() - start_time)
        # start_time = time.time()

        ml_weights = self.meta_learner.update_parameters()
        task_emb = self.meta_learner.get_task_embd(**task_embd_dict)
        task_weights = self.task_specific(task_emb, ml_weights)

        # Step 2: Compute initial support prediction and loss
        support_pred = self.meta_learner(support_item_emb, support_user_emb, vars_dict=task_weights, **task_embd_dict)
        loss = F.mse_loss(support_pred, support_y)
        grads = torch.autograd.grad(loss, task_weights.values())


        # Step 4: Compute Fast Adaptation Step using Implicit Gradients
        fast_ml_weights = {name: (param - self.local_lr * grad) for (name, param), grad in zip(task_weights.items(), grads)}



        for _ in range(1, self.config['local_update']):
            support_pred = self.meta_learner(support_item_emb, support_user_emb, vars_dict=fast_ml_weights, **task_embd_dict)
            loss = F.mse_loss(support_pred, support_y)
            grads = torch.autograd.grad(loss, fast_ml_weights.values(), retain_graph=True)
            fast_ml_weights = {name: (param - self.local_lr * grad) for (name, param), grad in zip(fast_ml_weights.items(), grads)}
        
        query_pred = self.meta_learner(query_item_emb, query_user_emb, vars_dict=fast_ml_weights, **task_embd_dict)
        loss_query = F.mse_loss(query_pred, query_y)
        query_grads = torch.autograd.grad(loss_query, fast_ml_weights.values(), retain_graph=True)

        self.query_y_real = query_y.data.cpu().numpy()
        self.query_y_pred = query_pred.data.cpu().numpy()

        return loss_query, fast_ml_weights, query_grads       
    

    def global_update(self, global_step, batch_data):
        
        # 调整收敛条件, tol以及最大迭代次数, max_iters，从而使得共轭梯度，cg更快计算
        def conjugate_gradient(A, b, x_init=None, max_iters=2, tol=1e-1):
            x = x_init if x_init is not None else torch.randn_like(b, requires_grad=True)
            r = b - A(x)  
            p = r.clone()
            rs_old = torch.dot(r, r)

            for _ in range(max_iters):
                Ap = A(p)
                alpha = rs_old / (torch.dot(p, Ap) + 1e-6)
                x = x + alpha * p
                r = r - alpha * Ap
                rs_new = torch.dot(r, r)
                if torch.sqrt(rs_new) < tol:
                    break
                p = r + (rs_new / rs_old) * p
                rs_old = rs_new

            return x

        # def hessian_vector_product(v):

        #     param_list = [param for fast_weights in fast_ml_weights for param in fast_weights.values() if param.requires_grad]
        #     split_sizes = [p.numel() for p in param_list]
        #     v_split = torch.split(v, split_sizes)
        #     v_reshaped = [v_part.view_as(p) for v_part, p in zip(v_split, param_list)]

        #     grads = torch.autograd.grad(task_losses, param_list, create_graph=True)
        #     grad_v = torch.sum(torch.stack([torch.sum(g * v_i) for g, v_i in zip(grads, v_reshaped)]))
        #     hvp = torch.autograd.grad(grad_v, param_list, retain_graph=True, allow_unused=True)
            
        #     return torch.cat([g.view(-1) for g in hvp])

        def finite_difference_hvp(v, epsilon=1e-3):
            param_list = [param for fast_weights in fast_ml_weights for param in fast_weights.values() if param.requires_grad]
            split_sizes = [p.numel() for p in param_list]
            v_split = torch.split(v, split_sizes)
            v_reshaped = [v_part.view_as(p) for v_part, p in zip(v_split, param_list)]
            original_params = [p.clone() for p in param_list]

            # 计算 f(x + epsilon * v)
            for p, v_i in zip(param_list, v_reshaped):
                p.data.add_(epsilon, v_i)
            loss_plus = task_losses
            grad_plus = torch.autograd.grad(loss_plus, param_list, create_graph=True)

            # 计算 f(x - epsilon * v)
            for p, v_i in zip(param_list, v_reshaped):
                p.data.sub_(2 * epsilon, v_i)
            loss_minus = task_losses
            grad_minus = torch.autograd.grad(loss_minus, param_list, create_graph=True)

            # 恢复原始参数
            for p, orig_p in zip(param_list, original_params):
                p.data.copy_(orig_p)

            # 计算有限差分近似
            hvp = [(g_plus - g_minus) / (2 * epsilon) for g_plus, g_minus in zip(grad_plus, grad_minus)]
            
            # 将结果拼接成一个向量
            return torch.cat([g.view(-1) for g in hvp])

        task_losses, fast_ml_weights, query_grads_list = [], [], []
        for i in range(len(batch_data['supp_xs'])):  
            task_data = {'supp_x': batch_data['supp_xs'][i],
                         'supp_y': batch_data['supp_ys'][i],
                         'query_x': batch_data['query_xs'][i],
                         'query_y': batch_data['query_ys'][i],
                         'task_self': batch_data['task_self_s'][i],
                         'task_social': batch_data['task_social_s'][i],
                         'task_implicit': batch_data['task_implicit_s'][i],
                         'task_coclick': batch_data['task_coclick_s'][i]
                         }

            loss_query, fast_weights, query_grads = self.local_update(task_data)
            task_losses.append(loss_query)
            fast_ml_weights.append(fast_weights)
            query_grads_list.append(query_grads)

        start_time = time.time()
        meta_loss = torch.stack(task_losses).mean()

        meta_grads = {name: torch.zeros_like(param) for name, param in self.meta_learner.update_parameters().items()}
        
        for query_grads in tqdm(query_grads_list, desc="Processing query_grads_list"):
            b = torch.cat([g.view(-1) for grads in query_grads_list for g in grads])
            hvp_approx = conjugate_gradient(finite_difference_hvp, b)
            offset = 0
            for name, param in meta_grads.items():
                param_size = param.numel()
                meta_grads[name] += hvp_approx[offset:offset + param_size].view(param.shape) / len(query_grads_list)
                offset += param_size

        self.meta_optimizer.zero_grad()
        for name, param in self.meta_learner.update_parameters().items():
            param.grad = meta_grads[name]
        self.meta_optimizer.step()
        self.add_time('global gradient', time.time() - start_time)

        return meta_loss.detach().cpu().numpy()

    def evaluation(self, task_data):
        """
        """
        # local_update
        loss = self.local_update(task_data)[0]
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

