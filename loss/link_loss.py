import torch
import torch.nn as nn


def loss_aux(f_pool, adj):
        # conducted only in training phase
        f_pool_t = f_pool.permute(0, 2, 1)
        link_loss = adj - torch.matmul(f_pool, f_pool_t)
        #print(f_pool_t[0])

        link_loss = torch.norm(link_loss, p=2)
        link_loss = link_loss /(64) #/ adj.numel()

        eps = 1e-10
        ent_loss = (-f_pool * torch.log(f_pool + eps)).sum(dim=-1).mean()

        #print(ent_loss)
        #print(link_loss)

        return link_loss,ent_loss


def loss_normf(features):
        # conducted only in training phase
        features_t = features.permute(0, 2, 1)
        norm_f = torch.matmul(features, features_t)
        #print(f_pool_t[0])
        zero = torch.zeros_like(norm_f)

        norm_f= torch.where(norm_f<0,zero,norm_f)
        f_loss = torch.norm(norm_f, p=2)
        f_loss = f_loss / adj.numel()

        #eps = 1e-10
        #ent_loss = (-f_pool * torch.log(f_pool + eps)).sum(dim=-1).mean()

        return f_loss