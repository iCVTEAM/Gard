import torch
import torch.nn as nn
from .backbones.resnet import ResNet, BasicBlock, Bottleneck, resnet50_bap
from loss.arcface import ArcFace
from model.sync_batchnorm import SynchronizedBatchNorm2d, SynchronizedBatchNorm1d
from utils.adj_matrix import *
import torch.nn.functional as F
from model.backbones.Attention import *
import numpy as np
from model.GCN.GCN import norm_adj_batch as norm_adj, GraphConvolution as GCNConv, GraphAttentionLayer as  GATConv


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        # nn.init.kaiming_normal_(m.weight, a=0, model='fan_out')
        nn.init.kaiming_normal_(m.weight, a=0)
        nn.init.constant_(m.bias, 0.0)

    elif classname.find('Conv') != -1:
        # nn.init.kaiming_normal_(m.weight, a=0, model='fan_in')
        nn.init.kaiming_normal_(m.weight, a=0)
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)
    elif classname.find('BatchNorm') != -1:
        if m.affine:
            nn.init.constant_(m.weight, 1.0)
            nn.init.constant_(m.bias, 0.0)


def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        nn.init.normal_(m.weight, std=0.001)
        if m.bias:
            nn.init.constant_(m.bias, 0.0)




class Backbone(nn.Module):
    def __init__(self, num_classes, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.LAST_STRIDE
        model_path = cfg.PRETRAIN_PATH
        self.num_classes = num_classes

        self.DBAP = GDA(n_part=512)
        self.DBAP1 = GDA(n_part=512)
        self.DBAP2 = GDA(n_part=512)
        self.DBAP3 = GDA(n_part=512)

        self.block_num = 64  # 32

        self.mask = torch.autograd.Variable(torch.ones(self.block_num + 1, self.block_num + 1), requires_grad=True)

        self.seg = True

        self.ATT = ATT()

        self.cos_layer = cfg.COS_LAYER
        model_name = cfg.MODEL_NAME
        pretrain_choice = cfg.PRETRAIN_CHOICE
        if model_name == 'resnet50':
            self.in_planes = 2048
            # self.base = ResNet(last_stride=last_stride,
            #                    block=Bottleneck,use_bap=True,
            #                    layers=[3, 4, 6, 3])
            self.base = resnet50_bap(pretrained=True)

        else:
            print('unsupported backbone! only support resnet50, but got {}'.format(model_name))

        if pretrain_choice == 'imagenet':
            # self.base.load_param(model_path)
            print('Loading pretrained ImageNet model......')

        self.gap = nn.AdaptiveAvgPool2d(1)

        self.dropout = 0.3  # 0.6
        self.nheads = 1  # recommend 8

        self.block_size = 28
        # self.block_num=self.block_size*self.block_size+1

        self.gap_dense = nn.AdaptiveAvgPool2d(self.block_size)

        self.grouptrans = nn.Sequential(
            nn.Conv2d(784, 512, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
            # nn.ReLU(inplace=True)

            # nn.ReLU(inplace=True)
        )

        self.transdim = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True)
        )

        self.transdim2 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, padding=0, bias=False),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True)
        )
        self.transdim3 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, padding=0, bias=True),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
            nn.ReLU(inplace=True)

        )
        self.linear = nn.Linear(256 * self.block_num, 1024, bias=False)
        self.linear.apply(weights_init_classifier)
        self.flat = nn.Flatten()

        self.stage1 = nn.Sequential(
            nn.Conv2d(512, 512, 1, 1, padding=0),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
        )

        self.stage2 = nn.Sequential(
            nn.Conv2d(1024, 512, 1, 1, padding=0),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
            nn.Conv2d(512, 512, 3, 1, padding=1),
            SynchronizedBatchNorm2d(512, momentum=cfg.TRAIN_BN_MOM),
        )

        self.max1 = nn.AdaptiveAvgPool2d(1)

        self.max2 = nn.MaxPool2d(kernel_size=28, stride=28)

        self.max3 = nn.MaxPool2d(kernel_size=14, stride=14)

        self.fuse = nn.Sequential(
            SynchronizedBatchNorm1d(512 * 3, momentum=cfg.TRAIN_BN_MOM),
            nn.Linear(512 * 3, 512),
            # SynchronizedBatchNorm1d(512, momentum=cfg.TRAIN_BN_MOM),
            nn.ELU(inplace=True),
            # nn.Linear(512, self.num_classes, bias=False)
        )

        hidden = 512
        # self.attentions = [GATConv(512,hidden,dropout=self.dropout,alpha=0.2) for _ in range (self.nheads)] # abandon for multi-GPUs

        # for i,attention in enumerate(self.attentions):
        #    self.add_module('attention_{}'.format(i),attention) # abandon for multi-GPUs

        self.attention1 = GCNConv(512, hidden)

        self.attention2 = GCNConv(512, hidden)


        self.node_pool1 = 64

        self.graphpool = GCNConv(hidden, self.node_pool1)
        self.graphpool2 = GCNConv(hidden, self.node_pool1)

        self.graphconv1 = GCNConv(hidden,
                                  hidden)  # GATConv(self.nheads * hidden, 512, dropout=self.dropout, alpha=0.2, concat=False, batch_size=8)

        self.graphconv2 = GCNConv(hidden, hidden)

        if self.cos_layer:
            print('using cosine layer')
            self.arcface = ArcFace(self.in_planes, self.num_classes, s=30.0, m=0.50)
        elif self.seg:
            print('using dense segmentation')

            self.classifier1 = nn.Linear(hidden, self.num_classes, bias=False)
            self.classifier1.apply(weights_init_classifier)

            self.classifier2 = nn.Linear(hidden, self.num_classes, bias=False)
            self.classifier2.apply(weights_init_classifier)

            self.classifier3 = nn.Linear(hidden, self.num_classes, bias=False)
            self.classifier3.apply(weights_init_classifier)

            self.classifier = nn.Linear(hidden*2, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        else:
            self.classifier = nn.Linear(self.in_planes, self.num_classes, bias=False)
            self.classifier.apply(weights_init_classifier)

        self.dropfeature = nn.Dropout(0.3)
        self.dropout2 = nn.Dropout2d(0.1)

        ##init
        self.graphconv1.apply(weights_init_kaiming)
        self.graphconv2.apply(weights_init_kaiming)

        self.transdim2.apply(weights_init_kaiming)
        self.transdim3.apply(weights_init_kaiming)
        self.transdim.apply(weights_init_kaiming)

        ##init

        self.relu = nn.LeakyReLU(0.2)
        self.softmax = nn.Softmax(dim=2)  # nn.LogSoftmax(dim=1)
        self.elu = nn.ELU()

        self.bottleneck = nn.BatchNorm1d(hidden*2)
        self.bottleneck.bias.requires_grad_(False)
        self.bottleneck.apply(weights_init_kaiming)

        self.bottleneck1 = nn.BatchNorm1d(hidden)
        self.bottleneck1.bias.requires_grad_(False)
        self.bottleneck1.apply(weights_init_kaiming)

        self.bottleneck2 = nn.BatchNorm1d(hidden)
        self.bottleneck2.bias.requires_grad_(False)
        self.bottleneck2.apply(weights_init_kaiming)

        self.bottleneck3 = nn.BatchNorm1d(hidden)
        self.bottleneck3.bias.requires_grad_(False)
        self.bottleneck3.apply(weights_init_kaiming)

        self.fuse.apply(weights_init_kaiming)

    def forward_backbone(self, inputs, label=None):

        attention_maps, features = self.base(inputs)

        # add new swap operation
        attention_y, feature_y = attention_maps.clone(), features.clone()
        idx = build_idx(attention_maps.shape[0]).long().cuda()
        attention_y = torch.index_select(attention_y, 0, idx)
        feature_y = torch.index_select(feature_y, 0, idx)

        # add cross attention
        b, c, w, h = attention_maps.size()

        coatt = torch.einsum('ian,inb->iab',
                             (attention_maps.view(b, c, w * h), attention_y.view(b, c, w * h).permute(0, 2, 1)))

        coatt = torch.div(coatt, float(w * h))
        coatt = torch.mul(torch.sign(coatt), torch.sqrt(torch.abs(coatt) + 1e-12))
        coatt = coatt.view(b, c, -1)
        coatt = torch.nn.functional.normalize(coatt, dim=-1)

        attention_x = torch.einsum('ibn,iab->ian', attention_maps.view(b, c, w * h), coatt).view(b, c, w, h)
        attention_y = torch.einsum('ibn,iab->ian', attention_y.view(b, c, w * h), coatt).view(b, c, w, h)

        sp_features, channel_features_x = self.DBAP(attention_x, features)
        sp_features, channel_features_y = self.DBAP(attention_y, feature_y)
        # b*c* 32 part

        b, c, p = channel_features_x.size()
        channel_features_x = channel_features_x.view(b, c, p, 1)
        channel_features_y = channel_features_y.view(b, c, p, 1)
        ## spatial down sample as 32

        channel_features_x = self.gap(channel_features_x)
        channel_features_y = self.gap(channel_features_y)

        featx = self.transdim(channel_features_x)

        featy = self.transdim(channel_features_y)

        featx = featx.view(b, -1)
        featy = featy.view(b, -1)

        if self.cos_layer:
            cls_score = self.arcface(featx, label)
        else:
            # feat = self.bottleneck1(feat)
            # cls_l = self.classifier1(feat)
            #
            featy = self.bottleneck3(featy)
            cls_y = self.classifier3(featy)
            featx = self.bottleneck3(featx)
            cls_x = self.classifier3(featx)

            # feat = self.bottleneck3(gfeature1)
            # cls_all = self.classifier3(feat)

        return cls_x, cls_y, channel_features_x  # global feature for triplet loss

    def forward_test(self, inputs, label=None):

        attention_maps, features = self.base(inputs)

        # add new swap operation

        # add cross attention
        b, c, w, h = attention_maps.size()

        coatt = torch.einsum('ian,inb->iab',
                             (attention_maps.view(b, c, w * h), attention_maps.view(b, c, w * h).permute(0, 2, 1)))

        coatt = torch.div(coatt, float(w * h))
        coatt = torch.mul(torch.sign(coatt), torch.sqrt(torch.abs(coatt) + 1e-12))
        coatt = coatt.view(b, c, -1)
        coatt = torch.nn.functional.normalize(coatt, dim=-1)

        attention_maps = torch.einsum('ibn,iab->ian', attention_maps.view(b, c, w * h), coatt).view(b, c, w, h)

        sp_features, channel_features_x = self.DBAP(attention_maps, features)
        # b*c* 32 part

        b, c, p = channel_features_x.size()
        channel_features_x = channel_features_x.view(b, c, p, 1)
        ## spatial down sample as 32

        channel_features_x = self.gap(channel_features_x)

        featx = self.transdim(channel_features_x)

        featx = featx.view(b, -1)

        if self.cos_layer:
            cls_score = self.arcface(featx, label)
        else:
            # feat = self.bottleneck1(feat)
            # cls_l = self.classifier1(feat)

            featx = self.bottleneck3(featx)
            cls_x = self.classifier3(featx)

            # feat = self.bottleneck3(gfeature1)
            # cls_all = self.classifier3(feat)

        return cls_x, cls_x, channel_features_x  # global feature for triplet loss

    # @torch.cuda.amp.autocast()
    def forward(self, inputs, label=None, mode='train'):  # label is unused if self.cos_layer == 'no'
        if mode == 'train':
            # return self.forward_gcn(inputs)
            return self.forward_gcn_pool(inputs, label)
        else:
            return self.forward_gcn_pool(inputs)

    def forward_gcn_pool(self, inputs, label=None):
        # unify test and train

        attention_maps, features, out = self.base(inputs)

        layers = self.base.get_layers()
        ## spatial down sample as 32
        # attention_maps = self.gap_dense(attention_maps)
        # features = self.gap_dense(features)
        b, c, w, h = features.size()
        # self.block_num = 64 * 3
        # self.block_num = 64
        #features = self.dropout1(features)

        ### add part feature
        #features_ori = self.dropfeature(features) #.clone()
        crop_part, vis_ret = self.ATT.NMS_crop(attention_maps, features)
        ###

        _, channel_features_ori = self.DBAP(attention_maps, features)
        _, channel_features_trans = self.DBAP2(features, attention_maps)

        channel_features_trans = channel_features_trans.view(b, c, 1, -1)

        channel_features = channel_features_ori.view(b, c, 1, -1)


        gfeature = self.transdim3(channel_features).reshape(b, c, -1).permute(0, 2, 1)


        edge = gen_adj_sim(512, gfeature, gfeature)

        adj = edge.cuda()
        adj = norm_adj(adj)
        # B*N1*C
        gfeature = self.graphconv1(gfeature, adj)
        gfeature = self.relu(gfeature)
        # B*N1*N2
        #print(gfeature.size())

        f_pool = self.graphpool(gfeature, adj)

        f_pool = F.softmax(f_pool, dim=-1)  # pool the cluster dim


        gpool = torch.einsum('inc,inm->imc', (gfeature, f_pool))

        ## second GCN LAYERS

        gfeature_t = self.transdim2(channel_features_trans).reshape(b, c, -1).permute(0, 2, 1)

        edge_t = gen_adj_sim(512, gfeature_t, gfeature_t)
        adj_t = edge_t.cuda()
        adj_t = norm_adj(adj_t)
        # B*N1*C
        gfeature_t = self.graphconv2(gfeature_t, adj_t)
        gfeature_t = self.relu(gfeature_t)
        # B*N1*N2
        # print(gfeature.size())

        f_pool_t = self.graphpool2(gfeature_t, adj_t)

        f_pool_t = F.softmax(f_pool_t, dim=-1)  # pool the cluster dim

        # print(f_pool.size())

        gpool_t = torch.einsum('inc,inm->imc', (gfeature_t, f_pool_t))

        gpool_cat = torch.cat([gpool,gpool_t],2)
        gfeature_cat = torch.cat([gfeature,gfeature_t],2)
        # shoule be 1024 dim

        # END OF SECOND gcn layer

        if self.cos_layer:
            cls_score = self.arcface(feat, label)
        else:

            part_1 = crop_part[:, 0, :, :, :]
            cls_1 = self.gap(part_1).view(b,-1)
            cls_1 = self.bottleneck1(cls_1)
            cls_1 = self.classifier1(cls_1)

            part_2 = crop_part[:, 1, :, :, :]
            cls_2 = self.gap(part_2).view(b,-1)
            cls_2 = self.bottleneck2(cls_2)
            cls_2 = self.classifier2(cls_2)

            part_3 = crop_part[:, 1, :, :, :]
            cls_3 = self.gap(part_3).view(b,-1)
            cls_3 = self.bottleneck3(cls_3)
            cls_3 = self.classifier3(cls_3)

            cls_g = gpool_cat.mean(1) + gfeature_cat.mean(1)
            cls_g = self.bottleneck(cls_g)
            cls_g = self.classifier(cls_g)

        return cls_g, cls_1, cls_2, cls_3, vis_ret
        # return cls_g, adj, f_pool, attention_maps, features  # global feature for triplet loss

    def forward_gcn_part(self, inputs, label=None):
        # unify test and train

        attention_maps, features, out = self.base(inputs)

        layers = self.base.get_layers()
        ## spatial down sample as 32

        b, c, w, h = features.size()


        ### add part feature
        features_ori = features
        crop_part, vis_ret = self.ATT.NMS_crop(attention_maps.clone(), features_ori)
        ###



        part_1 = crop_part[:, 0, :, :, :]
        part_2 = crop_part[:, 1, :, :, :]
        part_3 = crop_part[:, 2, :, :, :]

        _, channel_features_ori = self.DBAP(attention_maps, features)

        channel_features = channel_features_ori.view(b, c, 1, -1)
        # gfeature_1 = self.transdim(channel_features).view(b, c, -1).permute(0, 2, 1)
        # gfeature_2 = self.transdim2(channel_features).view(b, c, -1).permute(0, 2, 1)



        if self.cos_layer:
            cls_score = self.arcface(feat, label)
        else:
            cls_1 = self.gap(part_1).view(b, -1)
            cls_1 = self.bottleneck1(cls_1)
            cls_1 = self.classifier1(cls_1)


            cls_2 = self.gap(part_2).view(b, -1)
            cls_2 = self.bottleneck2(cls_2)
            cls_2 = self.classifier2(cls_2)


            cls_3 = self.gap(part_3).view(b, -1)
            cls_3 = self.bottleneck3(cls_3)
            cls_3 = self.classifier3(cls_3)

            cls_g = channel_features_ori.mean(1)
            cls_g = self.bottleneck(cls_g)
            cls_g = self.classifier(cls_g)

        return cls_g, cls_1, cls_2, cls_3, vis_ret
        # return cls_g, adj, f_pool, attention_maps, features  # global feature for triplet loss


    def forward_gcn(self, inputs, label=None):  # label is unused if self.cos_layer == 'no'
        # unify test and train

        attention_maps, features = self.base(inputs)
        # b*c* 32 part
        layers = self.base.get_layers()
        ## spatial down sample as 32

        _, channel_features = self.DBAP(attention_maps, features)

        cf_2 = self.stage1(layers[2])
        # cf_2 = self.max1(cf_2)
        _, cf_2 = self.DBAP2(cf_2, cf_2)

        cf_3 = self.stage2(layers[3])
        # cf_3 = self.max1(cf_3)
        _, cf_3 = self.DBAP3(cf_3, cf_3)

        b, c, p = channel_features.size()

        # root = self.gap(channel_features)

        feat = self.transdim(channel_features.view(b, c, p, 1))

        b, c, _, _ = feat.size()

        flatten = feat.view(b, c, -1).permute(0, 2, 1)

        cf_2 = cf_2.view(b, c, -1).permute(0, 2, 1)
        cf_3 = cf_3.view(b, c, -1).permute(0, 2, 1)

        gfeature = flatten.clone()

        gfeature_ori = torch.cat([gfeature, cf_2, cf_3], dim=1)
        # end calc adj
        self.block_num = 64 * 3
        edge = gen_adj_sim(64 * 3, gfeature_ori)
        edge = edge.cuda()
        # graph cross
        adj_c = get_sim_cross2(edge)
        adj_c = norm_adj(adj_c)
        # print(adj_c[0])

        # build subgraph
        edge1 = gen_adj_sim(64, gfeature)
        edge2 = gen_adj_sim(64, cf_2)
        edge3 = gen_adj_sim(64, cf_3)
        sub_edge = get_sim_local(edge1, edge2)
        sub_edge = get_sim_local(sub_edge, edge3)
        sub_edge = sub_edge.cuda()
        adj_s = norm_adj(sub_edge)

        #

        # gfeature_all = F.dropout(gfeature_all, self.dropout, training=self.training)

        # gfeature = torch.cat([att(gfeature,adj) for att in self.attentions], dim=1)
        gfeature_all = self.attention1(gfeature_ori, adj_c)
        gfeature_local = self.attention2(gfeature_ori, adj_s)

        gfeature = torch.cat([gfeature_all, gfeature_local], 2)  # concat in channel

        # gfeature_all = F.dropout(gfeature_all, self.dropout, training=self.training)
        # gfeature_all = F.elu(self.graphconv(gfeature_all, adj))

        # gfeature = gfeature.permute(0, 2, 1)

        if self.cos_layer:
            cls_score = self.arcface(feat, label)
        else:
            gfeature_all = gfeature_all.mean(1)
            cls_1 = self.bottleneck1(gfeature_all)
            cls_1 = self.classifier1(cls_1)

            gfeature_local = gfeature_local.mean(1)
            cls_2 = self.bottleneck2(gfeature_local)
            cls_2 = self.classifier2(cls_2)

            cls_3 = cf_3.mean(1)
            # cls_3 = self.bottleneck3(cls_3)
            # cls_3 = self.classifier3(cls_3)
            #

            cls_g = gfeature.mean(1)
            cls_g = self.bottleneck(cls_g)
            cls_g = self.classifier(cls_g)

            # feat = self.bottleneck3(gfeature1)
            # cls_all = self.classifier3(feat)

        return cls_g, cls_1, cls_2, cls_3  # global feature for triplet loss

    def forward_pure(self, inputs, label=None):  # label is unused if self.cos_layer == 'no'
        # unify test and train

        attention_maps, features = self.base(inputs)
        # b*c* 32 part

        ## spatial down sample as 32

        c_features = self.gap(attention_maps)
        b, c, w, h = c_features.size()
        c_features = c_features.view(b, -1)

        if self.cos_layer:
            cls_score = self.arcface(feat, label)
        else:

            cls_g = self.bottleneck3(c_features)
            cls_g = self.classifier3(cls_g)

        return cls_g, cls_g, c_features  # global feature for triplet loss

    def load_param(self, trained_path):
        param_dict = torch.load(trained_path)
        self.load_state_dict(param_dict)
        # for i in param_dict:
        #     if 'classifier' in i or 'arcface' in i:
        #         continue
        #     self.state_dict()[i].copy_(param_dict[i])
        print('Loading pretrained model from {}'.format(trained_path))

    def reinit_param(self):
        for m in self.named_modules():
            if "base" in m:
                continue
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, SynchronizedBatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                nn.init.constant_(m.bias, 0.0)



def make_model(cfg, num_class):
    model = Backbone(num_class, cfg)
    return model
