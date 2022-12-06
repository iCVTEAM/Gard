import logging
import numpy as np
import os
import time
import torch
import torch.nn as nn
from utils.logger import label2colormap
from utils.meter import AverageMeter
from utils.metrics import R1_mAP
from tensorboardX import SummaryWriter
from model.backbones.Attention import *
from tqdm import tqdm, trange
from loss.link_loss import loss_aux
from model.sync_batchnorm import *
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]





def do_train(cfg,
             model,
             center_criterion,
             train_loader,
             val_loader,
             optimizer,
             optimizer_center,
             scheduler,
             loss_fn,
             num_query):
    """

    Args:
        training function inputs

        cfg: configuration file, passed from /config/configs.py or /default.py

        model: initialized deep model

        center_criterion: could be enabled if using center loss, implemented for further updating

        train_loader: training dataloader

        val_loader: validation dataloader

        optimizer: SGD or ADAM optimizer

        optimizer_center: SGD or ADAM optimizer  for center loss

        scheduler: updating scheduler

        loss_fn: loss function for learning

        num_query: number of learning samples

    Returns:

    """



    log_period = cfg.LOG_PERIOD
    checkpoint_period = cfg.CHECKPOINT_PERIOD
    eval_period = cfg.EVAL_PERIOD

    device = "cuda"
    epochs = cfg.MAX_EPOCHS

    logger = logging.getLogger('{}.train'.format(cfg.PROJECT_NAME))
    logger.info('start training')

    # Set the Tensorboard logger
    tblogger = SummaryWriter(cfg.LOG_DIR)
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for training'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model, device_ids=[0, 1])
            #patch_replication_callback(model)
        model.to(device)




    Best_acc = 0
    Best_epoch = -1
    loss_meter = AverageMeter()
    ent_loss_meter = AverageMeter()
    link_loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    acc1_meter = AverageMeter()
    #torch.autograd.set_detect_anomaly(True)
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    att = ATT()
    tfstep = 0

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        link_loss_meter.reset()
        ent_loss_meter.reset()
        acc_meter.reset()
        acc1_meter.reset()

        if epoch == 101:
            model_dict = model.state_dict()
            path = os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch-1))
            pretrained_dict = torch.load(path, map_location=device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if ('base' in k) }
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)
            #model.reinit_param().
        model.train()



        for n_iter, (img, vid) in enumerate(train_loader):
            tfstep += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)



            cls_g = model(img, target, mode='train')
            cls_g = cls_g.to(device)

            loss_g = loss_fn(cls_g, cls_g, target)

            loss = loss_g
            loss.backward()

            optimizer.step()



            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT)
                optimizer_center.step()

            acc = (cls_g.max(1)[1] == target).float().mean()
            acc1 = (cls_g.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])

            acc_meter.update(acc, 1)
            acc1_meter.update(acc1, 1)
            FLAG_VISIBLE = False
            if (n_iter + 1) % log_period == 0:
                logger.info(
                    "Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f}, Base Lr: {:.2e}"
                    .format(epoch, (n_iter + 1), len(train_loader),
                            loss_meter.avg, acc_meter.avg,
                            scheduler.get_lr()[0]))

                if FLAG_VISIBLE == True and (n_iter + 1) % (4*log_period) == 0:

                    img_vis = img[0].cpu().numpy().transpose(1, 2, 0) * std + mean
                    #attention_vis = vis_ret.detach().cpu().numpy()
                    # score_img = torch.argmax(score[0], dim=0).cpu().numpy()
                    # score_img_color = label2colormap(score_img).transpose((2, 0, 1))
                    # label_color = label2colormap(score_img == target[0].cpu().numpy()).transpose((2, 0, 1))
                    #print (part_vis.size())
                    attention_vis1 = part_vis[0].unsqueeze(0).repeat(1,1,1).detach().cpu().numpy().transpose(1, 2, 0) #* std + mean
                    attention_vis2 = part_vis[1].unsqueeze(0).repeat(1,1,1).detach().cpu().numpy().transpose(1, 2, 0) #* std + mean
                    #attention_vis3 = part_vis[2].unsqueeze(0).repeat(1,1,1).detach().cpu().numpy().transpose(1, 2, 0) #* std + mean

                    tblogger.add_image('Img', img_vis.transpose(2, 0, 1), tfstep)
                    #tblogger.add_image('Part', part_vis.transpose(2, 0, 1), tfstep)
                    tblogger.add_image('attention_vis_1', attention_vis1.transpose(2, 0, 1) , tfstep)
                    tblogger.add_image('attention_vis_2', attention_vis2.transpose(2, 0, 1) , tfstep)
                    #tblogger.add_image('attention_vis_3', attention_vis3.transpose(2, 0, 1) , tfstep)

                    # tblogger.add_image('label', label_color, tfstep)
                    # tblogger.add_image('pred', score_img_color, tfstep)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Current Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s], Best Acc is {:.3f}, best Epoch is {}."
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch,Best_acc, Best_epoch))
        scheduler.step()

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0  and epoch>60:
            model.eval()
            evaluator.reset()
            logger.info("Start val ")
            count = 0
            for n_iter, (img, vid, imgpath) in enumerate(tqdm(val_loader)):
                with torch.no_grad():
                    img = img.to(device)
                    if cfg.FLIP_FEATS == 'on':
                        feat = torch.FloatTensor(img.size(0), cfg.CLASSNUM).zero_().to(device)
                        for i in range(2):
                            if i == 1:
                                inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(device)
                                img = img.index_select(3, inv_idx)

                            score= model(img, mode='test')

                            score = score.to(device)
                            f = score
                            feat = feat + f
                            evaluator.update((feat, vid))
                    else:
                            cls_g = model(img, mode='test')

                            evaluator.update((cls_g, vid))
            acc = evaluator.compute_acc()

            if cfg.MODE == "validation_debug":
                # enable this if we have validation set or the test set can be taken as validation
                if acc>=Best_acc:
                    Best_epoch = epoch
                    Best_acc = acc
                    torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_bestval.pth'))
            elif cfg.MODE == "customized_test":
                torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_current.pth'))

            # cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("no enhancement Validation Results - Epoch: {}".format(epoch))
            logger.info("Using Validation samples: {}".format(count))
            logger.info("validation acc: {:.1%}".format(acc))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       )
    evaluator.reset()
    if device:
        if torch.cuda.device_count() > 1:
            print('Using {} GPUs for inference'.format(torch.cuda.device_count()))
            model = nn.DataParallel(model)
        model.to(device)

    # model.load_param(cfg.TEST_WEIGHT,map_location=device)
    model_dict = torch.load(cfg.TEST_WEIGHT, map_location=device)
    model.load_state_dict(model_dict)

    att = ATT()
    model.eval()
    img_path_list = []
    for n_iter, (img, pid, imgpath) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            img = img.to(device)
            logger.info("Niter: {} ".format(n_iter))

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), cfg.CLASSNUM).zero_().to(device)
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().to(device)
                        img = img.index_select(3, inv_idx)

                    score = model(img, mode='test')

                    score = score.to(device)
                    f = score
                    feat = feat + f
            else:
                with torch.no_grad():
                    img = img.to(device)
                    feat = model(img, mode='test')


            evaluator.update((feat, pid))
            img_path_list.extend(imgpath)

    # cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()

    acc = evaluator.compute_acc()
    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Test acc: {:.1%}".format(acc))


