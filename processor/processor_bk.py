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
        model.to(device)

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    acc1_meter = AverageMeter()
    torch.autograd.set_detect_anomaly(True)
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM)
    # train
    att = ATT()
    tfstep = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        loss_meter.reset()
        acc_meter.reset()
        acc1_meter.reset()
        evaluator.reset()
        model.train()
        for n_iter, (img, vid) in enumerate(train_loader):
            tfstep += 1
            optimizer.zero_grad()
            optimizer_center.zero_grad()
            img = img.to(device)
            target = vid.to(device)

            cls_g, cls_1, cls_2, cls_3 = model(img, target, mode='train')
            cls_g = cls_g.to(device)
            cls_1 = cls_1.to(device)
            cls_2 = cls_2.to(device)
            cls_3 = cls_3.to(device)

            #loss_1 = loss_fn(cls_1, cls_1, target)
            loss_2 = loss_fn(cls_2, cls_2, target)
            loss_3 = loss_fn(cls_3, cls_3, target)
            loss_g = loss_fn(cls_g, cls_g, target)

            loss = loss_g #+ loss_2 * 0.2 + loss_3 * 0.2

            loss.backward()
            optimizer.step()



            if 'center' in cfg.LOSS_TYPE:
                for param in center_criterion.parameters():
                    param.grad.data *= (1. / cfg.CENTER_LOSS_WEIGHT) 
                optimizer_center.step()

            acc = (cls_g.max(1)[1] == target).float().mean()
            acc1 = (cls_1.max(1)[1] == target).float().mean()
            loss_meter.update(loss.item(), img.shape[0])
            acc_meter.update(acc, 1)
            acc1_meter.update(acc1, 1)
            vis = False
            if (n_iter + 1) % log_period == 0:
                logger.info("Epoch[{}] Iteration[{}/{}] Loss: {:.3f}, Acc: {:.3f} , Acc2: {:.3f}, Base Lr: {:.2e}"
                            .format(epoch, (n_iter + 1), len(train_loader),
                                    loss_meter.avg, acc_meter.avg, acc1_meter.avg, scheduler.get_lr()[0]))

                if vis == 100:
                    img_vis = img[0].cpu().numpy().transpose(1, 2, 0) * std + mean
                    score_img = torch.argmax(score[0], dim=0).cpu().numpy()
                    score_img_color = label2colormap(score_img).transpose((2, 0, 1))
                    label_color = label2colormap(score_img == target[0].cpu().numpy()).transpose((2, 0, 1))
                    tblogger.add_image('Img', img_vis.transpose(2, 0, 1), tfstep)
                    tblogger.add_image('label', label_color, tfstep)
                    tblogger.add_image('pred', score_img_color, tfstep)
        end_time = time.time()
        time_per_batch = (end_time - start_time) / (n_iter + 1)
        logger.info("Epoch {} done. Time per batch: {:.3f}[s] Speed: {:.1f}[samples/s]"
                    .format(epoch, time_per_batch, train_loader.batch_size / time_per_batch))
        scheduler.step()

        if not os.path.exists(cfg.OUTPUT_DIR):
            os.mkdir(cfg.OUTPUT_DIR)

        if epoch % checkpoint_period == 0:
            torch.save(model.state_dict(), os.path.join(cfg.OUTPUT_DIR, cfg.MODEL_NAME + '_{}.pth'.format(epoch)))

        if epoch % eval_period == 0:
            model.eval()
            logger.info("Start val ")
            for n_iter, (img, vid, _) in enumerate(val_loader):
                with torch.no_grad():
                    img = img.to(device)
                    score, _, _, _ = model(img, mode='test')
                    evaluator.update((score, vid))

            acc = evaluator.compute_acc()
            # cmc, mAP, _, _, _, _, _ = evaluator.compute()
            logger.info("no enahcnement Validation Results - Epoch: {}".format(epoch))
            logger.info("validation acc: {:.1%}".format(acc))


def do_inference(cfg,
                 model,
                 val_loader,
                 num_query):
    device = "cuda"
    logger = logging.getLogger('{}.test'.format(cfg.PROJECT_NAME))
    logger.info("Enter inferencing")
    evaluator = R1_mAP(num_query, max_rank=50, feat_norm=cfg.FEAT_NORM, \
                       method=cfg.TEST_METHOD, reranking=cfg.RERANKING)
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
    for n_iter, (img, pid, imgpath) in enumerate(val_loader):
        with torch.no_grad():
            img = img.to(device)
            logger.info("Niter: {:.1%} ".format(n_iter))

            if cfg.FLIP_FEATS == 'on':
                feat = torch.FloatTensor(img.size(0), cfg.CLASSNUM).zero_().cuda()
                for i in range(2):
                    if i == 1:
                        inv_idx = torch.arange(img.size(3) - 1, -1, -1).long().cuda()
                        img = img.index_select(3, inv_idx)

                    score_g, score, raw_features, _ = model(img, mode='test')

                    # attention_maps=attention_maps.to(device)

                    # img_crop, img_drop = att.attention_crop_drop(attention_maps, img)
                    # img_crop =img_crop.to(1)

                    # score2, _, _ = model(img_crop)

                    score = score.to(device)
                    # score2=score2.to(device)
                    # f= (score+score2)
                    f = score
                    # f = model(img)
                    feat = feat + f
            else:
                score, attention_maps, raw_features = model(img)

                attention_maps = attention_maps.to(device)

                img_crop, img_drop = att.attention_crop_drop(attention_maps, img)
                score2, _, _ = model(img_crop)

                score = score.to(device)
                score2 = score2.to(device)

                feat = (score + score2)

                # feat = model(img)

            evaluator.update((feat, pid))
            img_path_list.extend(imgpath)

    # cmc, mAP, distmat, pids, camids, qfeats, gfeats = evaluator.compute()

    acc = evaluator.compute_acc()
    # cmc, mAP, _, _, _, _, _ = evaluator.compute()
    logger.info("Test acc: {:.1%}".format(acc))

    # np.save(os.path.join(cfg.LOG_DIR, cfg.DIST_MAT) , distmat)
    # np.save(os.path.join(cfg.LOG_DIR, cfg.PIDS), pids)
    # np.save(os.path.join(cfg.LOG_DIR, cfg.CAMIDS), camids)
    # np.save(os.path.join(cfg.LOG_DIR, cfg.IMG_PATH), img_path_list[num_query:])
    # torch.save(qfeats, os.path.join(cfg.LOG_DIR, cfg.Q_FEATS))
    # torch.save(gfeats, os.path.join(cfg.LOG_DIR, cfg.G_FEATS))

    # logger.info("Validation Results")
    # logger.info("mAP: {:.1%}".format(mAP))
    # for r in [1, 5, 10]:
    #     logger.info("CMC curve, Rank-{:<3}:{:.1%}".format(r, cmc[r - 1]))
