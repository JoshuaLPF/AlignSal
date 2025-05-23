# -*- coding: utf-8 -*-
import os
import torch.nn.functional as F
import sys
import random
import numpy as np
from datetime import datetime
from model import AlignSal
from SCAL import ContrastiveLossWithConv
from torchvision.utils import make_grid
from data import get_loader, test_dataset
from utils import clip_gradient, adjust_lr
from tensorboardX import SummaryWriter
import logging
import torch.backends.cudnn as cudnn
from options import opt
import torch, gc

gc.collect()
torch.cuda.empty_cache()

cl_loss = ContrastiveLossWithConv(k=3, temperature=0.07, threshold=0.4)


def bce_iou_loss(pred, mask):
    size = pred.size()[2:]
    mask = F.interpolate(mask, size=size, mode='bilinear')
    wbce = F.binary_cross_entropy_with_logits(pred, mask)
    pred = torch.sigmoid(pred)
    inter = (pred * mask).sum(dim=(2, 3))
    union = (pred + mask).sum(dim=(2, 3))
    wiou = 1 - (inter + 1) / (union - inter + 1)
    return (wbce + wiou).mean()


def seed_torch(seed=42):  ## 42, 3407
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

seed_torch()

# set the path
image_root = opt.rgb_root
gt_root = opt.gt_root
depth_root = opt.depth_root

test_image_root = opt.test_rgb_root
test_gt_root = opt.test_gt_root
test_depth_root = opt.test_depth_root
save_path = opt.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

logging.basicConfig(filename=save_path + 'AlignSal.log',
                    format='[%(asctime)s-%(filename)s-%(levelname)s:%(message)s]', level=logging.INFO, filemode='a',
                    datefmt='%Y-%m-%d %I:%M:%S %p')
logging.info("AlignSal-Train_4_pairs")

# build the model
model = AlignSal()

# load gpu
gpu_num = torch.cuda.device_count()
if gpu_num == 1:
    print("Use Single GPU -", opt.gpu_id)
elif gpu_num > 1:
    print("Use multiple GPUs -", opt.gpu_id)
    model = torch.nn.DataParallel(model)

model.cuda()

if (opt.load is not None):
    # model.load_state_dict(torch.load(opt.load))
    model.load_pre(opt.load) # single GPU
    # model.module.load_pre(opt.load)  # multi-GPU  https://blog.csdn.net/MarrieChen/article/details/105969847
    print('load model from ', opt.load)

num_parms = 0
for p in model.parameters():
    num_parms += p.numel()
logging.info("Total Parameters (For Reference): {}".format(num_parms))
print("Total Parameters (For Reference): {}".format(num_parms))

params = model.parameters()
optimizer = torch.optim.Adam(params, opt.lr)


# load data
print('load data...')
train_loader = get_loader(image_root, gt_root,depth_root, batchsize=opt.batchsize, trainsize=opt.trainsize)
test_loader = test_dataset(test_image_root, test_gt_root,test_depth_root, opt.trainsize)
total_step = len(train_loader)

logging.info("Config")
logging.info(
    'epoch:{};lr:{};batchsize:{};trainsize:{};clip:{};decay_rate:{};load:{};save_path:{};decay_epoch:{}'.format(
        opt.epoch, opt.lr, opt.batchsize, opt.trainsize, opt.clip, opt.decay_rate, opt.load, save_path,
        opt.decay_epoch))


step = 0
writer = SummaryWriter(save_path + 'summary')
best_mae = 1
best_epoch = 0


# train function
def train(train_loader, model, optimizer, epoch, save_path):
    global step
    model.train()
    loss_all = 0
    epoch_step = 0

    try:
        for i, (images, gts, depth) in enumerate(train_loader, start=1):
            optimizer.zero_grad()

            images = images.cuda()
            gts = gts.cuda()
            depth = depth.repeat(1,3,1,1).cuda()

            s, r4cl, t4cl = model(images, depth)

            sal_loss = bce_iou_loss(s, gts)
            cl_total = cl_loss(t4cl, r4cl)
            loss = sal_loss + cl_total

            # loss.backward()
            loss.backward(retain_graph=True)

            clip_gradient(optimizer, opt.clip)
            optimizer.step()
            step += 1
            epoch_step += 1
            loss_all += loss.data
            memory_used = torch.cuda.max_memory_allocated() / (1024.0 * 1024.0)
            if i % 100 == 0 or i == total_step or i == 1:
                print('{} Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f} || sal_loss:{:4f}, cl_loss:{:4f}'.
                      format(datetime.now(), epoch, opt.epoch, i, total_step,
                             optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data, cl_total.data))
                logging.info(
                    '#TRAIN#:Epoch [{:03d}/{:03d}], Step [{:04d}/{:04d}], LR:{:.7f}, |sal_loss:{:4f}, cl_loss:{:4f}, mem_use:{:.0f}MB'.
                        format(epoch, opt.epoch, i, total_step, optimizer.state_dict()['param_groups'][0]['lr'], sal_loss.data, cl_total.data, memory_used))
                writer.add_scalar('Loss', loss.data, global_step=step)
                grid_image = make_grid(images[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('RGB', grid_image, step)
                grid_image = make_grid(gts[0].clone().cpu().data, 1, normalize=True)
                writer.add_image('Ground_truth', grid_image, step)
                res = s[0].clone()
                res = res.sigmoid().data.cpu().numpy().squeeze()
                res = (res - res.min()) / (res.max() - res.min() + 1e-8)
                writer.add_image('res', torch.tensor(res), step, dataformats='HW')
        # sal_loss_all /= epoch_step
        loss_all /= epoch_step
        logging.info('#TRAIN#:Epoch [{:03d}/{:03d}],Loss_AVG: {:.4f}'.format(epoch, opt.epoch, loss_all))
        writer.add_scalar('Loss-epoch', loss_all, global_step=epoch)
        if (epoch) % 10 == 0:
            torch.save(model.state_dict(), save_path + 'AlignSal_{}.pth'.format(epoch))
    except KeyboardInterrupt:
        print('Keyboard Interrupt: save model and exit.')
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        torch.save(model.state_dict(), save_path + 'AlignSal_{}.pth'.format(epoch + 1))
        print('save checkpoints successfully!')
        raise


# Initialize the counter for best models
best_model_count = 1  # Start from 1

# test function
def test(test_loader, model, epoch, save_path):
    model.eval()
    with torch.no_grad(): # aim to save memory
        mae_sum = 0    global best_mae, best_epoch

        for i in range(test_loader.size):
            image, gt, depth, name, img_for_post = test_loader.load_data()

            gt = np.asarray(gt, np.float32)
            gt /= (gt.max() + 1e-8)
            image = image.cuda()
            depth = depth.repeat(1,3,1,1).cuda()
            res,  r4cl, t4cl = model(image,depth)
            res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
            res = res.sigmoid().data.cpu().numpy().squeeze()
            res = (res - res.min()) / (res.max() - res.min() + 1e-8)
            mae_sum += np.sum(np.abs(res - gt)) * 1.0 / (gt.shape[0] * gt.shape[1])
        mae = mae_sum / test_loader.size
        writer.add_scalar('MAE', torch.tensor(mae), global_step=epoch)
        print('Epoch: {} MAE: {} ####  bestMAE: {} bestEpoch: {}'.format(epoch, mae, best_mae, best_epoch))
        if epoch == 1:
            best_mae = mae
        else:
            if mae < best_mae:
                best_mae = mae
                best_epoch = epoch
                # Save each best model with an incrementing counter
                torch.save(model.state_dict(), save_path + f'AlignSal_best_{best_model_count}.pth')
                best_model_count += 1
                print('Saved best model: AlignSal_epoch_best_{}.pth'.format(best_model_count - 1))
            logging.info('#TEST#:Epoch:{} MAE:{} bestEpoch:{} bestMAE:{}'.format(epoch, mae, best_epoch, best_mae))


if __name__ == '__main__':
    print("Start train...")
    for epoch in range(1, opt.epoch):
        cur_lr = adjust_lr(optimizer, opt.lr, epoch, opt.decay_rate, opt.decay_epoch)
        writer.add_scalar('learning_rate', cur_lr, global_step=epoch)
        train(train_loader, model, optimizer, epoch, save_path)

        test(test_loader, model, epoch, save_path)
