from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.ops 
import numpy as np
import argparse
from net import DATdetector
from datasets import VideoDataset

import sys
import os
import torchvision
from videonet.network import resnet101
from tqdm import tqdm

def train(model, backbone, d_type, train_loader, criterion, optimizer, epoch, stage):
    model.train()
    
    # p = 1000 if stage == 1 else 1
    p = 1000 if stage == 1 else 1000
    
    for batch_idx, (rgb, flow, roi, target) in enumerate(train_loader):
        rgb, flow = rgb.cuda(), flow.cuda()
        clip = rgb if d_type == 'rgb' else flow
        
        clip = clip.cuda()
        target = target.cuda()
        
        b, n, c, h, w = clip.size()
        clip = clip.reshape(b*n, c, h, w)
        roi = roi.reshape(b*n, p, 4)
        roi = [roi[i, :, :] for i in range(b*n)]
        
        with torch.no_grad():
            x = backbone(clip)
            x_roi = torchvision.ops.roi_pool(x, roi, 7)
            
        optimizer.zero_grad()
        output = model(x_roi, train=True, b=b)
        
        loss = criterion(output, target)
        loss.backward()
        
        optimizer.step()
        
        if batch_idx % 50 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * b, len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def image_collate_fn(batch):
    rgbs, paths = [], []
    for rgb, path in batch:
        rgbs.append(rgb)
        paths.append(path)

    return rgbs, paths

datasets = ['ucf_sports', 'ucf24', 'jhmdb']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--beta", default=0.9, type=float)
    parser.add_argument("--spatial_ckpt", default='ckpts/spatial_cnn.pth.tar', type=str)
    parser.add_argument("--motion_ckpt", default='ckpts/motion_cnn.pth.tar', type=str)
    parser.add_argument("--dataset", default='ucf_sports', choices=datasets, type=str)
    parser.add_argument("--dataroot_dir", default='./data/ucf_sports', type=str)
    parser.add_argument("--n_batch", default=1, type=int)
    parser.add_argument("--select_n", default=300, type=float)
    parser.add_argument("--n_frame", default=6, type=int)
    parser.add_argument("--d_type", default='rgb', type=str)
    parser.add_argument("--atp_ckpt", default='ckpts/stage1_best.pt', type=str)
    opt = parser.parse_args()

    test_dataset = VideoDataset(
        dataset_name=opt.dataset, root_dir=opt.dataroot_dir, 
        mode='test', n_frame=1, stage=1, load_all=True)
    train_dataset = VideoDataset(
        dataset_name=opt.dataset, root_dir=opt.dataroot_dir, 
        mode='train', n_frame=1, stage=1, load_all=True)
    
    all_dataset = torch.utils.data.ConcatDataset([train_dataset, test_dataset])

    frame_loader = torch.utils.data.DataLoader(all_dataset, batch_size=opt.n_batch, shuffle=False)
    
    if opt.d_type == 'rgb':
        spatial_ckpt = torch.load(opt.spatial_ckpt)
        backbone = resnet101(pretrained= True, channel=3).cuda()
        backbone.load_state_dict(spatial_ckpt['state_dict'])
    elif opt.d_type == 'flow':
        motion_ckpt = torch.load(opt.motion_ckpt)
        backbone = resnet101(pretrained= True, channel=20).cuda()
        backbone.load_state_dict(motion_ckpt['state_dict'])
            
    model_ckpt = torch.load(opt.atp_ckpt)
    model = DATdetector(mode=1, n_class=2).cuda()
    model.load_state_dict(model_ckpt)
    model.eval()
    
    sim = nn.CosineSimilarity()
    tubelets = []
    with torch.no_grad():
        for batch_idx, (rgb, flow, roi, _, vname) in enumerate(frame_loader):
            x_roi = []
            clip = rgb if opt.d_type == 'rgb' else flow
            
            clip = clip.cuda()
            b, n, c, h, w = clip.size()
            clip = clip.reshape(b*n, c, h, w)
            roi = roi.reshape(b*n, 1000, 4)
            roi = [roi[i, :, :] for i in range(n)]
            
            x = backbone(clip)
            best_links = torch.zeros(1000, 1000)
            best_scores = torch.zeros(1000, 1000)
            for i in range(n):
                x_roi = torchvision.ops.roi_pool(x[[i], :, :, :], [roi[i]], 7)
                actor_score = model(x_roi, train=False, b=b)
                if i > 0:
                    iou = torchvision.ops.box_iou(roi[i-1], roi[i])
                    best_link, best_score = [], []
                    for k in range(1000):
                        link_score = pre_actor_score[k] \
                                        + actor_score \
                                        + opt.beta*(
                                            iou[k, :]
                                            + sim(pre_x_roi[k], x_roi.reshape(1000, -1))
                                            )
                        best_link.append(np.argsort(link_score)[-1])
                        best_score.append(np.max(link_score))
                    best_links[i-1] = best_link
                    best_scores[i-1] = best_score
                        
                if i >= opt.n_frame:
                    tubelet_link = torch.zeros(1000, 6)
                    tubelet_score = torch.zeros(1000)
                    tubelet_link[:, 0] = torch.tensor(list(range(1000)))
                    for k in range(1000):
                        for n in reversed(range(1, opt.n_frame)):
                            tubelet_link[k,opt.n_frame-n] = best_links[i-n][k]
                            tubelet_score[k] += best_scores[i-n][k]
                    idx = np.argsort(-tubelet_score)[:opt.select_n]
                    tubelets.append(tubelet_link[idx])
                    
                pre_actor_score = actor_score
                pre_x_roi = x_roi.reshape(1000, -1)

            np.save(vname+'_tubelets.npy', np.array(tubelets))
            
    print('Done linking roi proposals!')
    