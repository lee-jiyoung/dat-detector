from __future__ import absolute_import

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import torchvision.ops 
import numpy as np
import argparse
from net import DATdetector
from datasets import VideoDataset, ImageDataset

import sys
import os
import detectron2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.modeling import build_model
from detectron2.structures.image_list import ImageList
from typing import List, Tuple, Union
from detectron2.layers import batched_nms, cat, move_device_like
from detectron2.structures import Boxes, Instances

from videonet.network import resnet101
from tqdm import tqdm

def train(model, backbone, d_type, train_loader, criterion, optimizer, epoch, stage):
    model.train()
    
    p = 1000 if stage == 1 else 300
    
    for batch_idx, (rgb, flow, roi, target, _) in enumerate(train_loader):
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


def test(model, rcnn, test_loader, criterion):
    with torch.no_grad():
        model.eval()
        p = 1000 if stage == 1 else 300
        test_loss = 0
        correct = 0
        for (rgb, flow, roi, target, _) in test_loader:
            rgb, flow = rgb.cuda(), flow.cuda()
            target = target.cuda()
            
            clip = rgb if d_type == 'rgb' else flow
            clip = clip.cuda()
            
            b, n, c, h, w = clip.size()
            clip = clip.reshape(b*n, c, h, w)
            roi = roi.reshape(b*n, p, 4)
            roi = [roi[i, :, :] for i in range(b*n)]

            x = backbone(clip)
            x_roi = torchvision.ops.roi_pool(x, roi, 7)
        
            output = model(x_roi, train=True, b=b)

            test_loss += criterion(output, target, size_average=False).item()
            pred = output.max(1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        test_loss /= len(test_loader.dataset)
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'
              .format(test_loss, correct, len(test_loader.dataset),
                      100. * correct / len(test_loader.dataset)))

def _is_tracing():
    # (fixed in TORCH_VERSION >= 1.9)
    if torch.jit.is_scripting():
        # https://github.com/pytorch/pytorch/issues/47379
        return False
    else:
        return torch.jit.is_tracing()

def find_top_rpn_proposals(
    proposals: List[torch.Tensor],
    pred_objectness_logits: List[torch.Tensor],
    image_sizes: List[Tuple[int, int]],
    nms_thresh: float,
    pre_nms_topk: int,
    post_nms_topk: int,
    min_box_size: float,
):
    """
    For each feature map, select the `pre_nms_topk` highest scoring proposals,
    apply NMS, clip proposals, and remove small boxes. Return the `post_nms_topk`
    highest scoring proposals among all the feature maps for each image.

    Args:
        proposals (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A, 4).
            All proposal predictions on the feature maps.
        pred_objectness_logits (list[Tensor]): A list of L tensors. Tensor i has shape (N, Hi*Wi*A).
        image_sizes (list[tuple]): sizes (h, w) for each image
        nms_thresh (float): IoU threshold to use for NMS
        pre_nms_topk (int): number of top k scoring proposals to keep before applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is per
            feature map.
        post_nms_topk (int): number of top k scoring proposals to keep after applying NMS.
            When RPN is run on multiple feature maps (as in FPN) this number is total,
            over all feature maps.
        min_box_size (float): minimum proposal box side length in pixels (absolute units
            wrt input images).

    Returns:
        list[Boxes]: list of N proposal boxes.
    """
    num_images = len(image_sizes)
    device = (
        proposals[0].device
        if torch.jit.is_scripting()
        else ("cpu" if torch.jit.is_tracing() else proposals[0].device)
    )

    # 1. Select top-k anchor for every level and every image
    topk_scores = []  # #lvl Tensor, each of shape N x topk
    topk_proposals = []
    level_ids = []  # #lvl Tensor, each of shape (topk,)
    batch_idx = move_device_like(torch.arange(num_images, device=device), proposals[0])
    for level_id, (proposals_i, logits_i) in enumerate(zip(proposals, pred_objectness_logits)):
        Hi_Wi_A = logits_i.shape[1]
        if isinstance(Hi_Wi_A, torch.Tensor):  # it's a tensor in tracing
            num_proposals_i = torch.clamp(Hi_Wi_A, max=pre_nms_topk)
        else:
            num_proposals_i = min(Hi_Wi_A, pre_nms_topk)

        topk_scores_i, topk_idx = logits_i.topk(num_proposals_i, dim=1)

        # each is N x topk
        topk_proposals_i = proposals_i[batch_idx[:, None], topk_idx]  # N x topk x 4

        topk_proposals.append(topk_proposals_i)
        topk_scores.append(topk_scores_i)
        level_ids.append(
            move_device_like(
                torch.full((num_proposals_i,), level_id, dtype=torch.int64, device=device),
                proposals[0],
            )
        )

    # 2. Concat all levels together
    topk_scores = cat(topk_scores, dim=1)
    topk_proposals = cat(topk_proposals, dim=1)
    level_ids = cat(level_ids, dim=0)  
    b = []

    for n, image_size in enumerate(image_sizes):
        boxes = Boxes(topk_proposals[n])
        scores_per_img = topk_scores[n]
        lvl = level_ids

        valid_mask = torch.isfinite(boxes.tensor).all(dim=1) & torch.isfinite(scores_per_img)
        if not valid_mask.all():
            if training:
                raise FloatingPointError(
                    "Predicted boxes or scores contain Inf/NaN. Training has diverged."
                )
            boxes = boxes[valid_mask]
            scores_per_img = scores_per_img[valid_mask]
            lvl = lvl[valid_mask]
        boxes.clip(image_size)
        save_proposal = boxes[:post_nms_topk]
        save_proposal.tensor = save_proposal.tensor / 800.0
        b.append(save_proposal)

    return b


def run_rcnn(rcnn, data_loader):
    with torch.no_grad():
        for data, paths in tqdm(data_loader):
            data = ImageList.from_tensors(data, size_divisibility=32).to("cuda")
            features = rcnn.backbone(data.tensor)

            rpn_features = [features[f] for f in rcnn.proposal_generator.in_features]
            anchors = rcnn.proposal_generator.anchor_generator(rpn_features)

            pred_objectness_logits, pred_anchor_deltas = rcnn.proposal_generator.rpn_head(rpn_features)
            # Transpose the Hi*Wi*A dimension to the middle:
            pred_objectness_logits = [
                # (N, A, Hi, Wi) -> (N, Hi, Wi, A) -> (N, Hi*Wi*A)
                score.permute(0, 2, 3, 1).flatten(1)
                for score in pred_objectness_logits
            ]
            pred_anchor_deltas = [
                # (N, A*B, Hi, Wi) -> (N, A, B, Hi, Wi) -> (N, Hi, Wi, A, B) -> (N, Hi*Wi*A, B)
                x.view(x.shape[0], -1, rcnn.proposal_generator.anchor_generator.box_dim, x.shape[-2], x.shape[-1])
                .permute(0, 3, 4, 1, 2)
                .flatten(1, -2)
                for x in pred_anchor_deltas
            ]

            pred_proposals = rcnn.proposal_generator._decode_proposals(anchors, pred_anchor_deltas)

            proposals = find_top_rpn_proposals(
                pred_proposals,
                pred_objectness_logits,
                data.image_sizes,
                rcnn.proposal_generator.nms_thresh,
                rcnn.proposal_generator.pre_nms_topk[rcnn.proposal_generator.training],
                rcnn.proposal_generator.post_nms_topk[rcnn.proposal_generator.training],
                rcnn.proposal_generator.min_box_size,
            )
            
            for i, p in tqdm(enumerate(paths)):
                if not os.path.exists('/'.join(p.replace('rgb', 'rois').split('/')[0:-1])):
                    os.makedirs('/'.join(p.replace('rgb', 'rois').split('/')[0:-1]))
                np.save(p.replace('rgb', 'rois').replace('.jpg', '.npy'), {'proposal': proposals[i]}) 
            
    return 'done'

def image_collate_fn(batch):
    rgbs, paths = [], []
    for rgb, path in batch:
        rgbs.append(rgb)
        paths.append(path)

    return rgbs, paths

datasets = ['ucf_sports', 'ucf24', 'jhmdb', 'imagenet_vod']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0001, type=float)
    parser.add_argument("--n_epoch", default=10, type=int)
    parser.add_argument("--mode", default='train', type=str)
    parser.add_argument("--gamma", default=0.9, type=float)
    parser.add_argument("--stage", default=1, type=int)
    parser.add_argument("--spatial_ckpt", default='ckpts/spatial_cnn.pth.tar', type=str)
    parser.add_argument("--motion_ckpt", default='ckpts/motion_cnn.pth.tar', type=str)
    parser.add_argument("--dataset", default='ucf_sports', choices=datasets, type=str)
    parser.add_argument("--dataroot_dir", default='./data/ucf_sports', type=str)
    parser.add_argument("--n_batch", default=2, type=int)
    parser.add_argument("--n_frame", default=6, type=int)
    parser.add_argument("--d_type", default='rgb', type=str)
    parser.add_argument("--save_dir", default='ckpts/', type=str)
    opt = parser.parse_args()

    train_dataset = VideoDataset(
        dataset_name=opt.dataset, root_dir=opt.dataroot_dir, mode='train', n_frame=opt.n_frame, stage=opt.stage)
    val_dataset = VideoDataset(
        dataset_name=opt.dataset, root_dir=opt.dataroot_dir, mode='test', n_frame=opt.n_frame, stage=opt.stage)
    
    
    # Load RPN
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml"))
    # Find a model from detectron2's model zoo. You can use the https://dl.fbaipublicfiles... url as well
    cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url("COCO-Detection/faster_rcnn_R_50_FPN_1x.yaml")
    rcnn = build_model(cfg)
    rcnn.proposal_generator.nms_thresh=0.0
    rcnn.eval()
        
    if opt.stage == 0:
        img_dataset = ImageDataset(dataset_name='ucf_sports', root_dir='./data/ucf_sports')
        rcnn_train_loader = torch.utils.data.DataLoader(img_dataset, batch_size=opt.n_batch, shuffle=False, collate_fn=image_collate_fn)
        rpns = run_rcnn(rcnn, rcnn_train_loader)

        print('Done extracting roi proposals!')
    
    else:
        if opt.d_type == 'rgb':
            spatial_ckpt = torch.load(opt.spatial_ckpt)
            backbone = resnet101(pretrained= True, channel=3).cuda()
            backbone.load_state_dict(spatial_ckpt['state_dict'])
        elif opt.d_type == 'flow':
            motion_ckpt = torch.load(opt.motion_ckpt)
            backbone = resnet101(pretrained= True, channel=20).cuda()
            backbone.load_state_dict(motion_ckpt['state_dict'])
        

        if opt.stage == 1:
            model = DATdetector(opt.stage, 2).cuda()
            train_dataset_neg = VideoDataset(
                dataset_name=opt.dataset, root_dir=opt.dataroot_dir, mode='train', n_frame=opt.n_frame, stage=opt.stage)
            val_dataset_neg = VideoDataset(
                dataset_name=opt.dataset, root_dir=opt.dataroot_dir, mode='test', n_frame=opt.n_frame, stage=opt.stage)
    
            train_dataset = torch.utils.data.ConcatDataset([train_dataset, train_dataset_neg])
            val_dataset = torch.utils.data.ConcatDataset([val_dataset, val_dataset_neg])
            loss = nn.BCELoss()

        elif opt.stage == 2:
            model = DATdetector(opt.stage, train_dataset.n_class).cuda()
            loss = nn.CrossEntropyLoss()

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.n_batch, shuffle=True)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=opt.n_batch, shuffle=False)

        optimizer = optim.Adam(model.parameters(), lr=opt.lr)
        
        for i in range(opt.n_epoch):
            
            train(model, backbone, opt.d_type, train_loader, loss, optimizer, i, opt.stage)

            test(model, backbone, opt.d_type, test_loader, loss)
            
            torch.save(model.state_dict(), os.path.join(opt.save_dir, f'stage{opt.stage}_epoch_{i}.pt'))

        print('Done Training!')
        