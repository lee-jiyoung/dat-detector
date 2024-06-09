import torch
from torch.utils.data import Dataset
import os
import glob
import random 
import cv2
from torchvision import transforms
from PIL import Image
import numpy as np
# import shutil


class VideoDataset(Dataset):
    def __init__(self, 
                dataset_name='ucf24', root_dir='./data',\
                mode='train', n_frame=6, transform=None, stage=1, load_all=False):
        self.dataset_name = dataset_name
        self.root_dir = root_dir
        self.transform = transform
        self.mode = mode
        self.n_frame = n_frame
        self.stage = stage
        self.all = load_all
        
        with open('./data/'+dataset_name+'.txt', 'r') as f:
            vlist = f.readlines()

        self.keys = []
        self.c = []
        for v in vlist:
            x = v.split()
            if x[2] == mode:
                self.keys.append(('/').join(x[-1].split('/')[0:2]))
                self.c.append(x[1])
        
        self.class_to_idx = dict()
        for i, name in enumerate(list(set(self.c))):
            self.class_to_idx[name] = i

        self.n_class = len(self.class_to_idx.keys())

    def __len__(self):
        return len(self.keys)
    
    def load_frame(self, fpath):
        img = cv2.imread(fpath)
        img = cv2.resize(img, (256, 256))
        img = img.astype("float32").transpose(2,0,1)
        if self.transform is not None:
            transformed_img = self.transform(img)
            return transformed_img
        else:
            return img
    
    def __getitem__(self, idx):
        rgbs, flows, rois = [], [], []
        
        v_name = self.keys[idx]
        
        if (self.stage == 1) & (self.dataset_name != 'imagenet_vod'):
            c = 1
            c = torch.FloatTensor([c])
        elif (self.stage == 1) & (self.dataset_name == 'imagenet_vod'):
            c = 0
            c = torch.FloatTensor([c])
        else:
            c = self.class_to_idx[self.c[idx]]
        
        frames = glob.glob(os.path.join(self.root_dir, v_name, 'rgb', '*.jpg'))
        frames.sort()
        v_len = len(frames)
        
        
        if self.all:
            sidx = 0
            n = v_len
        else:
            sidx = random.randint(0, v_len - self.n_frame - 1)
            n = sidx + self.n_frame
        
        for i in range(sidx, n):
            rgbs.append(
                torch.from_numpy(self.load_frame(frames[i]))
                )
            flows.append(
                torch.from_numpy(self.load_frame(frames[i]))
                )
            
            roifile = np.load(frames[i].replace('/rgb/', '/rois/').replace('.jpg', '.npy'), allow_pickle=True)
            rois.append(roifile.item()['proposal'].tensor)
        
        if self.stage == 2:
            tubelets = np.load(os.path.join(self.root_dir, v_name+'_tubelets.npy'), allow_pickle=True)
            s_tube = tubelets[sidx]
            new_rois = []
            for i in range(self.n_frame):
                new_rois.append(rois[i][s_tube[:,i]])
            
            rois = new_rois
        return torch.stack(rgbs,dim=0), torch.stack(flows,dim=0), torch.stack(rois,dim=0), c, v_name
        

class ImageDataset(Dataset):
    def __init__(self, dataset_name='ucf24', root_dir='./data'):
        self.root_dir = root_dir
        
        
        with open('./data/'+dataset_name+'.txt', 'r') as f:
            vlist = f.readlines()

        self.keys = []
        self.c = []
        for v in vlist:
            x = v.split()
            v_name = ('/').join(x[-1].split('/')[0:2])
            frames = glob.glob(os.path.join(self.root_dir, v_name, 'rgb', '*.jpg'))
            self.keys += frames
        

    def __len__(self):
        return len(self.keys)
    
    def load_frame(self, fpath):
        img = cv2.imread(fpath)
        img = cv2.resize(img, (800, 800))
        # img_tensor = torch.from_numpy(np.transpose(img,(2,0,1)))
        img_tensor = torch.as_tensor(img.astype("float32").transpose(2,0,1))
        return img_tensor
    
    def __getitem__(self, idx):
        rgbs, flows = [], []
        
        fname = self.keys[idx]
        rgb = self.load_frame(fname)
            
        return rgb, fname

