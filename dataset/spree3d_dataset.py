from email.mime import base
from math import ceil
from sympy import intervals
import torch
import os
import json
import numpy as np
import random
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as F
from glob import glob
from tqdm import tqdm
import pickle as pkl
from skimage.io import imread
import cv2

class Spree3DDataset(Dataset):
    def __init__(self, base_dir, mask_dir, seq_length=4, split='train', transform=None):
        self.base_dir = base_dir
        self.mask_dir = mask_dir
        self.split = split
        self.picked_paths = f"{self.base_dir}/pickled_paths.pkl"
        self.ratio = 0.85 if split == 'train' else 0.15
        self.load_or_create_pkl()

        self.seq_length = seq_length
        self.transform = transform
        
    def load_or_create_pkl(self):
        if os.path.isfile(self.picked_paths):
            ret = pkl.load(open(self.picked_paths, 'rb'))
            self.frame_paths = ret[0]
            self.mask_paths = ret[1]
            self.vid_bounds = ret[2]
        else:
            all_id_dirs = sorted(os.listdir(self.base_dir))
            size = int(len(all_id_dirs)*self.ratio)
            split_id_dirs = all_id_dirs[:size] if self.split=='train' else all_id_dirs[-size:]

            frame_paths = []
            mask_paths = []
            vid_bounds = []
            for id_dir in split_id_dirs:
                vid_frames_paths = sorted(glob(f"{self.base_dir}/{id_dir}/*", recursive=True))
                mask_paths = sorted(glob(f"{self.mask_dir}/{id_dir}/*", recursive=True))
                interval = (len(frame_paths), len(frame_paths)+len(mask_paths))
                frame_paths.extend(vid_frames_paths)
                mask_paths.extend(mask_paths)
                vid_bounds.extend([interval]*len(mask_paths))

            to_save = [frame_paths, mask_paths, vid_bounds]
            pkl.dump(to_save, open(self.picked_paths, 'wb'))
            
            self.frame_paths = frame_paths
            self.mask_paths = mask_paths
            self.vid_bounds = vid_bounds
            return frame_paths, mask_paths, vid_bounds

    def __len__(self):
        all_sizes = [len(vid) for vid in self.frame_paths]
        total_size = sum(all_sizes)
        return total_size
    

    def process_frame(self, f_img):
        return f_img
        if f_img.shape[:2] != self.img_size:
            f_img = cv2.resize(f_img, self.img_size)
        #f_img = np.array(f_img, dtype=np.uint8)
        f_img = torch.from_numpy(f_img.astype(np.float32)).permute(2, 0, 1)/255.0
        return f_img

    def process_mask(self, mask_old):
        return mask_old
        mask = np.zeros_like(mask_old)
        mask[mask_old == 1] = 1
        mask[mask_old == 2] = 2
        mask[mask_old == 13] = 3
        mask[mask_old == 4] = 4
        mask[mask_old == 20] = 5
        mask[mask_old == 10] = 6
        mask[mask_old == 5] = 7
        mask[mask_old == 7] = 7
        mask[mask_old == 14] = 8
        mask[mask_old == 15] = 8

        
        if mask.shape != self.img_size:
            mask = cv2.resize(mask, self.img_size, cv2.INTER_NEAREST)
        
        mask = torch.from_numpy(mask.astype(np.int))
        return mask

    def __getitem__(self, idx):

        interval = self.vid_bounds[idx]
        idx -=  (idx + self.seq_length) - interval[1]

        frames = []
        masks = []

        for t in range(self.seq_length):
            frame_path = self.frame_paths[idx+t]
            mask_path = self.mask_paths[idx+t]

            frame = imread(frame_path)
            frame = self.process_frame(frame)

            mask = imread(mask_path, as_gray=True)
            mask = self.process_mask(mask)

            frames.append(frame)
            masks.append(mask)
        
        return frames, masks

