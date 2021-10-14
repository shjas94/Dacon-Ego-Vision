import os
import json
import glob
import numpy as np
import pandas as pd
import cv2
from torch.utils.data import Dataset


class EvDataset(Dataset):
    '''
    paths : paths splitted by Kfold 
    additional : set excluded for stratified split
    mode : train/valid/test
    '''

    def __init__(self, root, paths, img_size, transforms=None, augmentations=None, additional=None, img_padding=0, mode='train'):
        super().__init__()
        self.root = root
        self.paths = paths
        self.additional = additional
        self.mode = mode
        self.data = self.get_img_meta_list()
        self.transforms = transforms
        self.augmentations = augmentations
        self.img_size = img_size
        self.img_padding = img_padding

    def get_img_meta_list(self):
        data = []
        for path in self.paths.values:
            imgs = glob.glob(os.path.join(path[0], '*.png'))
            imgs = sorted(imgs, key=lambda x: int(
                x.split('/')[5].split('.')[0]))
            meta = glob.glob(os.path.join(path[0], '*.json'))
            temp_meta = json.load(open(meta[0]))['annotations']
            temp_img_id_list = [i['image_id'] for i in temp_meta]
            if self.mode == 'train':
                for i in range(len(imgs)):
                    if int(imgs[i].split('/')[5].split('.')[0]) not in temp_img_id_list:
                        continue
                    data.append((imgs[i], meta[0]))
            else:
                for i in range(len(imgs)):
                    if int(imgs[i].split('/')[5].split('.')[0]) not in temp_img_id_list:
                        continue
                    data.append((imgs[i], meta[0]))
        if self.additional:
            for a in self.additional:
                imgs = glob.glob(os.path.join(a, '*.png'))
                imgs = sorted(imgs, key=lambda x: int(
                    x.split('/')[5].split('.')[0]))
                meta = glob.glob(os.path.join(a, '*.json'))
                temp_meta = json.load(open(meta[0]))['annotations']
                temp_img_id_list = [i['image_id'] for i in temp_meta]
                if self.mode == 'train':
                    for i in range(1, len(imgs)//2):
                        if int(imgs[i].split('/')[5].split('.')[0]) not in temp_img_id_list:
                            continue
                        data.append((imgs[i], meta[0]))
                elif self.mode == 'valid':
                    for i in range(len(imgs)//2, len(imgs)):
                        if int(imgs[0].split('/')[5].split('.')[0]) not in temp_img_id_list:
                            continue
                        data.append((imgs[i], meta[0]))
        return data

    def preprocess_img(self, img_path, annotation):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)

        annot_i = annotation['data']
        y_min = int(min(annot_i, key=lambda x: x[1])[1]) - self.img_padding if int(min(annot_i, key=lambda x: x[1])[1]) - self.img_padding > 0 \
            else int(min(annot_i, key=lambda x: x[1])[1])
        y_max = int(max(annot_i, key=lambda x: x[1])[1]) + self.img_padding if int(max(annot_i, key=lambda x: x[1])[1]) + self.img_padding < img.shape[1] \
            else int(max(annot_i, key=lambda x: x[1])[1])
        x_min = int(min(annot_i, key=lambda x: x[0])[0]) - self.img_padding if int(min(annot_i, key=lambda x: x[0])[0]) - self.img_padding > 0 \
            else int(min(annot_i, key=lambda x: x[0])[0])
        x_max = int(max(annot_i, key=lambda x: x[0])[0]) + self.img_padding if int(max(annot_i, key=lambda x: x[0])[0]) + self.img_padding < img.shape[0] \
            else int(max(annot_i, key=lambda x: x[0])[0])
        x_min = 0 if x_min < 0 else x_min
        y_min = 0 if y_min < 0 else y_min
        img = img[y_min:y_max, x_min:x_max, :]  # crop
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.

        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))  # resize

        return img

    def __getitem__(self, index: int):
        img_path, meta_path = self.data[index]
        img_id = int(img_path.split('/')[5].split('.')[0])
        meta = json.load(open(meta_path))

        label = meta['action'][0]
        for annotation in meta['annotations']:
            if annotation['image_id'] == img_id:
                keypoint = annotation

        img = self.preprocess_img(img_path, annotation)

        pose_data = pd.read_csv(os.path.join(
            self.root, 'hand_gesture_pose.csv'))
        pose_id = pose_data['pose_id'].values
        label = np.squeeze(np.where(pose_id == label)[0])

        data = {'image': img, 'label': label}

        if self.augmentations and self.mode != 'valid':
            data['image'] = self.augmentations(image=data['image'])['image']

        if self.transforms:
            data['image'] = self.transforms(image=data['image'])

        return data

    def __len__(self):
        return len(self.data)


class EvTestDataset(Dataset):
    def __init__(self, paths, img_size, img_padding=0, transforms=None):
        super().__init__()
        self.paths = paths
        self.data = self.get_img_meta_list()
        self.transforms = transforms
        self.img_size = img_size
        self.img_padding = img_padding

    def get_img_meta_list(self):
        data = []
        for path in self.paths.values:
            imgs = glob.glob(os.path.join(path[0], '*.png'))
            imgs = sorted(imgs, key=lambda x: int(
                x.split('/')[5].split('.')[0]))
            meta = glob.glob(os.path.join(path[0], '*.json'))
            for i in range(len(imgs)):
                data.append((imgs[i], meta[0]))

        return data

    def preprocess_img(self, img_path, annotation):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(np.float32)

        annot_i = annotation['data']
        y_min = int(min(annot_i, key=lambda x: x[1])[1]) - self.img_padding if int(min(annot_i, key=lambda x: x[1])[1]) - self.img_padding > 0 \
            else int(min(annot_i, key=lambda x: x[1])[1])
        y_max = int(max(annot_i, key=lambda x: x[1])[1]) + self.img_padding if int(max(annot_i, key=lambda x: x[1])[1]) + self.img_padding < img.shape[1] \
            else int(max(annot_i, key=lambda x: x[1])[1])
        x_min = int(min(annot_i, key=lambda x: x[0])[0]) - self.img_padding if int(min(annot_i, key=lambda x: x[0])[0]) - self.img_padding > 0 \
            else int(min(annot_i, key=lambda x: x[0])[0])
        x_max = int(max(annot_i, key=lambda x: x[0])[0]) + self.img_padding if int(max(annot_i, key=lambda x: x[0])[0]) + self.img_padding < img.shape[0] \
            else int(max(annot_i, key=lambda x: x[0])[0])
        x_min = 0 if x_min < 0 else x_min
        y_min = 0 if y_min < 0 else y_min
        img = img[y_min:y_max, x_min:x_max, :]  # crop
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)/255.
        img = cv2.resize(img, (self.img_size[0], self.img_size[1]))  # resize

        return img

    def __getitem__(self, index: int):
        img_path, meta_path = self.data[index]
        img_id = int(img_path.split('/')[5].split('.')[0])
        meta = json.load(open(meta_path))

        for annotation in meta['annotations']:
            if annotation['image_id'] == img_id:
                keypoint = annotation

        img = self.preprocess_img(img_path, annotation)

        data = {'image': img}

        if self.transforms:
            data['image'] = self.transforms(image=data['image'])

        return data

    def __len__(self):
        return len(self.data)
