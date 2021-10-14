import os
import glob
import json
import yaml
import numpy as np
import pandas as pd
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
from tqdm import tqdm
import argparse
import wandb

from modules.dataset import *
from modules.models import *
from modules.utils import *
from modules.schedulers import *


def get_labels_with_directory(cfg):  # return dataframe with path, label
    train_dir = sorted(glob.glob(os.path.join(
        cfg['PATH']['DATA'], 'train', '*')), key=lambda x: int(x.split('/')[4]))
    paths = []
    labels = []
    for t in train_dir:
        json_file = glob.glob(os.path.join(t, '*.json'))
        json_file = json.load(open(json_file[0]))
        action = json_file['action'][0]
        paths.append(t)
        labels.append(action)
    labels_with_directory = pd.DataFrame({'path': paths, 'label': labels})
    return labels_with_directory


# return total dataframe(except for path with scarce label), excluded paths
def exclude_dirs(labels_with_directory, target_count):
    lab_counts = labels_with_directory['label'].value_counts()

    lab_counts_one = np.array(lab_counts[lab_counts == target_count].index)
    excluded_dir = []
    for idx in range(len(labels_with_directory)):
        for i in lab_counts_one:
            if labels_with_directory.loc[idx]['label'] == i:
                excluded_dir.append(labels_with_directory.loc[idx].path)
    for one in excluded_dir:
        labels_with_directory = labels_with_directory.drop(
            labels_with_directory[labels_with_directory['path'] == one].index)
    return labels_with_directory, excluded_dir


def get_fold(cfg, df):
    root = cfg['PATH']['DATA']
    train_root_dir = os.path.join(root, 'train')
    train_dirs = os.listdir(train_root_dir)
    train_dirs = sorted(train_dirs, key=lambda x: int(x))
    train_dirs = [os.path.join(train_root_dir, train_dirs[i])
                  for i in range(len(train_dirs))]

    skf = StratifiedKFold(n_splits=5, shuffle=True,
                          random_state=cfg['TRAIN']['SEED'])
    skf.get_n_splits(df, df.label)
    train_indices, valid_indices = [], []
    for train_index, valid_index in tqdm(skf.split(df, df.label)):
        train_indices.append(train_index)
        valid_indices.append(valid_index)
    return train_indices, valid_indices


def train(cfg, train_path, val_path, excluded, fold_num):
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    root = cfg['PATH']['DATA']
    pose_data = pd.read_csv(os.path.join(root, 'hand_gesture_pose.csv'))
    num_classes = len(pose_data['pose_id'].unique())
    torch.cuda.empty_cache()
    wandb.init(project='Ego_vision',
               group=cfg['MODEL']['MODEL_NAME'], name=cfg['TRAIN']['RUN_NAME']+'_'+str(fold_num), config=cfg)

    transforms = A.Compose([
                           ToTensorV2(p=1.0)
                           ])

    augmentations = A.Compose([
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05,
                           rotate_limit=30, p=0.5),
        A.RandomBrightnessContrast(p=0.5),
        A.CoarseDropout(max_height=10, max_width=10, p=0.5)
    ])

    train_set = EvDataset(transforms=transforms, img_size=cfg['TRAIN']['IMG_SIZE'],
                          augmentations=augmentations, root=cfg['PATH']['DATA'], paths=train_path, additional=excluded, img_padding=cfg['TRAIN']['TRAIN_IMG_PADDING'], mode='train')
    val_set = EvDataset(transforms=transforms, root=cfg['PATH']['DATA'],
                        img_size=cfg['TRAIN']['IMG_SIZE'], paths=val_path, additional=excluded, img_padding=cfg['TRAIN']['TEST_IMG_PADDING'], mode='valid')

    train_loader = DataLoader(
        train_set,
        batch_size=cfg['TRAIN']['TRAIN_BATCH_SIZE'],
        num_workers=3,
        shuffle=True,
        pin_memory=use_cuda,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=cfg['TRAIN']['VAL_BATCH_SIZE'],
        num_workers=3,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )

    model = get_model(cfg, num_classes)
    model.to(device)
    criterion1, criterion2 = None, None
    if cfg['TRAIN']['MULTI_CRITERION']:
        criterion1, criterion2 = get_criterion(
            cfg['TRAIN']['CRITERION1'], cfg), get_criterion(cfg['TRAIN']['CRITERION2'], cfg)
    else:
        criterion1 = get_criterion(cfg['TRAIN']['CRITERION1'])
    max_norm = 1.
    optimizer = get_optimizer(cfg, model)

    # Add Scheduler!!
    if cfg['TRAIN']['SCHEDULER'] == 'cosinewarmup':
        scheduler = CosineAnnealingWarmUpRestart(
            optimizer=optimizer, T_0=4, T_mult=1, eta_max=2e-4,  T_up=1, gamma=0.5)

    best_valid_loss = np.inf
    best_model = None
    wandb.watch(model)
    early_stopping = EarlyStopping(
        patience=cfg['TRAIN']['PATIENCE'], verbose=True)

    for epoch in range(cfg['TRAIN']['EPOCH']):
        train_loss_list = []
        with tqdm(train_loader,
                  total=train_loader.__len__(),
                  unit='batch') as train_bar:
            for sample in train_bar:
                train_bar.set_description(f"Train Epoch: {epoch}")
                optimizer.zero_grad()
                images, labels = sample['image']['image'].float(
                ), sample['label'].long()
                images = images.to(device)
                labels = labels.to(device)

                if cfg['TRAIN']['MIXUP'] == True:
                    images, label_a, label_b, lam = mixup_data(images, labels,
                                                               cfg['TRAIN']['ALPHA'], use_cuda)
                    images, label_a, label_b = map(Variable, (images,
                                                              label_a, label_b))
                model.train()

                with torch.set_grad_enabled(True):
                    preds = model(images)

                    if cfg['TRAIN']['MIXUP'] == True:
                        if cfg['TRAIN']['MULTI_CRITERION']:
                            loss = lam * (0.5*criterion1(preds, label_a) + 0.5*criterion2(preds, label_a)) + \
                                (1-lam)*(0.5*criterion1(preds, label_b) +
                                         0.5*criterion2(preds, label_b))
                        else:
                            loss = lam * \
                                criterion1(preds, label_a) + \
                                (1 - lam) * criterion1(preds, label_b)
                    else:
                        if cfg['TRAIN']['MULTI_CRITERION']:
                            loss = 0.5 * \
                                criterion1(preds, labels) + 0.5 * \
                                criterion2(preds, labels)
                        else:
                            loss = criterion1(preds, labels)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm)

                    if cfg['TRAIN']['OPTIMIZER'] == 'sam' and not cfg['TRAIN']['MULTI_CRITERION'] and cfg['TRAIN']['MIXUP']:
                        optimizer.first_step(zero_grad=True)
                        criterion1(model(images), label_a).backward()
                        criterion1(model(images), label_b).backward()
                        optimizer.second_step(zero_grad=True)
                    else:
                        optimizer.step()

                    preds = preds.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = np.argmax(preds, axis=-1)

                    train_loss_list.append(loss.item())
                    train_loss = np.mean(train_loss_list)
                    learning_rate = get_lr(optimizer)
                    wandb.log({
                        "Train Loss": train_loss,
                        "Learning Rate": learning_rate
                    })
                    train_bar.set_postfix(train_loss=train_loss)

        valid_loss_list = []
        valid_acc_list = []
        valid_f1_list = []
        with tqdm(val_loader,
                  total=val_loader.__len__(),
                  unit="batch") as valid_bar:
            for sample in valid_bar:
                valid_bar.set_description(f"Valid Epoch: {epoch}")
                optimizer.zero_grad()

                images, labels = sample['image']['image'].float(
                ), sample['label'].long()

                images = images.to(device)
                labels = labels.to(device)

                model.eval()
                with torch.no_grad():
                    preds = model(images)

                    valid_loss = criterion1(preds, labels)

                    preds = preds.cpu().detach().numpy()
                    labels = labels.cpu().detach().numpy()
                    preds = np.argmax(preds, axis=-1)
                    batch_acc = (labels == preds).mean()
                    valid_acc_list.append(batch_acc)
                    valid_acc = np.mean(valid_acc_list)

                    batch_f1 = f1_score(labels, preds, average='macro')
                    valid_f1_list.append(batch_f1)
                    valid_f1 = np.mean(valid_f1_list)

                    valid_loss_list.append(valid_loss.item())
                    valid_loss = np.mean(valid_loss_list)

                    valid_bar.set_postfix(valid_loss=valid_loss,
                                          valid_acc=valid_acc,
                                          valid_f1=valid_f1)
        wandb.log({
            "Valid Loss": valid_loss,
            "Valid Acc": valid_acc,
            "Valid F1": valid_f1,
        })

        early_stopping(valid_loss, model)

        if early_stopping.early_stop:
            print("stopped Earlier than Expected!!!!!!!!!!!!")
            save_dir = cfg['PATH']['SAVE']
            model_name = cfg['TRAIN']['MODEL_NAME']
            torch.save(
                best_model, f'{save_dir}_{model_name}_{fold_num}_{best_valid_loss:2.4f}_epoch_{best_epoch}.pth')
            wandb.join()
            break
        if best_valid_loss > valid_loss:
            print()
            print(
                f"Best Model Changed!!, Previous: {best_valid_loss} VS current: {valid_loss}")
            best_valid_loss = valid_loss
            best_model = model
            best_epoch = epoch
        if cfg['TRAIN']['SCHEDULER']:
            scheduler.step()

    save_dir = cfg['PATH']['SAVE']
    model_name = cfg['TRAIN']['MODEL_NAME']
    torch.save(
        best_model, f'{save_dir}_{model_name}_{fold_num}_{best_valid_loss:2.4f}_epoch_{best_epoch}.pth')
    wandb.join()
    return best_model


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="configuration file name")
    args = parser.parse_args()

    with open(os.path.join(os.getcwd(), 'config', args.config)) as f:
        cfg = yaml.safe_load(f)

    seed_everything(cfg['TRAIN']['SEED'])
    df = get_labels_with_directory(cfg)
    trimmed_set, excluded_set = exclude_dirs(df, 1)

    train_sets, valid_sets = get_fold(cfg, trimmed_set)
    best_model = []
    i = 1
    for train_set, valid_set in zip(train_sets, valid_sets):
        best_model.append(train(
            cfg, trimmed_set.iloc[train_set], trimmed_set.iloc[valid_set], excluded_set, fold_num=i))
        i += 1
