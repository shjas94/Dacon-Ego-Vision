import os
import json
import glob
import yaml
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from modules.dataset import *
from modules.utils import *
from modules.models import *


def inference(cfg, model_path):
    submission = pd.read_csv(os.path.join(
        cfg['PATH']['DATA'], 'sample_submission.csv'))
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    torch.cuda.empty_cache()

    model = torch.load(model_path).to(device)
    test_df = get_test_paths(cfg)

    transforms = A.Compose([
                           ToTensorV2(p=1.0)
                           ])
    test_set = EvTestDataset(
        test_df, cfg['DATA']['IMG_SIZE'], img_padding=cfg['INFERENCE']['TEST_IMG_PADDING'], transforms=transforms)
    test_loader = DataLoader(
        test_set,
        batch_size=cfg['INFERENCE']['BATCH_SIZE'],
        num_workers=3,
        shuffle=False,
        pin_memory=use_cuda,
        drop_last=False
    )
    prediction_array = np.zeros([len(test_set),
                                 submission.shape[1] - 1])

    for idx, sample in enumerate(test_loader):
        with torch.no_grad():
            model.eval()
            images = sample['image']['image']
            images = images.to(device)
            probs = model(images)
            probs = probs.cpu().detach().numpy()
            batch_index = cfg['INFERENCE']['BATCH_SIZE'] * idx
            prediction_array[batch_index: batch_index + images.shape[0], :]\
                = probs
    return prediction_array


def k_fold_ensemble(cfg, prediction_arrays):
    prediction_arrays_probs = softmax(prediction_arrays[0])
    for i in range(1, len(prediction_arrays)):
        prediction_arrays_probs += softmax(prediction_arrays[i])
    prediction_arrays_probs /= len(prediction_arrays)

    test_dirs = sorted(glob.glob(os.path.join(
        cfg['PATH']['DATA'], 'test', '*')), key=lambda x: int(x.split('/')[4]))
    submission_arr = np.zeros(
        (len(test_dirs), prediction_arrays_probs.shape[1]))

    for i, test_dir in enumerate(test_dirs):
        meta = glob.glob(os.path.join(test_dir, '*.json'))[0]
        meta = json.load(open(meta))
        annot_len = len(meta['annotations'])
        submission_arr[i, :] = np.mean(
            prediction_arrays_probs[:annot_len, :], axis=0)
        prediction_arrays_probs = prediction_arrays_probs[annot_len:, :]
    submission = pd.read_csv(os.path.join(
        cfg['PATH']['DATA'], 'sample_submission.csv'))
    submission.iloc[:, 1:] = submission_arr
    return submission


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, help="configuration file name")
    args = parser.parse_args()

    with open(os.path.join(os.getcwd(), 'config', args.config)) as f:
        cfg = yaml.safe_load(f)

    model_path = glob.glob(os.path.join(cfg['PATH']['SAVE'], '*.pth'))
    prediction_arrays = []
    for m in model_path:
        prediction_arrays.append(inference(cfg, m))
    submission = k_fold_ensemble(cfg, prediction_arrays)
    submission.to_csv(os.path.join(
        os.getcwd(), 'submissions', cfg['INFERENCE']['SUBMISSION_NAME']), index=False)
