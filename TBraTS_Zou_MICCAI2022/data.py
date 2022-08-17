import os
from abc import ABC
from glob import glob
import numpy as np
from typing import *

import pytorch_lightning as pl
import torchio as tio
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch


def list_split(list_: list, split: tuple) -> List[List]:
    n = list(split)
    n.insert(0, 0)
    return [list_[i:i + n[i + 1]] for i in range(len(n) - 1)]


class BraTS2019Data:
    def __init__(self, dataset_dir='<YOUR_DATASET_DIRECTORY>', split=(265, 35, 35)):
        self.dataset_dir = dataset_dir
        self.split = split

        if not os.path.exists(os.path.join(self.dataset_dir, 'subjects')):
            os.mkdir(os.path.join(self.dataset_dir, 'subjects'))
            subjects = [os.path.dirname(s)
                        for s in glob(os.path.join(self.dataset_dir, '*', '*', "*seg.nii*"))]
            np.random.shuffle(subjects)

            self.train_subjects, self.val_subjects, self.test_subjects = list_split(
                subjects, self.split)

            # save the train, val, test subjects into the subjects folder
            with open(os.path.join(self.dataset_dir, 'subjects', 'train.txt'), 'w') as f:
                for s in self.train_subjects:
                    f.write(s + '\n')
            with open(os.path.join(self.dataset_dir, 'subjects', 'val.txt'), 'w') as f:
                for s in self.val_subjects:
                    f.write(s + '\n')
            with open(os.path.join(self.dataset_dir, 'subjects', 'test.txt'), 'w') as f:
                for s in self.test_subjects:
                    f.write(s + '\n')

        else:
            # read the train, val, test subjects from the subjects folder
            with open(os.path.join(self.dataset_dir, 'subjects', 'train.txt'), 'r') as f:
                self.train_subjects = [s.strip() for s in f.readlines()]
            with open(os.path.join(self.dataset_dir, 'subjects', 'val.txt'), 'r') as f:
                self.val_subjects = [s.strip() for s in f.readlines()]
            with open(os.path.join(self.dataset_dir, 'subjects', 'test.txt'), 'r') as f:
                self.test_subjects = [s.strip() for s in f.readlines()]

    def get_paths(self, split='train'):
        if split == 'train':
            return self.train_subjects
        elif split == 'val':
            return self.val_subjects
        elif split == 'test':
            return self.test_subjects
        else:
            raise ValueError('split must be either train, val, or test')

    def get_data(self, split='train') -> List[Dict]:
        paths = self.get_paths(split)

        flair_paths = [os.path.join(
            p, os.path.basename(p) + '_flair.nii') for p in paths]
        t1_paths = [os.path.join(p, os.path.basename(p) + '_t1.nii')
                    for p in paths]
        t1ce_paths = [os.path.join(
            p, os.path.basename(p) + '_t1ce.nii') for p in paths]
        t2_paths = [os.path.join(p, os.path.basename(p) + '_t2.nii')
                    for p in paths]
        seg_paths = [os.path.join(
            p, os.path.basename(p) + '_seg.nii') for p in paths]

        # return {'flair': flair_paths, 't1': t1_paths, 't1ce': t1ce_paths, 't2': t2_paths, 'seg': seg_paths}
        return [{'flair': flair, 't1': t1, 't1ce': t1ce, 't2': t2, 'seg': seg}
                for flair, t1, t1ce, t2, seg in zip(flair_paths, t1_paths, t1ce_paths, t2_paths, seg_paths)]


class BraTS2019DataModule(pl.LightningDataModule, ABC):
    def __init__(self, batch_size: int = 4):
        super().__init__()

        self.test_set = None
        self.val_set = None
        self.train_set = None
        self.test_subjects = None
        self.val_subjects = None
        self.train_subjects = None
        self.batch_size = batch_size

    def prepare_data(self) -> None:
        all_data = BraTS2019Data()

        self.train_subjects, self.val_subjects, self.test_subjects = [
            [
                tio.Subject(
                    flair=tio.ScalarImage(d['flair']),
                    t1=tio.ScalarImage(d['t1']),
                    t1ce=tio.ScalarImage(d['t1ce']),
                    t2=tio.ScalarImage(d['t2']),
                    seg=tio.LabelMap(d['seg'])
                ) for d in dset
            ]
            for dset in [all_data.get_data(split) for split in ['train', 'val', 'test']]]

    @staticmethod
    def get_preprocessing_transform() -> tio.Transform:
        return tio.Compose(
            [
                # Rescale each modality to the range [0, 1]
                tio.RescaleIntensity(out_min_max=(0, 1), in_min_max=(
                    0, 500), include=['flair', 't1ce']),
                tio.RescaleIntensity(out_min_max=(
                    0, 1), in_min_max=(200, 700), include="t1"),
                tio.RescaleIntensity(out_min_max=(
                    0, 1), in_min_max=(200, 800), include="t2"),

                tio.ToCanonical(),
                tio.EnsureShapeMultiple(8),  # for the U-Net
                tio.RemapLabels({4: 3}),  # for OneHot
                tio.OneHot(num_classes=4),
            ])

    @staticmethod
    def get_augmentation_transform() -> tio.Transform:
        return tio.Compose([
            tio.CropOrPad(target_shape=(128, 128, 128), mask_name='seg'),
            tio.RandomFlip(axes=(0, 1, 2)),
            tio.RandomGamma(p=0.5),
            tio.RandomNoise(p=0.5),
            tio.RandomMotion(p=0.1),
            tio.RandomBiasField(p=0.25),
        ])

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_set = tio.SubjectsDataset(subjects=self.train_subjects,
                                             transform=tio.Compose(
                                                 [self.get_preprocessing_transform(),
                                                  self.get_augmentation_transform()]))
        self.val_set = tio.SubjectsDataset(subjects=self.val_subjects,
                                           transform=self.get_preprocessing_transform())
        self.test_set = tio.SubjectsDataset(subjects=self.test_subjects,
                                            transform=self.get_preprocessing_transform(),
                                            load_getitem=False)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True, num_workers=8)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False, num_workers=8)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False, num_workers=8)


if __name__ == '__main__':
    test = BraTS2019Data()
    print("--- Test `BraTS2019Data` ---")
    print("Length of each split:")
    print(f"train: {len(test.get_paths('train'))}")
    print(f"val:   {len(test.get_paths('val'))}")
    print(f"test:  {len(test.get_paths('test'))}")

    print("\n--- Test `BraTS2019DataModule` ---")
    data = BraTS2019DataModule()
    data.prepare_data()
    data.setup()
    print("Length of each dataset:")
    print(f"train: {len(data.train_set)}")
    print(f"val:   {len(data.val_set)}")
    print(f"test:  {len(data.test_set)}")
