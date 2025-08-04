"""This file contains the definition of different heterogeneous datasets used for training."""
import joblib
import numpy as np
import torch
from loguru import logger


class AMASSDataset(torch.utils.data.Dataset):
    def __init__(self, is_train=True):
        logger.info(f"Loading AMASS dataset: --> training: {is_train}")

        amass_subsets = "./data/DFaust_67_train.pth.tar"

        self.marker_data = []
        self.rotations = []
        self.positions = []
        self.translation = []
        self.fnames = []
        self.betas = []
        self.gender = []

        data = joblib.load(amass_subsets)

        if is_train:
            logger.info(f"Loaded dataset: ----> {amass_subsets} ")
            logger.info(f"Number of Sequence: ----> {len(data)} ")

        for i, x in enumerate(data):
            self.fnames.append(data[i]["fname"])
            self.marker_data.append(data[i]["markers"])
            seq_length = data[i]["markers"].shape[0]

            self.rotations.append(data[i]["poses"])
            self.translation.append(data[i]["trans"])

            self.betas.append(np.repeat(data[i]["betas"][np.newaxis, :], seq_length, axis=0))

        self.marker_data = np.concatenate(self.marker_data)
        self.rotations = np.concatenate(self.rotations)
        self.translation = np.concatenate(self.translation)
        self.betas = np.concatenate(self.betas)

        logger.info(f"Finished loading all the datasets. Total number of samples: {len(self.fnames)}")

    def __len__(self):
        return len(self.marker_data)

    def __getitem__(self, index):
        item = dict()

        motion_rotations = self.rotations[index].reshape(-1, 3)
        translation = self.translation[index]
        body_shape = self.betas[index]

        item["rotations"] = torch.from_numpy(motion_rotations).type(dtype=torch.float32)
        item["translation"] = torch.from_numpy(translation).type(dtype=torch.float32)
        item["body_shape"] = torch.from_numpy(body_shape).type(dtype=torch.float32)

        return item
