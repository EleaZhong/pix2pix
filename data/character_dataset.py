"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
from data.base_dataset import BaseDataset, get_transform
import torchvision
import numpy as np
from io import BytesIO
import clip
import torch
import random
import torchvision
import webdataset as wds
import albumentations as A
from PIL import Image
from torch.utils.data import DataLoader, IterableDataset
from torch.utils.data import DistributedSampler
from albumentations.pytorch import ToTensorV2
from torch.utils.tensorboard import SummaryWriter
from webdataset.handlers import warn_and_continue
from torchvision.utils import save_image

# from data.image_folder import make_dataset
# from PIL import Image




class ProcessDataFT:
    def __init__(self,):
        self.transforms = torchvision.transforms.Compose([
            # torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(256),
            torchvision.transforms.RandomCrop(256),
            #torchvision.transforms.RandomResizedCrop(size=(256,256),scale=(0.75, 1.0)),
            
        ])

    def __call__(self, data):
        pil_image = Image.open(BytesIO(data["webp"])).convert("RGB")
        webpimg = torchvision.transforms.functional.to_tensor(pil_image).unsqueeze(0)
        # print(webpimg.size())
        sketch_image = data["png"] # Image.open(BytesIO(data["png"])).convert("RGB")
        sketchimg = torchvision.transforms.functional.to_tensor(sketch_image).unsqueeze(0)
        sketchimg = torchvision.transforms.functional.resize(sketchimg, [webpimg.size(2),webpimg.size(3)])
        # print(sketchimg.size())
        allim = torch.cat([webpimg, sketchimg])
        trans_im = self.transforms(allim)
        data["webp"] = trans_im[0]
        data["sketch"]= trans_im[1]
        return data


def collate(batch):
    images = torch.stack([i[0] for i in batch], dim=0)
    captions = [i[1] for i in batch]
    sketches = torch.stack([i[2] for i in batch], dim=0)
    return [images, captions, sketches]



dataset = wds.WebDataset(args.dataset_path, resampled=True, handler=warn_and_continue).decode("rgb", handler=warn_and_continue).map(
            ProcessDataFT(), handler=warn_and_continue).to_tuple("webp", "txt", "sketch", handler=warn_and_continue).shuffle(690, handler=warn_and_continue)

class CharacterDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--dataset_path', type=str, default="pipe:aws s3 cp s3://s-laion-wand/laion-aesthetic/data_res_x/laion2B-en_filter_sketch_open/{0..699}.tar -")
        parser.add_argument('--dataset_length', type=int, default=250_000)
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """

        self.data = wds.WebDataset(opt.dataset_path, resampled=True, handler=warn_and_continue).decode("rgb", handler=warn_and_continue).map(
            ProcessDataFT(), handler=warn_and_continue).to_tuple("webp", "txt", "sketch", handler=warn_and_continue).shuffle(690, handler=warn_and_continue)
        self.data = iter(self.data)

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        d = next(self.data)
        return {'data_A': d["webp"], 'data_B': d["sketch"], 'caption': d["txt"]}

    def __len__(self):
        """Return the total number of images."""
        return len(self.dataset_length)
