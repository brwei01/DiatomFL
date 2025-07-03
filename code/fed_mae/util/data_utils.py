# --------------------------------------------------------
# Based on BEiT and MAE code bases
# Pretrain and Finetune datasets for FL SSL.
# https://github.com/microsoft/unilm/tree/master/beit
# https://github.com/facebookresearch/mae
# Author: Rui Yan
# --------------------------------------------------------

import numpy as np
import pandas as pd
import ast
import os
from .datasets import DataAugmentationForPretrain, build_transform

from PIL import Image, ImageOps
from skimage.transform import resize
import cv2
import torch.utils.data as data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import random


def parse_logits(logits_str):
    try:
        if not logits_str.startswith("[") or not logits_str.endswith("]"):
            raise ValueError(f"Malformed logits string: {logits_str}")
        # Use ast.literal_eval to safely evaluate the string into a Python object

        parsed = ast.literal_eval(logits_str)
        # Ensure it's a list of floats
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
        else:
            raise ValueError(f"Parsed object is not a list: {parsed}")
    except Exception as e:
        print(f"Error parsing logits string: {logits_str}, Error: {e}")
        return []  # Return an empty list or handle the error appropriately
class DatasetFLPretrain_my(data.Dataset):
    """ data loader for pre-training """

    def __init__(self, args):

        if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients',
                                          args.split_type, args.single_client)

        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})

        self.labels = 0

        self.transform = DataAugmentationForPretrain(args)
        self.args = args

    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join(self.args.data_path, 'train', self.img_paths[index])
        name = self.img_paths[index]

        target = 0
        target = np.asarray(target).astype('int64')

        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))

        if img.ndim < 3:
            img = np.stack((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


class Dataset_build_pseudo(data.Dataset):
    " data loader for pre-training "
    def __init__(self, client_unlabeled_dir, args):

        # Load image paths from server CSV file
        self.img_paths = list({line.strip().split(',')[0] for line in open(client_unlabeled_dir)})

        self.labels = None
        self.transform = build_transform(False, 'finetune', args)

        self.args = args

    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join('/homes/xwang/mtan/NewExp_4clients_cls48/SSLpretrain/train', self.img_paths[index])
        if not os.path.exists(path):
            print('path')
        name = self.img_paths[index]

        try:
            img = np.array(Image.open(path).convert("RGB"))

            if img.ndim < 3:
                img = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] >= 3:
                img = img[:, :, :3]

            if self.transform is not None:
                img = Image.fromarray(np.uint8(img))
                sample = self.transform(img)

        except Exception as e:
            print(f"Error loading image: {name}, index: {index}, error: {e}")
            return None, None

        return sample, name
    def __len__(self):
        return len(self.img_paths)

class Dataset_load_pseudo(data.Dataset):
    """ data loader for pre-training """
    def __init__(self, pseudo_csv_path, args):

        self.args = args
        self.transform = build_transform(True, 'finetune', args)
        self.img_paths = list({line.strip().split(',', 2)[0] for line in open(pseudo_csv_path)})
        self.labels = {line.strip().split(',', 2)[0]: float(line.strip().split(',', 2)[1]) for line in
                       open(pseudo_csv_path)}

        self.logits = {line.strip().split(',', 2)[0]:
                                                   line.strip().split(',', 2)[2].strip()[1:-1]
                                                  if (line.strip().split(',', 2)[2].strip().startswith('"') and
                                                      line.strip().split(',', 2)[2].strip().endswith('"'))
                                                  else line.strip().split(',', 2)[2].strip()

                                                 for line in open(pseudo_csv_path)
                                                 }

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        index = index % len(self.img_paths)
        path = os.path.join('/homes/xwang/mtan/NewExp_4clients_cls48/SSLpretrain/train', self.img_paths[index])
        if not os.path.exists(path):
            print('path')
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(f"Missing label for image {name} at index {index}.")
            print(name, index)

        try:
            logits_list = parse_logits(self.logits[name])
            logits_list = np.asarray(logits_list).astype('float32')
            if len(logits_list) != 48:
                print(
                    f"Invalid logits length for image {name} at index {index}. Expected {48}, got {len(logits_list)}.")
        except Exception as e:
            print(name, index)
            print(f"Error parsing logits for image {name} at index {index}: {e}")

        try:
            img = np.array(Image.open(path).convert("RGB"))

            if img.ndim < 3:
                img = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] >= 3:
                img = img[:, :, :3]

            if self.transform is not None:
                img = Image.fromarray(np.uint8(img))
                sample = self.transform(img)

        except Exception as e:
            print(f"Error loading image: {name}, index: {index}, error: {e}")
            return None, None, None

        return sample, target, logits_list


class DatasetSecondFinetune(data.Dataset):
    """ data loader for fine-tuning """

    def __init__(self, cur_single_client, args):

        cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type, cur_single_client)

        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})

        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                       open(os.path.join(args.data_path, 'labels.csv'))}

        self.transform = build_transform(True, 'finetune', args)

        self.args = args

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join(self.args.data_path, 'train', self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)

        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))

        if img.ndim < 3:
            img = np.concatenate((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


class DatasetFLPretrain_KD(data.Dataset):
    """ data loader for pre-training """

    def __init__(self, args):

        # Load server data path
        server_data_path = os.path.join(args.server_data_path, 'server4student.csv')
        # Load image paths from server CSV file
        self.img_paths = list({line.strip().split(',')[0] for line in open(server_data_path)})

        self.labels = None

        self.transform = DataAugmentationForPretrain(args)
        self.args = args

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join(self.args.server_data_path, 'server4student', self.img_paths[index])
        name = self.img_paths[index]

        try:
            if self.args.data_set == 'Retina':
                img = np.load(path)
                img = resize(img, (256, 256))
            else:
                img = np.array(Image.open(path).convert("RGB"))

            if img.ndim < 3:
                img = np.stack((img,) * 3, axis=-1)
            elif img.shape[2] >= 3:
                img = img[:, :, :3]

            if self.transform is not None:
                img = Image.fromarray(np.uint8(img))
                sample = self.transform(img)

        except Exception as e:
            print(f"Error loading image: {name}, index: {index}, error: {e}")
            return None, None

        return sample, -1

    def __len__(self):
        return len(self.img_paths)


class DatasetFLPretrain(data.Dataset):
    """ data loader for pre-training """
    def __init__(self, args):

                
        if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients', 
                                            args.split_type, args.single_client)

        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
        
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                        open(os.path.join(args.data_path, 'labels.csv'))}
    
        self.transform = DataAugmentationForPretrain(args)
        self.args = args
    
    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)
        
        path = os.path.join(self.args.data_path, 'train', self.img_paths[index])
        name = self.img_paths[index]

        target = self.labels[name]
        target = np.asarray(target).astype('int64')
        
        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))
        
        if img.ndim < 3:
            img = np.stack((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        if self.transform is not None:
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)


class DatasetFLKD(data.Dataset):
    """ Data loader for server-side unlabeled data """

    def __init__(self, args, phase, mode='finetune'):
        super(DatasetFLKD, self).__init__()
        self.phase = phase  # Server phase
        is_train = (phase == 'train')  # Server data is typically not for training

        # Load server data path

        unlabeled_data_path = os.path.join(args.server_data_path, 'train.csv')

        # Load image paths from server CSV file
        self.img_paths = list({line.strip().split(',')[0] for line in open(unlabeled_data_path)})

        # No labels, initialize empty placeholder
        self.labels = None  # Server data is unlabeled

        # Apply transformations
        self.transform = build_transform(is_train, mode, args)

        self.args = args

    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, dummy_target), where dummy_target is -1 for unlabeled data.
        """
        index = index % len(self.img_paths)

        # Image path
        path = os.path.join(self.args.server_data_path, 'server4student', self.img_paths[index])
        name = self.img_paths[index]

        try:
            # Load image
            if self.args.data_set == 'Retina':
                img = np.load(path)
                img = resize(img, (256, 256))  # Resize if needed
            else:
                img = np.array(Image.open(path).convert("RGB"))  # Load RGB image

            # Handle grayscale images
            if img.ndim < 3:
                img = np.concatenate((img,) * 3, axis=-1)
            elif img.shape[2] >= 3:
                img = img[:, :, :3]

            # Convert to PIL image for transformations
            img = Image.fromarray(np.uint8(img))
            sample = self.transform(img)

        except Exception as e:
            print(f"Error loading image: {name}, index: {index}, error: {e}")
            return None, None

        # Return image and dummy target (-1 for unlabeled data)
        return sample, -1

    def __len__(self):
        return len(self.img_paths)


class DatasetFLFinetune(data.Dataset):
    """ data loader for fine-tuning """

    def __init__(self, args, phase, mode='finetune'):

        super(DatasetFLFinetune, self).__init__()
        self.phase = phase
        is_train = (phase == 'train')

        if not is_train:
            args.single_client = os.path.join(args.data_path, f'{self.phase}.csv')

        if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients',
                                          args.split_type, args.single_client)

        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})

        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                       open(os.path.join(args.data_path, 'labels.csv'))}

        self.transform = build_transform(is_train, mode, args)

        self.args = args

    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)

        path = os.path.join(self.args.data_path, self.phase, self.img_paths[index])
        name = self.img_paths[index]

        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)

        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))

        if img.ndim < 3:
            img = np.concatenate((img,) * 3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:, :, :3]

        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)

class DatasetFLFinetune_getlabels(data.Dataset):
    """ data loader for fine-tuning """
    def __init__(self, args, phase, mode='finetune'):

        super(DatasetFLFinetune_getlabels, self).__init__()
        self.phase = phase
        is_train = (phase == 'train')

        if not is_train: 
            args.single_client = os.path.join(args.data_path, f'{self.phase}.csv')
        
        if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
            cur_clint_path = os.path.join(args.data_path, args.split_type, args.single_client)
        else:
            cur_clint_path = os.path.join(args.data_path, f'{args.n_clients}_clients', 
                                            args.split_type, args.single_client)
        
        self.img_paths = list({line.strip().split(',')[0] for line in open(cur_clint_path)})
        
        self.labels = {line.strip().split(',')[0]: float(line.strip().split(',')[1]) for line in
                        open(os.path.join(args.data_path, 'labels.csv'))}
        
        self.transform = build_transform(is_train, mode, args)
        
        self.args = args

    def __getitem__(self, index):

        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        index = index % len(self.img_paths)
        
        path = os.path.join(self.args.data_path, self.phase, self.img_paths[index])
        name = self.img_paths[index]
        
        try:
            target = self.labels[name]
            target = np.asarray(target).astype('int64')
        except:
            print(name, index)
        
        if self.args.data_set == 'Retina':
            img = np.load(path)
            img = resize(img, (256, 256))
        else:
            img = np.array(Image.open(path).convert("RGB"))
        
        if img.ndim < 3:
            img = np.concatenate((img,)*3, axis=-1)
        elif img.shape[2] >= 3:
            img = img[:,:,:3]
        
        # if self.transform is not None:
        img = Image.fromarray(np.uint8(img))
        sample = self.transform(img)

        return sample, target

    def __len__(self):
        return len(self.img_paths)

    def get_label(self, index):
        """
        返回第 index 条样本的标签
        """
        name = self.img_paths[index]
        label = self.labels[name]  # float -> int 转换可在这里做
        return int(label)

    def get_all_categories(self):
        """
        返回当前数据集中所有出现的类别ID，适用于单标签分类场景
        """
        categories_set = set()
        for path in self.img_paths:
            label = self.labels[path]
            categories_set.add(int(label))
        return categories_set

def create_dataset_and_evalmetrix(args, mode='pretrain'):


    ## get the joined clients
    if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
        args.dis_cvs_files = ['central']

    if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, args.split_type))
    else:
        args.dis_cvs_files = os.listdir(os.path.join(args.data_path, f'{args.n_clients}_clients', args.split_type))
    
    args.clients_with_len = {}
    
    for single_client in args.dis_cvs_files:
        if args.split_type == 'central' or args.split_type == 'iid_local_1' or args.split_type == 'iid_local_2' or args.split_type == 'iid_local_3' or args.split_type == 'iid_local_4':
            img_paths = list({line.strip().split(',')[0] for line in
                            open(os.path.join(args.data_path, args.split_type, single_client))})
        else:
            img_paths = list({line.strip().split(',')[0] for line in
                                open(os.path.join(args.data_path, f'{args.n_clients}_clients',
                                                args.split_type, single_client))})
        args.clients_with_len[single_client] = len(img_paths)
    
    
    ## step 2: get the evaluation matrix
    args.learning_rate_record = []
    args.record_val_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.record_test_acc = pd.DataFrame(columns=args.dis_cvs_files)
    args.save_model = False # set to false donot save the intermeidate model
    args.best_eval_loss = {}
    
    for single_client in args.dis_cvs_files:
        if mode == 'pretrain':
            args.best_mlm_acc[single_client] = 0 
            args.current_mlm_acc[single_client] = []

        if mode == 'finetune':
            args.best_acc[single_client] = 0 if args.nb_classes > 1 else 999
            args.current_acc[single_client] = 0
            args.current_test_acc[single_client] = []
            args.best_eval_loss[single_client] = 9999


def crop_top(img, percent=0.15):
    offset = int(img.shape[0] * percent)
    return img[offset:]

def central_crop(img):
    size = min(img.shape[0], img.shape[1])
    offset_h = int((img.shape[0] - size) / 2)
    offset_w = int((img.shape[1] - size) / 2)
    return img[offset_w:offset_w + size, offset_h:offset_h + size]

def process_covidx_image(img, size=224, top_percent=0.08, crop=False):
    img = crop_top(img, percent=top_percent)
    if crop:
        img = central_crop(img)
    img = resize(img, (size, size))
    img = img * 255
    return img

def process_covidx_image_v2(img, size=224):
    img = cv2.resize(img, (size, size))
    img = img.astype('float64')
    img -= img.mean()
    img /= img.std()
    return img
    
def random_ratio_resize(img, prob=0.3, delta=0.1):

    if np.random.rand() >= prob:
        return img
    ratio = img.shape[0] / img.shape[1]
    ratio = np.random.uniform(max(ratio - delta, 0.01), ratio + delta)

    if ratio * img.shape[1] <= img.shape[1]:
        size = (int(img.shape[1] * ratio), img.shape[1])
    else:
        size = (img.shape[0], int(img.shape[0] / ratio))

    dh = img.shape[0] - size[1]
    top, bot = dh // 2, dh - dh // 2
    dw = img.shape[1] - size[0]
    left, right = dw // 2, dw - dw // 2

    if size[0] > 224 or size[1] > 224:
        print(img.shape, size, ratio)
    
    img = cv2.resize(img, size)
    
    padding = (left, top, right, bot)
    new_im = ImageOps.expand(img, padding)
    
    return img
