import copy
from datetime import timedelta, datetime
import imageio
import matplotlib.pyplot as plt
from matplotlib import cm
import multiprocessing
import numpy as np
import os
from pathlib import Path
import pydicom
import pytest
import scipy.ndimage as ndimage
from scipy.ndimage.interpolation import zoom
from skimage import measure, morphology, segmentation
from time import time, sleep
from tqdm import trange
from tqdm.auto import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, random_split, DistributedSampler, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import warnings
import glob
from math import sqrt
from joblib import Parallel, delayed
import importlib
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader
import numpy as np
import torch
import os
import pandas as pd
from tqdm.auto import tqdm
from matplotlib import pyplot as plt
import cv2
from PIL import Image
import glob
import os
from Resnet34Unet import ResnetSuperVision
import os
import cv2
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader,Dataset
import albumentations as albu
#from albumentations import torch as AT
from skimage.color import gray2rgb
import numpy as np
import functools



#CTSCANNER Dataset Load
class CTScansDataset(Dataset):
    def __init__(self, root_dir, transform=None,transform2=None):
        self.root_dir = Path(root_dir)
        self.patients = [p for p in glob.glob(root_dir+'*')]
        self.transform = transform
        self.transform2 = transform2

    def __len__(self):
        return len(self.patients)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        image, metadata = self.load_scan(self.patients[idx])
        sample = {'image': image, 'metadata': metadata}
        return sample

    def get_patient(self, patient_id):
        patient_ids = [str(p.stem) for p in self.patients]
        return self.__getitem__(patient_ids.index(patient_id))

    @staticmethod
    def load_scan(path):
        T = [row for row in os.listdir(path)]
        slices = [pydicom.dcmread(path + "/" + file) for file in T]
        try:
            slices.sort(key = lambda x: float(x.ImagePositionPatient[2]))
        except:
            pass
        images = np.stack([s.pixel_array.astype(float) for s in slices])
        images = images.astype(np.int16)
        for n in range(len(slices)):
            intercept = slices[n].RescaleIntercept
            slope = slices[n].RescaleSlope
            if slope != 1:
                images[n] = slope * images[n].astype(np.float64)
                images[n] = images[n].astype(np.int16)
            images[n] += np.int16(intercept)
        image = images
        if image.shape[1]!=512 or image.shape[2]!=512:
            mask = image>image[0,0,0]
            z,y,x = np.where(mask)
            z_min,z_max,y_min,y_max,x_min,x_max = z.min(),z.max(),y.min(),y.max(),x.min(),x.max()
            image = image[:,y_min:y_max,x_min:x_max]
        return image, slices[0]

def get_img(path,x, folder: str='train_images'):
    """
    Return image based on image name and folder.
    """
    data_folder = os.path.join(path,folder)
    image_path = os.path.join(data_folder, x)
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img

def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    return x.transpose(2, 0, 1).astype('float32')


sigmoid = lambda x: 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
    # don't remember where I saw it
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((350, 525), np.float32)
    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    return predictions, num


def get_training_augmentation(y=704,x=1024):
    train_transform = [albu.RandomBrightnessContrast(p=0.3),
#                            albu.RandomGamma(p=0.3),
                           albu.VerticalFlip(p=0.5),
                           albu.HorizontalFlip(p=0.5),
#                            albu.ShiftScaleRotate(scale_limit=0, rotate_limit=10, shift_limit=0.0625, p=0.5, border_mode=0),
                           albu.Downscale(p=1.0,scale_min=0.35,scale_max=0.75,),
                           albu.Resize(y, x),
                           albu.RandomCrop(height=400, width=400,p=0.5),
#                            albu.GridDistortion(p=0.5),
#                            albu.OpticalDistortion(p=0.5, distort_limit=2, shift_limit=0.5),
                           albu.Resize(y, x)]
    return albu.Compose(train_transform)


def get_validation_augmentation(y=704,x=1024):
    """Add paddings to make image shape divisible by 32"""
    test_transform = [albu.Resize(y, x)]
    return albu.Compose(test_transform)


def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform
    
    Args:
        preprocessing_fn (callbale): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    
    """
    
    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)


def set_lungwin(img, hu=[-1200., 600.]):
    lungwin = np.array(hu)
    newimg = (img-lungwin[0]) / (lungwin[1]-lungwin[0])
    newimg[newimg < 0] = 0
    newimg[newimg > 1] = 1
    return newimg

class CTDataset2D(Dataset):
    def __init__(self, x_train,y_train,y_train2,transforms = albu.Compose([albu.HorizontalFlip()]),preprocessing=None,size=512):
        self.x_train = x_train
        self.y_train = y_train
        self.y_train2 = y_train2
        self.transforms = transforms
        self.preprocessing = preprocessing
        self.size=size
        self.hu =[[-1200., 600.],[-1800., 1000.],[-1500., 800.]]

    def __getitem__(self, idx):
        img = self.x_train[idx]
        mask = self.y_train[idx]
        mask2 = self.y_train2[idx]
        mask = np.expand_dims(mask,axis=-1)
        mask2 = np.expand_dims(mask2,axis=-1)
        mask = np.concatenate([mask,mask2],axis=-1)
        img = set_lungwin(img,self.hu[np.random.randint(3)])
        img = img*255
        img = img.astype(np.uint8)
        img = gray2rgb(img)
        
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        if self.y_train2[idx].sum()==0:
            label = np.array(0.0)
        else:
            label = np.array(self.y_train[idx].sum()/self.y_train2[idx].sum())
        return img, mask,torch.from_numpy(label.reshape(1,1))

    def __len__(self):
        return len(self.x_train)
    

    
def norm(img):
    img-=img.min()
    return img/img.max()


def preprocess_input(
    x, mean=None, std=None, input_space="RGB", input_range=None, **kwargs
):

    if input_space == "BGR":
        x = x[..., ::-1].copy()

    if input_range is not None:
        if x.max() > 1 and input_range[1] == 1:
            x = x / 255.0

    if mean is not None:
        mean = np.array(mean)
        x = x - mean

    if std is not None:
        std = np.array(std)
        x = x / std

    return x

formatted_settings = {
            'input_size': [3, 224, 224],
            'input_range': [0, 1],
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],}

class config:
    model_name="resnet34"
    batch_size = 1
    WORKERS = 1
    DataLoder = {}
    DataLoder['PY'] = ""
    DataLoder["CLASS"] = "CTDataset2D"
    DataLoder["x_size"] = 512
    DataLoder["y_size"] = 512
    DataLoder["margin"] = 512
    preprocessing_fn = functools.partial(preprocess_input, **formatted_settings)
    seg_classes =2
    resume = True
    MODEL_PATH = '../input/osic-model/'
    if not os.path.exists(MODEL_PATH):
        os.makedirs(MODEL_PATH)


def get_train_loder(x,y,y2):
    train_dataset = CTDataset2D(x, y,y2,
                                  transforms=get_training_augmentation(x=config.DataLoder["x_size"], y=config.DataLoder["y_size"]),
                                  preprocessing=get_preprocessing(config.preprocessing_fn))
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    return train_loader,train_dataset
def get_val_loder(x,y,y2,b=18):
    val_dataset = CTDataset2D(x, y,y2,
                                  transforms=get_validation_augmentation(x=config.DataLoder["x_size"], y=config.DataLoder["y_size"]),
                                  preprocessing=get_preprocessing(config.preprocessing_fn))
    valid_loader = DataLoader(val_dataset, batch_size=b, shuffle=False, num_workers=config.WORKERS, pin_memory=True)
    return valid_loader,val_dataset

model = ResnetSuperVision(config.seg_classes, backbone_arch='resnet34').cuda()
model.load_state_dict(torch.load("../input/osic-model/resnet34_fib_best.pth"))

from torch.optim.lr_scheduler import ReduceLROnPlateau
class trainer:
    def __init__(self,model):
        self.model = model
        if config.resume==True:
            self.load_best_model()
    
    def batch_valid(self, batch_imgs,get_fet):
        self.model.train()
        batch_imgs = batch_imgs.cuda()
        with torch.no_grad():
            predicted = self.model(batch_imgs,get_fet)
        predicted2 = torch.sigmoid(predicted[0])
        if not get_fet:
            return predicted2.cpu().numpy().astype(np.float32),predicted[1].cpu().numpy().astype(np.float32)
        else:
            return predicted2.cpu().numpy().astype(np.float32),predicted[1].cpu().numpy().astype(np.float32),predicted[2].cpu().numpy().astype(np.float32)
         
    def load_best_model(self):
        if os.path.exists(config.MODEL_PATH+"/{}_fib_best.pth".format(config.model_name)):
            self.model.load_state_dict(torch.load(config.MODEL_PATH+"/{}_fib_best.pth".format(config.model_name)))
        
    def predict(self,imgs_tensor,get_fet = False):
        self.model.train()
        with torch.no_grad():
            return self.batch_valid(imgs_tensor,get_fet=get_fet)
        
Trainer = trainer(model)
Trainer.load_best_model()

my_lung_image = cv2.imread('/kaggle/input/lungimages/11 (139).jpg')
import numpy as np

# Resize and normalize your image
resized_image = cv2.resize(my_lung_image, (512, 512))  # Use the desired dimensions
normalized_image = resized_image.astype(np.float32) / 255.0  # Normalize to [0, 1]
# Assuming you have the model 'Trainer' initialized somewhere in your code
res = Trainer.predict(torch.tensor([normalized_image.transpose(2, 0, 1)]).cuda(), get_fet=True)
import matplotlib.pyplot as plt

# Plotting the results
lung_mask = res[0][0, 1] > 0.9
fibrosis_mask = res[0][0, 0] > 0.9

plt.figure(figsize=[20, 10])
plt.subplot(131)
plt.imshow(my_lung_image)  # Show your own image
plt.subplot(132)
plt.imshow(my_lung_image)
plt.imshow(lung_mask, cmap='hot', alpha=0.5)
plt.title("lung segmentation")
plt.subplot(133)
plt.imshow(my_lung_image)
plt.imshow(fibrosis_mask, cmap='hot', alpha=0.5)
plt.title("fibrosis segmentation")
plt.show()
