# MEDICAL IMAGING DIAGNOSTIC PROJECT - Config.py
# CORSO LAURA 230485

"""
File that contains:
* imports 
* global variables shared among all the files of the project.
"""


"""
--------------------------- Imports ---------------------------
"""
import torch
import torch.nn as nn 
import torchvision
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import Dataset
import random
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import os
import re
from tqdm.notebook import tqdm
from skimage import color
from PIL import Image, ImageOps
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
import pandas as pd
import cv2
import shutil
from zipfile import ZipFile
from functools import partial
#from google.colab.files import download



"""
--------------------------- Constants ---------------------------
"""
root_path_GDrive = "/content/gdrive/MyDrive/MedicalImagingDiagnostic/"  #path on GDrive were the complete dataset needed for the code will be saved
Pdataset_GDrive = "/content/gdrive/MyDrive/MedicalImagingDiagnostic/Dati San Matteo Dataset 2"  #path on GDrive were the dataset "Dati San Matteo Dataset 2"
root_path = "/Users/laura/Documents/UniTN/SecondYear/MedicalImagingDiagnostic/MedicalImagingDiagnostic_Shared"  #local path containig the zipped version of the dataset. It will contain also the files produced by the whole code
ds_path = "/Users/laura/Documents/UniTN/SecondYear/MedicalImagingDiagnostic/MedicalImagingDiagnostic_Shared/Dataset" #local path containing the unzipped version of the dataset
result_path = "/Users/laura/Documents/UniTN/SecondYear/MedicalImagingDiagnostic/MedicalImagingDiagnostic_Shared/Result"  #local path containing the files produced by the whole code
HS = 256 #the number of features in the hidden state of the Bi-LSTM
CLASSES = 4 #the number of classes
BATCH_SIZE = 4 #number of videos per batch
LEARNING_RATE = 0.0001 #learning rate for the optimizer
WEIGHT_DECAY = 0.000001 #weight decay for the optimizer
MOMENTUM = 0.9 #momentum for the optimizer
T_DEVICE = 'mps' #device used for training and testing
EPOCHS = 20 #number of epochs used for training 