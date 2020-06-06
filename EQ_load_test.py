import numpy as np
import keras, sklearn
import tensorflow as tf
import matplotlib.pyplot as plt
import glob
import shutil
from sklearn.model_selection import train_test_split
import pickle

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten,BatchNormalization
from keras.layers import Conv2D, MaxPooling2D,SeparableConv2D,DepthwiseConv2D,Conv1D


with open("model_history_save12.pkl", "rb") as f:
    model_history1 = pickle.load(f)
print('loaded')
