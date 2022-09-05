#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import layers, regularizers, initializers
import tensorflow_datasets as tfds
import tensorflow_addons as tfa

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2
import pandas as pd

from tqdm import tqdm
import warnings
warnings.simplefilter('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')

print(f'''Import Versions:-
* TensorFlow: {tf.__version__}
* TensorFlow Datasets: {tfds.__version__}
* TensorFlow Addons: {tfa.__version__}
* NumPy: {np.__version__}
* Pandas: {pd.__version__}
* cv2: {cv2.__version__}
* Matplotlib: {matplotlib.__version__}''')

