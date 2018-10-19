import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
import keras
import pandas as pd
from PIL import Image

test = tf.constant('Yesss! Tensorflow have successfully installed on this environment!')
sess = tf.Session()
print(sess.run(test))