import numpy as np
import tensorflow as tf

test = tf.constant('Yesss! Tensorflow have successfully installed on this environment!')
sess = tf.Session()
print(sess.run(test))