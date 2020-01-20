from __future__ import absolute_import, division, print_function, unicode_literals
try:
  # %tensorflow_version only exists in Colab.
  %tensorflow_version 2.x
except Exception:
  pass
import tensorflow as tf

import IPython.display as display
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = (12,12)
mpl.rcParams['axes.grid'] = False
import numpy as np
import PIL.Image
import time
import functools

def tensor_to_image(tensor):
  tensor = tensor*255
  tensor = np.array(tensor, dtype=np.uint8)
  if np.ndim(tensor)>3:
    assert tensor.shape[0] == 1
    tensor = tensor[0]
  return PIL.Image.fromarray(tensor)


#download style and content images
content_path = tf.keras.utils.get_file('Rabbit.jpg','https://github.com/uttam1216/stylization/blob/master/images/Rabbit.jpg')
style_path = tf.keras.utils.get_file('1_Femme.jpg','https://github.com/uttam1216/stylization/blob/master/images/1_Femme.jpg')
content2_path = tf.keras.utils.get_file('Dog.jpg','https://github.com/uttam1216/stylization/blob/master/images/Dog.jpg')
style2_path = tf.keras.utils.get_file('2_Vassily_Kandinsky.jpg','https://github.com/uttam1216/stylization/blob/master/images/2_Vassily_Kandinsky.jpg')
content3_path = tf.keras.utils.get_file('MonaLisa.jpg','https://github.com/uttam1216/stylization/blob/master/images/MonaLisa.jpg')
style3_path = tf.keras.utils.get_file('3_the_scream.jpg','https://github.com/uttam1216/stylization/blob/master/images/3_the_scream.jpg')

#define a funct. to load image & limit its max. dim. to 512 pxls
def load_img(path_to_img):
  max_dim = 512
  img = tf.io.read_file(path_to_img)
  img = tf.image.decode_image(img, channels=3)
  img = tf.image.convert_image_dtype(img, tf.float32)

  shape = tf.cast(tf.shape(img)[:-1], tf.float32)
  long_dim = max(shape)
  scale = max_dim / long_dim

  new_shape = tf.cast(shape * scale, tf.int32)

  img = tf.image.resize(img, new_shape)
  img = img[tf.newaxis, :]
  return img

#A simple function to display an image:
def imshow(image, title=None):
  if len(image.shape) > 3:
    image = tf.squeeze(image, axis=0)

  plt.imshow(image)
  if title:
    plt.title(title)

content_image = load_img(content_path)
style_image = load_img(style_path)

plt.subplot(1, 2, 1)
imshow(content_image, 'Content Image')

plt.subplot(1, 2, 2)
imshow(style_image, 'Style Image')

#style transfer
import tensorflow_hub as hub
hub_module = hub.load('https://tfhub.dev/google/magenta/arbitrary-image-stylization-v1-256/1')

stylized_image = hub_module(tf.constant(content_image), tf.constant(style_image))[0]
tensor_to_image(stylized_image)

stylized_image2 = hub_module(tf.constant(content2_image), tf.constant(style2_image))[0]
tensor_to_image(stylized_image2)

stylized_image3 = hub_module(tf.constant(content3_image), tf.constant(style3_image))[0]
tensor_to_image(stylized_image3)


