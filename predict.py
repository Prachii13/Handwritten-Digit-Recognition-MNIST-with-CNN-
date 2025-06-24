import tensorflow as tf
from PIL import Image
import numpy as np
import sys

model = tf.keras.models.load_model("model.h5")
img_path = sys.argv[1]

img = Image.open(img_path).convert('L').resize((28,28))
img_arr = np.array(img) / 255.0
img_arr = img_arr.reshape(1, 28, 28, 1)

pred = model.predict(img_arr)
print("Predicted digit:", np.argmax(pred))
