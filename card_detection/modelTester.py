import cv2
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from keras.models import load_model

source = "./dataset/testing/"
image = cv2.imread(source + "9.jpg")
image = tf.keras.utils.img_to_array(image)
image = np.expand_dims(image, axis=0)

model_rank = load_model("./models/modelDeteksiRankKartu")
model_suit = load_model("./models/modelDeteksiSuitKartu")

rank_class_names = ['10', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'J', 'K', 'Q']
suit_class_names = ['Clubs', 'Diamonds', 'Hearts', 'Spades']

# Predict
rank_pred = model_rank.predict(image, verbose=0)
print("Rank Prediction: ", rank_class_names[np.argmax(rank_pred[0])])