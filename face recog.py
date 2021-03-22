import tensorflow as tf
from keras.models import load_model
from inception_model import model
import os
import numpy as np
import random


data_path = r'/home/jap01/PycharmProjects/face recogiition original'

train_images = os.listdir(data_path + '/database')

def get_triplet(self, shape):
        class_1 = random.randint(0, shape[0] - 1)
        class_2 = random.randint(0, shape[0] - 1)
        a, p = (class_1, random.randint(0, shape[1] - 1)), (class_1, random.randint(0, shape[1] - 1))
        n = (class_2, random.randint(0, shape[1] - 1))
        # print(a, p, n)
        return a, p, n

def get_triplet_batch(self, batch_size, train_data=True):
	anchor_image = []
	positive_image = []
	negative_image = []
	if train_data:
		X = train_images
	else:
		X = val_images

	for _ in range(batch_size):
		ida, idp, idn = self.get_triplet(X.shape)
		anchor_image.append(X[ida])
		positive_image.append(X[idp])
		negative_image.append(X[idn])

	ai = np.array(anchor_image)
	pi = np.array(positive_image)
	ni = np.array(negative_image)
	return [ai, pi, ni]

def triplet_loss_function(y_true,y_pred,alpha = 0.3):
	anchor = y_pred[0]
	positive = y_pred[1]
	negative = y_pred[2]
	pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
	neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
	basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
	loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
	return loss


def create_batch(batch_size=256):
	x_anchors = np.zeros((batch_size, 784))
	x_positives = np.zeros((batch_size, 784))
	x_negatives = np.zeros((batch_size, 784))

	for i in range(0, batch_size):
		# We need to find an anchor, a positive example and a negative example
		random_index = random.randint(0, x_train.shape[0] - 1)
		x_anchor = x_train[random_index]
		y = y_train[random_index]

		indices_for_pos = np.squeeze(np.where(y_train == y))
		indices_for_neg = np.squeeze(np.where(y_train != y))

		x_positive = x_train[indices_for_pos[random.randint(0, len(indices_for_pos) - 1)]]
		x_negative = x_train[indices_for_neg[random.randint(0, len(indices_for_neg) - 1)]]

		x_anchors[i] = x_anchor
		x_positives[i] = x_positive
		x_negatives[i] = x_negative

	return [x_anchors, x_positives, x_negatives]


model =tf.keras.applications.InceptionV3(
        include_top=False, weights='imagenet', input_tensor=None,
        input_shape=(96,96,3), pooling=None, classes=5,
        classifier_activation='softmax'
    )

model.compile(optimizer = 'adam', loss = triplet_loss_function, metrics = ['accuracy'])

save_path = os.path.join(data_path, "model_weights_triplet_loss_2048.h5")

epochs = 50
steps_per_epoch = 100

model_history = model.fit(
    ,
    epochs=epochs, steps_per_epoch=steps_per_epoch,
    verbose=True
)

model.save_weights(save_path)
