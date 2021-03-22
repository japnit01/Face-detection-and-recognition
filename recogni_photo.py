import tensorflow as tf
import imutils
from imutils.video import VideoStream
from main import *
from keras.models import load_model
from keras.layers import Flatten,Dense,Input,concatenate,Conv2D,Dropout,BatchNormalization,MaxPooling2D,Lambda
from keras.models import Model,Sequential
from keras.activations import sigmoid

#v1 = VideoStream(src=0).start()

emb_size = 2048
alpha = 0.2

def triplet_loss(y_true, y_pred):
    anchor, positive, negative = y_pred[:,:emb_size], y_pred[:,emb_size:2*emb_size], y_pred[:,2*emb_size:]
    positive_dist = tf.reduce_mean(tf.square(anchor - positive), axis=1)
    negative_dist = tf.reduce_mean(tf.square(anchor - negative), axis=1)
    return tf.maximum(positive_dist - negative_dist + alpha, 0.)


def recognise_face(img,database,model):
    encoding = img_to_encoding_webcam(img,model)
    identity = None
    min_dist = 100
    for (name, db_enc) in database.items():
        dist= tf.sqrt(tf.reduce_sum(tf.pow(db_enc - encoding, 2), 1, keepdims=True)).numpy()
        #dist = np.linalg.norm(db_enc - encoding)
        print('distance for %s is %s' % (name, dist))
        if dist < min_dist:
            min_dist = dist
            identity = name

    # if min_dist > 0.6:
    #     # speak('cant recognisethe face', 2)
    #     return str(0)
    # else:
    return str(identity)


def prepare_database(model):
    database = {}
    for file in glob.glob("/home/jap01/PycharmProjects/face recogiition original/database1/*"):
        identity = os.path.splitext(os.path.basename(file))[0]
        #print(identity)
        database[identity] = img_to_encoding(file, model)

    return database

def img_to_encoding_webcam(image,model):
    image = cv2.resize(image, (224, 224))
    embedding = model.predict(np.expand_dims(image, axis=0))
    return embedding


def img_to_encoding(image_path, model):
    image = cv2.imread(image_path, 1)
    image = cv2.resize(image, (224,224))
    #print(image.shape)
    #x_train = np.array([img])

    embedding = model.predict(np.expand_dims(image,axis=0))
    return embedding

#def recogonise:

def load_my_model():
    embedding_model = Sequential()
    embedding_model.add(Conv2D(64, (10, 10), activation='relu', input_shape=(224, 224, 3)))
    embedding_model.add(BatchNormalization())
    embedding_model.add(MaxPooling2D())
    embedding_model.add(Dropout(0.5))
    embedding_model.add(Conv2D(128, (7, 7), activation='relu'))
    embedding_model.add(BatchNormalization())
    embedding_model.add(MaxPooling2D())
    embedding_model.add(Dropout(0.5))
    embedding_model.add(Conv2D(128, (4, 4), activation='relu'))
    embedding_model.add(BatchNormalization())
    embedding_model.add(MaxPooling2D())
    embedding_model.add(Dropout(0.5))
    embedding_model.add(Conv2D(256, (4, 4), activation='relu'))
    embedding_model.add(BatchNormalization())
    embedding_model.add(Flatten())
    embedding_model.add(Dense(emb_size, activation='sigmoid'))
    embedding_model.add(Lambda(lambda x: tf.keras.backend.l2_normalize(x, axis=1)))

    input_anchor = Input(shape=(224, 224, 3))
    input_positive = Input(shape=(224, 224, 3))
    input_negative = Input(shape=(224, 224, 3))

    embedding_anchor = embedding_model(input_anchor)
    embedding_positive = embedding_model(input_positive)
    embedding_negative = embedding_model(input_negative)

    output = concatenate([embedding_anchor, embedding_positive, embedding_negative], axis=1)

    net = Model([input_anchor, input_positive, input_negative], output)

    return net,embedding_model




net,embmodel = load_my_model()
net.summary()
net.compile(optimizer='adam', loss=triplet_loss, metrics=['accuracy'])
net.load_weights("/home/jap01/Downloads/mode19_weights_siamese_vgg16.h5")

while True:
    frame = cv2.imread("/home/jap01/PycharmProjects/face recogiition original/database/Tushant/tushant.jpg")
    frame = imutils.resize(frame, width=400)
    cv2.imshow("frame", frame)
    database = prepare_database(embmodel)
    face = recognise_face(frame, database, embmodel)
    print(face)
    #print(database)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break




