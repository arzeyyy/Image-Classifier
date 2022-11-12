from gc import callbacks
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import layers, models
from keras.models import load_model
import numpy as np
import os
import cv2
import imghdr



def limit_memory():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    
def remove_corrupted_files():
    num_skipped = 0
    for folder_name in ("Cat", "Dog"):
        folder_path = os.path.join("Image Classifier\data", folder_name)
        for fname in os.listdir(folder_path):
            fpath = os.path.join(folder_path, fname)
            try:
                fobj = open(fpath, "rb")
                is_jfif = tf.compat.as_bytes("JFIF") in fobj.peek(10)
            finally:
                fobj.close()

            if not is_jfif:
                num_skipped += 1
                # Delete corrupted image
                os.remove(fpath)

    print("Deleted %d images" % num_skipped)

limit_memory()
remove_corrupted_files()


def showImage():
    img = cv2.imread('image Classifier/data_dir/dogs/14-36-18-cdrd2e5a4ztz.jpg')
    print(img.shape)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()

#load data
tf.data.Dataset
data = tf.keras.utils.image_dataset_from_directory('Image Classifier/data')
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()
print(batch[0].shape)

def showImages():
    fig, ax = plt.subplots(ncols=4, figsize=(20,20))
    for i, img in enumerate(batch[0][:4]):
        ax[i].imshow(img)
        ax[i].title.set_text(batch[1][i])
    plt.show()
    

#scale
data = data.map(lambda x, y: (x/255, y))
scaled_iterator = data.as_numpy_iterator()
batch = scaled_iterator.next()

#split data
print(len(data))
train_size = int(len(data) /14 *8)
val_size = int(len(data) /14 *4)
test_size = int(len(data) /14 *2)


#allocate data
train = data.take(train_size)
val = data.skip(train_size).take(val_size)
test = data.skip(train_size + val_size).take(test_size)

print()
#build
model = models.Sequential()
model.add(layers.Conv2D(16, (3,3), 1, activation='relu', input_shape=(256,256,3)))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(32, (3,3), 1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(16, (3,3), 1, activation='relu'))
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(256, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

model.compile(optimizer="adam", loss=tf.losses.BinaryCrossentropy(), metrics=["accuracy"])
print(model.summary())

#train
def train_model():
    logdir = 'Image Classifier/logs'
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=logdir)
    model.fit(train, epochs=11, validation_data=val, callbacks=[tensorboard_callback])

train_input = input("Would you like to train your model? (Y/N)")
if(train_input =="Y" or train_input =="y"):
    train_model()

if(train_input =="Y" or train_input =="y"):
    again_input = input("Would you like to train again? (Y/N)")
    if(again_input =="Y" or again_input =="y"):
        train_model()

#evaluate performance
pre = tf.keras.metrics.Precision()
re = tf.keras.metrics.Recall()
acc = tf.keras.metrics.BinaryAccuracy()

for batch in test.as_numpy_iterator():
    x, y = batch
    prediction = model.predict(x)
    pre.update_state(y, prediction)
    re.update_state(y, prediction)
    acc.update_state(y, prediction)

print(f'Precission:{pre.result().numpy()}, Recall:{re.result().numpy()}, Accuracy:{pre.result().numpy()}')

#save model
if(train_input =="Y" or train_input =="y"):
    save_Input = input("do you want to save your model? (Y/N)")
    if(save_Input =='Y' or save_Input == "y"):
        model.save('Image Classifier/models/imageClassifier.model')
#load model
model = tf.keras.models.load_model('Image Classifier/models/imageClassifier.model')

os.system("cls")
print("---------------------------Testing---------------------------")

def loop():
    img_path = input("your image path: ")
    test_img = cv2.imread(img_path)
    resize = tf.image.resize(test_img, (256,256))
    plt.imshow(resize.numpy().astype(int))
    plt.show()

    #prediction
    prediction = model.predict(np.expand_dims(resize/255, 0))
    print(np.argmax(prediction))

    if prediction > 0.5:
        print("it's dog")
    else:
        print("it's cat")
        
    loop()
    
loop()


limit_memory()
remove_corrupted_files()
#showImages()
