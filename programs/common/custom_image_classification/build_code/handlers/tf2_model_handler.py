
#!/usr/bin/env python

import urllib
import tensorflow as tf

from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, ZeroPadding2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

import os
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime

from handlers.data_handler import DataHandler, DataSet

class ModelHandler(object):
    def __init__(self, CONFIG):
        self.name = "tensorflow2.0"
        self.CONFIG = CONFIG
        self.data_handler = self.get_data_handler()

    def get_data_handler(self):
        return DataHandler(self.CONFIG)

    # Private method
    def __prepare_data(self):
        dataset = self.data_handler.dataset
        print(len(dataset.train_dirs))
        print(len(dataset.validation_dirs))
        print("Total training images:", dataset.total_train)
        print("Total validation images:", dataset.total_val)
        print(self.data_handler.train_dir)
        print(self.data_handler.validation_dir)
        if self.CONFIG['MODEL_CONFIG']['augmentaion'] == True:
            print('Apply Augmentaion ...')
            train_image_generator = ImageDataGenerator(
                    rescale=1./255,
                    rotation_range=45,
                    width_shift_range=.15,
                    height_shift_range=.15,
                    horizontal_flip=True,
                    zoom_range=0.2,
                    fill_mode='nearest'
                    )
        else:
            print('NO Augmentaion ...')
            train_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our training data

        validation_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
        train_data_gen = train_image_generator.flow_from_directory(batch_size=self.CONFIG['MODEL_CONFIG']['batch_size'],
                                                           directory=self.data_handler.train_dir,
                                                           shuffle=True,
                                                           target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']),
                                                           class_mode='categorical')
        val_data_gen = validation_image_generator.flow_from_directory(batch_size=self.CONFIG['MODEL_CONFIG']['batch_size'],
                                                              directory=self.data_handler.validation_dir,
                                                              target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH']),
                                                              class_mode='categorical')
        
        # sample_training_images, _ = next(train_data_gen)
        return train_data_gen, val_data_gen

    def create_model(self):
        total_labels = len(self.data_handler.dataset.validation_dirs)
        # for image_batch, label_batch in train_data_gen:
        #     print("Image batch shape: ", image_batch.shape)
        #     print("Label batch shape: ", label_batch.shape)
        #     total_labels = label_batch.shape[1]
        #     print("Total Label: ", total_labels)
        #     break

        if self.CONFIG['MODEL_CONFIG']['dropouts'] == True:
            print('Apply Dropouts ...')
            model = Sequential([
                # first convolutional layer
                Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'], 3)),
                MaxPooling2D((2,2), strides=(2,2)),
                Dropout(0.2),

                # second convolutional layer
                Conv2D(32, 3, padding='same', activation='relu'),
                MaxPooling2D((2,2), strides=(2,2)),

                # third convolutional layer
                Conv2D(64, 3, padding='same', activation='relu'),
                MaxPooling2D((2,2), strides=(2,2)),

                # fourth convolutional layer
                Conv2D(128, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Dropout(0.2),
                # Conv2D(256, 3, padding='same', activation='relu'),
                # MaxPooling2D(),
                # Conv2D(512, 3, padding='same', activation='relu'),
                # MaxPooling2D(),
                
                Flatten(),
                Dense(512, activation='relu'),
                Dense(256, activation='relu'),
                Dense(total_labels, activation=self.CONFIG['MODEL_CONFIG']['activation'])
            ])            
        else:
            print('NO Dropouts ...')
            model = Sequential([
                Conv2D(16, 3, padding='same', activation='relu', input_shape=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'] ,3)),
                MaxPooling2D(),
                Conv2D(32, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Conv2D(64, 3, padding='same', activation='relu'),
                MaxPooling2D(),
                Flatten(),
                Dense(512, activation='relu'),
                Dense(2, activation=self.CONFIG['MODEL_CONFIG']['activation'])
            ])

        base_learning_rate = 0.0001
        # model.compile(optimizer=self.CONFIG['MODEL_CONFIG']['optimizer'],
        #       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #       metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])

        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate),
            loss=tf.keras.losses.CategoricalCrossentropy(),
            metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
        # model.compile(
        #     optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
        #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #     metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
        # model.compile(
        #     optimizer=tf.keras.optimizers.SGD(lr=0.0001, momentum=0.9),
        #     loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        #     metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
        model.summary()
        return model
    
    """
    def create_model2(self):
        IMG_SHAPE = (self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'], 3)
        VGG16_MODEL=tf.keras.applications.VGG16(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')
        VGG16_MODEL.trainable=False
        global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
        # global_average_layer = tf.keras.layers.GlobalMaxPooling2D()
        # label_names={'minor': 0, 'moderate': 1, 'severe': 2}
        # prediction_layer = tf.keras.layers.Dense(len(label_names),activation='softmax')
        prediction_layer = tf.keras.layers.Dense(1, activation='softmax')

        model = tf.keras.Sequential([
            VGG16_MODEL,
            global_average_layer,
            prediction_layer
        ])

        # model.compile(optimizer='adam',
        #       loss=tf.keras.losses.sparse_categorical_crossentropy,
        #       metrics=['accuracy'])
        # model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate), 
        #             loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
        #             metrics=['accuracy']
        # )

        base_learning_rate = 0.0001
        model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                    metrics=['accuracy'])
        model.summary()
        return model
    
    def create_model3(self):
        IMG_SHAPE = (self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'], 3)
        # Pre-trained model with MobileNetV2
        base_model = tf.keras.applications.MobileNetV2(
            input_shape=IMG_SHAPE,
            include_top=False,
            weights='imagenet'
        )
        # Freeze the pre-trained model weights
        base_model.trainable = False
        # Trainable classification head
        maxpool_layer = tf.keras.layers.GlobalMaxPooling2D()
        # maxpool_layer = tf.keras.layers.GlobalAveragePooling2D()
        prediction_layer = tf.keras.layers.Dense(1, activation=self.CONFIG['MODEL_CONFIG']['activation'])
        # Layer classification head with feature detector
        model = tf.keras.Sequential([
            base_model,
            maxpool_layer,
            prediction_layer
        ])

        base_learning_rate = 0.0001
        # Compile the model
        model.compile(optimizer=tf.keras.optimizers.Adam(lr=base_learning_rate), 
                    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                     metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
        
        model.summary()
        return model

    def load_vgg16(self, weights_path='../vgg16_weights.h5'):
        model = Sequential()
        model.add(ZeroPadding2D((1,1),input_shape=(3, self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'])))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(64, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(128, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(256, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))

        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(ZeroPadding2D((1,1)))
        model.add(Conv2D(512, 3, 3, activation='relu'))
        model.add(MaxPooling2D((2,2), strides=(2,2)))
        
        # assert os.path.exists(weights_path), 'Model weights not found (see "weights_path")'
        
        if weights_path:
        # note: this chops off the last layers of VGG16 
            f = h5py.File(weights_path)
            for k in range(f.attrs['nb_layers']):
                if k >= len(model.layers): 
                    # we don't look at the last (fully-connected) layers in the savefile
                    break
                g = f['layer_{}'.format(k)]
                weights = [g['param_{}'.format(p)] for p in range(g.attrs['nb_params'])]
                model.layers[k].set_weights(weights)
            f.close()
            print('VGG16 Model with partial weights loaded.')
        else:
            print('VGG16 Model with no weights Loaded.')

        model.compile(optimizer=self.CONFIG['MODEL_CONFIG']['optimizer'],
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=[self.CONFIG['MODEL_CONFIG']['metrics']])
        model.summary()
        return model
    """

    def clean(self):
        dirpath = self.CONFIG['MODEL_CONFIG']['log_dir']
        if os.path.exists(dirpath) and os.path.isdir(dirpath):
            shutil.rmtree(dirpath)

    def get_callbacks(self):
        # Include the epoch in the file name (uses `str.format`)
        checkpoint_path = self.CONFIG['CHECKPOINTS_PATH']
        checkpoint_dir = os.path.dirname(checkpoint_path+'/cp-{epoch:04d}.ckpt')

        log_dir = self.CONFIG['MODEL_CONFIG']['log_dir'] + "/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

        # Create a callback that saves the model's weights every 5 epochs
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_dir, 
            monitor='val_acc',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto',
            save_freq=5)
        return [tensorboard_callback]
    
    def train_model(self, model):
        self.clean()
        dataset = self.data_handler.dataset
        train_data_gen, val_data_gen = self.__prepare_data()
        # Use the fit_generator method of the ImageDataGenerator class to train the network.
        history = model.fit_generator(
            train_data_gen,
            steps_per_epoch = dataset.total_train // self.CONFIG['MODEL_CONFIG']['batch_size'],
            epochs=self.CONFIG['MODEL_CONFIG']['epochs'],
            validation_data = val_data_gen,
            validation_steps = dataset.total_val // self.CONFIG['MODEL_CONFIG']['batch_size'],
            callbacks = self.get_callbacks()
        )
        model.save(self.CONFIG["MODEL_PATH"])
        print("<<<<<<<< ML MODEL CREATED AND SAVED LOCALLY AT: ", self.CONFIG["MODEL_PATH"])
        # model.evaluate(val_data_gen)
        return model, history

    def check_accuracy(self, history):
        acc = history.history['accuracy']
        val_acc = history.history['val_accuracy']

        loss=history.history['loss']
        val_loss=history.history['val_loss']

        epochs_range = range(self.CONFIG['MODEL_CONFIG']['epochs'])

        plt.figure(figsize=(8, 8))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, acc, label='Training Accuracy')
        plt.plot(epochs_range, val_acc, label='Validation Accuracy')
        plt.legend(loc='lower right')
        plt.title('Training and Validation Accuracy')

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, loss, label='Training Loss')
        plt.plot(epochs_range, val_loss, label='Validation Loss')
        plt.legend(loc='upper right')
        plt.title('Training and Validation Loss')
        plt.show()

    def to_label(self, value):
        if value == 0:
            return 'Banana'
        elif value == 1:
            return 'Blueberry'
        elif value == 2:
            return 'Kiwi'
        elif value == 3:
            return 'Lemon'
        else:
            return 'Unknown'

    def load_model(self):
        return load_model(self.CONFIG["MODEL_PATH"])

    def classify_image(self):
        model = self.load_model()
        classify_image_generator = ImageDataGenerator(rescale=1./255) # Generator for our validation data
        classify_data_gen = classify_image_generator.flow_from_directory(batch_size=self.CONFIG['MODEL_CONFIG']['batch_size'],
                                                              directory=self.data_handler.validation_dir,
                                                              target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'])
                                                              )
        test_imgs, test_labels = next(classify_data_gen)
        predictions = model.predict(test_imgs)
        print(predictions)
        df = pd.DataFrame()
        df['actual'] = test_labels[:,1]
        df['predicted'] = np.round(predictions[:,1])
        df['predicted_labels']=df['predicted'].map(lambda x: self.to_label(x))
        print(df['predicted_labels'])
        # plots(test_imgs, titles=df['predicted_labels'])
    
    def predict_image(self, image_path):
        print("Determining Image...")
        model = self.load_model()
        # urllib.urlretrieve(image_path, 'save.jpg') # or other way to upload image
        img = load_img(image_path, target_size=(self.CONFIG['MODEL_CONFIG']['IMG_HEIGHT'], self.CONFIG['MODEL_CONFIG']['IMG_WIDTH'])) # this is a PIL image 
        x = img_to_array(img) # this is a Numpy array with shape (3, 256, 256)
        x = x.reshape((1,) + x.shape)/255 # this is a Numpy array with shape (1, 3, 256, 256)
        pred = model.predict(x)
        pred_label = np.argmax(pred, axis=1)
        
        d = {0: 'Banana', 1: 'Blueberry', 2: 'Kiwi', 3: 'Lemon'}
        for key in d.keys():
            if pred_label[0] == key:
                print("Assessment: {} Image ".format(d[key]))
        print("Prediction complete.")

    
    # This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
    def plotImages(self, images_arr):
        fig, axes = plt.subplots(1, 5, figsize=(20,20))
        axes = axes.flatten()
        for img, ax in zip( images_arr, axes):
            ax.imshow(img)
            ax.axis('off')
        plt.tight_layout()
        plt.show()
   
