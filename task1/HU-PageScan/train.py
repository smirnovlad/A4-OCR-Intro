"""Usage: my_program.py [options]

Options:
   -t --train-folder FOLDER      specify output file       [default: ]
   -v --validation-folder FOLDER      specify output file [default: ]
   -m --model NAME      specify output file [default: model]
   --gpu F specify gpu allocation                         [default: 0.6]
   --bs F batch size                         [default: 12]
   --valid-steps F specify gpu allocation                         [default: 20]
   --train-steps F specify gpu allocation                         [default: 20]
   --train-samples NUMBER
   --valid-samples NUMBER
   --lr FLOAT [default: 1e-5]
   --no-aug
   --epochs NUMBER

"""

from docopt import docopt
import tensorflow as tf
import tensorflow.keras.backend as K
import numpy as np
import glob
import os.path as P
# import dataaug
from keras.layers import *
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint, CSVLogger, Callback
import keras
import cv2
import threading
from course_intro_ocr_t1.data import MidvPackage
from pathlib import Path
import json
from matplotlib import pyplot as plt


path_to_save_new_model = './checkpoint'
output_refined = './output'

# uncertain why this hack is needed on the GPU machine
if __name__ == "__main__":
    import os, sys

    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

arguments = docopt(__doc__, version='FIXME')
print(arguments)
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.per_process_gpu_memory_fraction = float(arguments['--gpu'])
sess = tf.compat.v1.Session(config=config)

try:
    K.set_session(sess)
except:
    pass

__DEF_HEIGHT = 512
__DEF_WIDTH = 512

train_steps = int(arguments['--train-steps'])
valid_steps = int(arguments['--valid-steps'])
train_samples = arguments['--train-samples']
valid_samples = arguments['--valid-samples']
bs = int(arguments['--bs'])
lr = float(arguments['--lr'])

DATASET_PATH = Path().absolute() / 'midv500' / 'midv500_compressed'
data_packs = MidvPackage.read_midv500_dataset(DATASET_PATH)

train_items = []
valid_items = []

for i, item in enumerate(DATASET_PATH.iterdir()):
    if item.is_dir():
        package_path = DATASET_PATH / item.name
        items = MidvPackage.collect_items(package_path)
        if i < 43:
            for j in range(10):
                l = j * 30
                r = (j + 1) * 30
                train_items.extend(items[l:l + 25])
                valid_items.extend(items[r - 5:r])
        else:
            valid_items.extend(items)

print(f"Train items: {len(train_items)}, valid items: {len(valid_items)}")
# ignore CLI arguments
train_samples = len(train_items)
valid_samples = len(valid_items)

def generator_batch(items, bs, validation=False, stroke=True):
    batches = []
    for i in range(0, len(items), bs):
        batches.append(items[i: i + bs])

    print("Batching {} batches of size {} each for {} total files".format(len(batches), bs, len(items)))
    while True:
        for batch in batches:
            imgs_batch = []
            masks_batch = []
            for item in batch:
                img_path = str(item.img_path)
                _img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                _img_size = _img.shape
                if _img is None:
                    print(item)
                    continue

                _img = cv2.resize(_img, (__DEF_WIDTH, __DEF_HEIGHT), interpolation=cv2.INTER_CUBIC)
                _img = _img.astype('float32')

                gt_path = str(item.gt_path)
                with open(gt_path, 'r') as file:
                    gt_data = json.load(file)

                mask = np.zeros(_img_size, dtype=np.uint8)
                corner_coordinates = np.array(gt_data['quad'])
                mask = cv2.fillPoly(mask, [corner_coordinates], color=255)
                mask = cv2.resize(mask, (__DEF_WIDTH, __DEF_HEIGHT), interpolation=cv2.INTER_CUBIC)

                _img = 1 - (_img.reshape((__DEF_WIDTH, __DEF_HEIGHT, 1)) / 255)
                mask = mask.reshape((__DEF_WIDTH, __DEF_HEIGHT, 1)) / 255
                mask = mask > 0.3
                # plt.imshow(mask, cmap='gray')
                # plt.show(True)

                mask = mask.astype('float32')
                imgs_batch.append(_img)
                masks_batch.append(mask)

            imgs_batch = np.asarray(imgs_batch).reshape((bs, __DEF_WIDTH, __DEF_HEIGHT, 1)).astype('float32')
            masks_batch = np.asarray(masks_batch).reshape((bs, __DEF_WIDTH, __DEF_HEIGHT, 1)).astype('float32')

            yield imgs_batch, masks_batch


def dice_coef(y_true, y_pred, smooth=1000.0):
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


def get_model():
    inputs = Input((__DEF_WIDTH, __DEF_HEIGHT, 1))

    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(inputs)
    conv1 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool1)
    conv2 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool2)
    conv3 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool3)
    conv4 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(pool4)
    conv5 = Conv2D(512, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv5)

    up6 = concatenate([Conv2DTranspose(512, (2, 2), strides=(2, 2), padding='same')(conv5), conv4], axis=3)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up6)
    conv6 = Conv2D(256, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv6)

    up7 = concatenate([Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(conv6), conv3], axis=3)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up7)
    conv7 = Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv7)

    up8 = concatenate([Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(conv7), conv2], axis=3)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up8)
    conv8 = Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv8)

    up9 = concatenate([Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(conv8), conv1], axis=3)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(up9)
    conv9 = Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same')(conv9)

    conv10 = Conv2D(1, (1, 1), activation='sigmoid', kernel_initializer='he_uniform')(conv9)

    model = Model(inputs=[inputs], outputs=[conv10])

    model.compile(optimizer=Adam(learning_rate=lr), loss=dice_coef_loss, metrics=[dice_coef])
    return model


def main(train_steps, valid_steps, train_samples, valid_samples, bs, train_items, valid_items):
    if train_samples:
        train_items = train_items[:int(train_samples)]

    if valid_samples:
        valid_items = valid_items[:int(valid_samples)]
    train_samples = len(train_items) - (len(train_items) % bs)
    valid_samples = len(valid_items) - (len(valid_items) % bs)

    train_items = train_items[:train_samples]
    valid_items = valid_items[:valid_samples]

    np.random.seed(0)
    np.random.shuffle(train_items)
    np.random.shuffle(valid_items)
    np.random.seed()

    callbacks = []
    monitor = 'val_loss'
    monitor_mode = 'min'

    model = get_model()

    # For use pre trained model
    model = keras.models.load_model("C:\\Users\\Vlad\\A4-OCR-Intro\\task1\\checkpoint\\model.keras", custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})

    print(model.summary())

    callbacks.append(EarlyStopping(
        monitor=monitor, patience=30, verbose=1, mode=monitor_mode,
    ))

    exitlog = CSVLogger('training-model.txt')

    train_gen = generator_batch(train_items, bs=bs, stroke=False)
    valid_gen = generator_batch(valid_items, bs=bs, validation=True)

    class SaveImageCallback(Callback):
        def __init__(self, stroke=False):
            super(SaveImageCallback, self).__init__()
            self.lock = threading.Lock()

        def on_epoch_end(self, epoch, logs={}):
            self.lock = threading.Lock()
            with self.lock:
                data, gt = next(train_gen)
                mask = self.model.predict_on_batch(data)
                for i in range(mask.shape[0]):
                    cv2.imwrite(output_refined + '/%d-%d-0.png' % (epoch, i), mask[i, :, :, 0] * 255)
                    cv2.imwrite(output_refined + '/output_refined/%d-%d-1.png' % (epoch, i), gt[i, :, :, 0] * 255)
                    cv2.imwrite(output_refined + '/output_refined/%d-%d-2.png' % (epoch, i), data[i, :, :, 0] * 255)

    save_net = SaveImageCallback()

    checkpoint_model = ModelCheckpoint(filepath='./checkpoint/model_{epoch:02d}.keras',
        monitor='val_loss',
        save_best_only=False,
        save_weights_only=False,
        mode='auto',
        verbose=1)

    model.fit(
        train_gen,
        steps_per_epoch=train_steps,
        epochs=int(arguments['--epochs']),
        verbose=1,
        validation_data=valid_gen, validation_steps=valid_steps,
        callbacks=[save_net, checkpoint_model, exitlog],
    )

main(train_steps, valid_steps, train_samples, valid_samples, bs, train_items, valid_items)
