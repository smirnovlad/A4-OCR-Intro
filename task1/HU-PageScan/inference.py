import time

start = time.time()

import tensorflow as tf
import cv2
import random
import numpy as np
import os
import sys
from keras.layers import *
import keras
import json
import imutils
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from course_intro_ocr_t1.data import MidvPackage
from course_intro_ocr_t1.metrics import measure_crop_accuracy, dump_results_dict
from pathlib import Path


def dice_coef(y_true, y_pred, smooth=1000.0):
    y_true_f = Flatten()(y_true)
    y_pred_f = Flatten()(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)

    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)


DATASET_PATH = Path().absolute() / 'midv500' / 'midv500_compressed'
data_packs = MidvPackage.read_midv500_dataset(DATASET_PATH)

test_items = []
for i, item in enumerate(DATASET_PATH.iterdir()):
    if item.is_dir():
        package_path = DATASET_PATH / item.name
        items = MidvPackage.collect_items(package_path)
        for item in items:
            if item.is_test_split():
                test_items.append(item)

print(len(test_items))

# random.shuffle(test_items)

model_path = "checkpoint/model.keras"
model = keras.models.load_model(model_path, custom_objects={'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef})
print(model.summary())

count1 = 0
SAIDA = np.empty((1, 512, 512, 1))
ENTRADA = np.empty((1, 512, 512, 1))
GTIMG = np.empty((1, 512, 512, 1))


def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Load image in grayscale
    img = 1 - (img / 255)  # Invert pixel values and normalize
    img = cv2.resize(img, (512, 512), interpolation=cv2.INTER_CUBIC)  # Resize image
    img = img[None, :, :, np.newaxis]
    img_tensor = tf.convert_to_tensor(img, dtype=tf.float32)  # Convert numpy array to TensorFlow tensor
    return img_tensor  # Add singleton dimension for channel


def get_gt_mask(item):
    gt_path = str(item.gt_path)
    with open(gt_path, 'r') as file:
        gt_data = json.load(file)

    gt_mask = np.zeros((800, 450), dtype=np.uint8)
    corner_coordinates = np.array(gt_data['quad'])
    gt_mask = cv2.fillPoly(gt_mask, [corner_coordinates], color=255) / 255
    gt_mask = gt_mask > 0.3

    return gt_mask


def get_output_mask(prediction):
    output_mask = prediction[0, :, :, 0] * 255
    output_mask = output_mask.numpy()
    output_mask = output_mask.astype(np.uint8)
    output_mask = cv2.resize(output_mask, (450, 800), interpolation=cv2.INTER_CUBIC)  # Resize image

    return output_mask


def get_mask_contour(mask):
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    contour = max(cnts, key=cv2.contourArea)
    contour = np.squeeze(contour)

    return contour


# file_jaccard = open('jaccard.txt', 'w')
# file_time = open('time_run.txt', 'w')
# jaccard_result = []

results_dict = dict()
for i, item in tqdm(enumerate(test_items)):
    img = load_and_preprocess_image(str(item.img_path))
    start_time = time.time()
    prediction = model(img, training=False)
    end_time = time.time()
    # print("Inference time: {:.4f} seconds".format(end_time - start_time))

    gt_mask = get_gt_mask(item)
    output_mask = get_output_mask(prediction)
    cv2.imwrite(f"./outputs/output_{i}.jpg", output_mask)


    contour = get_mask_contour(output_mask)

    # fig, axs = plt.subplots(2, 3, figsize=(12, 8))

    # axs[0][0].imshow(cv2.imread(str(item.img_path)))
    # axs[0][1].imshow(mask, cmap='gray')
    # axs[0][2].imshow(result_img, cmap='gray')

    x = contour[:, 0]
    y = contour[:, 1]
    # axs[1][0].set_title('Original contour')
    # axs[1][0].imshow(result_img, cmap='gray')
    # axs[1][0].scatter(c[:, 0],c[:, 1], s=5, c='green')

    peri = cv2.arcLength(contour, True)

    eps_1 = None
    eps_2 = None
    for eps in np.linspace(0.001, 1, 1000):
        approx = cv2.approxPolyDP(contour, eps * peri, True)
        if len(approx) < 32 and len(approx) > 4:
            eps_1 = eps
        elif len(approx) == 4:
            eps_2 = eps
            break

    if eps_1 is None or eps_2 is None:
        print(f"Item unique key: {item.unique_key}")
        print(f"original contour size: {len(contour)}")
        print(f"eps_1: {eps_1}")
        print(f"eps_2: {eps_2}")
        continue

    approx_1 = cv2.approxPolyDP(contour, eps_1 * peri, True)
    # axs[1][1].set_title(f'{len(approx)} key points')
    # axs[1][1].imshow(result_img, cmap='gray')
    # axs[1][1].scatter(approx_1[:, 0, 0], approx_1[:, 0, 1], s=25, c='red')

    approx_2 = cv2.approxPolyDP(contour, eps_2 * peri, True)
    # axs[1][2].set_title('4 key points')
    # axs[1][2].imshow(result_img, cmap='gray')
    # axs[1][2].scatter(approx_2[:, 0, 0], approx_2[:, 0, 1], s=25, c='red')
    #
    # plt.show()

    abs_crop = approx_2[:, 0, :]
    h, w = 800, 450
    rel_crop = abs_crop / np.array([[w, h]])

    output = cv2.imread(str(item.img_path))
    cv2.drawContours(output, [abs_crop], -1, (0, 255, 0), 3)
    cv2.imwrite(f"./outputs/output_{i}.jpg", output)

    try:
        results_dict[item.unique_key] = rel_crop
    except Exception as exc:
        # Для пропущенных в словаре ключей в метриках автоаматически засчитается IoU=0
        print(exc)

    # Визуализация результатов (если необходимо)
    # plt.imshow(result_img, cmap='gray')
    # plt.show()

    # result_name = file.replace('_in', '_result')
    # cv2.imwrite(localization_save_output + result_name, saida)
    #
    # y_true_path = impath.replace('_in', '_gt')
    # y_true = cv2.imread(y_true_path)
    # y_pred = cv2.imread(localization_save_output + result_name)
    #
    # jaccardScore = jaccard_score(y_true.flatten(), y_pred.flatten())
    # file_jaccard.write(str(jaccardScore) + '\n')

# file_jaccard.close()
# file_time.close()


dump_results_dict(results_dict, Path() / 'pred.json')
acc = measure_crop_accuracy(
    Path() / 'pred.json',
    Path() / 'gt.json'
)
print("Точность кропа: {:1.4f}".format(acc))
