import cv2
import glob
import matplotlib.pyplot as plt
import pickle
from sklearn.model_selection import train_test_split
from descriptor_generation import *


def read_image(image_path):
    img = cv2.imread(image_path)  # Read image on image_path
    img = cv2.GaussianBlur(img, (3, 3), 1)  # Apply Gaussian blur
    img = img[:, :, 1]  # Green channel BGR
    # img = 255 - img  # Invert green channel
    return img


def read_mask(mask_path):
    img = cv2.imread(mask_path, 0)  # Read mask on mask_path
    ret, img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)  # Apply threshold
    img = img.astype('float32')
    img /= 255  # Scale to range [0,1]
    return img


def read_all_images(data_path):
    images = []
    for image_path in glob.glob(data_path + '/ex_images/*.png'):
        images.append(read_image(image_path=image_path))
    return images


def read_all_masks(data_path):
    masks = []
    for mask_path in glob.glob(data_path + '/ex_masks/*.png'):
        masks.append(read_mask(mask_path=mask_path))
    return masks


def generate_data_set_features(data_path, save_path=''):
    print 'Reading images from {}'.format(data_path)
    images = read_all_images(data_path=data_path)

    print 'Reading masks from {}'.format(data_path)
    masks = read_all_masks(data_path=data_path)

    x, y = [], []
    for image, mask in zip(images, masks):
        desc, label = generate_descriptors(image=image, mask=mask)
        x.extend(desc)
        y.extend(label)

    test_size = 0.2
    random_state = 7
    print 'Splitting data set with ratio: {}'.format(test_size)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=random_state)

    print 'Writing data to .pkl files...'
    pickle.dump(x_train, open('ex_x_train.pkl', 'wb'))
    pickle.dump(y_train, open('ex_y_train.pkl', 'wb'))
    pickle.dump(x_test, open('ex_x_test.pkl', 'wb'))
    pickle.dump(y_test, open('ex_y_test.pkl', 'wb'))


def test_image(image_path):
    img = cv2.imread(image_path)  # Read image on image_path
    img = cv2.GaussianBlur(img, (3, 3), 0.5)  # Apply Gaussian blur
    img = img[:, :, 1]  # Green channel BGR, RGB
    img = 255 - img  # Invert green channel
    img = cv2.resize(img, (925, 614))  # Resize for easier representation

    # Show image
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()


def load_data_and_train():
    x_train = pickle.load(open('ex_x_train.pkl', 'rb'))
    y_train = pickle.load(open('ex_y_train.pkl', 'rb'))
    x_test = pickle.load(open('ex_x_test.pkl', 'rb'))
    y_test = pickle.load(open('ex_y_test.pkl', 'rb'))

    train_model(x_train, y_train, x_test, y_test)


if __name__ == '__main__':
    # generate_data_set_features(data_path='../exudates')
    load_data_and_train()
