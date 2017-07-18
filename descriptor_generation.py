import cv2
import numpy as np
import random
import sklearn.svm as svm

def is_mask_point_considerable(x, y, mask):
    if x - 2 >= 0 and x + 2 < mask.shape[1] and \
        y - 2 >= 0 and y + 2 < mask.shape[0]:
        return sum(mask[y-2 : y+2, x-2:x+2]) >= 10
    return False


def is_background_point_considerable(x, y, mask):
    if x - 2 >= 0 and x + 2 < mask.shape[1] and \
        y - 2 >= 0 and y + 2 < mask.shape[0]:
        return sum(mask[y-2 : y+2, x-2:x+2]) < 10
    return False


def get_descriptor(image, x, y):
    points = image[y-2:y+2, x-2:x+2, 1].flatten()
    hist = cv2.calcHist([result], [0], None, [256], [0, 256])
    return hist


def generate_descriptors(image, mask):
    mask_points_coords = np.nonzero(mask)
    mask_points = zip(mask_points_coords[0], mask_points_coords[1])

    descriptors = []
    labels = []

    # create lesion descriptors
    for y, x in mask_points:
        if is_mask_point_considerable(x, y, mask):
            descriptor = get_descriptor(image, x, y)
            descriptors.append(descriptor)
            labels.append(1)

    # create background descriptors
    num_pf_lesion_descriptors = len(descriptors)
    for _ in range(num_pf_lesion_descriptors):
        got_point = False
        for ___ in range(5):
            x = random.randint(0, image.shape[1]-1)
            y = random.randint(0, image.shape[0]-1)

            if is_background_point_considerable(x, y, mask):
                got_point = True
                break

        if got_point:
            descriptor = get_descriptor(image, x, y)
            descriptors.append(descriptor)
            labels.append(0)

    return descriptors, labels


def train_model(descriptors, labels):
    svm_model = svm.SVC(kernel='poly')
    trained_model = svm_model.fit(descriptors, labels)

    return trained_model
