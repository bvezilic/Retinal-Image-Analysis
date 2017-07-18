import cv2
import numpy as np
from matplotlib import pyplot as plt


def test(image_path):
    img = cv2.imread(image_path)  # Read image on image_path
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    #img = img[:, :, 1]  # Green channel BGR
    # img = 255 - img  # Invert green channel
    img = cv2.resize(img, (925, 614))  # Resize for easier representation
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
    gray = np.float32(gray)  # Convert to 32bit float

    result = cv2.cornerHarris(gray,2,3,0.08)
    result = cv2.dilate(result, None)

    img[result > 0.095 * result.max()] = [0, 0, 255]  # Set treshold

    # Show image
    cv2.imshow('img', img)
    #cv2.imshow('result', result)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test(image_path="C:/Users/NT1/Desktop/ssip/SSIP_17/e_optha_EX/EX/E0007442/DS000FGD.jpg")
        # mask_path="C:/Users/NT1/Desktop/ssip/SSIP_17/e_optha_EX/Annotation_EX/E0019457/C0001273_EX.png")





