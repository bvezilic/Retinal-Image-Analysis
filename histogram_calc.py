import cv2
from matplotlib import pyplot as plt


def test(image_path, mask_path):
    img = cv2.imread(image_path)  # Read image on image_path
    mask = cv2.imread(mask_path, 0)  # Read mask on mask_path
    img = cv2.GaussianBlur(img, (3, 3), 0)  # Apply Gaussian blur
    img = img[:, :, 1]  # Green channel BGR
    # img = 255 - img  # Invert green channel
    # img = cv2.resize(img, (925, 614))  # Resize for easier representation
    # mask = cv2.resize(mask, (925, 614))

    result = img * mask  #apply mask on image

    hist = cv2.calcHist([result], [0], None, [256], [0, 256]) #calculate histogram

    plt.hist(result.ravel(), 256, [1, 256])
    plt.show()
    # Show image
    #cv2.imshow('mask', mask)
    # cv2.imshow('img', img)
    #
    # cv2.imshow('result', result)
    # cv2.waitKey()
    # cv2.destroyAllWindows()

if __name__ == '__main__':
    test(image_path="d:/Projects/SSIP/SSIP_17/e_optha_EX/EX/E0019457/C0001273.jpg",
         mask_path="d:/Projects/SSIP/SSIP_17/e_optha_EX/Annotation_EX/E0019457/C0001273_EX.png")




