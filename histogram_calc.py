import cv2
from matplotlib import pyplot as plt



def test(image_path):
    img = cv2.imread(image_path)  # Read image on image_path
    mask = cv2.imread("C:/Users/NT1/Desktop/ssip/SSIP_17/e_optha_EX/Annotation_EX/E0006265/DS000F4K_EX.png", 0)  # Read mask on mask_path
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
    img = img[:, :, 1]  # Green channel BGR
    img = 255 - img  # Invert green channel
    #img = cv2.resize(img, (925, 614))  # Resize for easier representation
    #mask = cv2.resize(mask, (925, 614))

    result = img * mask #apply mask on image

    hist = cv2.calcHist([result], [0], None, [256], [0, 256]) #calculate histogram

    # Show image
    #cv2.imshow('mask', mask)
    #cv2.imshow('img', img)
    print(hist)
    #plt.imshow(hist, interpolation='nearest') #show histogram
    #plt.show()
    #cv2.imshow('histogram', hist)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test(image_path="C:/Users/NT1/Desktop/ssip/SSIP_17/e_optha_EX/EX/E0006265/DS000F4K.jpg")
         #mask_path="C:/Users/NT1/Desktop/ssip/SSIP_17/e_optha_MA/Annotation_MA/E0002477/C0022963.png" )



