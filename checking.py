import cv2
import glob
import matplotlib.pyplot as plt


def calculate_treshlod(data_path):
    images, masks = [], []
    print 'Reading ex_images...'
    for image_path in glob.glob(data_path + '/ex_images/*.png'):
        img = cv2.imread(image_path)
        img = cv2.resize(img, (921, 612))
        temp = img == 200
        temp = temp.astype('float32')
        # img = img[:, :, 1]
        # print image_path
        # images.append(img)
        cv2.imshow('img', temp)
        cv2.waitKey()
        cv2.destroyAllWindows()

    print "Reading ex_masks..."
    for mask_path in glob.glob(data_path + '/ex_masks/*.png'):
        mask = cv2.imread(mask_path, 0)
        print mask_path
        masks.append(mask)

    for img, mask in zip(images, masks):
        print "Image {} with mask: {}".format(img, mask)
        result = img * mask
        plt.hist(result.ravel(), 256, [1, 256])
        plt.show()


def test(image_path):
    img = cv2.imread(image_path)  # Read image on image_path
    img = cv2.GaussianBlur(img, (3, 3), 0.5)  # Apply Gaussian blur
    img = img[:, :, 1]  # Green channel BGR, RGB
    img = 255 - img  # Invert green channel
    img = cv2.resize(img, (925, 614))  # Resize for easier representation

    # Show image
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # test("D:/Projects/SSIP/SSIP_17/e_optha_EX/EX/E0000404/C0021833.jpg")
    calculate_treshlod('../exudates/exudates')
