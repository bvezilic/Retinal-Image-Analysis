import cv2


def test(image_path):
    img = cv2.imread(image_path)  # Read image on image_path
    img = cv2.GaussianBlur(img, (5, 5), 0)  # Apply Gaussian blur
    img = img[:, :, 1]  # Green channel BGR
    img = 255 - img  # Invert green channel
    img = cv2.resize(img, (925, 614))  # Resize for easier representation

    # Show image
    cv2.imshow('img', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    test("D:/Projects/SSIP/SSIP_17/e_optha_MA/MA/E0002477/C0022963.jpg")
