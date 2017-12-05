import cv2


def read_image(path):
    return cv2.imread(path)


def write_image(path, img):
    cv2.imwrite(path, img)


def crop_image(img):
    assert img.ndim == 3
    assert img.shape[2] == 3

    padsize = 256
    cropsize = 224

    if img.shape[0] < img.shape[1]:
        other = (img.shape[1] * padsize) // img.shape[0]
        resized = cv2.resize(img, (other, padsize))
    else:
        other = (img.shape[0] * padsize) // img.shape[1]
        resized = cv2.resize(img, (padsize, other))
    pad0 = (resized.shape[0] - cropsize) // 2
    pad1 = (resized.shape[1] - cropsize) // 2

    cropped = resized[pad0:pad0 + cropsize, pad1:pad1 + cropsize, :]
    return cropped
