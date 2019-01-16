# coding=utf-8
from numba import jit
import numpy as np
import cv2
import time


@jit(nopython=True)
def getDownSampleValue(img, i, j, win_size):
    end_x = i + win_size
    end_y = j + win_size
    win = img[i:end_x, j:end_y]
    sum_val = 0
    for i in range(win.shape[0]):
        for j in range(win.shape[1]):
            sum_val += win[i, j]
    mean_val = int(sum_val / (win_size * win_size))
    return mean_val


@jit(nopython=True)
def localMean(img_new, new_height, new_width, down_size, img):
    # 计算局部均值方法，效果较好，速度相对较慢
    for i in range(new_height):
        for j in range(new_width):
            img_new[i, j] = getDownSampleValue(img, i * down_size, j * down_size, down_size)


@jit(nopython=True)
def directAssign(img_new, new_height, new_width, down_size, img):
    # 直接赋值，速度更快
    for i in range(new_height):
        for j in range(new_width):
            img_new[i, j] = img[i * down_size, j * down_size]


def downResample(img, down_size, method=1):
    height = img.shape[0]
    width = img.shape[1]

    # print "old size:", height, width

    new_height = height / down_sample
    new_width = width / down_sample

    # print "new size:", new_height, new_width

    img_new = np.zeros([new_height, new_width], img.dtype)

    if method == 1:
        directAssign(img_new, new_height, new_width, down_size, img)
    elif method == 2:
        localMean(img_new, new_height, new_width, down_size, img)
    return img_new


if __name__ == '__main__':
    down_sample = 2
    img = cv2.imread("img.jpg", cv2.IMREAD_GRAYSCALE)

    for i in range(5):
        t1 = time.time()
        img_res = downResample(img, down_sample, method=1)
        t2 = time.time()
        print 'cost time:', t2 - t1

    cv2.imwrite("img_res.jpg", img_res)
