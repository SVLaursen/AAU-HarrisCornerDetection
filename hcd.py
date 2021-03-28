import cv2 as cv
import numpy as np
import math
import scipy.ndimage as ndimage


def rgb2gray(img):
    rgb_weights = [0.2989, 0.5870, 0.1140]
    return np.dot(img[..., :3], rgb_weights)

def gradient_sobel_filter(img):
    img_x = ndimage.convolve(img, np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]))
    img_y = ndimage.convolve(img, np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]))

    result_x = img_x / np.max(img_x)
    result_y = img_y / np.max(img_y)
    return result_x, result_y


def harris_corner_detection(src_img, k, threshold, dist: 10):
    print('Starting Harris Corner Detection')

    gray = rgb2gray(src_img)
    blurred = ndimage.gaussian_filter(gray, sigma=1.0)
    sobel_x, sobel_y = gradient_sobel_filter(blurred)

    # FINDING CORNERS
    print('Finding corners...')

    xx = sobel_x * sobel_x
    yy = sobel_y * sobel_y
    xy = sobel_x * sobel_y

    corners = []
    max_count = 0

    for i in range(1, int(src_img.shape[0] - 1)):
        for j in range(1, int(src_img.shape[1] - 1)):
            window_x = xx[i - 4: i + 5, j - 4: j + 5]
            window_y = yy[i - 4: i + 5, j - 4: j + 5]
            window_xy = xy[i - 4: i + 5, j - 4: j + 5]

            sum_x = np.sum(window_x)
            sum_y = np.sum(window_y)
            sum_xy = np.sum(window_xy)

            determinant = (sum_x * sum_y) - (sum_xy * sum_xy)
            trace = sum_x + sum_y
            r = determinant - (k * trace * trace)
            corners.append((i, j, r))

            if r > max_count:
                max_count = r

    # COMPARING TO THRESHOLD
    print('Comparing corners to threshold...')

    L = []

    for res in corners:
        i, j, r = res
        if r > threshold:
            L.append([i, j, r])


    # NON-MAXIMAL SUPPRESSION
    print('Performing non-maximal suppression...')

    sorted_l = sorted(L, key=lambda x: x[2], reverse=True)
    final = [sorted_l[0][:-1]]
    xc, yc = [], []

    for i in sorted_l:
        for j in sorted_l:
            if abs(i[0] - j[0] <= dist) and abs(i[1] - j[1] <= dist):
                break
            else:
                final.append(i[:-1])
                xc.append(i[1])
                yc.append(i[0])

    result = np.zeros(src_img.shape)

    for i in final:
        y, x = i[0], i[1]
        result[y][x] = 1

    return result
