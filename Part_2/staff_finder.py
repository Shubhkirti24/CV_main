# coding: utf-8
import numpy as np
import cv2
from PIL import Image, ImageDraw
import math
import sys

def generate_gaussian_mask(filter_size, s):
    # s is the sigma here.
    mask = np.zeros((filter_size, filter_size))
    for i in range(filter_size):
        for j in range(filter_size):
            # Gaussian Kernel formula: ( 1/(2.pi.sigma^2) ).e^(-1/2 ( (i^2 + j^2) / sigma^2 ))
            mask[i][j] = (1 / (2 * np.pi * s ** 2)) * (np.exp(-0.5 * ((i ** 2 + j ** 2) / s ** 2)))
    return mask

def non_maximal_supp(gradient_direction):
    row_length, col_length = gradient_direction.shape
    # In the below line, we are first dividing the angle by 45 and since the maximum angle can
    # be 180, maximum value of 180/45 will be 4. So this will range between -4 to 4 and since
    # we will be rounding these values this will be either of these - -4, -3, -2, -1, 0, 1, 2, 3, 4
    # And then we again multiply this by 45 to get an angle which is either of these values: -180, -135,
    # -90, 0, 90, 135, 180. This helps us in determining in which direction to move in the matrix while
    # choosing the neighbor with the highest value.
    gradient_direction_rounded = 45 * (np.round(gradient_direction / 45))
    grad_non_maximal_supp_matrix = np.zeros(gradient_direction.shape)
    nbr_1, nbr_2 = 0.0, 0.0
    for i in range(1, row_length - 1):
        for j in range(1, col_length - 1):
            direction = gradient_direction_rounded[i][j]
            if direction == 180.0 or direction == -180.0 or direction == 0:
                # We move horizontally, i.e., along the row
                nbr_1, nbr_2 = gradient_direction_rounded[i - 1, j], gradient_direction_rounded[i + 1, j]
            elif direction == -45.0 or direction == 135.0:
                # We move diagonally, in the left diagonal direction.
                nbr_1, nbr_2 = gradient_direction_rounded[i - 1, j + 1], gradient_direction_rounded[i + 1, j - 1]
            elif direction == 45.0 or direction == -135.0:
                # We move diagonally, in the right diagonal direction.
                nbr_1, nbr_2 = gradient_direction_rounded[i + 1, j + 1], gradient_direction_rounded[i - 1, j - 1]
            elif direction == 90.0 or direction == -90.0:
                # We move vertically, i.e., along the column
                nbr_1, nbr_2 = gradient_direction_rounded[i, j - 1], gradient_direction_rounded[i, j + 1]
            else:
                print(">>>>Something went wrong!! This is not supposed to happen!!")

            # If the current cell's value is greater than its neighbor, we take it.
            if gradient_direction_rounded[i][j] > nbr_1 and gradient_direction_rounded[i][j] > nbr_2:
                grad_non_maximal_supp_matrix[i][j] = gradient_direction_rounded[i][j]
    return grad_non_maximal_supp_matrix

def hysteresis_thresholding(img):
    high_thresh = img.max() * 0.09
    low_thresh = high_thresh * 0.03
    n_rows, n_cols = img.shape
    image_with_strong_edges = np.zeros_like(img)
    for i in range(n_rows):
        for j in range(n_cols):
            if img[i][j] >= high_thresh:
                image_with_strong_edges[i][j] = 255
            elif img[i][j] >= low_thresh:
                image_with_strong_edges[i][j] = 10

    for i in range(1, n_rows - 1):
        for j in range(1, n_cols - 1):
            if image_with_strong_edges[i][j] == 10:
                if image_with_strong_edges[i][j - 1] == 255 or image_with_strong_edges[i - 1][j - 1] == 255 or \
                        image_with_strong_edges[i - 1][j] == 255 or image_with_strong_edges[i - 1][j + 1] == 255 or \
                        image_with_strong_edges[i][j + 1] == 255 or image_with_strong_edges[i + 1][j + 1] == 255 or \
                        image_with_strong_edges[i + 1][j] == 255 or image_with_strong_edges[i + 1][j - 1] == 255:
                    image_with_strong_edges[i][j] = 255
    return image_with_strong_edges

def hough_transformation(canny_edge_detected_image, orig_image):
    # rho's range is (-diagonal length of input image, diagonal length of input image)
    # Where diagonal length of input image = (number_of_rows^2 + number_of_columns^2)^0.5
    n_rows, n_cols = canny_edge_detected_image.shape
    r_max = math.ceil((n_rows ** 2 + n_cols ** 2) ** 0.5)
    r_list = [i for i in range(-r_max, r_max+1)]
    r_ndarray = np.array(r_list)
    # Theta ranges from -90 to 90
    t_list = [i for i in range(-90, 90)]
    t_ndarray = np.array(t_list)

    hm_n_rows = len(r_ndarray)
    hm_n_cols = len(t_ndarray)
    hough_matrix = np.zeros((hm_n_rows, hm_n_cols))
    thresh = 200
    for i in range(n_rows):
        for j in range(n_cols):
            if canny_edge_detected_image[i][j] > thresh:
                for hm_col_i, t in enumerate(t_ndarray):
                    # Line is given by - rho = x.cos(theta) + y.sin(theta), we calculate rho using this equation.
                    hm_row_i = int(j * np.cos(np.deg2rad(t)) + i * np.sin(np.deg2rad(t)))
                    # After calculating rho, we increment the hough_matrix for the given rho and theta values.
                    hough_matrix[hm_row_i][hm_col_i] += 1

    hough_list_of_tuples = []
    for i in range(hm_n_rows):
        for j in range(hm_n_cols):
            hough_list_of_tuples.append((hough_matrix[i][j], i, j))

    hough_list_of_tuples_sorted = sorted(hough_list_of_tuples, key=lambda x: x[0], reverse=True)
    # initialize x1 and x2 for each line as a large and small value and calculate corresponding y values using
    # the equation - rho = x.cos(theta) + y.sin(theta) => y = (rho - x.cos(theta)) / sin(theta)
    image_with_lines_highlighted = orig_image
    image_draw = ImageDraw.Draw(image_with_lines_highlighted)
    num_of_iterations = 15

    for i in range(num_of_iterations):
        x1, x2 = -1500, 1500
        r = hough_list_of_tuples_sorted[i][1]
        t = t_ndarray[hough_list_of_tuples_sorted[i][2]]
        x_coeff = np.cos(np.deg2rad(t))
        denom = np.sin(np.deg2rad(t))
        if denom == 0:
            x1 = r / x_coeff
            x2 = x1
            y1 = -1500
            y2 = 1500
        else:
            y1 = (r - x1 * x_coeff) / denom
            y2 = (r - x2 * x_coeff) / denom
        image_draw.line([(x1, y1), (x2, y2)], fill=(255, 0, 0), width=5)

    image_with_lines_highlighted.save('detected_staff.png')

    return image_with_lines_highlighted


def detect_staff(img):
    orig_image = img
    orig_image = orig_image.convert('RGB')
    img = img.convert("L")
    img = np.array(img)

    # Applying canny edge detection steps.

    # Applying gaussian smoothning
    mask = generate_gaussian_mask(3, 11)
    img_after_gaussian_smoothning = cv2.filter2D(img, -1, mask)

    # Calculate gradient along x and y directions
    # We are creating a mask with 2 values -1 and 1, when we apply this to the whole image
    # it will give us the difference between 2 pixels and hence the gradient.
    x_derivative_mask = np.array([[-1.0], [1.0]])
    y_derivative_mask = np.array([[-1.0, 1.0]])
    x_derived_image = cv2.filter2D(img_after_gaussian_smoothning, -1, x_derivative_mask)
    y_derived_image = cv2.filter2D(img_after_gaussian_smoothning, -1, y_derivative_mask)

    # Calculate gradient direction
    # given by: arctan(Image after getting derivative along y / Image after getting derivative along x)
    gradient_dir = np.arctan2(y_derived_image, x_derived_image) * (180 / np.pi)

    # Apply non-maximal suppression
    non_maximal_supp_img = non_maximal_supp(gradient_dir)

    # Apply hysteresis thresholding
    canny_edge_detected_image = hysteresis_thresholding(non_maximal_supp_img)

    # Canny edge detection complete

    # Apply hough transformation
    image_with_lines_highlighted = hough_transformation(canny_edge_detected_image, orig_image)
    image_with_lines_highlighted.save('detected_staff.png')

if __name__ == '__main__':
    img = Image.open(sys.argv[1])
    detect_staff(img)