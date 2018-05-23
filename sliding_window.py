import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

image = mpimg.imread('bbox-example-image.jpg')


# Here is your draw_boxes function from the previous exercise
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy


# Define a function that takes an image,
# start and stop positions in both x and y,
# window size (x and y dimensions),
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
    _, w, h = img.shape[::-1]
    x_start, x_stop = x_start_stop
    y_start, y_stop = y_start_stop
    # If x and/or y start/stop positions not defined, set to image size
    if x_start_stop[0] is None:
        x_start = 0
    if x_start_stop[1] is None:
        x_stop = w
    if y_start_stop[0] is None:
        y_start = 0
    if y_start_stop[1] is None:
        y_stop = h
    # Compute the span of the region to be searched
    window_span_x = x_stop - x_start - xy_window[0]
    window_span_y = y_stop - y_start - xy_window[1]
    # Compute the number of pixels per step in x/y
    pixels_per_x = int(xy_window[0] * xy_overlap[0])
    pixels_per_y = int(xy_window[1] * xy_overlap[1])
    # Compute the number of windows in x/y
    windows_x = 1 + round(window_span_x / pixels_per_x)
    windows_y = 1 + round(window_span_y / pixels_per_y)
    # Initialize a list to append window positions to
    window_list = []
    # Loop through finding x and y window positions
    for window_y in range(windows_y):
        for window_x in range(windows_x):
            top_left_x = x_start + (window_x * pixels_per_x)
            bottom_right_x = top_left_x + xy_window[0]
            top_left_y = y_start + (window_y * pixels_per_y)
            bottom_right_y = top_left_y + xy_window[1]

            window_list.append(((top_left_x, top_left_y), (bottom_right_x, bottom_right_y)))
    #     Note: you could vectorize this step, but in practice
    #     you'll be considering windows one by one with your
    #     classifier, so looping makes sense
    # Calculate each window position
    # Append window position to list
    # Return the list of windows
    return window_list


windows = slide_window(image, x_start_stop=[None, None], y_start_stop=[None, None],
                       xy_window=(128, 128), xy_overlap=(0.5, 0.5))

window_img = draw_boxes(image, windows, color=(0, 0, 255), thick=6)
plt.imshow(window_img)
plt.show()