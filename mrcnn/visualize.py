"""
Mask R-CNN
Display and Visualization Functions.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE_MATTERPORT for details)
Written by Waleed Abdulla

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Updated/Modified by Adrien JAUGEY
"""

import os
import sys
import random
import colorsys
import cv2

import numpy as np
from skimage.measure import find_contours
import matplotlib.pyplot as plt
from matplotlib import patches
from matplotlib.patches import Polygon

# Root directory of the project
from mrcnn.datasetDivider import CV2_IMWRITE_PARAM
from mrcnn.Config import Config

ROOT_DIR = os.path.abspath("../../")

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils


############################################################
#  Visualization
############################################################
def random_colors(N, bright=True, shuffle=True):
    """
    Generate random colors.
    To get visually distinct colors, generate them in HSV space then
    convert to RGB.
    """
    brightness = 1.0 if bright else 0.7
    hsv = [(i / N, 1, brightness) for i in range(N)]
    colors = list(map(lambda c: colorsys.hsv_to_rgb(*c), hsv))
    if shuffle:
        random.shuffle(colors)
    return colors


def apply_mask(image, mask, color, alpha=0.5, bbox=None):
    """Apply the given mask to the image.
    """
    # Define bbox as whole image if not given
    if bbox is None:
        y1, x1, y2, x2 = 0, 0, image.shape[0], image.shape[1]
    else:
        y1, x1, y2, x2 = bbox

    # Take only mask part if not already
    if (y2 - y1) != mask.shape[0] or (x2 - x1) != mask.shape[1]:
        _mask = mask[y1:y2, x1:x2]
    else:
        _mask = mask

    if type(color) in [tuple, list] and len(image.shape) > 2:
        # Color conversion if given in percentage instead of raw value
        if type(color[0]) is float:
            _color = [round(channelColor * 255) for channelColor in color]
        else:
            _color = color

        # Apply mask on each channel
        for channel in range(3):
            image[y1:y2, x1:x2, channel] = np.where(
                _mask > 0,
                (image[y1:y2, x1:x2, channel].astype(np.uint32) * (1 - alpha) + alpha * _color[channel]).astype(
                    np.uint8),
                image[y1:y2, x1:x2, channel]
            )
    elif type(color) in [int, float] and len(image.shape) == 2:
        # Color conversion if given in percentage instead of raw value
        _color = (color * 255) if type(color) is float else color

        image[y1:y2, x1:x2] = np.where(
            _mask > 0,
            (image[y1:y2, x1:x2].astype(np.uint32) * (1 - alpha) + alpha * _color).astype(np.uint8),
            image[y1:y2, x1:x2]
        )
    return image


def get_text_color(r, g, b, light_threshold=0.5):
    """
    Return black or white text color depending on the background color
    :param r: amount of red color
    :param g: amount of green color
    :param b: amount of blue color
    :param light_threshold: the threshold used to determine whether or not a color is dark or light
    :return: "k" if background color is light else "w"
    """
    _, light, _ = colorsys.rgb_to_hls(r, g, b)
    return "k" if light >= light_threshold else "w"


def display_instances(image, boxes, masks, class_ids, class_names,
                      scores=None, title="",
                      figsize=(16, 16), ax=None, fig=None, image_format="jpg",
                      show_mask=True, show_bbox=True,
                      colors=None, colorPerClass=False, captions=None,
                      fileName=None, save_cleaned_img=False, silent=False, config: Config = None):
    """
    boxes: [num_instance, (y1, x1, y2, x2, class_id)] in image coordinates.
    masks: [height, width, num_instances]
    class_ids: [num_instances]
    class_names: list of class names of the dataset
    scores: (optional) confidence scores for each box
    title: (optional) Figure title
    show_mask, show_bbox: To show masks and bounding boxes or not
    figsize: (optional) the size of the image
    colors: (optional) An array or colors to use with each object
    captions: (optional) A list of strings to use as captions for each object
    """
    # Number of instances
    N = boxes.shape[0]
    if not N:
        print("\n*** No instances to display *** \n")
    else:
        assert boxes.shape[0] == masks.shape[-1] == class_ids.shape[0]

    # If no axis is passed, create one and automatically call show()
    auto_show = False
    ownFig = False
    if ax is None or fig is None:
        ownFig = True
        fig, ax = plt.subplots(1, figsize=figsize)
        auto_show = not silent

    # Generate random colors
    nb_color = (len(class_names) - 1) if colorPerClass else N
    colors = colors if colors is not None else random_colors(nb_color, shuffle=(not colorPerClass))
    if type(colors[0][0]) is int:
        _colors = []
        for color in colors:
            _colors.append([c / 255. for c in color])
    else:
        _colors = colors
    # Show area outside image boundaries.
    height, width = image.shape[:2]
    ax.set_ylim(height + 10, -10)
    ax.set_xlim(-10, width + 10)
    ax.axis('off')
    ax.set_title(title)

    # To be usable on Google Colab we do not make a copy of the image leading to too much ram usage if it is a biopsy
    # or nephrectomy image
    masked_image = image
    # masked_image = image.astype(np.uint32).copy()
    for i in range(N):
        if colorPerClass:
            color = _colors[class_ids[i] - 1]
        else:
            color = _colors[i]
        # Bounding box
        if not np.any(boxes[i]):
            # Skip this instance. Has no bbox. Likely lost in image cropping.
            continue
        y1, x1, y2, x2 = boxes[i]
        if show_bbox:
            p = patches.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2,
                                  alpha=0.7, linestyle="dashed",
                                  edgecolor=color, facecolor='none')
            ax.add_patch(p)

        # Label
        if not captions:
            class_id = class_ids[i]
            score = scores[i] if scores is not None else None
            label = class_names[class_id]
            caption = "{} {:.3f}".format(label, score) if score else label
        else:
            caption = captions[i]
        ax.text(x1 + 4, y1 + 19, caption, color=get_text_color(color[0], color[1], color[2]),
                size=12, backgroundcolor=color)

        # Mask
        mask = masks[:, :, i]
        bbox = boxes[i]
        shift = np.array([0, 0])
        if config is not None and config.is_using_mini_mask():
            shifted_bbox = utils.shift_bbox(bbox)
            shift = bbox[:2]
            mask = utils.expand_mask(shifted_bbox, mask, tuple(shifted_bbox[2:]))
            mask = mask.astype(np.uint8) * 255
        if show_mask:
            masked_image = apply_mask(masked_image, mask, color, bbox=bbox)

        # Mask Polygon
        # Pad to ensure proper polygons for masks that touch image edges.
        padded_mask = np.zeros(
            (mask.shape[0] + 2, mask.shape[1] + 2), dtype=np.uint8)
        padded_mask[1:-1, 1:-1] = mask
        contours = find_contours(padded_mask, 0.5)
        for verts in contours:
            verts = verts + shift
            # Subtract the padding and flip (y, x) to (x, y)
            verts = np.fliplr(verts) - 1
            p = Polygon(verts, facecolor="none", edgecolor=color)
            ax.add_patch(p)
    # masked_image = masked_image.astype(np.uint8)
    ax.imshow(masked_image)
    fig.tight_layout()
    if fileName is not None:
        fig.savefig(f"{fileName}.{image_format}")
        if save_cleaned_img:
            BGR_img = cv2.cvtColor(masked_image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(f"{fileName}_clean.{image_format}", BGR_img, CV2_IMWRITE_PARAM)
    if auto_show:
        plt.show()
    fig.clf()
    if ownFig:
        del ax, fig
    return masked_image
