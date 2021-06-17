"""
Mask R-CNN
Common utility functions and classes.

Copyright (c) 2017 Matterport, Inc.
Licensed under the MIT License (see LICENSE_MATTERPORT for details)
Written by Waleed Abdulla

Copyright (c) 2021 Skinet Team
Licensed under the MIT License (see LICENSE for details)
Updated/Modified by Adrien JAUGEY
"""
import os
import shutil
import zipfile
import urllib.request
from distutils.version import LooseVersion

import cv2
import numpy as np
import skimage.color
import skimage.io
import skimage.transform

from mrcnn.Config import Config
from mrcnn import datasetDivider as dD

# URL from which to download the latest COCO trained weights
WEIGHTS_URL = [f"https://github.com/SkinetTeam/Skinet-MEST-C/releases/download/v{version}/skinet_{mode}.zip"
               for version, mode in [("1.0", "cortex"), ("1.0", "mest_main"), ("1.0", "mest_glom")]]


############################################################
#  Masks
############################################################

def reduce_memory(results, config: Config, allow_sparse=True):
    """
    Minimize all masks in the results dict from inference
    :param results: dict containing results of the inference
    :param config: the config object
    :param allow_sparse: if False, will only keep biggest region of a mask
    :return:
    """
    _masks = results['masks']
    _bbox = results['rois']
    if not allow_sparse:
        emptyMasks = []
        for idx in range(results['masks'].shape[-1]):
            mask = unsparse_mask(results['masks'][:, :, idx])
            if mask is None:
                emptyMasks.append(idx)
            else:
                results['masks'][:, :, idx] = mask
        if len(emptyMasks) > 0:
            results['scores'] = np.delete(results['scores'], emptyMasks)
            results['class_ids'] = np.delete(results['class_ids'], emptyMasks)
            results['masks'] = np.delete(results['masks'], emptyMasks, axis=2)
            results['rois'] = np.delete(results['rois'], emptyMasks, axis=0)
        results['rois'] = extract_bboxes(results['masks'])
    results['masks'] = minimize_mask(results['rois'], results['masks'], config.get_mini_mask_shape())
    return results


def get_mask_area(mask, verbose=0):
    """
    Computes mask area
    :param mask: the array representing the mask
    :param verbose: 0 : nothing, 1+ : errors/problems
    :return: the area of the mask and verbose output (None when nothing to print)
    """
    maskHistogram = dD.getBWCount(mask)
    display = None
    if verbose > 0:
        nbPx = mask.shape[0] * mask.shape[1]
        tempSum = maskHistogram[0] + maskHistogram[1]
        if tempSum != nbPx:
            display = "Histogram pixels {} != total pixels {}".format(tempSum, nbPx)
    return maskHistogram[1], display


def unsparse_mask(base_mask):
    """
    Return mask with only its biggest part
    :param base_mask: the mask image as np.bool or np.uint8
    :return: the main part of the mask as a same shape image and type
    """
    # http://www.learningaboutelectronics.com/Articles/How-to-find-the-largest-or-smallest-object-in-an-image-Python-OpenCV.php
    # https://stackoverflow.com/questions/19222343/filling-contours-with-opencv-python
    # Convert to np.uint8 if not before processing
    convert = False
    if type(base_mask[0, 0]) is np.bool_:
        convert = True
        base_mask = base_mask.astype(np.uint8) * 255
    # Padding the mask so that parts on edges will get correct area
    base_mask = np.pad(base_mask, 1, mode='constant', constant_values=0)
    res = np.zeros_like(base_mask, dtype=np.uint8)

    # Detecting contours and keeping only one with biggest area
    contours, _ = cv2.findContours(base_mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if len(contours) > 0:
        if len(contours) > 1:  # If only one region, reconstructing mask is useless
            biggest_part = sorted(contours, key=cv2.contourArea, reverse=True)[0]

            # Drawing the biggest part on the result mask
            cv2.fillPoly(res, pts=[biggest_part], color=255)
        else:
            res = base_mask
        # Removing padding of the mask
        res = res[1:-1, 1:-1]
        return res.astype(np.bool) if convert else res
    else:
        return None


############################################################
#  Bounding Boxes
############################################################
def in_roi(roi_to_test, roi, epsilon=0):
    """
    Tests if the RoI to test is included in the given RoI
    :param roi_to_test: the RoI/bbox to test
    :param roi: the RoI that should include the one to test
    :param epsilon: margin of the RoI to allow boxes that are not exactly inside
    :return: True if roi_to_test is included in roi
    """
    res = True
    i = 0
    while i < 4 and res:
        res = res and (roi[i % 2] - epsilon <= roi_to_test[i] <= roi[i % 2 + 2] + epsilon)
        i += 1
    return res


def get_bbox_area(roi):
    """
    Returns the bbox area
    :param roi: the bbox to use
    :return: area of the given bbox
    """
    return (roi[3] - roi[1]) * (roi[2] - roi[0])


def get_bboxes_intersection(roiA, roiB):
    """
    Computes the intersection area of two bboxes
    :param roiA: the first bbox
    :param roiB: the second bbox
    :return: the area of the intersection
    """
    xInter = min(roiA[3], roiB[3]) - max(roiA[1], roiB[1])
    yInter = min(roiA[2], roiB[2]) - max(roiA[0], roiB[0])
    return max(xInter, 0) * max(yInter, 0)


def global_bbox(roiA, roiB):
    """
    Returns the bbox enclosing two given bboxes
    :param roiA: the first bbox
    :param roiB: the second bbox
    :return: the enclosing bbox
    """
    return np.array([min(roiA[0], roiB[0]), min(roiA[1], roiB[1]), max(roiA[2], roiB[2]), max(roiA[3], roiB[3])])


def shift_bbox(roi, customShift=None):
    """
    Shifts bbox coordinates so that min x and min y equal 0
    :param roi: the roi/bbox to transform
    :param customShift: custom x and y shift as (yShift, xShift)
    :return: the shifted bbox
    """
    yMin, xMin, yMax, xMax = roi
    if customShift is None:
        return np.array([0, 0, yMax - yMin, xMax - xMin])
    else:
        return np.array([max(yMin - customShift[0], 0), max(xMin - customShift[1], 0),
                         max(yMax - customShift[0], 0), max(xMax - customShift[1], 0)])


def expand_masks(mini_mask1, roi1, mini_mask2, roi2):
    """
    Expands two masks while keeping their relative position
    :param mini_mask1: the first mini mask
    :param roi1: the first mask bbox/roi
    :param mini_mask2: the second mini mask
    :param roi2: the second mask bbox/roi
    :return: mask1, mask2
    """
    roi1And2 = global_bbox(roi1, roi2)
    shifted_roi1And2 = shift_bbox(roi1And2)
    shifted_roi1 = shift_bbox(roi1, customShift=roi1And2[:2])
    shifted_roi2 = shift_bbox(roi2, customShift=roi1And2[:2])
    mask1 = expand_mask(shifted_roi1, mini_mask1, shifted_roi1And2[2:])
    mask2 = expand_mask(shifted_roi2, mini_mask2, shifted_roi1And2[2:])
    return mask1, mask2


def extract_bboxes(mask):
    """Compute bounding boxes from masks.
    mask: [height, width, num_instances]. Mask pixels are either 1 or 0.

    Returns: bbox array [num_instances, (y1, x1, y2, x2)].
    """
    soleMask = False
    if len(mask.shape) != 3:
        _mask = np.expand_dims(mask, 2)
        soleMask = True
    else:
        _mask = mask
    boxes = np.zeros([_mask.shape[-1], 4], dtype=np.int32)
    for i in range(_mask.shape[-1]):
        m = _mask[:, :, i]
        # Bounding box.
        horizontal_indicies = np.where(np.any(m, axis=0))[0]
        vertical_indicies = np.where(np.any(m, axis=1))[0]
        if horizontal_indicies.shape[0]:
            x1, x2 = horizontal_indicies[[0, -1]]
            y1, y2 = vertical_indicies[[0, -1]]
            # x2 and y2 should not be part of the box. Increment by 1.
            x2 += 1
            y2 += 1
        else:
            # No mask for this instance. Might happen due to
            # resizing or cropping. Set bbox to zeros
            x1, x2, y1, y2 = 0, 0, 0, 0
        boxes[i] = np.array([y1, x1, y2, x2]).astype(np.int32)
    return boxes[0] if soleMask else boxes


############################################################
#  Dataset
############################################################
def minimize_mask(bbox, mask, mini_shape):
    """Resize masks to a smaller version to reduce memory load.
    Mini-masks can be resized back to image scale using expand_masks()

    See inspect_data.ipynb notebook for more details.
    """
    soleMask = False
    if len(bbox.shape) != 2 and len(mask.shape) != 3:
        soleMask = True
        _bbox = np.expand_dims(bbox, 0)
        _mask = np.expand_dims(mask, 2)
    else:
        _bbox = bbox
        _mask = mask
    mini_mask = np.zeros(mini_shape + (_mask.shape[-1],), dtype=bool)
    for i in range(_mask.shape[-1]):
        # Pick slice and cast to bool in case load_mask() returned wrong dtype
        m = _mask[:, :, i].astype(bool).astype(np.uint8) * 255
        y1, x1, y2, x2 = _bbox[i][:4]
        m = m[y1:y2, x1:x2]
        if m.size == 0:
            raise Exception("Invalid bounding box with area of zero")
        # Resize with bilinear interpolation
        m = resize(m, mini_shape)
        mini_mask[:, :, i] = np.around(m).astype(np.bool)
    return mini_mask[:, :, 0] if soleMask else mini_mask


def expand_mask(bbox, mini_mask, image_shape):
    """Resizes mini masks back to image size. Reverses the change
    of minimize_mask().

    See inspect_data.ipynb notebook for more details.
    """
    if type(image_shape) is not tuple:
        image_shape = tuple(image_shape)
    soleMask = False
    if len(bbox.shape) != 2 and len(mini_mask.shape) != 3:
        soleMask = True
        _bbox = np.expand_dims(bbox, 0)
        _mini_mask = np.expand_dims(mini_mask, 2)
    else:
        _bbox = bbox
        _mini_mask = mini_mask
    mask = np.zeros(image_shape[:2] + (_mini_mask.shape[-1],), dtype=bool)
    for i in range(mask.shape[-1]):
        m = _mini_mask[:, :, i].astype(bool).astype(np.uint8) * 255
        y1, x1, y2, x2 = _bbox[i][:4]
        h = y2 - y1
        w = x2 - x1
        # Resize with bilinear interpolation
        m = resize(m, (h, w))
        mask[y1:y2, x1:x2, i] = np.around(m).astype(np.bool)
    return mask[:, :, 0] if soleMask else mask


############################################################
#  Miscellaneous
############################################################
def download_trained_weights(weights=None, verbose=1):
    """ Download trained weights from Releases. """
    if weights is None:
        weights = WEIGHTS_URL
    if verbose > 0:
        print("Downloading weights files if needed ...", end='')
    for weightsUrl in weights:
        path = weightsUrl.split('/')[-1]
        if not os.path.exists(path) and not os.path.exists(path.replace(".zip", "")):
            with urllib.request.urlopen(weightsUrl) as resp, open(path, 'wb') as out:
                shutil.copyfileobj(resp, out)
        if not os.path.exists(path.replace(".zip", "")):
            with zipfile.ZipFile(path, 'r') as zip_ref:
                zip_ref.extractall(".")
    if verbose > 0:
        print(" Done !")


def resize(image, output_shape, order=1, mode='constant', cval=0, clip=True,
           preserve_range=False, anti_aliasing=False, anti_aliasing_sigma=None):
    """A wrapper for Scikit-Image resize().

    Scikit-Image generates warnings on every call to resize() if it doesn't
    receive the right parameters. The right parameters depend on the version
    of skimage. This solves the problem by using different parameters per
    version. And it provides a central place to control resizing defaults.
    """
    if LooseVersion(skimage.__version__) >= LooseVersion("0.14"):
        # New in 0.14: anti_aliasing. Default it to False for backward
        # compatibility with skimage 0.13.
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range, anti_aliasing=anti_aliasing,
            anti_aliasing_sigma=anti_aliasing_sigma)
    else:
        return skimage.transform.resize(
            image, output_shape,
            order=order, mode=mode, cval=cval, clip=clip,
            preserve_range=preserve_range)
