# -*- coding: utf-8 -*-
"""
Synthetic Image Generator Module for Presolar Grain Simulations

This module contains all functions dedicated to the generation of synthetic images
from original images. It includes mask creation, PG coordinate generation, Poisson
noise approximation, and core synthetic image generation logic.

Extracted from:
- PG_simulations_func.py: Mask creation, PG coordinate generation, Poisson approximation,
  and synthetic image generation logic

Functions:
- create_circular_mask: Create a single circular mask
- create_circular_mask_multiple: Create multiple circular masks
- PG_coor_mask: Generate presolar grain coordinates and masks
- approx_poisson: Approximate Poisson distribution using normal distribution
- generate_synthetic_image: Core function to generate synthetic images with PG modifications

@author: Maximilien Verdier-Paoletti
"""

import numpy as np
import cv2
from collections.abc import Iterable


# %% Mask creation function
def create_circular_mask(h, w, center=None, radius=None):
    """
    Create a circular mask for a given image size.

    Parameters
    ----------
    h : int
        Height of the image
    w : int
        Width of the image
    center : tuple, optional
        Center coordinates (x, y). If None, uses the middle of the image.
    radius : float, optional
        Radius of the circle. If None, uses the smallest distance from center to image walls.

    Returns
    -------
    mask : ndarray
        Boolean mask array where True indicates pixels inside the circle.
    """
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])
    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0]) ** 2 + (Y - center[1]) ** 2)
    mask = dist_from_center <= radius
    return mask


# ------------- Mask creation for multiple PG with multiple or single radius at once
def create_circular_mask_multiple(h, w, center=None, radius=None):
    """
    Create multiple circular masks for presolar grains.

    Parameters
    ----------
    h : int
        Height of the image
    w : int
        Width of the image
    center : array-like, optional
        Array of center coordinates [[x1, y1], [x2, y2], ...]. If None, uses the middle.
    radius : array-like or float, optional
        Radius or radii for each circle. If None, uses smallest distance from center.

    Returns
    -------
    mask : ndarray
        Integer mask array where 0 indicates background and i+1 indicates grain i.
    """
    if center is None:  # default to the middle of the image
        center = np.array([[int(w / 2), int(h / 2)]])
    else:
        center = np.array(center)

    if radius is None:  # default to smallest distance from center
        radius = np.array([min(c[0], c[1], w - c[0], h - c[1]) for c in center])
    elif not isinstance(radius, Iterable):
        radius = np.full(len(center), radius)

    # Create grid for coordinates
    Y, X = np.ogrid[:h, :w]

    # Initialize mask as zeros (no need for np.empty)
    mask = np.zeros((h, w), dtype=int)

    # Vectorized computation of mask
    for i, (cx, cy) in enumerate(center):
        dist_from_center = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
        mask[dist_from_center <= radius[i]] = (
            i + 1
        )  # Values attributed to PG have to start at 1 as 0 will be the non presolar material in the image

    return mask


# %% PG coordinates mask function
def PG_coor_mask(px, hr_coeff, Nb_PG, data, th, ind_OG_PG, PG_size, raster):
    """
    Generate presolar grain coordinates and create masks.

    This function randomly generates coordinates for presolar grains, ensuring they:
    - Are placed in regions with sufficient counts (above threshold)
    - Don't overlap with original grains (if provided)
    - Don't overlap with each other

    Parameters
    ----------
    px : int
        Original pixel dimension
    hr_coeff : int
        High resolution coefficient (multiplier for high-res image)
    Nb_PG : int
        Number of presolar grains to generate
    data : ndarray
        High resolution image data (3D array with isotopes)
    th : float
        Threshold fraction (0-1) for minimum counts relative to max
    ind_OG_PG : array-like
        Indices of original presolar grain pixels to avoid
    PG_size : array-like
        Size of each presolar grain in nm
    raster : float
        Raster size in microns

    Returns
    -------
    imhr_ini : ndarray
        Copy of high resolution image data
    PG_coor : ndarray
        Coordinates of presolar grains (Nb_PG, 2)
    radius : ndarray
        Radius of each grain in pixels
    mask_PG : ndarray
        Mask indicating presolar grain locations
    """
    # Randomly generate coordinates in the image space
    PG_coor = np.random.choice(px * hr_coeff, size=(Nb_PG, 2), replace=False)
    data_main = data[:, :, 0]
    data_max = data_main.max()  # Max value is computed once, reused
    radius = (PG_size * 1e-3 / (raster / (px * hr_coeff)) / 2).reshape(
        Nb_PG
    )  # Pre-compute radii

    # Flatten data once for efficient access in the loop
    flat_data_main = data_main.ravel()
    it = np.ravel_multi_index(PG_coor.T, data_main.shape)  # 1D index of the coordinates
    coor_verif = flat_data_main[it]  # Get 16O counts at generated positions

    ct = 0
    while True:
        # Check for bad coordinates (below threshold or overlap with OG_PG)
        ind_badcoor = np.where(coor_verif < data_max * th)[0]
        if len(ind_OG_PG) > 0:  # If OG_PG provided, check for overlaps
            ind_badcoor = np.union1d(
                ind_badcoor, np.where(np.isin(PG_coor, ind_OG_PG).all(axis=1))[0]
            )

        # Check for duplicates
        _, counts = np.unique(PG_coor, axis=0, return_counts=True)
        dup = np.where(counts > 1)[0]

        if len(ind_badcoor) == 0 and len(dup) == 0:
            break  # Exit if no bad coordinates and no duplicates

        if len(ind_badcoor) > 0:  # Replace bad coordinates
            PG_coor[ind_badcoor] = np.random.choice(
                px * hr_coeff, size=(len(ind_badcoor), 2), replace=False
            )
            it = np.ravel_multi_index(PG_coor.T, data_main.shape)
            coor_verif = flat_data_main[it]

        # Overlap detection
        dist_matrix = np.sqrt(
            (PG_coor[:, 0, None] - PG_coor[:, 0]) ** 2
            + (PG_coor[:, 1, None] - PG_coor[:, 1]) ** 2
        )
        overlap_matrix = dist_matrix < (radius[:, None] + radius[None, :])
        np.fill_diagonal(overlap_matrix, False)  # Ignore self-comparison

        if overlap_matrix.any():
            overlap_indices = np.argwhere(overlap_matrix)
            for i in overlap_indices:
                PG_coor[i[0], :] = np.random.choice(
                    px * hr_coeff, size=(1, 2), replace=False
                )
                it = np.ravel_multi_index(PG_coor.T, data_main.shape)
                coor_verif = flat_data_main[it]

        ct += 1
        if ct > 10:
            print("Overloop", ct)
            break

    mask_PG = create_circular_mask_multiple(
        px * hr_coeff, px * hr_coeff, center=PG_coor, radius=radius
    )
    imhr_ini = np.copy(data)  # Copy the HR images to avoid alteration

    return imhr_ini, PG_coor, radius, mask_PG


# %% Poisson approximation function
def approx_poisson(data):
    """
    Approximate Poisson distribution using normal distribution.

    This is more efficient than true Poisson sampling for large arrays.

    Parameters
    ----------
    data : ndarray
        Mean values for Poisson distribution

    Returns
    -------
    result : ndarray
        Random values sampled from approximate Poisson distribution
    """
    mean = data
    std_dev = np.sqrt(data)
    return np.random.normal(mean, std_dev).astype(int)


# %% Core synthetic image generation function
def generate_synthetic_image(
    extracted_cts,
    PG_coor,
    mask_PG,
    PG_delta,
    R,
    px,
    hr_coeff,
    beam_size,
    raster,
    boxcar_px,
):
    """
    Generate synthetic image with presolar grain modifications, beam blur, and boxcar smoothing.

    This function takes a high-resolution extracted image and:
    1. Modifies isotope counts at presolar grain locations based on PG_delta
    2. Applies Poisson noise
    3. Applies Gaussian blur (beam blur)
    4. Resizes to original resolution
    5. Applies boxcar smoothing

    Parameters
    ----------
    extracted_cts : ndarray
        High resolution extracted counts (3D array: height, width, isotopes)
    PG_coor : ndarray
        Coordinates of presolar grains (Nb_PG, 2)
    mask_PG : ndarray
        Mask indicating presolar grain locations
    PG_delta : array-like
        Delta values for each presolar grain (including background as first element)
    R : list
        Isotopic ratios [R_main, R_minor1, R_minor2]
    px : int
        Original pixel dimension
    hr_coeff : int
        High resolution coefficient
    beam_size : float
        Beam size in nm
    raster : float
        Raster size in microns
    boxcar_px : int
        Boxcar kernel size in pixels

    Returns
    -------
    imgauss_PG : ndarray
        Gaussian blurred image (after resizing to original resolution)
    imboxcar_PG : ndarray
        Boxcar smoothed image
    """
    # Copy the HR images to avoid alteration
    imhr_ini_PG = np.copy(extracted_cts)

    # Modifying maps counts on location of presolar grains
    R_minor = np.asarray(R[1::])
    imhr_ini_PG[mask_PG != 0, 1::] = (
        extracted_cts[mask_PG != 0, 0][:, None]
        * np.take((PG_delta * 1e-3 + 1) * R_minor, mask_PG, axis=0)[mask_PG != 0, :]
    )

    # Beam blur and Boxcar definitions
    # Defining the sigma parameters of the gaussian blurr
    fwhm_hr = (
        np.round((beam_size * 1e-3) / (raster / (px * hr_coeff))) / 2
    )  # the gaussian filter uses the given sigma as a radius for kernel size if radius is not specified

    # Poissonian random pixel using approximation function (Most efficient method)
    im_poiss = approx_poisson(imhr_ini_PG)

    # Image size reduction with beam blurr then boxcar
    gauss_ker = np.round(fwhm_hr * 2).astype(int)
    if gauss_ker % 2 != 1:
        gauss_ker = gauss_ker + 1
    boxcar_ker = np.ones((boxcar_px, boxcar_px)) / boxcar_px**2
    imgauss_PG = cv2.GaussianBlur(
        im_poiss * 1.0, (gauss_ker, gauss_ker), 0
    )  # int32 are not supported by open cv
    imgauss_PG = cv2.resize(imgauss_PG, (px, px), 0, 0)
    imboxcar_PG = cv2.filter2D(imgauss_PG, cv2.CV_64F, boxcar_ker)
    imgauss_PG.astype(int)  # Images are counts so integers
    imboxcar_PG.astype(int)

    return imgauss_PG, imboxcar_PG
