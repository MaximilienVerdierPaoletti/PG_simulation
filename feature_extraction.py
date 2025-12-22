# -*- coding: utf-8 -*-
"""
Feature extraction module for PG NanoSIMS Simulations

This module contains functions for extracting features from simulated grain images,
including mask creation, contour detection, diameter calculation, and delta value extraction.

@author: Maximilien Verdier-Paoletti
"""

import numpy as np
import skimage.measure
from synthetic_image_generator import create_circular_mask


def select_sigma_delta_maps(plots, plots_title, grain_delta):
    """
    Select sigma and delta maps based on the most anomalous ratio.
    
    Parameters
    ----------
    plots : list
        List of plot arrays/maps
    plots_title : list
        List of plot titles corresponding to plots
    grain_delta : pandas.DataFrame
        DataFrame containing delta values for the grain
        
    Returns
    -------
    sigma_map : list
        List of sigma maps
    delta_map : list
        List of delta maps
    sigma_anomalous_map_index : list
        Index of the anomalous sigma map
    delta_anomalous_map_index : list
        Index of the anomalous delta map
    anomalous_ratio_name : str
        Name of the anomalous ratio
    """
    # Locate most anomalous ratio. It will be the one used for contouring
    ind_anomalous = np.argmax(np.abs(grain_delta))
    anomalous_ratio_name = grain_delta.columns[ind_anomalous].replace("d-", "")
    
    # Find indices of sigma and delta maps
    sigma_map_index = [
        plots_title.index(n) for n in plots_title if "Sigma" in n
    ]
    delta_map_index = [
        plots_title.index(n) for n in plots_title if "Delta" in n
    ]
    
    # Find indices of anomalous sigma and delta maps
    sigma_anomalous_map_index = [
        sigma_map_index.index(n)
        for n in sigma_map_index
        if anomalous_ratio_name in plots_title[n]
    ]
    delta_anomalous_map_index = [
        delta_map_index.index(n)
        for n in delta_map_index
        if anomalous_ratio_name in plots_title[n]
    ]
    
    # Extract sigma and delta maps
    sigma_map = [plots[n] for n in sigma_map_index]
    delta_map = [plots[n] for n in delta_map_index]
    
    return (
        sigma_map,
        delta_map,
        sigma_anomalous_map_index,
        delta_anomalous_map_index,
        anomalous_ratio_name,
    )


def extract_grain_features(
    PG_size_batch,
    PG_coor,
    raster,
    px,
    sig_r,
    sigma_map,
    delta_map,
    sigma_anomalous_map_index,
    delta_anomalous_map_index,
    grain_index,
):
    """
    Extract features from a single grain simulation.
    
    Parameters
    ----------
    PG_size_batch : numpy.ndarray
        1D array of grain sizes for the current batch
    PG_coor : numpy.ndarray
        Array of grain coordinates
    raster : float
        Raster size
    px : int
        Pixel size
    sig_r : float
        Sigma ratio threshold
    sigma_map : list
        List of sigma maps
    delta_map : list
        List of delta maps
    sigma_anomalous_map_index : list
        Index of the anomalous sigma map
    delta_anomalous_map_index : list
        Index of the anomalous delta map
    grain_index : int
        Index of the grain to extract features from
        
    Returns
    -------
    Diam : float
        Measured diameter in nm
    delta_values : numpy.ndarray
        Array of mean delta values for each ratio
    mask : numpy.ndarray
        Binary mask of the grain
    mask_th : numpy.ndarray
        Thresholded mask
    contour_initial : tuple
        Initial contour coordinates (xsel, ysel)
    contour_thresholded : tuple
        Thresholded contour coordinates (x, y)
    """
    # Calculate radius for mask creation
    radius = (
        (np.asarray(PG_size_batch[grain_index]) / 2)
        * 1e-3
        / (raster / px)
        * 1.5
    )
    
    # Create circular mask for the grain
    mask = create_circular_mask(
        px,
        px,
        center=np.floor_divide(PG_coor, 8)[grain_index],
        radius=radius,
    )
    
    # Find initial contour
    contour = skimage.measure.find_contours(mask != 0, 0.5)
    ysel, xsel = contour[0].T
    
    # Extract coordinates of masked pixels
    x, y = np.nonzero(mask)
    
    # Apply sigma threshold to find significant pixels
    I = np.where(
        sigma_map[sigma_anomalous_map_index[0]][x, y].data
        >= sigma_map[sigma_anomalous_map_index[0]][x, y].data.max() * sig_r
    )
    X = x[I]
    Y = y[I]
    
    # Create thresholded mask
    mask_th = np.zeros_like(mask)
    mask_th[X, Y] = 1
    
    # Calculate diameter from thresholded pixels
    Diam = (
        np.sqrt(
            len(delta_map[delta_anomalous_map_index[0]][X, Y])
            * ((raster / px) ** 2)
            / np.pi
        )
        * 1000
        * 2
    )
    
    # Extract mean delta values for each ratio
    delta_values = np.array(
        [np.mean(delta_map[ratio_idx][X, Y]) for ratio_idx in range(len(delta_map))]
    )
    
    # Find thresholded contour
    contour_th = skimage.measure.find_contours(mask_th == 1, 0.5)
    y_th, x_th = contour_th[0].T
    
    return (
        Diam,
        delta_values,
        mask,
        mask_th,
        (xsel, ysel),
        (x_th, y_th),
    )


def extract_features_for_all_grains(
    PG_size,
    PG_coor,
    raster,
    px,
    sig_r,
    sigma_map,
    delta_map,
    sigma_anomalous_map_index,
    delta_anomalous_map_index,
):
    """
    Extract features for all grains in a simulation batch.
    
    Parameters
    ----------
    PG_size : numpy.ndarray
        Array of grain sizes (shape: [batch_size, num_grains])
    PG_coor : numpy.ndarray
        Array of grain coordinates
    raster : float
        Raster size
    px : int
        Pixel size
    sig_r : float
        Sigma ratio threshold
    sigma_map : list
        List of sigma maps
    delta_map : list
        List of delta maps
    sigma_anomalous_map_index : list
        Index of the anomalous sigma map
    delta_anomalous_map_index : list
        Index of the anomalous delta map
        
    Returns
    -------
    results : list
        List of dictionaries, each containing features for one grain:
        - 'diameter': Measured diameter in nm
        - 'delta_values': Array of mean delta values
        - 'mask': Binary mask
        - 'mask_th': Thresholded mask
        - 'contour_initial': Initial contour coordinates
        - 'contour_thresholded': Thresholded contour coordinates
    """
    results = []
    
    for batch_idx in range(PG_size.shape[0]):
        for grain_idx in range(PG_size.shape[1]):
            features = extract_grain_features(
                PG_size[batch_idx, :],
                PG_coor,
                raster,
                px,
                sig_r,
                sigma_map,
                delta_map,
                sigma_anomalous_map_index,
                delta_anomalous_map_index,
                grain_idx,
            )
            
            results.append({
                'diameter': features[0],
                'delta_values': features[1],
                'mask': features[2],
                'mask_th': features[3],
                'contour_initial': features[4],
                'contour_thresholded': features[5],
            })
    
    return results

