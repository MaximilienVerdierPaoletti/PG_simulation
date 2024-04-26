# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:44:58 2023

@author: mverdier
"""

# import sys
# sys.path.insert(0, 'E:/Work/Programmation/Presolar grains/Python functions/')

import os
import numpy as np
import pandas as pd  # enables the use of dataframe
import matplotlib.pyplot as plt  # Enables plotting of data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import random
import skimage.filters
import skimage.io
import skimage.color
import skimage.filters
import skimage.measure
import sims
import cv2
# from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes  # to insert subplot within plot
from collections.abc import Iterable

# from numba import jit


# Astronomy Specific Imports
from astropy.convolution import convolve as ap_convolve
from astropy.convolution import Box2DKernel


#%% Mask creation

def create_circular_mask(h, w, center=None, radius=None):
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
    if center is None:  # use the middle of the image
        center = (int(w / 2), int(h / 2))
    if radius is None:  # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w - center[0], h - center[1])

    Y, X = np.ogrid[:h, :w]
    mask = np.empty((h, w),int)
    for i in range(0, len(center)):
        dist_from_center = np.sqrt((X - center[i][0]) ** 2 + (Y - center[i][1]) ** 2)
        if isinstance(radius, Iterable) == False:
            print('here')
            m = dist_from_center <= radius
            mask = mask + m * (i + 1) # Values attributed to PG have to start at 1 as 0 will be the non presolar material in the image
        else:
            m = dist_from_center <= radius[i]
            mask = mask + m * (i + 1)
    return mask


#%% Isotopic ratio extraction function

def Iso_Ratio(elem):
    if isinstance(elem, list) == False:
        elem = [elem]

    ratio_list = pd.read_excel('Iso_Ratio_Table.xlsx', header=0, sheet_name='isotope ratios')

    R = [ratio_list[ratio_list['var_name'].str.contains(i)]['ratio'] for i in elem]

    return R


#%% Simulation v1

def PG_simulation(elem=None, Nb_PG=None, PG_delta=None, PG_size=None, beam_size=None,
                  raster=None, px=None, frames=None, countrates=None, dwell_time=None, boxcar_px=None,
                  display='OFF'):
    if display == 'OFF':
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem == None:
        R = Iso_Ratio('17O')[0]
        print('No isotope specified. Isotope selected : 17 oxygen')
    else:
        R = Iso_Ratio(elem)
        if len(R) == 1:
            R = R[0].item()

    # ---------------- Grain info
    if PG_delta == None:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print('No composition specified. Grain composition fixed to ' + str(PG_delta) + ' permil')
    else:
        PG_delta = [PG_delta]
    if PG_size == None:
        PG_size = np.random.randint(100, 600, Nb_PG)
        print('No size specified. Grain diameter fixed to ' + str(PG_size) + ' nm')
    else:
        PG_size = [PG_size]

    # ------------------- Acquisition variables

    if beam_size == None:
        beam_size = 100
        print('No beam size specified. Fixed to ' + str(beam_size) + ' nm')
    if countrates == None:
        countrates = 380000
        print('No countrates specified. Fixed to ' + str(countrates) + ' cps/s')
    if dwell_time == None:
        dwell_time = 3
        print('No dwell_time specified. Fixed to ' + str(dwell_time) + ' ms/px')
    if frames == None:
        frames = 20
        print('No number of frames specified. Fixed to ' + str(frames))
    if raster == None:
        raster = 15
        print('No raster size specified. Fixed to ' + str(raster) + ' microns')
    if px == None:
        px = 256
        print('No pixel size specified. Fixed to ' + str(px) + 'x' + str(px))
    if boxcar_px == None:
        boxcar_px = 3
        print('No boxcar size specified. Fixed to ' + str(boxcar_px) + 'x' + str(boxcar_px))

    # T=(dwell_time*1E-3*frames*px**2) #total counting time in sec
    T_px = (dwell_time * 1E-3 * frames)  # total counting per pixel time in sec

    # -------------------------------------------------- SIMULATION

    # --------------- Generation of a higher resolution simulated image
    # Real material will not be pixelated and blurred by the beam, so we first need to create a high resolution image to depict "reality"
    hr_coeff = 8
    T_pxhr = T_px / hr_coeff

    # Test for varying countrates for each pixel
    # im_arr=[]
    # im_arr_delta=[]
    # for i in range(0,(px*hr_coeff)**2):
    #     C=np.random.randint(50000,380000,1)
    #     px_value=np.random.poisson(C*T_pxhr*R,1)
    #     px_value_delta=(px_value/(C*T_pxhr*R)-1)*1000
    #     im_arr.append(px_value)
    #     im_arr_delta.append(px_value_delta)

    # im_hr=np.array(im_arr).reshape(px*hr_coeff,px*hr_coeff)
    # imhr_delta=np.array(im_arr_delta).reshape(px*hr_coeff,px*hr_coeff)

    imhr = np.random.poisson(countrates * T_pxhr * R, (px * hr_coeff, px * hr_coeff))
    imhr_delta = (imhr / (countrates * T_pxhr * R) - 1) * 1000

    # --------------- Generation of simulated image
    # It generates for each pixel a cumulated count of measured (17 or 18)O ions based on the 16O countrates, dwell time, number of frames and terrestrial ratio
    # im=np.random.poisson(countrates*T_px*R,(px,px)) # in cumulated counts
    # im_delta= (im/(countrates*T_px*R)-1)*1000 # Conversion of the simulated image to delta values

    # --------------- Beam blurr
    # sig_gaussian=np.round(beam_size/(raster/px*1000)) #sigma parameter (width of the gaussian) based on the beam size
    # sig_gaussian=beam_size/(np.sqrt(8*np.log(2))) or beam_size/2.35
    sig_gaussian = np.round((beam_size * 1E-3) / (raster / (px * hr_coeff))) / (np.sqrt(8 * np.log(2)))

    # ax2.imshow(scipy.ndimage.gaussian_filter(im, sigma=sig_gaussian),cmap="plasma")
    # im_gauss=skimage.filters.gaussian(imhr_delta, sigma=sig_gaussian,truncate=1,preserve_range=True,mode='constant') #convolution of the gaussian filter with a kernel of 2*sigma*truncate pixels
    # FOR MORE INFO : https://datacarpentry.org/image-processing/06-blurring/

    # Downscaling the High Resolution image while applying a gaussian filter to simulate the beam.
    im_gaussconv = skimage.transform.rescale(imhr_delta, scale=1 / hr_coeff, preserve_range=True, anti_aliasing=True,
                                             anti_aliasing_sigma=sig_gaussian)

    # --------------- Boxcar Blurr
    box_kernel = Box2DKernel(boxcar_px)
    im_boxcar = ap_convolve(im_gaussconv, box_kernel, normalize_kernel=True)

    ## PG addition

    # PG coordinates
    PG_coor = [random.sample(range(px * hr_coeff), 2) for i in range(Nb_PG)]

    hr_px_size = raster / (px * hr_coeff)
    imhr_delta_PG = imhr_delta.copy()

    for i in range(0, Nb_PG):
        m = create_circular_mask(px * hr_coeff, px * hr_coeff, center=PG_coor[i],
                                 radius=(PG_size[i] / 2) * 1E-3 / hr_px_size)
        imhr_delta_PG[m == True] = PG_delta[i]

    imgauss_PG = skimage.transform.rescale(imhr_delta_PG, scale=1 / hr_coeff, preserve_range=True, anti_aliasing=True,
                                           anti_aliasing_sigma=sig_gaussian)
    sig_gauss = np.abs((imgauss_PG - np.average(imgauss_PG)) / np.std(imgauss_PG))

    box_kernel = Box2DKernel(boxcar_px)
    imboxcar_PG = ap_convolve(imgauss_PG, box_kernel, normalize_kernel=True)
    sig_boxcar = np.abs((imboxcar_PG - np.average(imboxcar_PG)) / np.std(imboxcar_PG))

    ## Plots

    plots = [imhr_delta, im_gaussconv, im_boxcar, imhr_delta_PG, imgauss_PG, imboxcar_PG, sig_gauss, sig_boxcar]
    plots_title = ['Delta HR no PG', 'Delta Gaussian no PG', 'Delta Gaussian + Boxcar no PG', 'Delta HR', 'Gaussian',
                   'Gaussian + Boxcar', 'Sigma Gaussian', 'Sigma Boxcar']
    # titles=["Raw Simulation",str(beam_size)+" nm Beam Blurr Simulation",str(boxcar_px)+" x "+str(boxcar_px)+" Boxcar and Gaussian Blurr Simulation"]

    fig, axs = plt.subplots(3, 3, figsize=(12, 12))

    h = 0
    for j in range(3):
        for i in range(3):
            if j == 2 and i == 0:
                # axs[i,j].set_axis_off()
                axs[i, j].set_xticks([])
                axs[i, j].set_yticks([])
                # axs[i,j].text(0.5,0.65,"Presolar Grain Simulation",horizontalalignment='center',verticalalignment='center')
                axs[i, j].text(0.5, 0.55, "Grain size : " + str(PG_size) + ' nm', horizontalalignment='center',
                               verticalalignment='center')
                axs[i, j].text(0.5, 0.45, "Grain delta : " + str(PG_delta) + ' permil', horizontalalignment='center',
                               verticalalignment='center')
                axs[i, j].text(0.5, 0.35, "Raster : " + str(raster) + ' microns', horizontalalignment='center',
                               verticalalignment='center')
                axs[i, j].text(0.5, 0.25, "Boxcar : " + str(boxcar_px) + ' px', horizontalalignment='center',
                               verticalalignment='center')
            else:
                img_plot = axs[i, j].imshow(plots[h], cmap="plasma", interpolation='None', rasterized=True)
                divider = make_axes_locatable(axs[i, j])
                cax = divider.append_axes("right", size="5%", pad=0.05)
                plt.colorbar(img_plot, cax=cax)
                # axs[i,j].title.set_text(titles[h])
                axs[i, j].set_axis_off()
                plt.title(plots_title[h])
                h = h + 1

    # print('\n')
    # print('Plot list is :')
    # print('0: High-resolution image without grain')
    # print('1: Beam blurred image without grain')
    # print('2: Beam and boxcar blurred image without grain')
    # print('3: High-resolution image with grain')
    # print('4: Beam blurred image with grain')
    # print('5: Beam and boxcar blurred image with grain')
    # print('6: Beam blurred sigma unit image without grain')
    # print('7: Beam and boxcar blurred sigma unit image without grain')
    PG_coord = [int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff)]

    return fig, axs, plots, PG_coord


#%% Simulation v2 : Summed 16O counts
# @jit(fastmath=True)
def PG_simulationv2(file=None, elem=None, Nb_PG=None, PG_delta=None, PG_size=None, beam_size=None, boxcar_px=None,
                    OG_grain=None, standard=None, smart=None, verif=None,
                    display='OFF'):
    ## Check inputs

    np.seterr(divide='ignore')

    if display == 'OFF':
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem == None:
        elem = 'O'
        print('No element specified. Element selected : Oxygen')

    match elem:
        case 'O':
            Iso = ['16O', '17O', '18O']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case 'Mg':
            Iso = ['24Mg', '25Mg', '26Mg']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case _:
            print('The specified element is inadequat. Check if you did not specify an isotope instead of an element.')

    # ---------------- Grain info
    if PG_delta == None:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print('No composition specified. Grain composition fixed to ' + str(PG_delta) + ' permil')
    else:
        if type(PG_delta) != list:
            PG_delta = [PG_delta]

    if PG_size == None:
        PG_size = np.random.randint(100, 600, Nb_PG)
        print('No size specified. Grain diameter fixed to ' + str(PG_size) + ' nm')
    else:
        if type(PG_size) != list:
            PG_size = [PG_size]

    # ------------------- Acquisition variables

    if beam_size == None:
        beam_size = 100
        print('No beam size specified. Fixed to ' + str(beam_size) + ' nm')
    if boxcar_px == None:
        boxcar_px = 3
        print('No boxcar size specified. Fixed to ' + str(boxcar_px) + 'x' + str(boxcar_px))
    if smart == None:
        smart = 1
        print('Smart error activated')
    if standard == None:
        print(
            'Image will be relative to standard terrestrial values. Set "standard" to "average" for delta values relative to the average value of the region.')
    # if countrates==None:
    #     countrates=380000
    #     print('No countrates specified. Fixed to '+str(countrates)+' cps/s')
    # if dwell_time==None:
    #     dwell_time=3
    #     print('No dwell_time specified. Fixed to '+str(dwell_time)+' ms/px')
    # if frames==None:
    #     frames=20
    #     print('No number of frames specified. Fixed to '+str(frames))
    # if raster==None:
    #     raster=15
    #     print('No raster size specified. Fixed to '+str(raster)+' microns')
    # if px==None:
    #     px=256
    #     print('No pixel size specified. Fixed to '+str(px)+'x'+str(px))

    # --------------- Generation of a higher resolution simulated image
    # Real material will not be pixelated and blurred by the beam, so we first need to create a high resolution image to depict "reality"
    # T=(dwell_time*1E-3*frames*px**2) #total counting time in sec
    # T_px=(dwell_time*1E-3)  #total counting per pixel time in sec
    hr_coeff = 8
    # T_pxhr=T_px/hr_coeff
    # hr_px_size=raster/(px*hr_coeff)
    N_iso = len(R)

    ## Extraction of real image

    if ('file' not in globals()) or (
            file == ''):  # If the file path is not specified then, select randomly an image from the following folder.
        path_realim = 'E:/Work/Programmation/Presolar grains/Simulations PG/Real Images/'
        file = random.choice(os.listdir(path_realim))
        file = path_realim + file
    s = sims.SIMS(file)
    raster = s.header['Image']['raster'] / 1000
    realcts = np.zeros((s.header['Image']['width'], s.header['Image']['height'], N_iso))
    realcts[:, :, 0] = s.data.loc[Iso[0]].sum(axis=0).values  # extract maps of the main isotope and sum all frames
    realcts[:, :, 1] = s.data.loc[Iso[1]].sum(axis=0).values
    realcts[:, :, 2] = s.data.loc[Iso[2]].sum(axis=0).values
    realcts = realcts / 1

    # If we work on each frame of the data
    p = s.header['Image']['planes']
    raster = s.header['Image']['raster'] / 1000
    px = s.header['Image']['width']

    # print('')
    # print('Image is : '+str(file))
    # print(str(raster)+'x'+str(raster)+' µm, '+str(px)+'x'+str(px)+' pixels, '+str(p)+' frames')

    # realcts=np.zeros((s.header['Image']['width'],s.header['Image']['height'],len(R)*p))

    ## Verification

    realcts_verif = np.zeros((s.header['Image']['width'], s.header['Image']['height'], len(R)))
    realcts_verif[:, :, 0] = s.data.loc[Iso[0]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif[:, :, 1] = s.data.loc[Iso[1]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif[:, :, 2] = s.data.loc[Iso[2]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif = realcts_verif / 1

    box_kernel = Box2DKernel(boxcar_px)
    imboxcar_PG = np.zeros_like(realcts_verif)
    for i in range(0, 3):
        imboxcar_PG[:, :, i] = ap_convolve(realcts_verif[:, :, i], box_kernel, boundary='fill', fill_value=0.0)

    # ----- Threshold
    th = 0.05
    cts_th = int(imboxcar_PG[:, :, 0].max() * th)
    mask = np.zeros((px, px))
    mask[imboxcar_PG[:, :, 0] < cts_th] = 1

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)

    masked_image = np.ma.dstack((main, minor1, minor2))

    R_1st = minor1 / main
    D = minor1

    err_R1 = R_1st * np.sqrt(1 / minor1 + 1 / main) / boxcar_px
    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    err_d1 = err_R1 / R[1] * 1000

    R_2nd = minor2 / main
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000

    # Rsig=[R[1]]
    Rsig = [np.mean(R_1st)]
    Dmod = np.where(D < main * Rsig[0], main * Rsig[0], D)

    # imboxcar_sig1st=np.abs(R_1st-Rsig[0])/err_R1
    imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod) * 3
    # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*3
    # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*np.sqrt(3)

    if verif == 1:
        d1 = ((realcts_verif[:, :, 1] / realcts_verif[:, :, 0]) / R[1] - 1) * 1000
        f, [[ax, axbox], [axerr, axsig]] = plt.subplots(2, 2, sharex=True, sharey=True)
        im0 = ax.imshow(d1, cmap='gnuplot2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title('delta no boxcar')
        f.colorbar(im0, cax=cax)

        im1 = axbox.imshow(d_1st, cmap='gnuplot2')
        divider = make_axes_locatable(axbox)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axbox.set_title('delta boxcar')
        f.colorbar(im1, cax=cax)
        # plt.plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=20)

        # plt.figure()
        im2 = axsig.imshow(imboxcar_sig1st, cmap='gnuplot2')
        divider = make_axes_locatable(axsig)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axsig.set_title('sigma')
        f.colorbar(im2, cax=cax)
        # plt.colorbar()
        # plt.plot(PG_coor[0]/256,'.r',markersize=15)

        im3 = axerr.imshow(err_d1, cmap='gnuplot2')
        divider = make_axes_locatable(axerr)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axerr.set_title('delta error')
        f.colorbar(im3, cax=cax)

        plt.suptitle(file, fontsize=15)

    ## Simulated images

    # dwell_real=s.header['BFields'][0]['time per pixel']
    th = 0.05

    ####---- Extraction of countrates from Low Resolution (LR) to High Resolution image
    D = np.copy(realcts)  # We copy the image to make sure to not alter the extraction
    extracted_cts = cv2.resize(D, dsize=(px * hr_coeff, px * hr_coeff),
                               interpolation=cv2.INTER_AREA)  # Interpolate into larger dimensions (here from 256x256 px to 2064x2064 px)

    # Original PG coordinates
    if 'OG_grain' in globals():
        OG_PG_center = [OG_grain.ROIX.item(), OG_grain.ROIY.item()]
        m = create_circular_mask(px * hr_coeff, px * hr_coeff, center=OG_PG_center,
                                 radius=int(OG_grain.ROIDIAM / (2 * raster) * px * hr_coeff))
        ind_OG_PG = np.argwhere(m == True)
        del m
    else:
        ind_OG_PG = []

    ####---- PG coordinates
    PG_coor = [random.sample(range(px * hr_coeff), 2) for i in
               range(Nb_PG)]  # Get coordinates for the required number of grains
    it = np.ravel_multi_index(np.asarray(PG_coor).T,
                              extracted_cts[:, :, 0].shape)  # 1D index of the coordinates in the image
    coor_verif = extracted_cts[:, :, 0].take(it)  # Extracting the corresponding 16O counts of the coordinates
    while (any(coor_verif < extracted_cts[:, :, 0].max() * th) == True) or (
            coor_verif in ind_OG_PG):  # If any 16O counts select as the center of a grain is below the masking threshold
        ind_badcoor = np.where(
            coor_verif < extracted_cts[:, :, 0].max() * th)  # Location of the problematic coordinates
        for i in ind_badcoor[0]:  # Loop over the problematic coordinates
            PG_coor[i] = random.sample(range(px * hr_coeff), 2)  # Replacement of the problematic coordinates
        it = np.ravel_multi_index(np.asarray(PG_coor).T, extracted_cts[:, :, 0].shape)  # Update of the 1D index
        coor_verif = extracted_cts[:, :, 0].take(it)  # Update of the 16O counts
    radius = list((np.asarray(PG_size) / 2) * 1E-3 / (
                raster / (px * hr_coeff)))  # Radius calculation of the grains in the HR dimensions
    mask_PG = create_circular_mask_multiple(px * hr_coeff, px * hr_coeff, center=PG_coor,
                                            radius=radius)  # Mask creation of the grains' pixels

    imhr_ini = np.copy(extracted_cts)  # Copying the HR images channels to avoid altering them

    ####---- Modifying maps counts on location of presolar grains
    for j in range(1, N_iso):  # Loop on isotopes
        for i in range(1, Nb_PG + 1):  # Loop on PG
            imhr_ini[mask_PG == i, j] = extracted_cts[mask_PG == i, 0] * R[j] * (PG_delta[0] / 1E3 + 1)

    imhr_ini_PG = np.copy(imhr_ini)  # Copying the modified images

    ####---- Beam blurr and Boxcar definitions

    # Defining the sigma parameters of the gaussian blurr
    # sig_gaussian=beam_size/(np.sqrt(8*np.log(2))) or beam_size/2.35
    fwhm_hr = np.round((beam_size * 1E-3) / (raster / (
                px * hr_coeff))) / 2  # the gaussian filter uses the given sigma as a radius for kernel size if radius is not specified
    # fwhm_hr=np.round((beam_size*1E-3)/(raster/(px)))
    sig_gaussian = fwhm_hr / (np.sqrt(8 * np.log(2)))

    box_kernel = Box2DKernel(boxcar_px)  # Boxcar Kernel

    imgauss_PG = np.zeros((px, px, N_iso))  # Allocation of memory for the beam blurred image
    imboxcar_PG = np.zeros((px, px, N_iso))  # Allocation of memory for the boxcar smoothed image

    ####---- Beam blurr and Boxcar smoothing

    # Two options :
    # 1. A new image is created for each isotopes from the original ones. Each pixel value is used as a mean for a poisson distribution of which a new pixel value is interpolated. Then the images are beam blurred and boxcar smoothed.
    # 2. We started by applying the gaussian blurr before interpolation new values from a poisson distribution for each pixel of each isotope image. Then the image is boxcar smoothed.

    # Allocation of memory for the new Poisson interpolated images
    # #Option 1
    im_poiss = np.zeros_like(imhr_ini_PG)
    # Option 2
    # im_poiss=np.zeros_like(imgauss_PG)

    # Simulation including the presolar grains
    for i in range(0, N_iso):
        # # ----- Option 1 : First poisson then gaussian blurr
        im_poiss[:, :, i] = np.random.poisson(imhr_ini_PG[:, :, i])  # Poisson distribution interpolation
        imgauss_PG[:, :, i] = skimage.transform.rescale(im_poiss[:, :, i], order=0, scale=1 / hr_coeff,
                                                        preserve_range=True, anti_aliasing=True,
                                                        anti_aliasing_sigma=sig_gaussian)  # Gaussian blurring to simulate the beam while downscaling from 2064x2064 to 256x256 px. The "Anti_Aliasing" option applies a gaussian blurr which kernel radius will be sigma*truncate+1.
        imboxcar_PG[:, :, i] = ap_convolve(imgauss_PG[:, :, i], box_kernel, boundary='fill',
                                           fill_value=0.0)  # Boxcar smoothing

        # # ----- Option 2 : First gaussian blurr then poisson

        # imgauss_PG[:,:,i]=skimage.transform.rescale(imhr_ini_PG[:,:,i],order=0,scale=1/hr_coeff,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)
        # im_poiss[:,:,i]=np.random.poisson(imgauss_PG[:,:,i])
        # imboxcar_PG[:,:,i] = ap_convolve(imgauss_PG[:,:,i], box_kernel, boundary='fill', fill_value=0.0) # Create the boxcar image for this isotopes

    ####---- Masking low counts regions

    cts_th = int(imboxcar_PG[:, :,
                 0].max() * th)  # Define criterion as 5% of the max encountered for the main isotope counts in one pixel
    mask = np.zeros((px, px))  # Allocate memory for the mask with a 0 matrix to the image dimensions
    mask[imboxcar_PG[:, :,
         0] < cts_th] = 1  # Set all coordinates of pixels with lower counts than the criterion to 1 in 0 matrix mask.

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)  # Extract main isotope image
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)  # Extract first minor isotope image
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)  # Extract second minor isotope image

    # Stack masked images together again
    masked_image = np.ma.dstack((main, minor1, minor2))  # Stack the masked images together

    ####---- Ratio, delta and error calculations

    R_1st = minor1 / main
    R_2nd = minor2 / main

    # err_R1=R_1st*np.sqrt(1/minor1+1/main)/boxcar_px
    # err_R2=R_2nd*np.sqrt(1/minor2+1/main)/boxcar_px
    D = minor1
    Dmod = np.where(D < main * R[1], main * R[1], D)
    err_R1 = R_1st * np.sqrt(1 / Dmod + 1 / main) / boxcar_px
    err_R2 = R_2nd * np.sqrt(1 / minor2 + 1 / main) / boxcar_px

    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    # err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
    err_d1 = err_R1 / R[1] * 1000
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000
    err_d2 = err_R2 / R[2] * 1000

    ####---- Sigma images

    # Sigma images will be defined either relative to the standard value or the average ratio of the image
    if standard == "average":
        Rsig = [np.mean(R_1st), np.mean(R_2nd)]
    else:
        Rsig = R[1:]
    # Rsig=

    # Smart error ON
    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(main*Rsig[0])*boxcar_px
    # # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(minor2-main*Rsig[1])/np.sqrt(main*Rsig[1])*boxcar_px #smart error
    # # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    if smart == 1:
        Dmod = np.zeros((main.shape[0], main.shape[1], N_iso - 1))
        Dmod[:, :, 0] = minor1
        Dmod[:, :, 1] = minor2

        for g in range(0, N_iso - 1): Dmod[:, :, g] = np.where(masked_image[:, :, g + 1] < main * Rsig[g],
                                                               main * Rsig[g], minor1)

        imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod[:, :, 0]) * 3
        imboxcar_sig2nd = np.abs(minor2 - main * Rsig[1]) / np.sqrt(Dmod[:, :, 1]) * 3
    else:
        imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
        imboxcar_sig2nd = np.abs(R_2nd - Rsig[1]) / err_R2

    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(d_2nd/err_d2)#smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    # plt.subplot()
    # plt.imshow(realcts,cmap='gnuplot2')
    # plt.colorbar()

    if verif == 1:
        # f_verif,[axpoiss,axgauss,axbox]=plt.subplots(1,3,sharex=True,sharey=True)
        f_verif, [axpoiss, axgauss, axbox] = plt.subplots(1, 3)
        axpoiss.imshow(im_poiss[:, :, 1], cmap='gnuplot2')
        axpoiss.set_title('Poisson HR')
        axpoiss.plot(PG_coor[0][0], PG_coor[0][1], 'o', mfc='none', mec='r', markersize=15)
        axgauss.imshow(imgauss_PG[:, :, 1], cmap='gnuplot2')
        axgauss.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                     markersize=15)
        axgauss.set_title('Gaussian blurr')
        axbox.imshow(imboxcar_PG[:, :, 1], cmap='gnuplot2')
        axbox.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                   markersize=15)
        axbox.set_title('Gaussian + Boxcar')
        plt.suptitle('17O counts', fontsize=15)

    ## Plots

    plots = [imgauss_PG[:, :, 0], imgauss_PG[:, :, 1], imgauss_PG[:, :, 2],
             imboxcar_PG[:, :, 0], imboxcar_PG[:, :, 1], imboxcar_PG[:, :, 2],
             R_1st, d_1st, imboxcar_sig1st,
             R_2nd, d_2nd, imboxcar_sig2nd]
    plots_title = [str(Iso[0]) + ' counts', str(Iso[1]) + ' counts', str(Iso[2]) + ' counts',
                   str(Iso[0]) + ' counts boxcar', str(Iso[1]) + ' counts boxcar', str(Iso[2]) + ' counts boxcar',
                   'Ratio ' + str(Iso[1]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Ratio ' + str(Iso[2]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[2]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[2]) + '/' + str(Iso[0])]
    # titles=["Raw Simulation",str(beam_size)+" nm Beam Blurr Simulation",str(boxcar_px)+" x "+str(boxcar_px)+" Boxcar and Gaussian Blurr Simulation"]

    fig, axs = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    axinsert = []
    palette = 'gnuplot2'

    fontprops = fm.FontProperties(size=12)

    for i in range(0, len(plots)):
        axins = inset_axes(axs[i], width="60%", height="60%", loc="upper left", bbox_to_anchor=(-0.5, 0.2, 1, 1),
                           bbox_transform=axs[i].transAxes, borderpad=1)
        img_plot = axs[i].imshow(plots[i], cmap=palette, interpolation='None', rasterized=True, )
        # axs[i].plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=15)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img_plot, cax=cax)
        # axs[i,j].title.set_text(titles[h])
        axs[i].set_axis_off()

        plt.title(plots_title[i])

        axins.imshow(plots[i], cmap=palette, interpolation='None', rasterized=True, )
        axins.set_xlim(int(PG_coor[0][0] / hr_coeff) - 15, int(PG_coor[0][0] / hr_coeff) + 15)
        axins.set_ylim(int(PG_coor[0][1] / hr_coeff) - 15, int(PG_coor[0][1] / hr_coeff) + 15)
        axins.get_xaxis().set_visible(False)
        axins.get_yaxis().set_visible(False)
        axinsert.append(axins)

        scalebar = AnchoredSizeBar(axs[i].transData,
                                   int(px / raster * 2), r'2 $\mu m$', 'lower center',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        axs[i].add_artist(scalebar)

    return fig, axs, axinsert, plots, plots_title, PG_coor, raster, px


#%% Simulation v3
# @jit(fastmath=True)
def PG_simulationv3(elem=None, Nb_PG=None, PG_delta=None, PG_size=None, beam_size=None,
                    raster=None, px=None, frames=None, countrates=None, dwell_time=None, boxcar_px=None,
                    display='OFF'):
    ## Check inputs
    if display == 'OFF':
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem == None:
        elem = 'O'
        print('No element specified. Element selected : Oxygen')

    match elem:
        case 'O':
            Iso = ['16O', '17O', '18O']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case 'Mg':
            Iso = ['24Mg', '25Mg', '26Mg']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case _:
            print('The specified element is inadequat. Check if you did not specify an isotope instead of an element.')

    # ---------------- Grain info
    if PG_delta == None:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print('No composition specified. Grain composition fixed to ' + str(PG_delta) + ' permil')
    else:
        PG_delta = [PG_delta]
    if PG_size == None:
        PG_size = np.random.randint(100, 600, Nb_PG)
        print('No size specified. Grain diameter fixed to ' + str(PG_size) + ' nm')
    else:
        PG_size = [PG_size]

    # ------------------- Acquisition variables

    if beam_size == None:
        beam_size = 100
        print('No beam size specified. Fixed to ' + str(beam_size) + ' nm')
    if countrates == None:
        countrates = 380000
        print('No countrates specified. Fixed to ' + str(countrates) + ' cps/s')
    if dwell_time == None:
        dwell_time = 3
        print('No dwell_time specified. Fixed to ' + str(dwell_time) + ' ms/px')
    if frames == None:
        frames = 20
        print('No number of frames specified. Fixed to ' + str(frames))
    if raster == None:
        raster = 15
        print('No raster size specified. Fixed to ' + str(raster) + ' microns')
    if px == None:
        px = 256
        print('No pixel size specified. Fixed to ' + str(px) + 'x' + str(px))
    if boxcar_px == None:
        boxcar_px = 3
        print('No boxcar size specified. Fixed to ' + str(boxcar_px) + 'x' + str(boxcar_px))

    ## Extraction of real image

    path_realim = 'E:/Work/Programmation/Presolar grains/Simulations PG/Real Images/'
    file = random.choice(os.listdir(path_realim))
    s = sims.SIMS(path_realim + file)

    print('Selected initial image is : ' + str(file))

    # file='E:/Work/NanoSIMS Data/Presolar grains/Presolar silicates/Paris/Measurements with PGs/R1C1_3_corr.im'
    # s=sims.SIMS(file)

    # realcts=np.zeros((s.header['Image']['width'],s.header['Image']['height'],len(R)))
    # realcts[:,:,0]=s.data.loc[Iso[0]].sum(axis=0).values #extract maps of the main isotope and sum all frames
    # realcts[:,:,1]=s.data.loc[Iso[1]].sum(axis=0).values #extract maps of the main isotope and sum all frames
    # realcts[:,:,2]=s.data.loc[Iso[2]].sum(axis=0).values #extract maps of the main isotope and sum all frames
    # realcts=realcts/1

    # If we work on each frame of the data
    p = s.header['Image']['planes']
    raster = s.header['Image']['raster'] / 1000
    realcts = np.zeros((s.header['Image']['width'], s.header['Image']['height'], len(R) * p))
    realcts[:, :, 0:p] = np.swapaxes(s.data.loc[Iso[0]].values, 0, -1).swapaxes(1,
                                                                                0)  # extract maps of the main isotope and sum all frames
    realcts[:, :, p:p * 2] = np.swapaxes(s.data.loc[Iso[1]].values, 0, -1).swapaxes(1,
                                                                                    0)  # extract maps of the main isotope and sum all frames
    realcts[:, :, 2 * p:3 * p] = np.swapaxes(s.data.loc[Iso[2]].values, 0, -1).swapaxes(1,
                                                                                        0)  # extract maps of the main isotope and sum all frames
    realcts = realcts / 1

    frames = p

    ## Verification

    realcts_verif = np.zeros((s.header['Image']['width'], s.header['Image']['height'], len(R)))
    realcts_verif[:, :, 0] = s.data.loc[Iso[0]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif[:, :, 1] = s.data.loc[Iso[1]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif[:, :, 2] = s.data.loc[Iso[2]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif = realcts_verif / 1

    # ---------- Gaussian effect verification
    # hr_coeff=8
    # # Use the image directly and try to convert it to HR to use it (DOESN'T WORK BECAUSE COUNTRATE END UP NOT BEING INTEGERS FOR MINOR ISOTOPES)
    #  dwell_real=s.header['BFields'][0]['time per pixel']
    #  # D=np.copy(realcts_verif[:,:,0])
    #  D=np.copy(realcts_verif)
    #  extracted_cts=cv2.resize(D, dsize=(px*hr_coeff, px*hr_coeff),interpolation=cv2.INTER_AREA)

    #  #--------------- Beam blurr
    #  sig_gaussian=np.round((beam_size*1E-3)/(raster/(px*hr_coeff)))/(np.sqrt(8*np.log(2)))

    #  # Downscaling the High Resolution image while applying a gaussian filter to simulate the beam.
    #  imgauss_verif=np.zeros((px,px,3))
    #  for i in range(0,3):
    #      imgauss_verif[:,:,i]=skimage.transform.rescale(extracted_cts[:,:,i],scale=1/hr_coeff,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)

    # for i in range(0,3):
    #     im_poiss=np.random.poisson(realcts_verif[:,:,i])
    #     im

    # plt.subplot(1,3,1)
    # plt.imshow(realcts_verif[:,:,0],cmap='gnuplot2')
    # plt.title('Original')
    # plt.subplot(1,3,2)
    # plt.imshow(extracted_cts,cmap='gnuplot2')
    # plt.title('High resolution')
    # plt.subplot(1,3,3)
    # plt.imshow(imgauss_verif,cmap='gnuplot2')
    # plt.title('HR gaussian blur')
    # plt.suptitle('Gaussian blurr effect verification', fontsize=15)

    # from scipy import signal

    box_kernel = Box2DKernel(boxcar_px)
    # box_kernel = np.ones((boxcar_px,boxcar_px))
    # box_kernel_hr = np.ones((boxcar_px*hr_coeff+1,boxcar_px*hr_coeff+1))
    imboxcar_PG = np.zeros_like(realcts_verif)
    # imboxcar_PG=np.zeros_like(extracted_cts)
    for i in range(0, 3):
        # imboxcar_PG[:,:,i] = ap_convolve(extracted_cts[:,:,i], box_kernel_hr, boundary='fill', fill_value=0.0)
        imboxcar_PG[:, :, i] = ap_convolve(realcts_verif[:, :, i], box_kernel, boundary='fill', fill_value=0.0)
        # imboxcar_PG[:,:,i] = ap_convolve(imgauss_verif[:,:,i], box_kernel, boundary='fill', fill_value=0.0)
        # imboxcar_PG[:,:,i] = signal.convolve2d(realcts_verif[:,:,i], box_kernel, boundary='fill', mode='same')

    # ----- Threshold
    th = 0.05
    cts_th = int(imboxcar_PG[:, :, 0].max() * th)
    mask = np.zeros((px, px))
    # mask=np.zeros((px*hr_coeff,px*hr_coeff))
    mask[imboxcar_PG[:, :, 0] < cts_th] = 1

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)

    masked_image = np.ma.dstack((main, minor1, minor2))

    R_1st = minor1 / main
    D_smart = minor1
    # D_smart2=np.where(D_smart<10,D_smart,main*R[1])
    # err_R1=R_1st*np.sqrt(1/D_smart+1/main)/boxcar_px
    err_R1 = R_1st * np.sqrt(1 / D_smart + 1 / main) / boxcar_px
    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    err_d1 = err_R1 / R[1] * 1000

    R_2nd = minor2 / main
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000

    # err_R1=R_1st*np.sqrt(1/minor1+1/main)/np.sqrt(boxcar_px)
    # err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))

    # d_2nd=((minor2/main)/R[2]-1)*1000

    # # Delta sigma maps
    # sig_d1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # sig_d2nd=np.abs((d_2nd-np.average(d_2nd))/np.std(d_2nd))

    # imboxcar_d1st=ap_convolve(d_1st, box_kernel, normalize_kernel=True)
    # imboxcar_d2nd=ap_convolve(d_2nd, box_kernel, normalize_kernel=True)

    # ind=np.argwhere(imboxcar_PG[:,:,1]!=0)
    # imboxcar_sig1st=np.abs(imboxcar_PG[ind.T[0],ind.T[1],1]-imboxcar_PG[ind.T[0],ind.T[1],0]*R[1])/np.sqrt(imboxcar_PG[ind.T[0],ind.T[1],1])#smart error
    # ind=np.argwhere(imboxcar_PG[:,:,2]!=0)
    # imboxcar_sig2nd=np.abs(imboxcar_PG[ind.T[0],ind.T[1],2]-imboxcar_PG[ind.T[0],ind.T[1],0]*R[2])/np.sqrt(imboxcar_PG[ind.T[0],ind.T[1],2])

    # Rsig=[R[1]]
    Rsig = [np.mean(R_1st)]

    imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(minor1)

    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(minor1)*boxcar
    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(imboxcar_PG[:,:,2]-imboxcar_PG[:,:,0]*R[2])/np.sqrt(imboxcar_PG[:,:,0]*R[2]) #smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    d1 = ((realcts_verif[:, :, 1] / realcts_verif[:, :, 0]) / R[1] - 1) * 1000
    f, [[ax, axbox], [axerr, axsig]] = plt.subplots(2, 2, sharex=True, sharey=True)
    im0 = ax.imshow(d1, cmap='gnuplot2')
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    ax.set_title('delta no boxcar')
    f.colorbar(im0, cax=cax)

    im1 = axbox.imshow(d_1st, cmap='gnuplot2')
    divider = make_axes_locatable(axbox)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axbox.set_title('delta boxcar')
    f.colorbar(im1, cax=cax)
    # plt.plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=20)

    # plt.figure()
    im2 = axsig.imshow(imboxcar_sig1st, cmap='gnuplot2')
    divider = make_axes_locatable(axsig)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axsig.set_title('sigma')
    f.colorbar(im2, cax=cax)
    # plt.colorbar()
    # plt.plot(PG_coor[0]/256,'.r',markersize=15)

    im3 = axerr.imshow(err_d1, cmap='gnuplot2')
    divider = make_axes_locatable(axerr)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    axerr.set_title('delta error')
    f.colorbar(im3, cax=cax)

    plt.suptitle(file, fontsize=15)

    ## Simulated image

    # -------------------------------------------------- SIMULATION

    # --------------- Generation of a higher resolution simulated image
    # Real material will not be pixelated and blurred by the beam, so we first need to create a high resolution image to depict "reality"
    # T=(dwell_time*1E-3*frames*px**2) #total counting time in sec
    T_px = (dwell_time * 1E-3)  # total counting per pixel time in sec
    hr_coeff = 8
    T_pxhr = T_px / hr_coeff
    hr_px_size = raster / (px * hr_coeff)
    N_iso = len(R)

    # ------------ Extraction of countrates

    # Extract countrates randomly from image (WORKS)
    # extracted_cts=np.random.choice(np.ravel(realcts[realcts!=0]),(px*hr_coeff)**2)
    # extracted_cts=extracted_cts.reshape(px*hr_coeff,px*hr_coeff)

    # Use the image directly and try to convert it to HR to use it (DOESN'T WORK BECAUSE COUNTRATE END UP NOT BEING INTEGERS FOR MINOR ISOTOPES)
    # realcts[realcts==0]=1
    # dwell_real=s.header['BFields'][0]['time per pixel']
    # D=realcts/(dwell_real*len(s.data.frame))
    # D=np.copy(realcts)
    # extracted_cts=cv2.resize(D, dsize=(px*hr_coeff, px*hr_coeff),interpolation=cv2.INTER_AREA)

    # PG coordinates
    PG_coor = [random.sample(range(px), 2) for i in range(Nb_PG)]
    radius = list((np.asarray(PG_size) / 2) * 1E-3 / (raster / px))
    mask = create_circular_mask_multiple(px, px, center=PG_coor, radius=radius)

    imhr_ini = np.copy(realcts)
    # imhr_ini[:,:]=countrates*R*T_pxhr

    # --------------- Creation of maps of counts for each isotopes based on given countrates for the main isotopes

    for i in range(0, Nb_PG):
        for j in range(0, p):
            it = i + 1

            imhr_ini[mask == it, p + j] = realcts[mask == it, j] * R[1] * (PG_delta[i] / 1E3 + 1)
            imhr_ini[mask == it, 2 * p + j] = realcts[mask == it, j] * R[2] * (PG_delta[i] / 1E3 + 1)

    imhr_ini_PG = np.copy(imhr_ini)
    # imgauss_PG=np.copy(imhr_ini)

    # ----------------- Poisson simulation of counts for each isotopes

    # # boxcar kernel for boxcar images
    # box_kernel = Box2DKernel(boxcar_px)
    # imboxcar_PG=np.zeros_like(imhr_ini_PG)

    # for h in range(0,len(R)): # Loop on the isotopes maps
    #     im_ini_cycles=np.zeros((px*hr_coeff,px*hr_coeff,frames))
    #     for i in range(0,frames): # Loop on the cycles
    #         im_ini_cycles[:,:,i]=np.random.poisson(imhr_ini_PG[:,:,h]) # For each pixel count values extract a random value from a poisson distrubtion with a mu equal to the counts value of the pixel
    #     im=np.sum(im_ini_cycles,axis=2) # Once loop over cycles is over sum all the counts over the number of cycles
    #     imhr_ini_PG[:,:,h]=im # Write over the new counts over the one given by the user (basically, update the homemade image with Poisson randomness)
    #     # imboxcar_PG[:,:,h] = ap_convolve(imhr_ini_PG[:,:,h], box_kernel, normalize_kernel=True) # Create the boxcar image for this isotopes

    # --------------- Beam blurr
    # # sig_gaussian=np.round(beam_size/(raster/px*1000)) #sigma parameter (width of the gaussian) based on the beam size
    # # sig_gaussian=beam_size/(np.sqrt(8*np.log(2))) or beam_size/2.35
    # sig_gaussian=np.round((beam_size*1E-3)/(raster/(px*hr_coeff)))/(np.sqrt(8*np.log(2)))

    # # ax2.imshow(scipy.ndimage.gaussian_filter(im, sigma=sig_gaussian),cmap="plasma")
    # # im_gauss=skimage.filters.gaussian(imhr_ini_PG[:,:,1], sigma=sig_gaussian,truncate=1,preserve_range=True,mode='constant') #convolution of the gaussian filter with a kernel of 2*sigma*truncate pixels
    # # FOR MORE INFO : https://datacarpentry.org/image-processing/06-blurring.html

    # # Downscaling the High Resolution image while applying a gaussian filter to simulate the beam.
    # # imgauss_PG=np.zeros((px,px,N_iso))
    # # for i in range(0,N_iso):
    # #     imgauss_PG[:,:,i]=skimage.transform.rescale(imhr_ini_PG[:,:,i],scale=1/hr_coeff,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)

    # imgauss_PG=np.zeros((px,px,imhr_ini_PG.shape[2]))
    # for i in range(0,imhr_ini_PG.shape[2]):
    #     imgauss_PG[:,:,i]=skimage.transform.rescale(imhr_ini_PG[:,:,i],scale=1/hr_coeff,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)

    # #----------------- Poisson simulation of counts for each isotopes

    # boxcar kernel for boxcar images
    box_kernel = Box2DKernel(boxcar_px)
    im_counts = np.zeros((px, px, N_iso))
    imgauss_PG = np.zeros((px, px, N_iso))
    imboxcar_PG = np.zeros((px, px, N_iso))

    sig_gaussian = np.round((beam_size * 1E-3) / (raster / (px))) / (np.sqrt(8 * np.log(2)))
    gauss_kern_size = int(np.round((beam_size * 1E-3) / (raster / (px))))

    for h in range(0, N_iso):  # Loop on the isotopes maps
        im_ini_cycles = np.zeros((px, px, frames))
        for i in range(0, frames):  # Loop on the cycles
            im_ini_cycles[:, :, i] = np.random.poisson(imhr_ini_PG[:, :,
                                                       h * p + i])  # For each pixel count values extract a random value from a poisson distrubtion with a mu equal to the counts value of the pixel
        im = np.sum(im_ini_cycles, axis=2)  # Once loop over cycles is over sum all the counts over the number of cycles
        # imgauss_PG[:,:,h]=skimage.transform.rescale(im,scale=1,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)
        # imgauss_PG[:,:,h]=cv2.GaussianBlur(im,(gauss_kern_size,gauss_kern_size), sigmaX=sig_gaussian,sigmaY=sig_gaussian)
        im_counts[:, :,
        h] = im  # Write over the new counts over the one given by the user (basically, update the homemade image with Poisson randomness)
        # im_counts[:,:,h]=np.where(mask==1, im, imgauss_PG[:,:,h])
        imboxcar_PG[:, :, h] = ap_convolve(im_counts[:, :, h], box_kernel, boundary='fill', fill_value=0.0,
                                           normalize_kernel=True)  # Create the boxcar image for this isotopes

    # ----- Threshold
    th = 0.05
    cts_th = int(imboxcar_PG[:, :, 0].max() * th)
    mask = np.zeros((px, px))
    mask[imboxcar_PG[:, :, 0] < cts_th] = 1

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)

    # Stack masked images together again
    masked_image = np.ma.dstack((main, minor1, minor2))

    R_1st = minor1 / main
    err_R1 = R_1st * np.sqrt(1 / minor1 + 1 / main) / boxcar_px
    R_2nd = minor2 / main
    err_R2 = R_2nd * np.sqrt(1 / minor2 + 1 / main) / boxcar_px
    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    # err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
    err_d1 = err_R1 / R[1] * 1000
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000
    err_d2 = err_R2 / R[2] * 1000

    # # Delta sigma maps
    # sig_d1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # sig_d2nd=np.abs((d_2nd-np.average(d_2nd))/np.std(d_2nd))

    # imboxcar_d1st=ap_convolve(d_1st, box_kernel, normalize_kernel=True)
    # imboxcar_d2nd=ap_convolve(d_2nd, box_kernel, normalize_kernel=True)

    # ind=np.argwhere(imboxcar_PG[:,:,1]!=0)
    # imboxcar_sig1st=np.abs(imboxcar_PG[ind.T[0],ind.T[1],1]-imboxcar_PG[ind.T[0],ind.T[1],0]*R[1])/np.sqrt(imboxcar_PG[ind.T[0],ind.T[1],1])#smart error
    # ind=np.argwhere(imboxcar_PG[:,:,2]!=0)
    # imboxcar_sig2nd=np.abs(imboxcar_PG[ind.T[0],ind.T[1],2]-imboxcar_PG[ind.T[0],ind.T[1],0]*R[2])/np.sqrt(imboxcar_PG[ind.T[0],ind.T[1],2])

    Rsig = [R[1], R[2]]
    # Rsig=[np.mean(R_1st),np.mean(R_2nd)]

    # Smart error ON
    imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
    # # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    imboxcar_sig2nd = np.abs(R_2nd - Rsig[1]) / err_R2  # smart error
    # # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(d_2nd/err_d2)#smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    ## Plots


#%% Simulation v4 : Simulation v2 but only simulated image around grain for computation optimization
# @jit(fastmath=True)
def PG_simulationv4(file=None, elem=None, Nb_PG=None, PG_delta=None, PG_size=None, beam_size=None, boxcar_px=None,
                    OG_grain=None, standard=None, smart=None, verif=None,
                    display='OFF'):
    ## Check inputs

    np.seterr(divide='ignore')

    if display == 'OFF':
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem == None:
        elem = 'O'
        print('No element specified. Element selected : Oxygen')

    match elem:
        case 'O':
            Iso = ['16O', '17O', '18O']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case 'Mg':
            Iso = ['24Mg', '25Mg', '26Mg']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case _:
            print('The specified element is inadequat. Check if you did not specify an isotope instead of an element.')

    # ---------------- Grain info
    if PG_delta == None:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print('No composition specified. Grain composition fixed to ' + str(PG_delta) + ' permil')
    else:
        if type(PG_delta) != list:
            PG_delta = [PG_delta]

    if PG_size == None:
        PG_size = np.random.randint(100, 600, Nb_PG)
        print('No size specified. Grain diameter fixed to ' + str(PG_size) + ' nm')
    else:
        if type(PG_size) != list:
            PG_size = [PG_size]

    # ------------------- Acquisition variables

    if beam_size == None:
        beam_size = 100
        print('No beam size specified. Fixed to ' + str(beam_size) + ' nm')
    if boxcar_px == None:
        boxcar_px = 3
        print('No boxcar size specified. Fixed to ' + str(boxcar_px) + 'x' + str(boxcar_px))
    if smart == None:
        smart = 1
        print('Smart error activated')
    if standard == None:
        print(
            'Image will be relative to standard terrestrial values. Set "standard" to "average" for delta values relative to the average value of the region.')
    # if countrates==None:
    #     countrates=380000
    #     print('No countrates specified. Fixed to '+str(countrates)+' cps/s')
    # if dwell_time==None:
    #     dwell_time=3
    #     print('No dwell_time specified. Fixed to '+str(dwell_time)+' ms/px')
    # if frames==None:
    #     frames=20
    #     print('No number of frames specified. Fixed to '+str(frames))
    # if raster==None:
    #     raster=15
    #     print('No raster size specified. Fixed to '+str(raster)+' microns')
    # if px==None:
    #     px=256
    #     print('No pixel size specified. Fixed to '+str(px)+'x'+str(px))

    # --------------- Generation of a higher resolution simulated image
    # Real material will not be pixelated and blurred by the beam, so we first need to create a high resolution image to depict "reality"
    # T=(dwell_time*1E-3*frames*px**2) #total counting time in sec
    # T_px=(dwell_time*1E-3)  #total counting per pixel time in sec
    hr_coeff = 8
    # T_pxhr=T_px/hr_coeff
    # hr_px_size=raster/(px*hr_coeff)
    N_iso = len(R)

    ## Extraction of real image

    if ('file' not in globals()) or (
            file == ''):  # If the file path is not specified then, select randomly an image from the following folder.
        path_realim = 'E:/Work/Programmation/Presolar grains/Simulations PG/Real Images/'
        file = random.choice(os.listdir(path_realim))
        file = path_realim + file
    s = sims.SIMS(file)
    raster = s.header['Image']['raster'] / 1000
    realcts = np.zeros((s.header['Image']['width'], s.header['Image']['height'], N_iso))
    realcts[:, :, 0] = s.data.loc[Iso[0]].sum(axis=0).values  # extract maps of the main isotope and sum all frames
    realcts[:, :, 1] = s.data.loc[Iso[1]].sum(axis=0).values
    realcts[:, :, 2] = s.data.loc[Iso[2]].sum(axis=0).values
    realcts = realcts / 1

    # If we work on each frame of the data
    p = s.header['Image']['planes']
    raster = s.header['Image']['raster'] / 1000
    px = s.header['Image']['width']

    # print('')
    # print('Image is : '+str(file))
    # print(str(raster)+'x'+str(raster)+' µm, '+str(px)+'x'+str(px)+' pixels, '+str(p)+' frames')

    # realcts=np.zeros((s.header['Image']['width'],s.header['Image']['height'],len(R)*p))

    ## Verification

    realcts_verif = np.zeros((s.header['Image']['width'], s.header['Image']['height'], len(R)))
    realcts_verif[:, :, 0] = s.data.loc[Iso[0]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif[:, :, 1] = s.data.loc[Iso[1]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif[:, :, 2] = s.data.loc[Iso[2]].sum(
        axis=0).values  # extract maps of the main isotope and sum all frames
    realcts_verif = realcts_verif / 1

    box_kernel = Box2DKernel(boxcar_px)
    imboxcar_PG = np.zeros_like(realcts_verif)
    for i in range(0, 3):
        imboxcar_PG[:, :, i] = ap_convolve(realcts_verif[:, :, i], box_kernel, boundary='fill', fill_value=0.0)

    # ----- Threshold
    th = 0.05
    cts_th = int(imboxcar_PG[:, :, 0].max() * th)
    mask = np.zeros((px, px))
    mask[imboxcar_PG[:, :, 0] < cts_th] = 1

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)

    masked_image = np.ma.dstack((main, minor1, minor2))

    R_1st = minor1 / main
    D = minor1

    err_R1 = R_1st * np.sqrt(1 / minor1 + 1 / main) / boxcar_px
    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    err_d1 = err_R1 / R[1] * 1000

    R_2nd = minor2 / main
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000

    # Rsig=[R[1]]
    Rsig = [np.mean(R_1st)]
    Dmod = np.where(D < main * Rsig[0], main * Rsig[0], D)

    # imboxcar_sig1st=np.abs(R_1st-Rsig[0])/err_R1
    imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod) * 3
    # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*3
    # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*np.sqrt(3)

    if verif == 1:
        d1 = ((realcts_verif[:, :, 1] / realcts_verif[:, :, 0]) / R[1] - 1) * 1000
        f, [[ax, axbox], [axerr, axsig]] = plt.subplots(2, 2, sharex=True, sharey=True)
        im0 = ax.imshow(d1, cmap='gnuplot2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title('delta no boxcar')
        f.colorbar(im0, cax=cax)

        im1 = axbox.imshow(d_1st, cmap='gnuplot2')
        divider = make_axes_locatable(axbox)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axbox.set_title('delta boxcar')
        f.colorbar(im1, cax=cax)
        # plt.plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=20)

        # plt.figure()
        im2 = axsig.imshow(imboxcar_sig1st, cmap='gnuplot2')
        divider = make_axes_locatable(axsig)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axsig.set_title('sigma')
        f.colorbar(im2, cax=cax)
        # plt.colorbar()
        # plt.plot(PG_coor[0]/256,'.r',markersize=15)

        im3 = axerr.imshow(err_d1, cmap='gnuplot2')
        divider = make_axes_locatable(axerr)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axerr.set_title('delta error')
        f.colorbar(im3, cax=cax)

        plt.suptitle(file, fontsize=15)

    ## Simulated images

    # dwell_real=s.header['BFields'][0]['time per pixel']
    th = 0.05

    ####---- Extraction of countrates from Low Resolution (LR) to High Resolution image
    D = np.copy(realcts)  # We copy the image to make sure to not alter the extraction
    extracted_cts = cv2.resize(D, dsize=(px * hr_coeff, px * hr_coeff),
                               interpolation=cv2.INTER_AREA)  # Interpolate into larger dimensions (here from 256x256 px to 2064x2064 px)

    # Original PG coordinates
    if 'OG_grain' in globals():
        OG_PG_center = [OG_grain.ROIX.item(), OG_grain.ROIY.item()]
        m = create_circular_mask(px * hr_coeff, px * hr_coeff, center=OG_PG_center,
                                 radius=int(OG_grain.ROIDIAM / (2 * raster) * px * hr_coeff))
        ind_OG_PG = np.argwhere(m == True)
        del m
    else:
        ind_OG_PG = []

    ####---- PG coordinates
    PG_coor = [random.sample(range(px * hr_coeff), 2) for i in
               range(Nb_PG)]  # Get coordinates for the required number of grains
    it = np.ravel_multi_index(np.asarray(PG_coor).T,
                              extracted_cts[:, :, 0].shape)  # 1D index of the coordinates in the image
    coor_verif = extracted_cts[:, :, 0].take(it)  # Extracting the corresponding 16O counts of the coordinates
    while (any(coor_verif < extracted_cts[:, :, 0].max() * th) == True) or (
            coor_verif in ind_OG_PG):  # If any 16O counts select as the center of a grain is below the masking threshold
        ind_badcoor = np.where(
            coor_verif < extracted_cts[:, :, 0].max() * th)  # Location of the problematic coordinates
        for i in ind_badcoor[0]:  # Loop over the problematic coordinates
            PG_coor[i] = random.sample(range(px * hr_coeff), 2)  # Replacement of the problematic coordinates
        it = np.ravel_multi_index(np.asarray(PG_coor).T, extracted_cts[:, :, 0].shape)  # Update of the 1D index
        coor_verif = extracted_cts[:, :, 0].take(it)  # Update of the 16O counts
    radius = list((np.asarray(PG_size) / 2) * 1E-3 / (
            raster / (px * hr_coeff)))  # Radius calculation of the grains in the HR dimensions
    mask_PG = create_circular_mask_multiple(px * hr_coeff, px * hr_coeff, center=PG_coor,
                                            radius=radius)  # Mask creation of the grains' pixels

    #### ---- Cropping index
    reg_w = 30  # Width around the grain in pixels (for LR images) of the region
    ranges = [[x - reg_w * hr_coeff, x + reg_w * hr_coeff] for x in PG_coor[0]]
    ranges = [[np.sign(x) * px * hr_coeff if np.abs(x) > px * hr_coeff else x for x in R] for R in ranges]
    ranges = [[0 if x < 0 else x for x in R] for R in ranges]

    # -- Get new coordinated of the grain in the cropped image
    PG_coorLR = []
    for o in PG_coor[0]:
        if o - reg_w * hr_coeff < 0:
            PG_coorLR.append(int(o / hr_coeff))
        else:
            PG_coorLR.append(reg_w)

    imhr_ini = np.copy(extracted_cts)  # Copying the HR images channels to avoid altering them

    ####---- Modifying maps counts on location of presolar grains
    for j in range(1, N_iso):  # Loop on isotopes
        for i in range(1, Nb_PG + 1):  # Loop on PG
            imhr_ini[mask_PG == i, j] = extracted_cts[mask_PG == i, 0] * R[j] * (PG_delta[0] / 1E3 + 1)

    imhr_ini_PG = np.copy(
        imhr_ini[ranges[1][0]:ranges[1][1], ranges[0][0]:ranges[0][1], :])  # Copy the cropped region around the grain

    ####---- Beam blurr and Boxcar definitions

    # Defining the sigma parameters of the gaussian blurr
    # sig_gaussian=beam_size/(np.sqrt(8*np.log(2))) or beam_size/2.35
    fwhm_hr = np.round((beam_size * 1E-3) / (raster / (
                px * hr_coeff))) / 2  # the gaussian filter uses the given sigma as a radius for kernel size if radius is not specified
    # fwhm_hr=np.round((beam_size*1E-3)/(raster/(px)))
    sig_gaussian = fwhm_hr / (np.sqrt(8 * np.log(2)))

    box_kernel = Box2DKernel(boxcar_px)  # Boxcar Kernel

    imgauss_PG = np.zeros((int(np.round(imhr_ini_PG.shape[0] / hr_coeff)),
                           int(np.round(imhr_ini_PG.shape[1] / hr_coeff)),
                           N_iso))  # Allocation of memory for the beam blurred image
    imboxcar_PG = np.zeros_like(imgauss_PG)  # Allocation of memory for the boxcar smoothed image

    ####---- Beam blurr and Boxcar smoothing

    # Two options :
    # 1. A new image is created for each isotopes from the original ones. Each pixel value is used as a mean for a poisson distribution of which a new pixel value is interpolated. Then the images are beam blurred and boxcar smoothed.
    # 2. We started by applying the gaussian blurr before interpolation new values from a poisson distribution for each pixel of each isotope image. Then the image is boxcar smoothed.

    # Allocation of memory for the new Poisson interpolated images
    # #Option 1
    im_poiss = np.zeros_like(imhr_ini_PG)
    # Option 2
    # im_poiss=np.zeros_like(imgauss_PG)

    # Simulation including the presolar grains
    for i in range(0, N_iso):
        # # ----- Option 1 : First poisson then gaussian blurr
        im_poiss[:, :, i] = np.random.poisson(imhr_ini_PG[:, :, i])  # Poisson distribution interpolation
        imgauss_PG[:, :, i] = skimage.transform.rescale(im_poiss[:, :, i], order=0, scale=1 / hr_coeff,
                                                        preserve_range=True, anti_aliasing=True,
                                                        anti_aliasing_sigma=sig_gaussian)  # Gaussian blurring to simulate the beam while downscaling from 2064x2064 to 256x256 px. The "Anti_Aliasing" option applies a gaussian blurr which kernel radius will be sigma*truncate+1.
        imboxcar_PG[:, :, i] = ap_convolve(imgauss_PG[:, :, i], box_kernel, boundary='fill',
                                           fill_value=0.0)  # Boxcar smoothing

        # # ----- Option 2 : First gaussian blurr then poisson

        # imgauss_PG[:,:,i]=skimage.transform.rescale(imhr_ini_PG[:,:,i],order=0,scale=1/hr_coeff,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)
        # im_poiss[:,:,i]=np.random.poisson(imgauss_PG[:,:,i])
        # imboxcar_PG[:,:,i] = ap_convolve(imgauss_PG[:,:,i], box_kernel, boundary='fill', fill_value=0.0) # Create the boxcar image for this isotopes

    ####---- Masking low counts regions

    cts_th = int(imboxcar_PG[:, :,
                 0].max() * th)  # Define criterion as 5% of the max encountered for the main isotope counts in one pixel
    mask = np.zeros((int(np.round(imhr_ini_PG.shape[0] / hr_coeff)), int(np.round(
        imhr_ini_PG.shape[1] / hr_coeff))))  # Allocate memory for the mask with a 0 matrix to the image dimensions
    mask[imboxcar_PG[:, :,
         0] < cts_th] = 1  # Set all coordinates of pixels with lower counts than the criterion to 1 in 0 matrix mask.

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)  # Extract main isotope image
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)  # Extract first minor isotope image
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)  # Extract second minor isotope image

    # Stack masked images together again
    masked_image = np.ma.dstack((main, minor1, minor2))  # Stack the masked images together

    ####---- Ratio, delta and error calculations

    R_1st = minor1 / main
    R_2nd = minor2 / main

    # err_R1=R_1st*np.sqrt(1/minor1+1/main)/boxcar_px
    # err_R2=R_2nd*np.sqrt(1/minor2+1/main)/boxcar_px
    D = minor1
    Dmod = np.where(D < main * R[1], main * R[1], D)
    err_R1 = R_1st * np.sqrt(1 / Dmod + 1 / main) / boxcar_px
    err_R2 = R_2nd * np.sqrt(1 / minor2 + 1 / main) / boxcar_px

    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    # err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
    err_d1 = err_R1 / R[1] * 1000
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000
    err_d2 = err_R2 / R[2] * 1000

    ####---- Sigma images

    # Sigma images will be defined either relative to the standard value or the average ratio of the image
    if standard == "average":
        Rsig = [np.mean(R_1st), np.mean(R_2nd)]
    else:
        Rsig = R[1:]
    # Rsig=

    # Smart error ON
    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(main*Rsig[0])*boxcar_px
    # # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(minor2-main*Rsig[1])/np.sqrt(main*Rsig[1])*boxcar_px #smart error
    # # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    if smart == 1:
        Dmod = np.zeros((main.shape[0], main.shape[1], N_iso - 1))
        Dmod[:, :, 0] = minor1
        Dmod[:, :, 1] = minor2

        for g in range(0, N_iso - 1): Dmod[:, :, g] = np.where(masked_image[:, :, g + 1] < main * Rsig[g],
                                                               main * Rsig[g], minor1)

        imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod[:, :, 0]) * 3
        imboxcar_sig2nd = np.abs(minor2 - main * Rsig[1]) / np.sqrt(Dmod[:, :, 1]) * 3
    else:
        imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
        imboxcar_sig2nd = np.abs(R_2nd - Rsig[1]) / err_R2

    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(d_2nd/err_d2)#smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    # plt.subplot()
    # plt.imshow(realcts,cmap='gnuplot2')
    # plt.colorbar()

    if verif == 1:
        # f_verif,[axpoiss,axgauss,axbox]=plt.subplots(1,3,sharex=True,sharey=True)
        f_verif, [axpoiss, axgauss, axbox] = plt.subplots(1, 3)
        axpoiss.imshow(im_poiss[:, :, 1], cmap='gnuplot2')
        axpoiss.set_title('Poisson HR')
        axpoiss.plot(PG_coor[0][0], PG_coor[0][1], 'o', mfc='none', mec='r', markersize=15)
        axgauss.imshow(imgauss_PG[:, :, 1], cmap='gnuplot2')
        axgauss.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                     markersize=15)
        axgauss.set_title('Gaussian blurr')
        axbox.imshow(imboxcar_PG[:, :, 1], cmap='gnuplot2')
        axbox.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                   markersize=15)
        axbox.set_title('Gaussian + Boxcar')
        plt.suptitle('17O counts', fontsize=15)

    ## Plots

    plots = [imgauss_PG[:, :, 0], imgauss_PG[:, :, 1], imgauss_PG[:, :, 2],
             imboxcar_PG[:, :, 0], imboxcar_PG[:, :, 1], imboxcar_PG[:, :, 2],
             R_1st, d_1st, imboxcar_sig1st,
             R_2nd, d_2nd, imboxcar_sig2nd]
    plots_title = [str(Iso[0]) + ' counts', str(Iso[1]) + ' counts', str(Iso[2]) + ' counts',
                   str(Iso[0]) + ' counts boxcar', str(Iso[1]) + ' counts boxcar', str(Iso[2]) + ' counts boxcar',
                   'Ratio ' + str(Iso[1]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Ratio ' + str(Iso[2]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[2]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[2]) + '/' + str(Iso[0])]
    # titles=["Raw Simulation",str(beam_size)+" nm Beam Blurr Simulation",str(boxcar_px)+" x "+str(boxcar_px)+" Boxcar and Gaussian Blurr Simulation"]

    fig, axs = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    # axinsert=[]
    palette = 'gnuplot2'

    fontprops = fm.FontProperties(size=12)

    for i in range(0, len(plots)):
        # axins = inset_axes(axs[i], width="60%", height="60%", loc="upper left", bbox_to_anchor=(-0.5,0.2,1,1), bbox_transform=axs[i].transAxes, borderpad=1)
        img_plot = axs[i].imshow(plots[i], cmap=palette, interpolation='None', rasterized=True)
        # axs[i].plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=15)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img_plot, cax=cax)
        # axs[i,j].title.set_text(titles[h])
        axs[i].set_axis_off()

        plt.title(plots_title[i])

        # axins.imshow(plots[i],cmap=palette,interpolation='None',rasterized=True,)
        # axins.set_xlim(int(PG_coor[0][0]/hr_coeff)-15,int(PG_coor[0][0]/hr_coeff)+15)
        # axins.set_ylim(int(PG_coor[0][1]/hr_coeff)-15, int(PG_coor[0][1]/hr_coeff)+15)
        # axins.get_xaxis().set_visible(False)
        # axins.get_yaxis().set_visible(False)
        # axinsert.append(axins)

        scalebar = AnchoredSizeBar(axs[i].transData,
                                   int(px / raster * 2), r'2 $\mu m$', 'lower center',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        axs[i].add_artist(scalebar)

    return fig, axs, plots, plots_title, PG_coorLR, raster, px


#%% Simulation v5 : For automatization on PG from tables


# @jit(parallel=True)
def PG_simulationv5(file=None, elem=None, PG_delta=None, PG_size=None, beam_size=None, boxcar_px=None, OG_grain=None,
                    standard=None, smart=None, verif=None,
                    display='OFF'):
    ## Check inputs

    np.seterr(divide='ignore')

    if display == 'OFF':
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem == None:
        elem = 'O'
        print('No element specified. Element selected : Oxygen')

    match elem:
        case 'O':
            Iso = ['16O', '17O', '18O']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case 'Mg':
            Iso = ['24Mg', '25Mg', '26Mg']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case _:
            print('The specified element is inadequat. Check if you did not specify an isotope instead of an element.')

    # ---------------- Grain info

    try:
        PG_size
    except NameError:
        Nb_PG = 1
        PG_size = np.random.randint(100, 600, Nb_PG)
        print('No size specified. Grain diameter fixed to ' + str(PG_size) + ' nm')
    else:
        Nb_PG = len(PG_size)
        if isinstance(PG_size, Iterable) == False:
            PG_size = [PG_size]

    try:
        PG_delta
    except:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        while PG_delta in range(-200, 200, 1):
            PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print('No composition specified. Grain composition fixed to ' + str(PG_delta) + ' permil')
    else:
        if isinstance(PG_delta, Iterable) == False:
            PG_delta = [PG_delta]

    # ------------------- Acquisition variables

    if beam_size == None:
        beam_size = 100
        print('No beam size specified. Fixed to ' + str(beam_size) + ' nm')
    if boxcar_px == None:
        boxcar_px = 3
        print('No boxcar size specified. Fixed to ' + str(boxcar_px) + 'x' + str(boxcar_px))
    if smart == None:
        smart = 1
        print('Smart error activated')
    if standard == None:
        print(
            'Image will be relative to standard terrestrial values. Set "standard" to "average" for delta values relative to the average value of the region.')
    # if countrates==None:
    #     countrates=380000
    #     print('No countrates specified. Fixed to '+str(countrates)+' cps/s')
    # if dwell_time==None:
    #     dwell_time=3
    #     print('No dwell_time specified. Fixed to '+str(dwell_time)+' ms/px')
    # if frames==None:
    #     frames=20
    #     print('No number of frames specified. Fixed to '+str(frames))
    # if raster==None:
    #     raster=15
    #     print('No raster size specified. Fixed to '+str(raster)+' microns')
    # if px==None:
    #     px=256
    #     print('No pixel size specified. Fixed to '+str(px)+'x'+str(px))

    # --------------- Generation of a higher resolution simulated image
    # Real material will not be pixelated and blurred by the beam, so we first need to create a high resolution image to depict "reality"
    # T=(dwell_time*1E-3*frames*px**2) #total counting time in sec
    # T_px=(dwell_time*1E-3)  #total counting per pixel time in sec
    hr_coeff = 8
    # T_pxhr=T_px/hr_coeff
    # hr_px_size=raster/(px*hr_coeff)
    N_iso = len(R)

    ## Extraction of real image
    try:
        file
    except:  # If the file path is not specified then, select randomly an image from the following folder.
        print('No file specified')
        path_realim = 'E:/Work/Programmation/Presolar grains/Simulations PG/Real Images/'
        file = random.choice(os.listdir(path_realim))
        file = path_realim + file
    s = sims.SIMS(file)
    raster = s.header['Image']['raster'] / 1000
    realcts = np.zeros((s.header['Image']['width'], s.header['Image']['height'], N_iso))
    realcts[:, :, 0] = s.data.loc[Iso[0]].sum(axis=0).values  # extract maps of the main isotope and sum all frames
    realcts[:, :, 1] = s.data.loc[Iso[1]].sum(axis=0).values
    realcts[:, :, 2] = s.data.loc[Iso[2]].sum(axis=0).values
    realcts = realcts / 1

    # If we work on each frame of the data
    # p=s.header['Image']['planes']
    raster = s.header['Image']['raster'] / 1000
    px = s.header['Image']['width']

    # print('')
    # print('Image is : '+str(file))
    # print(str(raster)+'x'+str(raster)+' µm, '+str(px)+'x'+str(px)+' pixels, '+str(p)+' frames')

    # realcts=np.zeros((s.header['Image']['width'],s.header['Image']['height'],len(R)*p))

    ## Verification

    if verif == 1:

        realcts_verif = np.copy(realcts)
        realcts_verif = realcts_verif / 1

        box_kernel = Box2DKernel(boxcar_px)
        imboxcar_PG = np.zeros_like(realcts_verif)
        for i in range(0, 3):
            imboxcar_PG[:, :, i] = ap_convolve(realcts_verif[:, :, i], box_kernel, boundary='fill', fill_value=0.0)

        # ----- Threshold
        th = 0.05
        cts_th = int(imboxcar_PG[:, :, 0].max() * th)
        mask = np.zeros((px, px))
        mask[imboxcar_PG[:, :, 0] < cts_th] = 1

        main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)
        minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)
        minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)
        masked_image = np.ma.dstack((main, minor1, minor2))

        R_1st = minor1 / main
        D = minor1
        err_R1 = R_1st * np.sqrt(1 / minor1 + 1 / main) / boxcar_px
        d_1st = ((minor1 / main) / R[1] - 1) * 1000
        err_d1 = err_R1 / R[1] * 1000
        R_2nd = minor2 / main
        d_2nd = ((minor2 / main) / R[2] - 1) * 1000

        # Rsig=[R[1]]
        Rsig = [np.mean(R_1st)]
        Dmod = np.where(D < main * Rsig[0], main * Rsig[0], D)

        # imboxcar_sig1st=np.abs(R_1st-Rsig[0])/err_R1
        imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod) * 3
        # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*3
        # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*np.sqrt(3)

        d1 = ((realcts_verif[:, :, 1] / realcts_verif[:, :, 0]) / R[1] - 1) * 1000
        f, [[ax, axbox], [axerr, axsig]] = plt.subplots(2, 2, sharex=True, sharey=True)
        im0 = ax.imshow(d1, cmap='gnuplot2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title('delta no boxcar')
        f.colorbar(im0, cax=cax)

        im1 = axbox.imshow(d_1st, cmap='gnuplot2')
        divider = make_axes_locatable(axbox)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axbox.set_title('delta boxcar')
        f.colorbar(im1, cax=cax)
        # plt.plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=20)

        # plt.figure()
        im2 = axsig.imshow(imboxcar_sig1st, cmap='gnuplot2')
        divider = make_axes_locatable(axsig)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axsig.set_title('sigma')
        f.colorbar(im2, cax=cax)
        # plt.colorbar()
        # plt.plot(PG_coor[0]/256,'.r',markersize=15)

        im3 = axerr.imshow(err_d1, cmap='gnuplot2')
        divider = make_axes_locatable(axerr)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axerr.set_title('delta error')
        f.colorbar(im3, cax=cax)

        plt.suptitle(file, fontsize=15)

    ## Simulated images

    # dwell_real=s.header['BFields'][0]['time per pixel']
    th = 0.05

    ####---- Extraction of countrates from Low Resolution (LR) to High Resolution image
    D = np.copy(realcts)  # We copy the image to make sure to not alter the extraction
    extracted_cts = cv2.resize(D, dsize=(px * hr_coeff, px * hr_coeff),
                               interpolation=cv2.INTER_AREA)  # Interpolate into larger dimensions (here from 256x256 px to 2064x2064 px)

    # Original PG coordinates
    if 'OG_grain' in globals():
        OG_PG_center = [OG_grain.ROIX.item(), OG_grain.ROIY.item()]
        m = create_circular_mask(px * hr_coeff, px * hr_coeff, center=OG_PG_center,
                                 radius=int(OG_grain.ROIDIAM / (2 * raster) * px * hr_coeff))
        ind_OG_PG = np.argwhere(m == True)
        # del m
    else:
        ind_OG_PG = []

    ####---- PG coordinates

    PG_coor = [random.sample(range(px * hr_coeff), 2) for i in
               range(Nb_PG)]  # Get coordinates for the required number of grains
    it = np.ravel_multi_index(np.asarray(PG_coor).T,
                              extracted_cts[:, :, 0].shape)  # 1D index of the coordinates in the image
    coor_verif = extracted_cts[:, :, 0].take(it)  # Extracting the corresponding 16O counts of the coordinates
    while (any(coor_verif < extracted_cts[:, :, 0].max() * th) == True) or (
            coor_verif in ind_OG_PG):  # If any 16O counts select as the center of a grain is below the masking threshold
        ind_badcoor = np.where(
            coor_verif < extracted_cts[:, :, 0].max() * th)  # Location of the problematic coordinates
        for i in ind_badcoor[0]:  # Loop over the problematic coordinates
            PG_coor[i] = random.sample(range(px * hr_coeff), 2)  # Replacement of the problematic coordinates
        it = np.ravel_multi_index(np.asarray(PG_coor).T, extracted_cts[:, :, 0].shape)  # Update of the 1D index
        coor_verif = extracted_cts[:, :, 0].take(it)  # Update of the 16O counts
    radius = list((np.asarray(PG_size) / 2) * 1E-3 / (
                raster / (px * hr_coeff)))  # Radius calculation of the grains in the HR dimensions
    mask_PG = create_circular_mask_multiple(px * hr_coeff, px * hr_coeff, center=PG_coor,
                                            radius=radius)  # Mask creation of the grains' pixels

    imhr_ini = np.copy(extracted_cts)  # Copying the HR images channels to avoid altering them

    ####---- Modifying maps counts on location of presolar grains
    for j in range(1, N_iso):  # Loop on isotopes
        for i in range(1, Nb_PG + 1):  # Loop on PG
            imhr_ini[mask_PG == i, j] = extracted_cts[mask_PG == i, 0] * R[j] * (PG_delta[i - 1] / 1E3 + 1)

    imhr_ini_PG = np.copy(imhr_ini)  # Copying the modified images

    ####---- Beam blurr and Boxcar definitions

    # Defining the sigma parameters of the gaussian blurr
    # sig_gaussian=beam_size/(np.sqrt(8*np.log(2))) or beam_size/2.35
    fwhm_hr = np.round((beam_size * 1E-3) / (raster / (
                px * hr_coeff))) / 2  # the gaussian filter uses the given sigma as a radius for kernel size if radius is not specified
    # fwhm_hr=np.round((beam_size*1E-3)/(raster/(px)))
    sig_gaussian = fwhm_hr / (np.sqrt(8 * np.log(2)))

    box_kernel = Box2DKernel(boxcar_px)  # Boxcar Kernel

    imgauss_PG = np.zeros((px, px, N_iso))  # Allocation of memory for the beam blurred image
    imboxcar_PG = np.zeros((px, px, N_iso))  # Allocation of memory for the boxcar smoothed image

    ####---- Beam blurr and Boxcar smoothing

    # Two options :
    # 1. A new image is created for each isotopes from the original ones. Each pixel value is used as a mean for a poisson distribution of which a new pixel value is interpolated. Then the images are beam blurred and boxcar smoothed.
    # 2. We started by applying the gaussian blurr before interpolation new values from a poisson distribution for each pixel of each isotope image. Then the image is boxcar smoothed.

    # Allocation of memory for the new Poisson interpolated images
    # #Option 1
    im_poiss = np.zeros_like(imhr_ini_PG)
    # Option 2
    # im_poiss=np.zeros_like(imgauss_PG)

    # Simulation including the presolar grains
    im_poiss = np.random.poisson(imhr_ini_PG)
    for i in range(0, N_iso):
        # # ----- Option 1 : First poisson then gaussian blurr
        # im_poiss[:,:,i]=np.random.poisson(imhr_ini_PG[:,:,i]) # Poisson distribution interpolation
        imgauss_PG[:, :, i] = skimage.transform.rescale(im_poiss[:, :, i], order=0, scale=1 / hr_coeff,
                                                        preserve_range=True, anti_aliasing=True,
                                                        anti_aliasing_sigma=sig_gaussian)  # Gaussian blurring to simulate the beam while downscaling from 2064x2064 to 256x256 px. The "Anti_Aliasing" option applies a gaussian blurr which kernel radius will be sigma*truncate+1.
        imboxcar_PG[:, :, i] = ap_convolve(imgauss_PG[:, :, i], box_kernel, boundary='fill',
                                           fill_value=0.0)  # Boxcar smoothing

        # # ----- Option 2 : First gaussian blurr then poisson

        # imgauss_PG[:,:,i]=skimage.transform.rescale(imhr_ini_PG[:,:,i],order=0,scale=1/hr_coeff,preserve_range=True,anti_aliasing=True,anti_aliasing_sigma=sig_gaussian)
        # im_poiss[:,:,i]=np.random.poisson(imgauss_PG[:,:,i])
        # imboxcar_PG[:,:,i] = ap_convolve(imgauss_PG[:,:,i], box_kernel, boundary='fill', fill_value=0.0) # Create the boxcar image for this isotopes

    ####---- Masking low counts regions

    cts_th = int(imboxcar_PG[:, :,
                 0].max() * th)  # Define criterion as 5% of the max encountered for the main isotope counts in one pixel
    mask = np.zeros((px, px))  # Allocate memory for the mask with a 0 matrix to the image dimensions
    mask[imboxcar_PG[:, :,
         0] < cts_th] = 1  # Set all coordinates of pixels with lower counts than the criterion to 1 in 0 matrix mask.

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)  # Extract main isotope image
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)  # Extract first minor isotope image
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)  # Extract second minor isotope image

    # Stack masked images together again
    masked_image = np.ma.dstack((main, minor1, minor2))  # Stack the masked images together

    ####---- Ratio, delta and error calculations

    R_1st = minor1 / main
    R_2nd = minor2 / main

    # err_R1=R_1st*np.sqrt(1/minor1+1/main)/boxcar_px
    # err_R2=R_2nd*np.sqrt(1/minor2+1/main)/boxcar_px
    D = minor1
    Dmod = np.where(D < main * R[1], main * R[1], D)
    err_R1 = R_1st * np.sqrt(1 / Dmod + 1 / main) / boxcar_px
    err_R2 = R_2nd * np.sqrt(1 / minor2 + 1 / main) / boxcar_px

    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    # err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
    err_d1 = err_R1 / R[1] * 1000
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000
    err_d2 = err_R2 / R[2] * 1000

    ####---- Sigma images

    # Sigma images will be defined either relative to the standard value or the average ratio of the image
    if standard == "average":
        Rsig = [np.mean(R_1st), np.mean(R_2nd)]
    else:
        Rsig = R[1:]
    # Rsig=

    # Smart error ON
    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(main*Rsig[0])*boxcar_px
    # # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(minor2-main*Rsig[1])/np.sqrt(main*Rsig[1])*boxcar_px #smart error
    # # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    if smart == 1:
        Dmod = np.zeros((main.shape[0], main.shape[1], N_iso - 1))
        Dmod[:, :, 0] = minor1
        Dmod[:, :, 1] = minor2

        for g in range(0, N_iso - 1): Dmod[:, :, g] = np.where(masked_image[:, :, g + 1] < main * Rsig[g],
                                                               main * Rsig[g], minor1)

        imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod[:, :, 0]) * 3
        imboxcar_sig2nd = np.abs(minor2 - main * Rsig[1]) / np.sqrt(Dmod[:, :, 1]) * 3
    else:
        imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
        imboxcar_sig2nd = np.abs(R_2nd - Rsig[1]) / err_R2

    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(d_2nd/err_d2)#smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    # plt.subplot()
    # plt.imshow(realcts,cmap='gnuplot2')
    # plt.colorbar()

    if verif == 1:
        # f_verif,[axpoiss,axgauss,axbox]=plt.subplots(1,3,sharex=True,sharey=True)
        f_verif, [axpoiss, axgauss, axbox] = plt.subplots(1, 3)
        axpoiss.imshow(im_poiss[:, :, 1], cmap='gnuplot2')
        axpoiss.set_title('Poisson HR')
        axpoiss.plot(PG_coor[0][0], PG_coor[0][1], 'o', mfc='none', mec='r', markersize=15)
        axgauss.imshow(imgauss_PG[:, :, 1], cmap='gnuplot2')
        axgauss.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                     markersize=15)
        axgauss.set_title('Gaussian blurr')
        axbox.imshow(imboxcar_PG[:, :, 1], cmap='gnuplot2')
        axbox.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                   markersize=15)
        axbox.set_title('Gaussian + Boxcar')
        plt.suptitle('17O counts', fontsize=15)

    ## Plots

    plots = [imgauss_PG[:, :, 0], imgauss_PG[:, :, 1], imgauss_PG[:, :, 2],
             imboxcar_PG[:, :, 0], imboxcar_PG[:, :, 1], imboxcar_PG[:, :, 2],
             R_1st, d_1st, imboxcar_sig1st,
             R_2nd, d_2nd, imboxcar_sig2nd]
    plots_title = [str(Iso[0]) + ' counts', str(Iso[1]) + ' counts', str(Iso[2]) + ' counts',
                   str(Iso[0]) + ' counts boxcar', str(Iso[1]) + ' counts boxcar', str(Iso[2]) + ' counts boxcar',
                   'Ratio ' + str(Iso[1]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Ratio ' + str(Iso[2]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[2]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[2]) + '/' + str(Iso[0])]
    # titles=["Raw Simulation",str(beam_size)+" nm Beam Blurr Simulation",str(boxcar_px)+" x "+str(boxcar_px)+" Boxcar and Gaussian Blurr Simulation"]

    fig, axs = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    # axinsert=[]
    palette = 'gnuplot2'

    fontprops = fm.FontProperties(size=12)

    for i in range(0, len(plots)):
        # axins = inset_axes(axs[i], width="60%", height="60%", loc="upper left", bbox_to_anchor=(-0.5,0.2,1,1), bbox_transform=axs[i].transAxes, borderpad=1)
        img_plot = axs[i].imshow(plots[i], cmap=palette, interpolation='None', rasterized=True, )
        # axs[i].plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=15)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img_plot, cax=cax)
        # axs[i,j].title.set_text(titles[h])
        axs[i].set_axis_off()

        plt.title(plots_title[i])

        # axins.imshow(plots[i],cmap=palette,interpolation='None',rasterized=True,)
        # axins.set_xlim(int(PG_coor[0][0]/hr_coeff)-15,int(PG_coor[0][0]/hr_coeff)+15)
        # axins.set_ylim(int(PG_coor[0][1]/hr_coeff)-15, int(PG_coor[0][1]/hr_coeff)+15)
        # axins.get_xaxis().set_visible(False)
        # axins.get_yaxis().set_visible(False)
        # axinsert.append(axins)

        scalebar = AnchoredSizeBar(axs[i].transData,
                                   int(px / raster * 2), r'2 $\mu m$', 'lower center',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        axs[i].add_artist(scalebar)

    return fig, axs, plots, plots_title, PG_coor, raster, px  # ,axinsert

#%% Simulation v6 : For automatization on PG from tables using multiple isotopic ratios simultaneously

# Similar to v5 but enables to gives as many delta values per grain as there are isotopic ratios. This enables for 3D exploration of best match

# @jit(parallel=True)
def PG_simulationv6(file=None, elem=None, PG_delta=None, PG_size=None, beam_size=None, boxcar_px=None, OG_grain=None,
                    standard=None, smart=None, verif=None,
                    display='OFF'):
    ## Check inputs

    np.seterr(divide='ignore')

    if display == 'OFF':
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem == None:
        elem = 'O'
        print('No element specified. Element selected : Oxygen')

    match elem:
        case 'O':
            Iso = ['16O', '17O', '18O']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case 'Mg':
            Iso = ['24Mg', '25Mg', '26Mg']
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case _:
            print('The specified element is inadequat. Check if you did not specify an isotope instead of an element.')

    # ---------------- Grain info

    try:
        PG_size
    except NameError:
        Nb_PG = 1
        PG_size = np.random.randint(100, 600, Nb_PG)
        print('No size specified. Grain diameter fixed to ' + str(PG_size) + ' nm')
    else:
        Nb_PG = len(PG_size)
        if isinstance(PG_size, Iterable) == False:
            PG_size = [PG_size]

    try:
        PG_delta
    except:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        while PG_delta in range(-200, 200, 1):
            PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print('No composition specified. Grain composition fixed to ' + str(PG_delta) + ' permil')
    # else:
        if isinstance(PG_delta, Iterable) == False:
            PG_delta = [PG_delta]

    # ------------------- Acquisition variables

    if beam_size == None:
        beam_size = 100
        print('No beam size specified. Fixed to ' + str(beam_size) + ' nm')
    if boxcar_px == None:
        boxcar_px = 3
        print('No boxcar size specified. Fixed to ' + str(boxcar_px) + 'x' + str(boxcar_px))
    if smart == None:
        smart = 1
        print('Smart error activated')
    if standard == None:
        print('Image will be relative to standard terrestrial values. Set "standard" to "average" for delta values relative to the average value of the region.')
    # if countrates==None:
    #     countrates=380000
    #     print('No countrates specified. Fixed to '+str(countrates)+' cps/s')
    # if dwell_time==None:
    #     dwell_time=3
    #     print('No dwell_time specified. Fixed to '+str(dwell_time)+' ms/px')
    # if frames==None:
    #     frames=20
    #     print('No number of frames specified. Fixed to '+str(frames))
    # if raster==None:
    #     raster=15
    #     print('No raster size specified. Fixed to '+str(raster)+' microns')
    # if px==None:
    #     px=256
    #     print('No pixel size specified. Fixed to '+str(px)+'x'+str(px))


    # --------------- Generation of a higher resolution simulated image
    # Real material will not be pixelated and blurred by the beam, so we first need to create a high resolution image to depict "reality"
    # T=(dwell_time*1E-3*frames*px**2) #total counting time in sec
    # T_px=(dwell_time*1E-3)  #total counting per pixel time in sec
    hr_coeff = 8
    # T_pxhr=T_px/hr_coeff
    # hr_px_size=raster/(px*hr_coeff)
    N_iso = len(R)

    ## Extraction of real image
    try:
        file
    except:  # If the file path is not specified then, select randomly an image from the following folder.
        print('No file specified')
        path_realim = 'E:/Work/Programmation/Presolar grains/Simulations PG/Real Images/'
        file = random.choice(os.listdir(path_realim))
        file = path_realim + file
    s = sims.SIMS(file)
    raster = s.header['Image']['raster'] / 1000
    px = s.header['Image']['width']
    realcts = np.zeros((s.header['Image']['width'], s.header['Image']['height'], N_iso))
    realcts[:, :, 0] = s.data.loc[Iso[0]].sum(axis=0).values  # extract maps of the main isotope and sum all frames
    realcts[:, :, 1] = s.data.loc[Iso[1]].sum(axis=0).values
    realcts[:, :, 2] = s.data.loc[Iso[2]].sum(axis=0).values
    realcts = realcts / 1

    ## Verification

    if verif == 1:

        realcts_verif = np.copy(realcts)
        realcts_verif = realcts_verif / 1

        box_kernel = Box2DKernel(boxcar_px)
        imboxcar_PG = np.zeros_like(realcts_verif)
        for i in range(0, 3):
            imboxcar_PG[:, :, i] = ap_convolve(realcts_verif[:, :, i], box_kernel, boundary='fill', fill_value=0.0)

        # ----- Threshold
        th = 0.05
        cts_th = int(imboxcar_PG[:, :, 0].max() * th)
        mask = np.zeros((px, px))
        mask[imboxcar_PG[:, :, 0] < cts_th] = 1

        main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)
        minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)
        minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)
        masked_image = np.ma.dstack((main, minor1, minor2))

        R_1st = minor1 / main
        D = minor1
        err_R1 = R_1st * np.sqrt(1 / minor1 + 1 / main) / boxcar_px
        d_1st = ((minor1 / main) / R[1] - 1) * 1000
        err_d1 = err_R1 / R[1] * 1000
        R_2nd = minor2 / main
        d_2nd = ((minor2 / main) / R[2] - 1) * 1000

        # Rsig=[R[1]]
        Rsig = [np.mean(R_1st)]
        Dmod = np.where(D < main * Rsig[0], main * Rsig[0], D)

        # imboxcar_sig1st=np.abs(R_1st-Rsig[0])/err_R1
        imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod) * 3
        # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*3
        # imboxcar_sig1st=np.abs(minor1-R[1]*main)/np.sqrt(R[1]*main)*np.sqrt(3)

        d1 = ((realcts_verif[:, :, 1] / realcts_verif[:, :, 0]) / R[1] - 1) * 1000
        f, [[ax, axbox], [axerr, axsig]] = plt.subplots(2, 2, sharex=True, sharey=True)
        im0 = ax.imshow(d1, cmap='gnuplot2')
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        ax.set_title('delta no boxcar')
        f.colorbar(im0, cax=cax)

        im1 = axbox.imshow(d_1st, cmap='gnuplot2')
        divider = make_axes_locatable(axbox)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axbox.set_title('delta boxcar')
        f.colorbar(im1, cax=cax)
        # plt.plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=20)

        # plt.figure()
        im2 = axsig.imshow(imboxcar_sig1st, cmap='gnuplot2')
        divider = make_axes_locatable(axsig)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axsig.set_title('sigma')
        f.colorbar(im2, cax=cax)
        # plt.colorbar()
        # plt.plot(PG_coor[0]/256,'.r',markersize=15)

        im3 = axerr.imshow(err_d1, cmap='gnuplot2')
        divider = make_axes_locatable(axerr)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axerr.set_title('delta error')
        f.colorbar(im3, cax=cax)

        plt.suptitle(file, fontsize=15)

    ## Simulated images

    # dwell_real=s.header['BFields'][0]['time per pixel']
    th = 0.05

    ####---- Extraction of countrates from Low Resolution (LR) to High Resolution image
    D = np.copy(realcts)  # We copy the image to make sure to not alter the extraction
    extracted_cts = cv2.resize(D, dsize=(px * hr_coeff, px * hr_coeff),
                               interpolation=cv2.INTER_AREA)  # Interpolate into larger dimensions (here from 256x256 px to 2064x2064 px)

    # Original PG coordinates
    if 'OG_grain' in globals():
        OG_PG_center = [OG_grain.ROIX.item(), OG_grain.ROIY.item()]
        m = create_circular_mask(px * hr_coeff, px * hr_coeff, center=OG_PG_center,
                                 radius=int(OG_grain.ROIDIAM / (2 * raster) * px * hr_coeff))
        ind_OG_PG = np.argwhere(m == True)
        # del m
    else:
        ind_OG_PG = []

    ####---- PG coordinates
    PG_coor = np.random.choice(px * hr_coeff, size=(Nb_PG,2))
    it = np.ravel_multi_index(np.asarray(PG_coor).T, extracted_cts[:, :, 0].shape)  # 1D index of the coordinates in the image
    coor_verif = extracted_cts[:, :, 0].take(it)  # Extracting the corresponding 16O counts of the coordinates
    while (any(coor_verif < extracted_cts[:, :, 0].max() * th) == True) or (coor_verif in ind_OG_PG):  # If any 16O counts select as the center of a grain is below the masking threshold
        ind_badcoor = np.where(coor_verif < extracted_cts[:, :, 0].max() * th)  # Location of the problematic coordinates
        PG_coor[ind_badcoor] = np.random.choice(px*hr_coeff,size=(len(ind_badcoor),2))  # Replacement of the problematic coordinates
        it = np.ravel_multi_index(np.asarray(PG_coor).T, extracted_cts[:, :, 0].shape)  # Update of the 1D index
        coor_verif = extracted_cts[:, :, 0].take(it)  # Update of the 16O counts
    radius = ((PG_size/2) * 1E-3 / (raster/(px * hr_coeff))).reshape(Nb_PG)  # Radius calculation of the grains in the HR dimensions
    mask_PG = create_circular_mask_multiple(px * hr_coeff, px * hr_coeff, center=PG_coor,radius=radius)  # Mask creation of the grains' pixels
    imhr_ini = np.copy(extracted_cts)  # Copying the HR images channels to avoid altering them

    imhr_ini_PG = np.copy(imhr_ini)  # Copying the modified images



    ####---- Modifying maps counts on location of presolar grains
    PG_delta = np.insert(PG_delta[0],0,[0,0],axis=0) # Ensures the non PG areas remain solar
    R_minor = np.asarray(R[1::])
    imhr_ini_PG[mask_PG !=0, 1::] = extracted_cts[mask_PG !=0, 0][:,None] * np.take((PG_delta*1E-3+1)*R_minor,mask_PG,axis=0)[mask_PG!=0,:]

    ####---- Beam blurr and Boxcar definitions

    # Defining the sigma parameters of the gaussian blurr
    # sig_gaussian=beam_size/(np.sqrt(8*np.log(2))) or beam_size/2.35
    fwhm_hr = np.round((beam_size * 1E-3) / (raster / (px * hr_coeff))) / 2  # the gaussian filter uses the given sigma as a radius for kernel size if radius is not specified
    # fwhm_hr=np.round((beam_size*1E-3)/(raster/(px)))
    # sig_gaussian = fwhm_hr / (np.sqrt(8 * np.log(2)))

    ####---- Beam blurr and Boxcar smoothing
    # Two options :
    # 1. A new image is created for each isotopes from the original ones. Each pixel value is used as a mean for a poisson distribution of which a new pixel value is interpolated. Then the images are beam blurred and boxcar smoothed.
    # 2. We started by applying the gaussian blurr before interpolation new values from a poisson distribution for each pixel of each isotope image. Then the image is boxcar smoothed.

    # Simulation including the presolar grains
    im_poiss = np.random.poisson(imhr_ini_PG)

    #----- Image size reduction with beam blurr then boxcar
    gauss_ker=np.round(fwhm_hr*2).astype(int)
    if gauss_ker%2 !=1: gauss_ker=gauss_ker+1
    boxcar_ker=np.ones((boxcar_px,boxcar_px))/boxcar_px**2
    imgauss_PG = cv2.GaussianBlur(im_poiss*1.0,(gauss_ker,gauss_ker),0) # int32 are not supported by open cv
    imgauss_PG=cv2.resize(imgauss_PG,(px,px),0,0)
    imboxcar_PG = cv2.filter2D(imgauss_PG, cv2.CV_64F, boxcar_ker)
    imgauss_PG.astype(int) # Images are counts so integers
    imboxcar_PG.astype(int)

    ####---- Masking low counts regions
    cts_th = int(imboxcar_PG[:, :,0].max() * th)  # Define criterion as th% of the max encountered for the main isotope counts in one pixel
    mask = np.zeros((px, px))
    mask[imboxcar_PG[:, :,0] < cts_th] = 1  # Set all coordinates of pixels with lower counts than the criterion to 1 in 0 matrix mask.

    main = np.ma.masked_array(imboxcar_PG[:, :, 0], mask=mask)  # Extract main isotope image
    minor1 = np.ma.masked_array(imboxcar_PG[:, :, 1], mask=mask)  # Extract first minor isotope image
    minor2 = np.ma.masked_array(imboxcar_PG[:, :, 2], mask=mask)  # Extract second minor isotope image
    masked_image = np.ma.dstack((main, minor1, minor2))  # Stack the masked images together

    ####---- Ratio, delta and error calculations

    R_1st = minor1 / main
    R_2nd = minor2 / main

    # err_R1=R_1st*np.sqrt(1/minor1+1/main)/boxcar_px
    # err_R2=R_2nd*np.sqrt(1/minor2+1/main)/boxcar_px
    D = minor1
    Dmod = np.where(D < main * R[1], main * R[1], D)
    err_R1 = R_1st * np.sqrt(1 / Dmod + 1 / main) / boxcar_px
    err_R2 = R_2nd * np.sqrt(1 / minor2 + 1 / main) / boxcar_px

    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    # err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
    err_d1 = err_R1 / R[1] * 1000
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000
    err_d2 = err_R2 / R[2] * 1000

    ####---- Sigma images

    # Sigma images will be defined either relative to the standard value or the average ratio of the image
    if standard == "average":
        Rsig = [np.mean(R_1st), np.mean(R_2nd)]
    else:
        Rsig = R[1:]
    # Rsig=

    # Smart error ON
    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(main*Rsig[0])*boxcar_px
    # # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(minor2-main*Rsig[1])/np.sqrt(main*Rsig[1])*boxcar_px #smart error
    # # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    if smart == 1:
        Dmod = np.zeros((main.shape[0], main.shape[1], N_iso - 1))
        Dmod[:, :, 0] = minor1
        Dmod[:, :, 1] = minor2

        for g in range(0, N_iso - 1): Dmod[:, :, g] = np.where(masked_image[:, :, g + 1] < main * Rsig[g],
                                                               main * Rsig[g], minor1)

        imboxcar_sig1st = np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod[:, :, 0]) * 3
        imboxcar_sig2nd = np.abs(minor2 - main * Rsig[1]) / np.sqrt(Dmod[:, :, 1]) * 3
    else:
        imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
        imboxcar_sig2nd = np.abs(R_2nd - Rsig[1]) / err_R2

    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(d_2nd/err_d2)#smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    # plt.subplot()
    # plt.imshow(realcts,cmap='gnuplot2')
    # plt.colorbar()

    if verif == 1:
        # f_verif,[axpoiss,axgauss,axbox]=plt.subplots(1,3,sharex=True,sharey=True)
        f_verif, [axpoiss, axgauss, axbox] = plt.subplots(1, 3)
        axpoiss.imshow(im_poiss[:, :, 1], cmap='gnuplot2')
        axpoiss.set_title('Poisson HR')
        axpoiss.plot(PG_coor[0][0], PG_coor[0][1], 'o', mfc='none', mec='r', markersize=15)
        axgauss.imshow(imgauss_PG[:, :, 1], cmap='gnuplot2')
        axgauss.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                     markersize=15)
        axgauss.set_title('Gaussian blurr')
        axbox.imshow(imboxcar_PG[:, :, 1], cmap='gnuplot2')
        axbox.plot(int(PG_coor[0][0] / hr_coeff), int(PG_coor[0][1] / hr_coeff), 'o', mfc='none', mec='r',
                   markersize=15)
        axbox.set_title('Gaussian + Boxcar')
        plt.suptitle('17O counts', fontsize=15)

    ## Plots

    plots = [imgauss_PG[:, :, 0], imgauss_PG[:, :, 1], imgauss_PG[:, :, 2],
             imboxcar_PG[:, :, 0], imboxcar_PG[:, :, 1], imboxcar_PG[:, :, 2],
             R_1st, d_1st, imboxcar_sig1st,
             R_2nd, d_2nd, imboxcar_sig2nd]
    plots_title = [str(Iso[0]) + ' counts', str(Iso[1]) + ' counts', str(Iso[2]) + ' counts',
                   str(Iso[0]) + ' counts boxcar', str(Iso[1]) + ' counts boxcar', str(Iso[2]) + ' counts boxcar',
                   'Ratio ' + str(Iso[1]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[1]) + '/' + str(Iso[0]),
                   'Ratio ' + str(Iso[2]) + '/' + str(Iso[0]), 'Delta ' + str(Iso[2]) + '/' + str(Iso[0]),
                   'Sigma ' + str(Iso[2]) + '/' + str(Iso[0])]
    # titles=["Raw Simulation",str(beam_size)+" nm Beam Blurr Simulation",str(boxcar_px)+" x "+str(boxcar_px)+" Boxcar and Gaussian Blurr Simulation"]

    fig, axs = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    # axinsert=[]
    palette = 'gnuplot2'

    fontprops = fm.FontProperties(size=12)

    for i in range(0, len(plots)):
        # axins = inset_axes(axs[i], width="60%", height="60%", loc="upper left", bbox_to_anchor=(-0.5,0.2,1,1), bbox_transform=axs[i].transAxes, borderpad=1)
        img_plot = axs[i].imshow(plots[i], cmap=palette, interpolation='None', rasterized=True, )
        # axs[i].plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=15)
        divider = make_axes_locatable(axs[i])
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(img_plot, cax=cax)
        # axs[i,j].title.set_text(titles[h])
        axs[i].set_axis_off()

        plt.title(plots_title[i])

        # axins.imshow(plots[i],cmap=palette,interpolation='None',rasterized=True,)
        # axins.set_xlim(int(PG_coor[0][0]/hr_coeff)-15,int(PG_coor[0][0]/hr_coeff)+15)
        # axins.set_ylim(int(PG_coor[0][1]/hr_coeff)-15, int(PG_coor[0][1]/hr_coeff)+15)
        # axins.get_xaxis().set_visible(False)
        # axins.get_yaxis().set_visible(False)
        # axinsert.append(axins)

        scalebar = AnchoredSizeBar(axs[i].transData,
                                   int(px / raster * 2), r'2 $\mu m$', 'lower center',
                                   pad=0.1,
                                   color='white',
                                   frameon=False,
                                   size_vertical=1,
                                   fontproperties=fontprops)
        axs[i].add_artist(scalebar)

    return fig, axs, plots, plots_title, PG_coor, raster, px  # ,axinsert
