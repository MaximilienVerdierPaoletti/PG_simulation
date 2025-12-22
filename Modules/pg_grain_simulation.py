# -*- coding: utf-8 -*-
"""
Presolar grain simulation functions.

This module contains the core simulation function for generating synthetic
presolar grain images with isotopic ratios.

Created on Fri 3 May 2024

@author: mverdier
"""

# %% Modules
import os
from pickletools import uint8

import numpy as np
import pandas as pd  # enables the use of dataframe
import matplotlib.pyplot as plt  # Enables plotting of data
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.anchored_artists import AnchoredSizeBar
import matplotlib.font_manager as fm
import random
import sims
import cv2

# from scipy.ndimage import gaussian_filter
from mpl_toolkits.axes_grid1.inset_locator import (
    inset_axes,
)  # to insert subplot within plot
from collections.abc import Iterable

# Import synthetic image generation functions
from .synthetic_image_generator import (
    create_circular_mask,
    PG_coor_mask,
    generate_synthetic_image,
)

# Astronomy Specific Imports
from astropy.convolution import convolve as ap_convolve
from astropy.convolution import Box2DKernel


# %% Isotopic ratio extraction function
def Iso_Ratio(elem):
    if not isinstance(elem, list):
        elem = [elem]
    ratio_list = pd.read_excel(
        "Iso_Ratio_Table.xlsx", header=0, sheet_name="isotope ratios"
    )
    R = [ratio_list[ratio_list["var_name"].str.contains(i)]["ratio"] for i in elem]
    return R


# approx_poisson function moved to synthetic_image_generator.py module


# GD_AdamNesperov function moved to gradient_descent.py module


# %% Simulation v6 : For automatization on PG from tables using multiple isotopic ratios simultaneously
def PG_simulationv6(
    file=None,
    elem=None,
    PG_delta=None,
    PG_size=None,
    beam_size=None,
    boxcar_px=None,
    OG_grain=None,
    standard=None,
    smart=None,
    verif=None,
    display="OFF",
):
    # ---------------------- Check inputs

    np.seterr(divide="ignore")

    if display == "OFF":
        plt.ioff()
    else:
        plt.ion()

    # ---------------- Isotopic ratios

    if elem is None:
        elem = "O"
        print("No element specified. Element selected : Oxygen")

    match elem:
        case "O":
            Iso = ["16O", "17O", "18O"]
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case "Mg":
            Iso = ["24Mg", "25Mg", "26Mg"]
            R = [1, Iso_Ratio(Iso[1])[0].item(), Iso_Ratio(Iso[2])[0].item()]
        case _:
            print(
                "The specified element is inadequat. Check if you did not specify an isotope instead of an element."
            )

    # ---------------- Grain info

    if PG_size is None:
        Nb_PG = 1
        PG_size = np.random.randint(100, 600, Nb_PG)
        print("No size specified. Grain diameter fixed to " + str(PG_size) + " nm")
    else:
        Nb_PG = len(PG_size)
        if not isinstance(PG_size, Iterable):
            print("not iterable")
            PG_size = [PG_size]

    if PG_delta is None:
        PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        while PG_delta in range(-200, 200, 1):
            PG_delta = np.random.randint(-1000, 10000, Nb_PG)
        print(
            "No composition specified. Grain composition fixed to "
            + str(PG_delta)
            + " permil"
        )
    else:
        if not isinstance(PG_delta, Iterable):
            PG_delta = [PG_delta]

    # ------------------- Acquisition variables

    if beam_size is None:
        beam_size = 100
        print("No beam size specified. Fixed to " + str(beam_size) + " nm")
    if boxcar_px is None:
        boxcar_px = 3
        print(
            "No boxcar size specified. Fixed to "
            + str(boxcar_px)
            + "x"
            + str(boxcar_px)
        )
    if smart is None:
        smart = 1
        print("Smart error activated")
    if standard is None:
        standard = "average"
        print(
            'Image will be relative to standard terrestrial values. Set "standard" to "average" for delta values relative to the average value of the region.'
        )

    # --------------- Generation of a higher resolution simulated image
    hr_coeff = 8
    N_iso = len(R)
    th = 0.05

    ## Extraction of real image
    if (
        file is None
    ):  # If the file path is not specified then, select randomly an image from the following folder.
        print("No file specified")
        path_realim = (
            "E:/Work/Programmation/Presolar grains/Simulations PG/Real Images/"
        )
        file = random.choice(os.listdir(path_realim))
        file = path_realim + file
    s = sims.SIMS(file)
    # image_header=s.header['Image']
    # raster=image_header['raster']/1000
    # px=image_header['width']
    #
    # isotope_data = [s.data.loc[Iso[i]].sum(axis=0).values for i in range(N_iso)]
    # realcts = np.stack(isotope_data,axis=2).astype(np.uint8)  # Has to be converted into uint8 for the cv2.resize function

    raster = s.header["Image"]["raster"] / 1000
    px = s.header["Image"]["width"]
    realcts = np.zeros((s.header["Image"]["width"], s.header["Image"]["height"], N_iso))
    realcts[:, :, 0] = (
        s.data.loc[Iso[0]].sum(axis=0).values
    )  # extract maps of the main isotope and sum all frames
    realcts[:, :, 1] = s.data.loc[Iso[1]].sum(axis=0).values
    realcts[:, :, 2] = s.data.loc[Iso[2]].sum(axis=0).values
    realcts = realcts / 1

    ####---- Extraction of countrates from Low Resolution (LR) to High Resolution image
    D = np.copy(realcts)  # We copy the image to make sure to not alter the extraction
    extracted_cts = cv2.resize(
        D, dsize=(px * hr_coeff, px * hr_coeff), interpolation=cv2.INTER_AREA
    )  # Interpolate into larger dimensions (here from 256x256 px to 2064x2064 px)

    # Original PG coordinates
    if OG_grain is not None:
        OG_PG_center = [OG_grain.ROIX.item(), OG_grain.ROIY.item()]
        m = create_circular_mask(
            px * hr_coeff,
            px * hr_coeff,
            center=OG_PG_center,
            radius=int(OG_grain.ROIDIAM.iloc[0] / (2 * raster) * px * hr_coeff),
        )
        ind_OG_PG = np.argwhere(m == True)
    else:
        ind_OG_PG = []

    ####---- PG coordinates
    imhr_ini, PG_coor, radius, mask_PG = PG_coor_mask(
        px, hr_coeff, Nb_PG, extracted_cts, th, ind_OG_PG, PG_size, raster
    )

    ####---- Modifying maps counts on location of presolar grains and generating synthetic image
    PG_delta = np.insert(
        PG_delta[0], 0, [0, 0], axis=0
    )  # Ensures the non PG areas remain solar

    # Generate synthetic image with beam blur and boxcar smoothing
    imgauss_PG, imboxcar_PG = generate_synthetic_image(
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
    )

    ####---- Masking low counts regions
    cts_th = int(
        imboxcar_PG[:, :, 0].max() * th
    )  # Define criterion as th% of the max encountered for the main isotope counts in one pixel
    mask = np.zeros((px, px))
    mask[imboxcar_PG[:, :, 0] < cts_th] = (
        1  # Set all coordinates of pixels with lower counts than the criterion to 1 in 0 matrix mask.
    )

    main = np.ma.masked_array(
        imboxcar_PG[:, :, 0], mask=mask
    )  # Extract main isotope image
    minor1 = np.ma.masked_array(
        imboxcar_PG[:, :, 1], mask=mask
    )  # Extract first minor isotope image
    minor2 = np.ma.masked_array(
        imboxcar_PG[:, :, 2], mask=mask
    )  # Extract second minor isotope image
    masked_image = np.ma.dstack(
        (main, minor1, minor2)
    )  # Stack the masked images together

    ####---- Ratio, delta and error calculations
    R_1st = minor1 / main
    R_2nd = minor2 / main

    D_R1 = np.copy(minor1)
    D_R2 = np.copy(minor2)
    Dmod_R1 = np.where(D_R1 < main * R[1], main * R[1], D_R1)
    Dmod_R2 = np.where(D_R2 < main * R[2], main * R[2], D_R2)
    err_R1 = R_1st * np.sqrt(1 / Dmod_R1 + 1 / main) / boxcar_px
    err_R2 = R_2nd * np.sqrt(1 / Dmod_R2 + 1 / main) / boxcar_px

    d_1st = ((minor1 / main) / R[1] - 1) * 1000
    err_d1 = (
        err_R1 / R[1] * 1000
    )  # OR err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
    d_2nd = ((minor2 / main) / R[2] - 1) * 1000
    err_d2 = err_R2 / R[2] * 1000

    ####---- Sigma images

    # Sigma images will be defined either relative to the standard value or the average ratio of the original image (not the simulation with multiple PG)
    if standard == "average":
        cts_th = int(D[:, :, 0].max() * th)
        mask_OG = np.zeros((px, px))
        mask_OG[D[:, :, 0] < cts_th] = (
            1  # Set all coordinates of pixels with lower counts than the criterion to 1 in 0 matrix mask.
        )

        main_OG = np.ma.masked_array(D[:, :, 0], mask=mask_OG)
        minor1_OG = np.ma.masked_array(D[:, :, 1], mask=mask_OG)
        minor2_OG = np.ma.masked_array(D[:, :, 2], mask=mask_OG)

        R1st_OG = minor1_OG / main_OG
        R2nd_OG = minor2_OG / main_OG

        Rsig = [np.mean(R1st_OG), np.mean(R2nd_OG)]
    else:
        Rsig = R[1:]

    # Smart error ON
    # imboxcar_sig1st=np.abs(minor1-main*Rsig[0])/np.sqrt(main*Rsig[0])*boxcar_px
    # # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(minor2-main*Rsig[1])/np.sqrt(main*Rsig[1])*boxcar_px #smart error
    # # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    if smart == 1:
        Dmod = np.zeros((main.shape[0], main.shape[1], N_iso - 1))
        Dmod[:, :, 0] = minor1
        Dmod[:, :, 1] = minor2

        # Replace px whose counts are < to expected poissonian counts
        for g in range(0, N_iso - 1):
            Dmod[:, :, g] = np.where(
                masked_image[:, :, g + 1] < main * Rsig[g],
                main * Rsig[g],
                masked_image[:, :, g + 1],
            )

        imboxcar_sig1st = (
            np.abs(minor1 - main * Rsig[0]) / np.sqrt(Dmod[:, :, 0]) * boxcar_px
        )
        imboxcar_sig2nd = (
            np.abs(minor2 - main * Rsig[1]) / np.sqrt(Dmod[:, :, 1]) * boxcar_px
        )
    else:
        imboxcar_sig1st = np.abs(R_1st - Rsig[0]) / err_R1
        imboxcar_sig2nd = np.abs(R_2nd - Rsig[1]) / err_R2

    # imboxcar_sig1st=np.abs(d_1st/err_d1)
    # imboxcar_sig1st=np.abs((d_1st-np.average(d_1st))/np.std(d_1st))
    # imboxcar_sig2nd=np.abs(d_2nd/err_d2)#smart error
    # sig_boxcar=np.abs((imboxcar_PG-np.average(imboxcar_PG))/np.std(imboxcar_PG))

    if verif == 1:
        realcts_OG = np.copy(realcts)
        # realcts_OG = realcts_OG / 1

        # box_kernel_OG = Box2DKernel(boxcar_px)
        # imboxcar_PG_OG = np.zeros_like(realcts_OG)
        # for i in range(0, 3):
        #     imboxcar_PG_OG[:, :, i] = ap_convolve(realcts_OG[:, :, i], box_kernel_OG, boundary='fill', fill_value=0.0)

        boxcar_ker = np.ones((boxcar_px, boxcar_px)) / boxcar_px**2
        imboxcar_PG_OG = cv2.filter2D(realcts_OG, cv2.CV_64F, boxcar_ker)
        imboxcar_PG_OG.astype(int)

        # ----- Threshold
        cts_th = int(imboxcar_PG_OG[:, :, 0].max() * th)
        mask_OG = np.zeros((px, px))
        mask_OG[imboxcar_PG_OG[:, :, 0] < cts_th] = 1

        main_OG = np.ma.masked_array(imboxcar_PG_OG[:, :, 0], mask=mask)
        minor1_OG = np.ma.masked_array(imboxcar_PG_OG[:, :, 1], mask=mask)
        minor2_OG = np.ma.masked_array(imboxcar_PG_OG[:, :, 2], mask=mask)
        masked_image_OG = np.ma.dstack(
            (main_OG, minor1_OG, minor2_OG)
        )  # Stack the masked images together

        ####---- Ratio, delta and error calculations
        R_1st_OG = minor1_OG / main_OG
        R_2nd_OG = minor2_OG / main_OG
        delta1_OG = ((minor1_OG / main_OG) / R[1] - 1) * 1000
        delta2_OG = ((minor2_OG / main_OG) / R[2] - 1) * 1000

        D_R1_OG = np.copy(minor1_OG)
        D_R2_OG = np.copy(minor2_OG)
        Dmod_R1_OG = np.where(D_R1_OG < main_OG * R[1], main_OG * R[1], D_R1_OG)
        Dmod_R2_OG = np.where(D_R2_OG < main_OG * R[2], main_OG * R[2], D_R2_OG)
        err_R1_OG = R_1st_OG * np.sqrt(1 / Dmod_R1_OG + 1 / main_OG) / boxcar_px
        err_R2_OG = R_2nd_OG * np.sqrt(1 / Dmod_R2_OG + 1 / main_OG) / boxcar_px

        d_1st_OG = ((minor1_OG / main_OG) / R[1] - 1) * 1000
        err_d1_OG = (
            err_R1_OG / R[1] * 1000
        )  # OR err_d1=(1000/(R[1]*main))*np.sqrt((minor1*(minor1+main)/main))
        d_2nd_OG = ((minor2_OG / main_OG) / R[2] - 1) * 1000
        err_d2_OG = err_R2_OG / R[2] * 1000

        if smart == 1:
            label_err = "ON"
            Dmod_OG = np.zeros((main_OG.shape[0], main_OG.shape[1], N_iso - 1))
            Dmod_OG[:, :, 0] = minor1_OG
            Dmod_OG[:, :, 1] = minor2_OG

            for g in range(0, N_iso - 1):
                Dmod_OG[:, :, g] = np.where(
                    masked_image_OG[:, :, g + 1] < main_OG * Rsig[g],
                    main_OG * Rsig[g],
                    masked_image_OG[:, :, g + 1],
                )

            imboxcar_sig1st_OG = (
                np.abs(minor1_OG - main_OG * Rsig[0])
                / np.sqrt(Dmod_OG[:, :, 0])
                * boxcar_px
            )
            imboxcar_sig2nd_OG = (
                np.abs(minor2_OG - main_OG * Rsig[1])
                / np.sqrt(Dmod_OG[:, :, 1])
                * boxcar_px
            )
        else:
            label_err = "OFF"
            imboxcar_sig1st_OG = np.abs(R_1st_OG - Rsig[0]) / err_R1_OG
            imboxcar_sig2nd_OG = np.abs(R_2nd_OG - Rsig[1]) / err_R2_OG

        # Figure
        f_OG, [[axR1, axd1, axsig1], [axR2, axd2, axsig2]] = plt.subplots(
            2, 3, figsize=(12, 12), sharex=True, sharey=True
        )
        im0 = axR1.imshow(R_1st_OG, cmap="gnuplot2")
        divider = make_axes_locatable(axR1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axR1.set_title("Ratio 17O")
        f_OG.colorbar(im0, cax=cax)

        im1 = axd1.imshow(delta1_OG, cmap="gnuplot2")
        divider = make_axes_locatable(axd1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axd1.set_title("delta 17O")
        f_OG.colorbar(im1, cax=cax)
        # plt.plot(int(PG_coor[0][0]/hr_coeff),int(PG_coor[0][1]/hr_coeff),'o',mfc='none',mec='r',markersize=20)

        # plt.figure()
        im2 = axsig1.imshow(imboxcar_sig1st_OG, cmap="gnuplot2")
        divider = make_axes_locatable(axsig1)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axsig1.set_title("sigma 17O, smart error: " + label_err)
        f_OG.colorbar(im2, cax=cax)
        # plt.colorbar()
        # plt.plot(PG_coor[0]/256,'.r',markersize=15)

        im3 = axR2.imshow(R_2nd_OG, cmap="gnuplot2")
        divider = make_axes_locatable(axR2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axR2.set_title("Ratio 18O")
        f_OG.colorbar(im3, cax=cax)

        im4 = axd2.imshow(delta2_OG, cmap="gnuplot2")
        divider = make_axes_locatable(axd2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axd2.set_title("delta 18O")
        f_OG.colorbar(im4, cax=cax)

        im5 = axsig2.imshow(imboxcar_sig2nd_OG, cmap="gnuplot2")
        divider = make_axes_locatable(axsig2)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        axsig2.set_title("sigma 18O")
        f_OG.colorbar(im5, cax=cax)

        # plt.suptitle(file, fontsize=15)

    ## Plots

    plots = [
        imgauss_PG[:, :, 0],
        imgauss_PG[:, :, 1],
        imgauss_PG[:, :, 2],
        imboxcar_PG[:, :, 0],
        imboxcar_PG[:, :, 1],
        imboxcar_PG[:, :, 2],
        R_1st,
        d_1st,
        imboxcar_sig1st,
        R_2nd,
        d_2nd,
        imboxcar_sig2nd,
    ]
    plots_title = [
        str(Iso[0]) + " counts",
        str(Iso[1]) + " counts",
        str(Iso[2]) + " counts",
        str(Iso[0]) + " counts boxcar",
        str(Iso[1]) + " counts boxcar",
        str(Iso[2]) + " counts boxcar",
        "Ratio " + str(Iso[1]) + "/" + str(Iso[0]),
        "Delta " + str(Iso[1]) + "/" + str(Iso[0]),
        "Sigma " + str(Iso[1]) + "/" + str(Iso[0]),
        "Ratio " + str(Iso[2]) + "/" + str(Iso[0]),
        "Delta " + str(Iso[2]) + "/" + str(Iso[0]),
        "Sigma " + str(Iso[2]) + "/" + str(Iso[0]),
    ]
    # titles=["Raw Simulation",str(beam_size)+" nm Beam Blurr Simulation",str(boxcar_px)+" x "+str(boxcar_px)+" Boxcar and Gaussian Blurr Simulation"]

    fig, axs = plt.subplots(4, 3, figsize=(12, 12), sharex=True, sharey=True)
    axs = axs.ravel()
    # axinsert=[]
    palette = "gnuplot2"

    fontprops = fm.FontProperties(size=12)

    for i in range(0, len(plots)):
        # axins = inset_axes(axs[i], width="60%", height="60%", loc="upper left", bbox_to_anchor=(-0.5,0.2,1,1), bbox_transform=axs[i].transAxes, borderpad=1)
        img_plot = axs[i].imshow(
            plots[i],
            cmap=palette,
            interpolation="None",
            rasterized=True,
        )
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

        scalebar = AnchoredSizeBar(
            axs[i].transData,
            int(px / raster * 2),
            r"2 $\mu m$",
            "lower center",
            pad=0.1,
            color="white",
            frameon=False,
            size_vertical=1,
            fontproperties=fontprops,
        )
        axs[i].add_artist(scalebar)
    if verif == 1:
        return fig, axs, plots, plots_title, PG_coor, raster, px, f_OG
    return fig, axs, plots, plots_title, PG_coor, raster, px, None  # ,axinsert

