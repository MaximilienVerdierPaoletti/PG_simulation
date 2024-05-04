# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 10:27:25 2023

Version 1.0:
Uses image files and excel database of those files to automatically estimates dilution on size and  one isotope ratio.

Version 2.0:
Enables to explore multiple isotpic ratios per grains at once.

Version 3.0:
Implement gradient descent methodoly

@author: Maximilien Verdier-Paoletti
"""

#%% Modules

try:
    from IPython import get_ipython
    get_ipython().run_line_magic('reset','-f')
except:
    pass

import sys
# # sys.path.insert(0, 'F:/Work/Programmation/Presolar grains/Python functions/')
# sys.path.insert(0,'F:/Work/Programmation/Presolar grains/Simulations PG/')

import os
import numpy as np
import pandas as pd  # enables the use of dataframe
import matplotlib
import matplotlib.pyplot as plt  # Enables plotting of data
from matplotlib.backends.backend_pdf import PdfPages
import skimage.measure
from matplotlib.lines import Line2D
from matplotlib import cm
import time
import tkinter as tk
from tkinter import filedialog

from PG_simulations_func import PG_simulationv6, create_circular_mask


# plt.ioff()
matplotlib.rcParams['interactive'] = False


def GD_AdamNesperov(target, measured_simulations, initial_simulations, learning_rate):
    #FIXME : Pour les normes il est préférable d'utiliser des valeurs absolue plutôt que des racines carrés en terme de coût machine

    X = measured_simulations[0]
    Y = measured_simulations[1]
    Z = measured_simulations[2]
    #
    norm3D = np.asarray(((X-target[0])**2+(Y-target[1])**2+(Z-target[2])**2)**0.5) #FIXME might not be the best suited norm because of different unit and dimension. Might have to normalize
    # grad = np.array([(X-target[0])/norm3D, (Y-target[1])/norm3D, (Z-target[2])/norm3D])

    norm3D_norm = np.asarray((((X-target[0])/target[0])**2 +
                         ((Y-target[1])/target[1])**2 +
                         ((Z-target[2])/target[2])**2)**0.5)
    grad = np.array([(X-target[0])/(norm3D_norm*target[0]**2),
                     (Y-target[1])/(norm3D_norm*target[1]**2),
                     (Z-target[2])/(norm3D_norm*target[2]**2)])

    return initial_simulations.T - learning_rate, norm3D_norm, grad





if __name__ == "__main__":
    plt.close("all")

#%% Grains and acquisition characteristics
    #---- Grain parameters for simulations
    Nb_PG = 9
    nb_closest_match = 3  # Can't be bigger than Nb_PG
    num_final_selection = 6  # Number of closest matches select amongst all the simulation
    beam = 100
    boxcar = 3
    elem = 'O'
    N_mod=np.array([3, 4, 5, 6, 7, 8, 9])

    pdf_save = 'test_v2'

    delta_database = [*range(-900, 0, 50)]
    delta_database.extend(range(0, 200, 10))
    delta_database.extend(range(200, 2000, 100))
    delta_database.extend(range(2000, 21000, 1000))

    #---- Number of outer and inner iterations
    iterations = 3
    zoom_iteration = 10
    c = cm.rainbow(np.linspace(0, 1, Nb_PG))

    #---- Legend of summary figure (fres) for each measured grain
    lines = []
    labels = []
    point_inner = Line2D([0], [0], marker='s', mfc='None', mec='k', linestyle='', markersize=10)
    point_matchfinal = Line2D([0], [0], marker='o', mfc='None', mec='g', linestyle='', markersize=10)
    label_points = ['Best matches of this iteration', 'Best matches all iterations']

    #--- Initialization of PDF figure summary
    pp = PdfPages(pdf_save+'.pdf')

    # -----------------------------------------------------------------#
    #Data call
    root = tk.Tk()
    root.lift()
    root.attributes('-topmost', True)
    root.after_idle(root.attributes, '-topmost', False)

    file_list = filedialog.askopenfilename(multiple=True)
    root.withdraw()

    data = filedialog.askopenfilename()
    root.withdraw()
    data = pd.read_excel(data, header=0)
    data = data[data.NAME.str.contains("Bulk") == False]  # Drop bulk ROIs

    #---- Saving variable allocation
    all_data = []
    # Ratio_names=data.columns[data.columns.str.contains('^d-.*'+elem)].str.replace('d-','').to_list()
    Ratio_names=data.columns[data.columns.str.contains('^d-.*'+elem)].to_list()
    col = ['Image', 'Outer Iteration', 'Inner Iteration','Simulated grain index', 'Grain', 'Initial grain size (nm)', 'sigma R', 'Measured diameter']
    col.extend(Ratio_names)


    col_res=['Grain', 'sigma_r', 'Measured diameter',
             'Simu true delta', 'Simu true radius (nm)', 'Simu measured radius (nm)']
    for m in range(len(Ratio_names)):
        s = Ratio_names[m]
        col_res.extend(['Simu measured '+s, 'std', 'Dilution on '+s+' (%)'])
        col_res.insert(3+m, s)
        col.insert(4+m, 'Initial '+s)
    col_res.append('Dilution on size (%)')

    summary = pd.DataFrame(columns=col)

    # -----------------------------------------------------------------#
    #### Grain characteristics extraction
    start = time.time()
    for file in file_list:
        imagename = file.rsplit('/', 1)[1].replace('.im', '')
        if data.NAME.str.contains('_corr').any() ==False: imagename = imagename.replace('_corr', '')
        grains = data.loc[data.NAME.str.contains(imagename)]

        if grains.empty is True:
            print(f'No presolar grain were detected prior by the user in acquisition {imagename}')
            continue
        else:
            print(f'{grains.shape[0]} presolar grain detected by user in {imagename}')

        for z in range(grains.shape[0]): # Loop on number of detected grains by user in this file
            grain = grains.iloc[[z]]
            grain_delta = grain.filter(regex="^d-.*" + elem,axis=1)  # Extract automatically the delta composition of the grain based on the specified element
            ER_grain_delta = grain.filter(regex="^ER-d-.*" + elem,axis=1)
            grain_size = grain['ROIDIAM'].item() * 1000

            if grain.columns.str.contains('sig', case=False).any():
                sig_r = grain.iloc[:, np.where(grain.columns.str.contains('sig') == True)]
            else:
                sig_r = 0.5
                print(f'No information available on sigma ratio, fixed to {sig_r}')

            delta_range=[]
            for i in grain_delta.to_numpy()[0]:
                delta_range.append([h for h in delta_database if np.sign(i)*h > np.abs(i)])

            size_range = range(50, 900, 50)

#%% Loops on simulations

            it = 0
            fres, axres = plt.subplots(2, iterations,subplot_kw={"projection": "3d"})
            f_adnesp, ax_adnesp = plt.subplots(4,3)

            for k in range(0, iterations):
                axres[0, k].plot(grain_size, grain_delta.iloc[:, 0].item(),grain_delta.iloc[:, 1].item(), 'sk', markersize=12,
                                 label='Measured presolar grain',zorder=10)
                axres[1, k].plot(grain_size, grain_delta.iloc[:, 0].item(),grain_delta.iloc[:, 1].item(), 'sk', markersize=12,
                                 label='Measured presolar grain',zorder=10)

                # -----------------------------------------------------------------#
                # Grain simulations conditions

                PG_delta = [[[np.random.choice(i) for i in delta_range] for o in range(Nb_PG)]]  # Randomly select grain composition from initial ranges with shape
                PG_size = np.random.choice(size_range, Nb_PG).reshape(1, Nb_PG)

                #---- Gradient descent parameters
                eta = 10**np.round(np.log10(np.abs(np.concatenate((PG_size.T,np.array(PG_delta).reshape(Nb_PG,len(Ratio_names))), axis=1))))/10
                learning_rate = eta
                eps = 1E-8
                beta_decay = 0.9
                beta_momentum = 0.6
                decay_mat = np.zeros((3,Nb_PG)) # number of parameters (size, ratio1, ratio 2) x Nb of grains
                decay_adam = np.zeros((3,Nb_PG))
                momentum_mat = np.zeros((3,Nb_PG))
                momentum_adam = np.zeros((3,Nb_PG))

                for j in range(0, zoom_iteration):
                    for u in range(0, PG_size.shape[0]):
                        # ----------- Simulation of PG images

                        f, ax, plots, plots_title, PG_coor, raster, px = PG_simulationv6(file=file, elem=elem,
                                                                                         PG_delta=PG_delta,
                                                                                         PG_size=PG_size[u, :],
                                                                                         OG_grain=grain,
                                                                                         beam_size=beam, boxcar_px=boxcar,
                                                                                         smart=1, standard='average',
                                                                                         display='OFF')

                        ax = ax.ravel()

                        # Sigma and Delta map selection based on anomalous ratio
                        ind_anomalous = np.argmax(np.abs(grain_delta))  # Locate most anomalous ratio. It will be the one used for contouring
                        anomalous_ratio_name = grain_delta.columns[ind_anomalous].replace('d-', '')
                        sigma_map_index = [plots_title.index(n) for n in plots_title if "Sigma" in n]
                        delta_map_index = [plots_title.index(n) for n in plots_title if "Delta" in n]
                        sigma_anomalous_map_index = [sigma_map_index.index(n) for n in sigma_map_index if anomalous_ratio_name in plots_title[n]]
                        delta_anomalous_map_index = [delta_map_index.index(n) for n in delta_map_index if anomalous_ratio_name in plots_title[n]]
                        sigma_map = [plots[n] for n in sigma_map_index]
                        delta_map = [plots[n] for n in delta_map_index]

                        for i in range(0, PG_size.shape[1]):
                            radius = (np.asarray(PG_size[u, i]) / 2) * 1E-3 / (raster / px) * 1.5
                            mask = create_circular_mask(px, px, center=np.floor_divide(PG_coor, 8)[i],radius=radius)  # Mask creation of the grains' pixels
                            contour = skimage.measure.find_contours(mask != 0, 0.5)

                            ysel, xsel = contour[0].T

                            x, y = np.nonzero(mask)
                            I = np.where(sigma_map[sigma_anomalous_map_index[0]][x, y].data >= sigma_map[sigma_anomalous_map_index[0]][x, y].data.max() * sig_r)
                            X = x[I]
                            Y = y[I]
                            mask_th = np.zeros_like(mask)
                            mask_th[X, Y] = 1

                            Diam = np.sqrt(len(delta_map[delta_anomalous_map_index[0]][X, Y]) * ((raster / px) ** 2) / np.pi) * 1000 *2

                            axres[0, k].plot(Diam, np.mean(delta_map[0][X, Y]), np.mean(delta_map[1][X, Y]), 'o', color=c[i], markersize=10, alpha=0.5)
                            axres[1, k].plot(Diam, np.mean(delta_map[0][X, Y]), np.mean(delta_map[1][X, Y]), 'o', color=c[i], markersize=10, alpha=0.5)

                            contour = skimage.measure.find_contours(mask_th == 1, 0.5)
                            y, x = contour[0].T
                            ax = ax.ravel()
                            for h in range(0, len(ax)):
                                ax[h].plot(xsel, ysel, '--', color='w', linewidth=2)
                                ax[h].plot(x, y, '-', color='r', linewidth=2)
                                # axinsert[k].plot(x,y,'-',color='r',linewidth=2)

                            # final=r'$\sigma_{R}$ ='+str(sig_r[j])+r', $\delta^{17}O$ ='+str(int(np.mean(delta_map[X,Y])))+ r', $\delta^{18}O$ ='+str(int(np.mean(plots[10][X,Y])))+', Diameter ='+str(int(R*2))

                            # plt.gcf().text(0.7,0.98-j*0.01,final,fontsize=9)

                            S = pd.Series([imagename, k, j,i,grain.NAME, PG_size[u, i], sig_r,
                                           Diam, np.round(np.mean(delta_map[0][X, Y]), 2), np.round(np.mean(delta_map[1][X, Y]), 2)])
                            S = S.to_frame().T
                            for o in range(len(PG_delta[u][i])): S.insert(4+o, '', PG_delta[u][i][o], allow_duplicates=True)
                            S = S.set_axis(col, axis=1)
                            summary = pd.concat([summary, S], axis=0, ignore_index=True)

                        initial = 'Outer Iteration : ' + str(k) + '\n' + 'Inner Iteration : ' + str(j) + '\n' "Presolar Grain Simulation #" + str(
                            it) + '\n' + 'Image is: ' + str(file.rsplit('/', 1)[1]) + '\n' + 'Number of grains : ' + str(
                            PG_size.shape[1])

                        f.text(0.15, 0.92, initial, fontsize=11)

                        f.set_size_inches(16, 10)
                        it =+ 1

                        pp.savefig(f, transparent=True, dpi=100)
                        plt.close(f)

                        all_data.append(plots)

                    sim_outerin = summary.loc[(summary['Outer Iteration']==k) & (summary['Inner Iteration']==j)]

                    target = np.array([grain_size, grain_delta['d-17O/16O'].item(), grain_delta['d-18O/16O'].item()])
                    measured_simulations = np.array([sim_outerin['Measured diameter'], sim_outerin['d-17O/16O'], sim_outerin['d-18O/16O']])
                    initial_simulations = np.array([sim_outerin['Initial grain size (nm)'], sim_outerin['Initial d-17O/16O'], sim_outerin['Initial d-18O/16O']])

                    [new_simu, norm3D,grad] = GD_AdamNesperov(target, measured_simulations, initial_simulations, learning_rate)
                    decay_mat = decay_mat * beta_decay + (1-beta_decay) * grad**2
                    decay_adam = decay_mat / (1-beta_decay**(j+1))
                    momentum_mat = beta_momentum * momentum_mat + (1-beta_momentum) * grad
                    momentum_adam = momentum_mat / (1-beta_momentum**(j+1))
                    # learning_rate = (eta.T * momentum_adam / (decay_adam+eps)**0.5).T  # Learning rate for Adam Protocol
                    momentum_nesperov_adam = beta_momentum * momentum_adam + (1-beta_momentum) * grad
                    learning_rate = (eta.T * momentum_nesperov_adam / (decay_adam+eps)**0.5).T # Learning rate for Adam Nesperov protocol

                    new_simu.T[0] = np.where(new_simu.T[0] < 50, 100, new_simu.T[0])
                    new_simu.T[1::] = np.where(new_simu.T[1::] <= -1000, -999, new_simu.T[1::])

                    # Study of the behavior of parameters in gradient descent
                    for m in range(9): # Loop on simulated grains
                        ax_adnesp[0,k].plot(j,norm3D[m],'o',mec=c[m],mfc='none',linewidth=3)
                        ax_adnesp[1,k].plot(j,decay_adam[0].reshape(1,9)[0,m],'o',mec=c[m],mfc='none',linewidth=3)
                        ax_adnesp[1,k].plot(j,momentum_nesperov_adam[0].reshape(1,9)[0,m],'s',mec=c[m],mfc='none',linewidth=3)
                        ax_adnesp[2,k].plot(j,decay_adam[1].reshape(1,9)[0,m],'o',mec=c[m],mfc='None',linewidth=3)
                        ax_adnesp[2,k].plot(j,momentum_nesperov_adam[1].reshape(1,9)[0,m],'s',mec=c[m],mfc='None',linewidth=3)
                        ax_adnesp[3,k].plot(j,decay_adam[2].reshape(1,9)[0,m],'o',mec=c[m],mfc='None',linewidth=3)
                        ax_adnesp[3,k].plot(j,momentum_nesperov_adam[2].reshape(1,9)[0,m],'s',mec=c[m],mfc='None',linewidth=3)


                    del PG_size, PG_delta

                    PG_size = new_simu[:,0].reshape(1,new_simu.shape[0])
                    PG_delta = np.c_[new_simu[:,1],new_simu[:,2]]
                    PG_delta = np.where(PG_delta <= -1000, -999, PG_delta)
                    PG_delta = [PG_delta.tolist()]

                # Look for the closest match in all simulation of this outer iteration
                #FIXME change norm formulation (Cf gradient descent)
                sim_outer = summary.loc[summary['Outer Iteration'] == k]
                norm_outer = ((sim_outer['Measured diameter'].divide(grain_size) - 1) ** 2 +
                                 ((sim_outer[Ratio_names].div(grain_delta.values)-1)**2).sum(axis=1)) ** 0.5
                closest_match_index = norm_outer.argsort()[0:nb_closest_match]
                closest_match = sim_outer.iloc[closest_match_index, :]

                axres[0, k].plot(closest_match['Measured diameter'], closest_match[Ratio_names[0]], closest_match[Ratio_names[1]], 's', mec='k', mfc='None',
                                 markersize=10, zorder=5, linewidth=10)
                axres[1, k].plot(closest_match['Measured diameter'], closest_match[Ratio_names[0]], closest_match[Ratio_names[1]], 's', mec='k', mfc='None',
                                 markersize=10, zorder=5, linewidth=10)
                axres[0, k].set_title('Iteration #' + str(k), fontsize=12)


                ax_adnesp[0, k].set_title('Outer iteration : ' + str(k), fontsize=18)
                ax_adnesp[3, k].set_xlabel('Inner iteration', fontsize=15)

            # Look for the closest match throughout all the iterations
            norm = ((summary['Measured diameter'].divide(grain_size) - 1) ** 2 +
                             ((summary[Ratio_names].div(grain_delta.values)-1)**2).sum(axis=1)) ** 0.5
            closest_match_index = norm.argsort()[0:num_final_selection]
            closest_match_final = summary.iloc[closest_match_index, :]

            # for ax in axres:
            #     ax.plot(closest_match_final['Measured Radius']*2,closest_match_final['d17O'],'o',mec='g',mfc='None',markersize=10,zorder=5,linewidth=2)

            for l in range(0, iterations):
                axres[0, l].plot(closest_match_final['Measured diameter'], closest_match_final[Ratio_names[0]], closest_match_final[Ratio_names[1]], 'o', mec='g', mfc='None',
                                 markersize=10, zorder=5, linewidth=5)
                axres[1, l].plot(closest_match_final['Measured diameter'], closest_match_final[Ratio_names[0]], closest_match_final[Ratio_names[1]], 'o', mec='g', mfc='None',
                                 markersize=10, zorder=5, linewidth=5)
                axres[1, l].set_xlim3d([int(grain_size * 0.7), int(grain_size * 1.3)])
                axres[1, l].set_ylim3d([int(grain_delta[Ratio_names[0]].item() * 0.5), int(grain_delta[Ratio_names[0]].item() * 1.5)])
                axres[1, l].set_zlim3d([int(grain_delta[Ratio_names[1]].item() * 0.5), int(grain_delta[Ratio_names[1]].item() * 1.5)])

                axres[0, l].set_xlabel('Grain diameter (nm)', fontsize=14)
                axres[0, l].set_ylabel(Ratio_names[0], fontsize=14)
                axres[0, l].set_zlabel(Ratio_names[1], fontsize=14)
                axres[1, l].set_xlabel('Grain diameter (nm)', fontsize=14)
                axres[1, l].set_ylabel(Ratio_names[0], fontsize=14)
                axres[1, l].set_zlabel(Ratio_names[1], fontsize=14)


            lines.extend((point_inner, point_matchfinal))
            labels.extend(label_points)
            fres.suptitle(imagename+'\n grain : '+str(grain.NAME.item()),fontsize=15)
            axres[0, 0].legend(lines, labels, loc='best',ncol=2)


            # f_norm.suptitle(imagename+'\n grain : '+str(grain.NAME.item()),fontsize=15)
            # f_norm.set_size_inches(16, 10)
            # pp.savefig(f_norm, transparent=True, dpi=100)
            # plt.close(fres)

            if 'match_summary' not in globals():
                match_summary = closest_match_final
            else:
                match_summary = pd.concat([match_summary, closest_match_final])

            # Result variables
            Diam_res = closest_match_final['Initial grain size (nm)'].mean()
            ER_Diam_res = match_summary['Initial grain size (nm)'].std()
            Delta_res = match_summary.filter(regex='Initial d-*').mean()
            ER_Delta_res = match_summary.filter(regex='Initial d-*').std()
            Dilu_size = np.round((1-grain_size/Diam_res)*100,2)
            Dilu_delta = (1-grain_delta.div(Delta_res.values))*100
            ER_Dilu_delta = np.abs((Dilu_delta-100)*(np.divide(ER_Delta_res.values,Delta_res.values)**2+np.divide(ER_grain_delta.values,grain_delta.values)**2)**0.5)


            data_res = pd.DataFrame({'Estimated true diameter (nm)' : [int(Diam_res)], 'sigma': [int(ER_Diam_res)],'Estimated '+Ratio_names[0]: [int(Delta_res.iloc[0])],
                                     'sigma': [int(ER_Delta_res.iloc[0])], 'Estimated '+Ratio_names[1]: [int(Delta_res.iloc[1])], 'sigma': [int(ER_Delta_res.iloc[0])],
                                     'Dilution on size (%)': [Dilu_size], 'Dilution on ' + Ratio_names[0] + '(%)': [Dilu_delta.iloc[:,0]],
                                     'Dilution on ' + Ratio_names[1] + '(%)': [Dilu_delta.iloc[:,1]]})

            #FIXME create excel file at the end

            print(f'Estimated size {int(Diam_res)} nm ({Dilu_size} %) and compositions {"/".join(map(str, Delta_res.astype(int).values))} permil '
                  f'({str().join(map(str, Dilu_delta.astype(int).values))} %)')

            del closest_match, closest_match_index


        ax_adnesp[0, 0].set_ylabel('Norm',fontsize=15)
        ax_adnesp[1, 0].set_ylabel('Size',fontsize=15)
        ax_adnesp[2, 0].set_ylabel(Ratio_names[0],fontsize=15)
        ax_adnesp[3, 0].set_ylabel(Ratio_names[1],fontsize=15)
        for l in range(3): ax_adnesp[0,l].set_yscale('log')
        f_adnesp.suptitle('Evolution of the gradient descent parameters', fontsize=22)



        S=summary.loc[summary['Outer Iteration']==1]
        f, ax = plt.subplots(1, 1,subplot_kw={"projection": "3d"})
        ax.scatter(grain_size, grain_delta.iloc[:, 0].item(),grain_delta.iloc[:, 1].item(), 's',c='k',s=80)
        cmap=matplotlib.colors.ListedColormap(c)
        for i in S['Simulated grain index'].unique():
            Ssel=S.loc[S['Simulated grain index']==i]
            ax.scatter(Ssel['Measured diameter'], Ssel['d-17O/16O'], Ssel['d-18O/16O'], 'o-',s=200-i*20,linestyle='dashed',c=Ssel['Inner Iteration'],cmap=cmap,alpha=0.7,label=Ssel['Inner Iteration'])
            ax.plot(Ssel['Measured diameter'], Ssel['d-17O/16O'], Ssel['d-18O/16O'], '--', color=c[i])


        fres.set_size_inches(16, 10)
        f_adnesp.set_size_inches(16, 10)
        pp.savefig(fres, transparent=True, dpi=100)
        pp.savefig(f_adnesp, transparent=True, dpi=100)


    plt.ion()
    plt.show()

    # #------------ Saving the results of simulation

    pp.close()

    end = time.time()
    print('Elapsed time: ' + str(end - start) + ' s')

