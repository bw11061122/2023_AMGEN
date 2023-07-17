#!/usr/bin/env python3

# 2023-07-17
# This script takes a grid from COMSOL (image / sollution to simulation > export > regular grid)
# and converts it to a numerical image 
# This is the final script suitable for analysis of a series of images
# THIS IS THE FILE THAT YOU WANT TO USE 
# you need to supply: the file with your parameter screen
# parameters used are extracted and identified automatically 
# all results / plots saved will include date and filename supplied + parameters used in the screen 
# to run: python3 ~/Desktop/2023_internship_work/2D_simulations/scripts/20230711_image_segment_parameter_sweep_filename.py your_file.txt 
# python3 ~/Desktop/2023_internship_work/2D_simulations/scripts/20230717_image_segment_analysis_2paramsweep_filename.py ~/Desktop/2023_internship_work/2D_simulations/data/20230712_2D_carillo_sweep.txt

# %%
# import necessary packages for data analysis 
from curses.ascii import FF
from xml.dom.expatbuilder import parseString
import pandas as pd
import numpy as np
import skimage 
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from skimage.segmentation import clear_border
from skimage.measure import label, regionprops, regionprops_table
from skimage.color import label2rgb
from skimage import util 
import re 

# get the time the script was run 
import time
timestr = time.strftime("%Y%m%d") # get current date 

# get the filename 
import sys
sys.path.append("~/Desktop/2023_internhsip_work/2D_simulations/scripts/") ## if not in current working directory

my_file = sys.argv[1] # sys[0] is the first argument and it is the name of your file 
fp = open(my_file)

import os.path
file_name=os.path.basename(sys.argv[1]).split('.')[0]

# access functions (Malte's script)
from import_and_convert_comsol_grid_update import (import_pointcloud_from_comsol, pointcloud_to_image) 
# import_and_convert_comsol_grid is a file I received from Malte (see Barbara_Walkowiak folder on Iber drive > 
# /Volumes/iber/Users/Barbara_Walkowiak/import_and_convert_comsol_grid.py)
# it contains the two imported functions which I am using for image processing 
# I also have this file in the same directory as this script (/Desktop/2023_internship_work/2D_simulations/scripts)

# %%
## read the file into a pandas dataframe
comsol_df = import_pointcloud_from_comsol(fp)

# %%
# find the names of the parameters that were used in your simulation
## two first columns are X and Y and we don't want to consider them 
col_str = comsol_df.columns[3]
param = re.findall(r"\b([a-zA-Z])=\d+\.?\d*", col_str)
excluded = [0] # remove t which is always the first and also marked with the same pattern (index 0)
param = [x for i, x in enumerate(param) if i not in excluded]

columns = comsol_df.columns[2:] # columns which store value of parameters used in the simulation
parameters = [] # create an empty list 

for p in param:
    values = columns.str.extract(r"" +p+ r"=([^,]*)") # extract values of the specific parameter from column
    values.columns = ['a']
    values = values['a'].tolist()
    values = [x for x in values if x == x]
    parameters.append([values]) # add list to list of lists

# "parameters" is a list of lists that stores values of a and R (or any other parameters you will have)

# %%
image_df = pointcloud_to_image(comsol_df, mesh = None)
# returns a list of images from parameter sweep 

# %%
# will append results to this df (empty)
prop_df_results = pd.DataFrame([])

# %%
## processing each image in the image_df list of images 
for idx, df in enumerate(image_df):

    # identify parameters used in the simulation
    print(idx)
    par1 = param[0] # name of the first parameter
    par2 = param[1] # name of the second parameter
    val1 = parameters[0][0][idx] # value of the first parameter
    val2 = parameters[1][0][idx] # value of the second parameter 
    val = [parameters[0][0][idx], parameters[1][0][idx]]
    val_str = "_".join(map(str, val))

    threshold = skimage.filters.threshold_otsu(image=df) # threshold image 
    binary = df > threshold # binary image 
    
    # convert binary True-False to 1-0 format 
    def bool_to_float(v):
        if v == 'True':
            return '1'
        elif v == 'False':
            return '0'
        else:
            return v
    binary_num = np.vectorize(bool_to_float)(binary).astype(float)

    # convert white to black background such that regions of interest are white and will be labelled 
    whitePx = np.count_nonzero([binary_num]) #number of black pixels
    blackPx = binary_num.size - whitePx
    if whitePx > blackPx:
        # this means that most of the image is white and you want to invert it 
        binary_fin = np.where((binary_num==0)|(binary_num==1), 1-binary_num, binary_num)
    else:
        binary_fin = binary_num

    # compare before and after thresholding 
    fig, axes = plt.subplots(ncols=2, figsize=(6, 3))
    ax = axes.ravel()
    ax[0].imshow(df, cmap=plt.cm.gray)
    ax[0].set_title('Original image')
    ax[1].imshow(binary_fin, cmap=plt.cm.gray)
    ax[1].set_title('Result after applying threshold')
    for ab in ax:
        ab.axis('off')
    plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + '_' + file_name +  '_thresholding_' + str(val_str) + '_' + str(par1) + '_' + str(par2) + '.pdf')
  
    # label the finalised segmented binary image (only regions > 50 to avoid dealing with artefacts)
    # from skimage import morphology 
    # from skimage.util import img_as_int
    # binary_int = img_as_int(binary_fin)
    # clean_binary = morphology.remove_small_objects(binary_int, 500)
    labelled = label(binary_fin)

    # automatically measure properties of labelled regions
    properties = skimage.measure.regionprops(labelled)
    prop_of_interest = ['label', 'area', 'area_bbox', 'orientation', 'centroid', 'axis_major_length', 'axis_minor_length', 'solidity','eccentricity']
    prop_of_interest_nl = ['area', 'area_bbox', 'orientation', 'axis_major_length', 'axis_minor_length', 'solidity','eccentricity']

    props = regionprops_table(labelled, properties=prop_of_interest)
    prop_df = pd.DataFrame(props)

    nr_labels = prop_df.shape[0] # number of rows of the datacolumn

    # add parameter as a column
    prop_df[str(par1)] = val1
    prop_df[str(par2)] = val2
    prop_df["nr_labels"] = nr_labels
    prop_df_results = pd.concat([prop_df_results, prop_df])

    # save as CSV (for plotting etc.)
    prop_df.to_csv('~/Desktop/2023_internship_work/2D_simulations/results/measurements/' + timestr + '_' + file_name + '_region_properties_' + str(par1) + '_' + str(par2) + '_' + str(val_str) + '.csv') 
   
    # analysis of parameters for each simulation (not actually using this - not very useful I guess)
    # for poi in prop_of_interest_nl:
    #     y_col = prop_df[poi]
    #     fig, ax = plt.subplots()
    #     ax.bar(range(len(y_col)), y_col)
    #     ax.set_title(f"Plot for Variable: {poi}")
    #     ax.set_xlabel("Region")
    #     ax.set_ylabel(f"Value of {poi}")
    #     plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + '_' + file_name + '_' + str(poi) + '_'+ str(par1) + '_' + str(par2) + '_' + str(val_str) + '.pdf')
    
    # measure distances between regions
    coord = prop_df.apply(lambda row: [row['centroid-0']] + [row['centroid-1']], axis=1).to_list() ### extract coordinates of centroid 
    distances_matrix = np.array([np.linalg.norm((item*np.ones((len(coord),len(item))))-coord,axis=1) for item in coord]) ## distance matrix 
    distances_df = pd.DataFrame(distances_matrix)
    # distances_df.to_csv('~/Desktop/2023_internship_work/2D_simulations/results/measurements/' + timestr + '_' + file_name + '_distance_matrix_' + str(par1) + '_' + str(par2) + '_' + str(val_str) + '.csv')

    # extract I only want consecutive distances (0 -> 1, 1 -> 2, 2 -> 3, ...)
    df2 = distances_df.iloc[:-1, 1:] # drop first row and last column
    consecutive_distances = pd.Series(np.diag(df2), index=[df2.index, df2.columns])
    consecutive_distances.to_csv('~/Desktop/2023_internship_work/2D_simulations/results/measurements/' + timestr + '_' + file_name + '_consecutive_distances_' + str(par1) + '_' + str(par2) + '_' + str(val_str) + '.csv')
    
    # identify coordinates of centroids 
    coord2 = prop_df.apply(lambda row: [row['centroid-1']] + [row['centroid-0']], axis=1).to_list()
    coord2 = np.array(coord2)

    # create label overlay 
    image_np = np.asarray(df)
    image_label_overlay = label2rgb(labelled, image=image_np, bg_label=0)

    # indicate position of centroids on the image 
    # fig, ax = plt.subplots()
    # plt.scatter(coord2[: , 0], coord2[: , 1], marker="x", color="red", s=10)
    # plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + '_' + file_name + '_coord_' +str(par1) + '_' + str(par2) + '_' + str(val_str) + '.pdf')
    
    ## add labels and centroid onto overlay 
    fig, ax = plt.subplots()
    ax.imshow(image_label_overlay)
    small_prop_df = prop_df[['centroid-0', 'centroid-1', 'label']]
    small_prop_df.label = small_prop_df.label.astype(int)
    small_prop_df.plot('centroid-1', 'centroid-0', kind='scatter', ax=ax)
    small_prop_df[['centroid-1','centroid-0','label']].apply(lambda row: ax.text(*row),axis=1)
    plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + file_name + '_labelled_' + str(par1) + '_'+ str(par2) + '_' + str(val_str) + '.pdf')

# %%
## Previously, I generated separate plots for each parameter (looking at each region individually)

# I want a new dataframe which has: 
## parameter value (a = 20, 25, 30, ..., 70) in column 1
## name of property (label, area, min_axis, major_axis...) in column 2
## name of region (1, 2, 3, ...) in column 3
## value for the parameter, for the property, in a given region in column 4 

print(prop_df_results)
prop_df_results["area_bbox_ratio"] = prop_df_results["area"] / prop_df_results["area_bbox"]
## ratio closer to 1 should correspond to a shape closer to a rectangle 

### for plotting, I am making a copy of df to remove these with area > 1000 (not stripes)
prop_df_results_melted = pd.melt(prop_df_results, id_vars = [str(par1), str(par2)])
props = list(set(list(prop_df_results_melted['variable'])))

for prop in props:
    if prop in ['label', 'centroid-0', 'centroid-1']:
        pass
    elif prop in ['area', 'area_bbox']:   
        prop_df_select = prop_df_results.loc[:, [str(par1), str(par2), prop]]
        prop_df_select = prop_df_select.loc[prop_df_select[prop] < 300]
        fig = prop_df_select.plot.scatter(x = str(par1), y =str(par2), s = 200, colormap = "RdYlBu", c = prop, title = "parameter space_" + str(par1) + "_" + str(par2) + '_' + str(prop), 
        figsize=(10, 8), fontsize=22)
        plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + "_" + file_name  + "_parameter_space_" + str(prop) + "_" + str(par1) + '_' + str(par2) + '.pdf')
    else:   
        prop_df_select = prop_df_results.loc[:, [str(par1), str(par2), prop]]
        fig = prop_df_select.plot.scatter(x = str(par1), y =str(par2), s = 200, colormap = "RdYlBu", c = prop, title = "parameter space_" + str(par1) + "_" + str(par2) + '_' + str(prop), 
        figsize=(10, 8), fontsize=22)
        plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + "_" + file_name  + "_parameter_space_" + str(prop) + "_" + str(par1) + '_' + str(par2) + '.pdf')

for prop in props:
    prop_df_col = prop_df_results_melted.loc[prop_df_results_melted['variable'] == prop]
    if prop in ['label', 'centroid-0, centroid-1']:
        pass
    else:
        for par in [par1, par2]:
            boxplot = prop_df_col.boxplot(column=['value'], by = par,
            grid=False, rot=45, fontsize=12, return_type = 'axes') 
            boxplot[0].set_ylabel(str(prop))
            boxplot[0].set_title(f'Impact of {par} on ' + str(prop))
            plt.savefig('/Users/barbarawalkowiak/Desktop/2023_internship_work/2D_simulations/results/figures/' + timestr + file_name + '_boxplot_' + str(par) + '_impact_on_' + str(prop) + '.pdf')

# %%