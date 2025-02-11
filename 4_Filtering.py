# -*- coding: utf-8 -*-

########################
### Filtering script ###
########################

# Import libraries
from osgeo import gdal
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import seaborn as sns
import numpy as np
import os

##################
# Input parameters

# Path to stored input files: Probability_n.csv, disp.csv and master.tif
path_data = r"C:/Users/user/Desktop/test/"
path_img = path_data+"img/"
if os.path.exists(path_img) == False:
    os.mkdir(path_img)

cmap = 'RdYlBu_r'           # deafult color map
no_patch = 36               # number of master tiles
grid_locator = 100          # size of tile
master_tiff = "CAT1.png"    # master image filename
thres = 5                   # probability threshold


#####################
# Read intesity image
master = gdal.Open(path_data+master_tiff)
band_m = master.GetRasterBand(1)
img_master = band_m.ReadAsArray()

##########################################
# Show original velocity and probabilities
# Velocity
datacube = pd.read_csv(path_data+"disp.csv")
data_v = datacube.pivot(index='y',columns='x',values='vel')
sns.color_palette(cmap, as_cmap=True)

plt.figure(figsize=(10,8))
ax1 = sns.heatmap(data_v, annot=True, fmt=".0f", linewidth=.5, cmap=cmap, square=True) #vmax=1000
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set(xlabel="range [pxl]", ylabel="azimuth [pxl]")
ax1.set_title('velocity [m/day]')

plt.tight_layout()
plt.savefig(path_img+"_vel.png", dpi=300) 
plt.show()

# Probability
datacube['probab'] = datacube['probab']*100
data_p = datacube.pivot(index='y',columns='x',values='probab')
sns.color_palette(cmap, as_cmap=True)

plt.figure(figsize=(10,8))
ax1 = sns.heatmap(data_p, annot=True, fmt=".0f", linewidth=.5, cmap='rocket_r', square=True) #vmax=1000
ax1.set_xticklabels(ax1.get_xticklabels(), rotation=90)
ax1.set(xlabel="range [pxl]", ylabel="azimuth [pxl]")
ax1.set_title('distribution of confidence')

plt.tight_layout()
plt.savefig(path_img+"probabilities.png", dpi=300) 
plt.show()

##########################################
#  Remove patches with very low confidence

data_all_updated = datacube.copy()
data_all_updated['probab_test'] = pd.Series(np.NaN)

patches = list(range(0,no_patch))

for patch in patches:
    dataset = pd.read_csv(path_data+"Probability_"+str(patch)+".csv", sep=",")
    cmap = 'RdYlBu_r' 

    dataset['Probability'] = dataset['Probability']

    sred = dataset['Probability'].mean()
    odch = dataset['Probability'].std()
    maxim = dataset['Probability'].max()

    lower3 = sred-3*odch
    upper3 = sred+3*odch
    lower2 = sred-2*odch
    upper2 = sred+2*odch    
    
    if maxim > upper3:
        data_all_updated.at[patch,'probab_test'] = data_all_updated.at[patch,'probab']
        
data_clear = data_all_updated.dropna(axis=0)
data_clear.to_csv(path_data+"disp_low.csv")


########################################
#  Remove probabilities below threshold

data_filtered = data_clear[data_clear['probab']*100>thres]
data_filtered.to_csv(path_data+"disp_low_outlier_"+str(thres)+".csv")

################################
# Visualize after all operations

# Probability
what = 'vel'
pro = data_filtered.pivot(index='y',columns='x',values=what)

plt.figure(figsize=(10,8))
axx = sns.heatmap(pro, annot=True, fmt=".0f", linewidth=.5, cmap=cmap, square=True)
axx.set_xticklabels(axx.get_xticklabels(), rotation=90)
axx.set_yticklabels(axx.get_yticklabels())
axx.set(xlabel="range [pxl]", ylabel="azimuth [pxl]")
axx.set_title(what)

plt.tight_layout()
plt.savefig(path_img+what+"_"+str(thres)+".png", dpi=300) 
plt.show()
plt.close()
plt.clf

# Vectors 
plt.figure(figsize=(10,8))
fig, axy = plt.subplots()
axy.quiver(data_filtered['x'], data_filtered['y'], data_filtered['dX_m'], -data_filtered['dY_m'], color='white')

axy.xaxis.set_major_locator(MultipleLocator(grid_locator))
axy.yaxis.set_major_locator(MultipleLocator(grid_locator))
axy.xaxis.set_minor_locator(MultipleLocator(50))
axy.yaxis.set_minor_locator(MultipleLocator(50))
axy.set_xticklabels(axy.get_xticklabels(), rotation=90)
axy.set_yticklabels(axy.get_yticklabels())
plt.xlabel("range [pxl]", fontsize=6)
plt.ylabel("azimuth [pxl]", fontsize=6)
axy.tick_params(axis='both', which='major', labelsize=6)
plt.grid(which='major', color="white", linewidth=0.15)
plt.grid(which='minor', color="white", linewidth=0.05)

plt.tight_layout()
plt.imshow(img_master, cmap='Greys_r', vmax=1500)

plt.savefig(path_img+"direction_all_flt_"+str(thres)+".png", dpi=300)
plt.imshow(img_master, cmap='Greys_r', vmax=1500)
plt.show()
plt.close()
plt.clf

