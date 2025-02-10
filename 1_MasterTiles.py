# -*- coding: utf-8 -*-

### Step 1 ###
### Generate master tiles ###
# This script contains function to generate image tiles from master image
# Those tiles will not take part in training process but they will be used for model evaluation. 
# Grid size selected at this point determines the denisty of the output points with displacement infromation.


# Import libraries

from PIL import Image
import os
from itertools import product
import pandas as pd
from osgeo import gdal 

# Input parameters:
    # path - new tiles will be stored in this location
    # patch_size - size in pixels of the new tiles
    # grid_size - sampling density (final displacement gird density) 
    # master_img - path to master image (.tiff)
    # csv_path - path to .csv file where coordinates of tiles will be stored
        
tiles_path = r"C:/Users/user/Desktop/test/master_tiles/"
if os.path.exists(tiles_path) == False:
    os.mkdir(tiles_path)
patch_size = 100
grid_size = 100
master_img = r"C:/Users/user/Desktop/test/CAT1.png"
csv_path = r"C:/Users/user/Desktop/test/master_coord.csv"

# Function definition
def create_tiles(dir_in, dir_out, patch_size, grid_size):
    
    master = gdal.Open(dir_in)
    band1 = master.GetRasterBand(1)
    band1 = band1.ReadAsArray()
    band1 = Image.fromarray(band1)
    img = band1.convert("L")

    w, h = img.size             
    k = 0                      
    coord = []
    grid = product(range(0, h-(patch_size-1), grid_size), range(0, w-(patch_size-1), grid_size))  
    for i, j in grid:                                          
        coord.append([i+patch_size/2,j+patch_size/2])           
        coordinates = pd.DataFrame(coord, columns=['Y','X'])    
        
        box = (j, i, j+patch_size, i+patch_size)        
        out = os.path.join(dir_out+"/"+str(k)+'.png')   
        img.crop(box).save(out)                         

        k += 1                                          
        
    return coordinates

# Generation of tiles and coordinate files
coord = create_tiles(master_img, tiles_path, patch_size, grid_size)
coord.to_csv(csv_path)
