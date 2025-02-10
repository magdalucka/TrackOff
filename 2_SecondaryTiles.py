# -*- coding: utf-8 -*-

### Step 2 ###
### Generate secondary tiles ###
# This script contains function to generate image tiles from secondary image.
# Generated tiles are also processed to expand teh training dataset vy filtering, resizing and roatation.

# Import libraries

from PIL import Image, ImageFilter
import os
import pandas as pd
from itertools import product
from osgeo import gdal
import numpy as np

# Input parameters:
    # no_of_tiles - number of master tiles
    # r - searching radius for similar tiles
    # patch_size - size in pixels of the new tiles
    # step - denisty of image sampling within the searching radius
    # csv_MasterCoord - path to .csv file where coordinates of master tiles are stored
    # second_img - path to sceondary image (.tif/.png/...)
    # buff - resizing boundaries 

no_of_tiles = 36   
r = 10
step = 1
patch_size = 100
buff = 2
start_angle = -15
stop_angle = 16
step_angle = 5
work_dir = r"C:/Users/user/Desktop/test/"
csv_MasterCoord = work_dir+"master_coord.csv"
second_img = work_dir+"CAT2.png"

# Function definition for tiles generation and data augmenatation

def slave_tiles(dir_in, dir_out, patch_size, step_patch, buff, nr, r, start, stop, step_angle, csv_MasterCoord):
    patches = pd.read_csv(csv_MasterCoord)
    X_mid, Y_mid = int(patches.iloc[nr]['X']), int(patches.iloc[nr]['Y'])
    slave = gdal.Open(dir_in)
    band_s = slave.GetRasterBand(1)
    img_slave = band_s.ReadAsArray()
    img_slave = img_slave.astype(np.float32)
    img1 = Image.fromarray(img_slave)
    img = img1.convert("L")
    
    angle = list(range(start,stop,step_angle))  # okreslenie zakresu, w którym ma być obracane zdjęcia od, do, z jakim krokiem
    angle.remove(0) 
    bufor_crop = 15
    w, h = img.size
    k = 0
    coord = []
    grid = product(range(Y_mid-r-int(patch_size/2), Y_mid+r+int(patch_size/2)-(patch_size-1), step_patch), range(X_mid-r-int(patch_size/2), X_mid+r+int(patch_size/2)-(patch_size-1), step_patch))
                                                                                                    
    for i, j in grid:    
            if i>=0 and i<=(h-patch_size):
                if j>=0 and j<=(w-patch_size):                                         
                    os.mkdir(os.path.join(dir_out+"/"+str(k))) 
                    
                    coord.append([i+patch_size/2,j+patch_size/2])
                    coordinates = pd.DataFrame(coord, columns=['Y','X']) 
                    
                    box = (j, i, j+patch_size, i+patch_size)                   
                    out = os.path.join(dir_out+"/"+str(k)+"/"+str(k)+'.png')
                    img_org = img.crop(box)
                    img_org.save(out)    
                    
                    box_flt = (j-1, i-1, j+patch_size+1, i+patch_size+1)
                    im_gray = img.crop(box_flt).convert("L")
                    box_new = (1,1,patch_size+1,patch_size+1)
                    im_f2 = im_gray.filter(ImageFilter.CONTOUR).crop(box_new).save(out[:-4]+"_CONTOUR"+".png")
                    im_f6 = im_gray.filter(ImageFilter.FIND_EDGES).crop(box_new).save(out[:-4]+"_FIND_EDGES"+".png")  
                    im_f5 = im_gray.filter(ImageFilter.EMBOSS).crop(box_new).save(out[:-4]+"_EMBOSS"+".png")
                    
                    box_L = (j-buff, i-buff, j+patch_size+buff, i+patch_size+buff)              
                    out_L = os.path.join(dir_out+"/"+str(k)+"/"+str(k)+'_L'+str(buff)+'.png')   
                    img_L = img.crop(box_L).resize((patch_size,patch_size))                     
                    img_L.save(out_L)                                                             
                                                                                         
            
                    box_S = (j+buff, i+buff, j+patch_size-buff, i+patch_size-buff)              
                    out_S = os.path.join(dir_out+"/"+str(k)+"/"+str(k)+'_S'+str(buff)+'.png')  
                    img_S = img.crop(box_S).resize((patch_size,patch_size))                    
                    img_S.save(out_S)
                    
                    images = [img_org,img_L,img_S]
                    for image in images:
                        if image == img_org:
                            save_path = out
                        elif image == img_L:
                            save_path = out_L
                        elif image == img_S:
                            save_path = out_S
                        im_f3 = image.filter(ImageFilter.DETAIL).save(save_path[:-4]+"_DETAIL"+".png")  
                        im_f4 = image.filter(ImageFilter.EDGE_ENHANCE).save(save_path[:-4]+"_EDGE_ENHANCE"+".png")  
                        im_f7 = image.filter(ImageFilter.SHARPEN).save(save_path[:-4]+"_SHARPEN"+".png")
                        im_f8 = image.filter(ImageFilter.SMOOTH).save(save_path[:-4]+"_SMOOTH"+".png")
                                                
                    
                    box_rot = (j-bufor_crop, i-bufor_crop, j+patch_size+bufor_crop, i+patch_size+bufor_crop)    
                    temp = img.crop(box_rot)                                    
                    wid, hei = temp.size                                    
                    for x in angle:                                       
                        box2 = (bufor_crop,bufor_crop,patch_size+bufor_crop,patch_size+bufor_crop)        
                        im_rotate = temp.rotate(x)                          
                        im_final = im_rotate.crop(box2)                     
                        im_final.save(os.path.join(path+str(k)+"/"+str(k)+"_"+str(x)+".png"))
                        
                        im_gray = im_rotate.convert("L")
                        im_f1 = im_rotate.filter(ImageFilter.BLUR)
                        im_f1_r = im_f1.crop(box2)                    
                        im_f1_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_BLUR"+".png")
            
                        im_f2 = im_gray.filter(ImageFilter.CONTOUR)
                        im_f2_r = im_f2.crop(box2)
                        im_f2_r.save(path+str(k)+"/"+str(k)+"_"+str(x)+"_CONTOUR"+".png")
            
                        im_f3 = im_rotate.filter(ImageFilter.DETAIL)
                        im_f3_r = im_f3.crop(box2)
                        im_f3_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_DETAIL"+".png")
            
                        im_f4 = im_rotate.filter(ImageFilter.EDGE_ENHANCE)
                        im_f4_r = im_f4.crop(box2)
                        im_f4_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_EDGE_ENHANCE"+".png")
            
                        im_f5 = im_gray.filter(ImageFilter.EMBOSS)
                        im_f5_r = im_f5.crop(box2)
                        im_f5_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_EMBOSS"+".png")
            
                        im_f6 = im_gray.filter(ImageFilter.FIND_EDGES)
                        im_f6_r = im_f6.crop(box2)
                        im_f6_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_FIND_EDGES"+".png")
            
                        im_f7 = im_rotate.filter(ImageFilter.SHARPEN)
                        im_f7_r = im_f7.crop(box2)
                        im_f7_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_SHARPEN"+".png")
            
                        im_f8 = im_rotate.filter(ImageFilter.SMOOTH)
                        im_f8_r = im_f8.crop(box2)
                        im_f8_r.save(path+"/"+str(k)+"/"+str(k)+"_"+str(x)+"_SMOOTH"+".png")
                        
                        
                    k += 1
                else:
                    pass
            else:
                pass
    return coordinates

# Process all master tiles:     
patches = list(range(0,no_of_tiles))
for no in patches:
    patch_no = no
    path = work_dir+"slave_tiles_"+str(patch_no)+"/"       # generate new folder for each class
    os.mkdir(path)
          
    coord = slave_tiles(second_img, path, patch_size, step, buff, patch_no, r, start_angle, stop_angle, step_angle, csv_MasterCoord)
    coord.to_csv(work_dir+"slave_XY_"+str(patch_no)+".csv")  # generate new .csv file with coordinates
