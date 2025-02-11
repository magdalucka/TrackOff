# -*- coding: utf-8 -*-

### Step 3 ###
### Train CNN and calculate offsets ###
# This script contains functions: 
    # (1) to read master and secondary images,
    # (2) train CNN model,
    # (3) calculate offsets between images imported (1) and write to output .csv file.

# Import libraries
import cv2
import os
import numpy as np
import sys
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, cohen_kappa_score
import tensorflow as tf
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
import pandas
import gc
import math
import keras

# Use GPU (optional)
devices = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(devices[0], True)

################################
#### Define input parameters ###
################################

# Spatial resolution [m] in range and azimuth directions:
r = 1
az = 1
# Temporal baseline [days]:
days = 1
# Output directory:
out_dir =  r"C:/Users/user/Desktop/test/output/"
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)
# Working dir (contains .csv files and tiles folders for master and secondary)
work_dir =  r"C:/Users/user/Desktop/test/"
# Number of patches:
no_tiles = 36
patches = list(range(0,no_tiles))

######################################
### Definition of useful functions ###
######################################

def import_images(path):
    labels =[]
    images = []
    for folder in os.listdir(path):
        path2 = path+"/"+folder
        for im in os.listdir(path2):
            label = folder
            image = cv2.imread(path2+'/'+im)
            im_norm = cv2.normalize(image, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
            images.append(im_norm)
            labels.append(label)
    im = np.array(images, dtype='float32')
    label = np.array(labels, dtype='int32')
    output = shuffle(im,label)
    return output

def plot_accuracy_loss(history):
    plt.figure(figsize=(10,5))

    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--',label = 'acc')
    plt.plot(history.history['val_accuracy'],'ro--',label = 'val_acc')
    plt.title('train_acc vs val_acc')
    plt.ylabel('accuracy')
    plt.xlabel('epochs'+str(epoch))
    plt.legend()

    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--',label = 'loss')
    plt.plot(history.history['val_loss'],'ro--',label = 'val_loss')
    plt.title('train_loss vs val_loss')
    plt.ylabel('loss')
    plt.xlabel('epochs'+str(epoch))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir+str(no)+'_loss_acc.png', dpi=300)
    plt.close()

    plt.show
    
def final_fig(no, range_res, azimuth_res, days):
    Coord1 = pandas.read_csv(work_dir+"master_coord.csv")
    Coord2 = pandas.read_csv(work_dir+"slave_XY_"+str(no_patch)+".csv")
    im_test = cv2.imread(work_dir+"master_tiles/"+str(no)+".png")
    im_test = cv2.normalize(im_test, None, 0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    im_test = np.array([im_test], dtype='float32')
    result_all = model_best.predict(im_test)
    result_class = np.argmax(result_all,axis=1)
    probab = np.transpose(result_all)

    img1 = Image.open(work_dir+"master_tiles/"+str(no)+".png")

    print('Predicted class: ', result_class)
    print('Porbability: ', result_all[0].max())

    predict_path =work_dir+"slave_tiles_"+str(no_patch)+"/"+str(result_class[0])+"/"+str(result_class[0])+".png"
    img2 = Image.open(predict_path)
    
    X2 = Coord2.iloc[result_class[0]]['X']
    Y2 = Coord2.iloc[result_class[0]]['Y'] # slave
    X1 = Coord1.iloc[no]['X'] # master
    Y1 = Coord1.iloc[no]['Y']
   
    dX = X2 - X1
    dY = Y2 - Y1
    
    dX_m = dX*range_res
    dY_m = dY*azimuth_res
    dMAX = math.sqrt(dX**2+dY**2)
    dMAX_m = math.sqrt(dX_m**2+dY_m**2)
    vel = dMAX_m/days 
    df = [no,X1,Y1,dX,dY,dX_m,dY_m,dMAX, dMAX_m, vel, result_all[0].max()]
    print('Offset in pixels: dX=', dX,'dY=', dY)
    
    fig, (ax1,ax2) = plt.subplots(1,2)
    
    ax1.imshow(img1, cmap='gray')
    ax1.set_xticklabels([])
    ax1.set_xticks([])
    ax1.set_yticklabels([])
    ax1.set_yticks([])
    ax1.set_title('Master tile (serached): Input', size=8)

    ax2.imshow(img2, cmap='gray')
    ax2.set_xticklabels([])
    ax2.set_xticks([])
    ax2.set_yticklabels([])
    ax2.set_yticks([])
    ax2.set_title('Secondary tile: level=%.2f' %(result_all[0].max()), size=8)
    
    plt.text(0,-15,'Offset in pixels: dX= %.2f, dY=%.2f' %(dX,dY), size=8, ha='center')

    plt.savefig(out_dir+str(no)+'_fit.png', dpi=300)
    plt.close()
    return df,probab


###################################
### Go through all image tiles: ###
###################################

for no in patches:
    no_patch = no
    path = work_dir+"slave_tiles_"+str(no_patch)
    

    # import raining dataset
    X, Y = import_images(path)
    X_train, X_test, Y_train, Y_test = train_test_split(X,Y, random_state=0, train_size = .75)
    
    count = 0
    for file in os.scandir(path):
        count += 1
    no_classes = count
    del X,Y
    
    # build CNN architecture (based on AlexNet)
    gc.collect()

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Conv2D(filters=96, kernel_size=(11, 11),
                            strides=(4, 4), activation="relu",
                            input_shape=(100, 100, 3)))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides= (2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(5, 5),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=384, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3),
                            strides=(1, 1), activation="relu",
                            padding="same"))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)))
    model.add(tf.keras.layers.Flatten())
    model.add(tf.keras.layers.Dense(4096, activation="relu"))
    model.add(tf.keras.layers.Dropout(0.5))
    model.add(tf.keras.layers.Dense(no_classes, activation="softmax"))
    
    # compilation and training
    optimizer_type = tf.optimizers.Adam
    LR = 0.001
    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=optimizer_type(learning_rate=LR),
                  metrics=['accuracy'])
    model.summary()

    with open(out_dir + 'model_summary_'+str(no_patch)+'.txt','w') as fh:
        model.summary(print_fn=lambda x: fh.write(x + '\n'))
    
    epoch = 100
    model_path = out_dir+"model_best_"+str(no_patch)+".hdf5"

    his = keras.callbacks.History()
    es = keras.callbacks.EarlyStopping(monitor='val_loss', 
                                       mode='min', restore_best_weights=True) # important - otherwise you just return the last weigths...
    mcp_save = keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='accuracy', mode='max')
    history = model.fit(X_train, Y_train, epochs=epoch, shuffle=True, batch_size=128, validation_split=0.2, verbose=1, callbacks=[mcp_save,his])

    # accuracy and loss values over epochs
    plot_accuracy_loss(history)
   
    # check quality metrics and write to file
    gc.collect()
    model_best = keras.models.load_model(model_path)
    
    test_loss = model_best.evaluate(X_test, Y_test, batch_size=16)
    print('loss = ', test_loss[0], '\naccuracy = ', test_loss[1])
    
    predictions = model_best.predict(X_test)
    pred_labels = np.argmax(predictions,axis=1)
    report = classification_report(Y_test, pred_labels, output_dict=True)
    
    df = pandas.DataFrame(report).transpose()
    df.to_csv(out_dir+"class_report"+str(no_patch)+".csv")
    
    matrix = confusion_matrix(Y_test, pred_labels)
    
    with open(out_dir + 'model_summary_'+str(no_patch)+'.txt','a') as fh:
        fh.write('\n\nepochs = ' + str(epoch) + '\noptimizer: ' + str(optimizer_type)+
                 '\nlearning rate: ' + str(LR)+
                 '\n\nloss = ' + str(test_loss[0]) + 
                 '\naccuracy = ' + str(test_loss[1])) # +

    # display matched image tiles
    Coord1 = pandas.read_csv(work_dir+"master_coord.csv")
    Coord2 = pandas.read_csv(work_dir+"slave_XY_"+str(no_patch)+".csv")

    df, probab = final_fig(no_patch, r, az, days)

                   
    # add new displacements values
    data_all = pandas.read_csv(out_dir+"disp.csv")
    data_all.loc[len(data_all)]= df
    
    data_all.to_csv(out_dir+"disp.csv", index=False)
    Coord2['Probability'] = probab
    Coord2.to_csv(out_dir+"Probability_"+str(no_patch)+".csv")
    gc.collect()
    print('Processing patch ' + str(no) + (': done'))
    
