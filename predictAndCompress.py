import os
import zipfile
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
import pickle

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from transformers.boatFormat import ToTimeseriesFormat, RemoveCol, SelectCols, OneVsAll, AddDistanceBetweenPoints, AddTimeBetweenPoints
from loader import  loadData
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from numpy import genfromtxt
from sklearn.ensemble import RandomForestClassifier

def transform_directions(headings):
    transformed_directions = np.zeros_like(headings)

    if headings.size == 0:
        return transformed_directions

    differences = np.diff(headings)
    differences = np.where(differences > 180, differences - 360, differences)
    differences = np.where(differences < -180, differences + 360, differences)

    transformed_directions[1:] = differences

    return transformed_directions


def calculate_distance(longitudes,latitudes ):
    # Check if the input arrays have the same length
    if len(latitudes) != len(longitudes):
        raise ValueError("Latitude and longitude arrays must have the same length.")

    # Convert degrees to radians
    latitudes = np.radians(latitudes)
    longitudes = np.radians(longitudes)

    # Earth's radius in meters
    earth_radius = 6371000

    # Calculate the differences in latitude and longitude
    delta_lat = np.diff(latitudes)
    delta_lon = np.diff(longitudes)

    # Haversine formula
    a = np.sin(delta_lat / 2) ** 2 + np.cos(latitudes[:-1]) * np.cos(latitudes[1:]) * np.sin(delta_lon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    distances = earth_radius * c

    # Create an array with the same size as input arrays, with 0 as the first element
    distances = np.insert(distances, 0, 0)
    
    distances[distances>1000] = 1000

    return distances




def calculate_time_difference(timestamps):
    # Check if the input array has at least two elements
    if len(timestamps) < 2:
        raise ValueError("Timestamp array must have at least two elements.")

    # Calculate the time differences between consecutive timestamps
    time_diff = np.diff(timestamps)
    time_diff[time_diff>1000] = 1000
    time_diff[time_diff<-1000] = -1000

    # Insert a placeholder value (0) at the beginning of the time differences array
    time_diff = np.insert(time_diff, 0, 0)
    
    return time_diff



def loadOnlyData(boatFile, target=1):
    mmsi = int(boatFile.split("/")[-1].split("_")[0])

    #rec-timestamp, send-timestamp, message-type, mmsi, nav-status, lon, lat, sog, cog, true-heading, rot
    boatCounter = 0
    boatMatrixes = []
    totalLength = 0
    print(boatFile)
    boatData = genfromtxt(boatFile, delimiter=',')
    if boatData.shape[0]>100:
        boatData = boatData[~np.isnan(boatData[:, :]).any(axis=1)]
        container = np.zeros((boatData.shape[0], boatData.shape[1]+1))
        container[:, :-1] = boatData
        container[:, -1] = mmsi
        boatCounter += 1
        boatMatrixes.append(container)

        return container
    return None



def zip_files_in_folder(folder_path, c):
    if not os.path.exists(folder_path):
        print("Error: Folder path does not exist.")
        return

    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    num_files = len(files)

    if c <= 0 or num_files == 0:
        print("Error: Invalid count or no files to zip.")
        return

    files_per_zip = num_files // c
    if num_files % c != 0:
        files_per_zip += 1

    zip_file_count = 1
    file_count = 0

    for file in files:
        with zipfile.ZipFile(f"zip_{zip_file_count}.zip", "a", zipfile.ZIP_DEFLATED) as zipf:
            zipf.write(os.path.join(folder_path, file), file)

        file_count += 1
        if file_count == files_per_zip:
            file_count = 0
            zip_file_count += 1


if __name__ == "__main__":
    maneuver_1_conv1d = keras.models.load_model('maneuver_1_con1d_32.h5')
    maneuver_2_conv1d = keras.models.load_model('maneuver_2_con1d_32.h5')
    maneuver_3_conv1d = keras.models.load_model('maneuver_3_con1d_32.h5')
    maneuver_4_conv1d = keras.models.load_model('maneuver_4_con1d_32.h5')
    
    
    amountOfTimestampsPerRow = 32
    x_transformers = [#Transformers-----------------------------------
                 ('addDistanceBetweenPoints', AddDistanceBetweenPoints()),
                 ('addTimeBetweenPoints', AddTimeBetweenPoints()),
                 ('selectVariables', SelectCols([0,5,6,7,8,9,10,11,12,13])),
                 ('convertToTimeseriesFormat', ToTimeseriesFormat(amountOfTimestampsPerRow))
                 ]

    path =     "/data_path/"
    destPath = "/predictions_path/"
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = sorted(onlyfiles)

    np.set_printoptions(suppress=True)

    index=1
    slice_start = (len(onlyfiles) // 8 ) * (index-1)
    slice_end = (len(onlyfiles) // 8 ) * index
    onlyfiles = onlyfiles[slice_start:slice_end]
    x_transformersPipeline = Pipeline(x_transformers) # define the pipeline object.
    for filename in onlyfiles:
        if "label" not in filename and not isfile(destPath+filename.replace(".txt","_predictions.txt")):
            x = loadOnlyData(path+filename)

            if x is not None:
                try:
                    x = x[x[:,7]<30]
                    x = x[x[:,7]>=0]
                    x_transformersPipeline = Pipeline(x_transformers) # define the pipeline object.
                    xt = x_transformersPipeline.fit_transform(x)

                    #Transform time
                    for index in range(xt.shape[0]):
                        xt[index, :,0] = calculate_time_difference(xt[index, :,0])
                    #Transform directions
                    for index in range(xt.shape[0]):
                        xt[index, :,4] = transform_directions(xt[index, :,4])
                    #Transform distances
                    for index in range(xt.shape[0]):
                        xt[index, :,7] = calculate_distance(xt[index, :,1], xt[index, :,2])

                    xt = xt[:,:,[0,3,4,7]]

                    xt[:,:,0] = (xt[:,:,0]+1000)/2000
                    xt[:,:,1] = (xt[:,:,1])/20
                    xt[:,:,2] = (((xt[:,:,2])/180)+1)/2
                    xt[:,:,3] = (xt[:,:,3])/1000


                    #xt = xt.reshape((xt.shape[0], xt.shape[1]*xt.shape[2]))

                    container = np.zeros((x.shape[0], 4+4))
                    container[:,:-4] = x[:,[0,1,5,6]]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-4] = maneuver_1_conv1d.predict(xt)[:,0]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-3] = maneuver_2_conv1d.predict(xt)[:,0]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-2] = maneuver_3_conv1d.predict(xt)[:,0]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-1] = maneuver_4_conv1d.predict(xt)[:,0]
                    print(np.unique(container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-1], return_counts=True))
                    container.astype(np.float32)
                    np.savetxt(destPath+filename.replace(".txt","_predictions.txt"), container, fmt="%i,%i,%.8f,%.8f,%.6f,%.6f,%.6f,%.6f")

                except:
                    print("Error----------------")
                    continue
    
    
    compressed_path = "./compressedFolder"
    count_parameter = 5

    #zip_files_in_folder(folder_path, count_parameter)
