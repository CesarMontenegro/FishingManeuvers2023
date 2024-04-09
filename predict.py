import os
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy import genfromtxt
import pandas as pd
import pickle
import argparse

from sklearn.pipeline import Pipeline
from utils.boatFormat import ToTimeseriesFormat, SelectCols, AddDistanceBetweenPoints, AddTimeBetweenPoints
from utils.loader import  loadData
import tensorflow.keras as keras

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



#Calculates the time differences between consecutive timestamps
def calculate_time_difference(timestamps):
    # Check if the input array has at least two elements
    if len(timestamps) < 2:
        raise ValueError("Timestamp array must have at least two elements.")

    # Calculate the time differences between consecutive timestamps
    time_diff = np.diff(timestamps)
    
    # Clip the time differences to a maximum of 1000 and a minimum of -1000
    time_diff = np.clip(time_diff, -1000, 1000)

    # Insert a placeholder value (0) at the beginning of the time differences array
    time_diff = np.insert(time_diff, 0, 0)
    
    return time_diff



#Loads data from a boat file.
def loadOnlyData(boatFile, target=1):
    # Extract MMSI from file path
    mmsi = int(boatFile.split("/")[-1].split("_")[0])

    # Initialize variables
    boatCounter = 0
    boatMatrixes = []

    # Load boat data from file
    boatData = genfromtxt(boatFile, delimiter=',')
    
    # Check if boatData has more than 100 rows
    if boatData.shape[0] > 100:
        # Remove rows with NaN values
        boatData = boatData[~np.isnan(boatData[:, :]).any(axis=1)]
        
        # Create container array to hold data
        container = np.zeros((boatData.shape[0], boatData.shape[1]+1))
        
        # Populate container with boatData and MMSI
        container[:, :-1] = boatData
        container[:, -1] = mmsi
        
        # Increment boat counter and append container to boatMatrixes
        boatCounter += 1
        boatMatrixes.append(container)
        
        return container
    
    return None



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Load Keras models.')
    parser.add_argument('--data_path', type=str, default="data_path/", help='Path where the models are located.')
    parser.add_argument('--predictions_path', type=str, default="predictions_path/", help='Path for predictions.')
    parser.add_argument('--model_Longliner', type=str, default='models/maneuver_Longliner.h5', help='Path to model Longliner model.')
    parser.add_argument('--model_Trawler', type=str, default='models/maneuver_Trawler.h5', help='Path to model Trawler model.')
    parser.add_argument('--model_Seiner', type=str, default='models/maneuver_Seiner.h5', help='Path to model Seiner model.')
    parser.add_argument('--model_SquidTranship', type=str, default='models/maneuver_SquidTranship.h5', help='Path to model SquidTranship model.')


    # Set paths and model names
    args = parser.parse_args()
    
    path = args.data_path
    destPath = args.predictions_path    
    maneuver_1 = keras.models.load_model(args.model_Longliner)
    maneuver_2 = keras.models.load_model(args.model_Trawler)
    maneuver_3 = keras.models.load_model(args.model_Seiner)
    maneuver_4 = keras.models.load_model(args.model_SquidTranship)
    
    # Define transformers pipeline
    amountOfTimestampsPerRow = 32
    x_transformers = [#Transformers-----------------------------------
                 ('addDistanceBetweenPoints', AddDistanceBetweenPoints()),
                 ('addTimeBetweenPoints', AddTimeBetweenPoints()),
                 ('selectVariables', SelectCols([0,5,6,7,8,9,10,11,12,13])),
                 ('convertToTimeseriesFormat', ToTimeseriesFormat(amountOfTimestampsPerRow))
                 ]

    # Get files in the data path
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    onlyfiles = sorted(onlyfiles)

    # Set numpy printing options
    np.set_printoptions(suppress=True)

    # Define transformers pipeline
    x_transformersPipeline = Pipeline(x_transformers) # define the pipeline object.
    # Loop through files
    for filename in onlyfiles:
        print(filename)
        if "label" not in filename and not isfile(destPath+filename.replace(".txt","_predictions.txt")):
            x = loadOnlyData(path+filename)

            if x is not None:
                try:
                    # Filtering data
                    x = x[x[:,7]<30]
                    x = x[x[:,7]>=0]
                    
                    # Fit and transform data using transformers pipeline
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

                    # Select features
                    xt = xt[:,:,[0,3,4,7]]

                    # Normalize features
                    xt[:,:,0] = (xt[:,:,0]+1000)/2000
                    xt[:,:,1] = (xt[:,:,1])/20
                    xt[:,:,2] = (((xt[:,:,2])/180)+1)/2
                    xt[:,:,3] = (xt[:,:,3])/1000


                    # Predict using loaded models
                    container = np.zeros((x.shape[0], 4+4))
                    container[:,:-4] = x[:,[0,1,5,6]]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-4] = maneuver_1.predict(xt)[:,0]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-3] = maneuver_2.predict(xt)[:,0]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-2] = maneuver_3.predict(xt)[:,0]
                    container[amountOfTimestampsPerRow//2-1:-(amountOfTimestampsPerRow//2-1),-1] = maneuver_4.predict(xt)[:,0]
                    
                    # save to file
                    container.astype(np.float32)
                    np.savetxt(destPath+filename.replace(".txt","_predictions.txt"), container, fmt="%i,%i,%.8f,%.8f,%.6f,%.6f,%.6f,%.6f")

                except:
                    print("Error----------------")
                    continue
    
    

