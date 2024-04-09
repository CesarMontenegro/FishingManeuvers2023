import numpy as np
from numpy import genfromtxt
import pickle
from os import listdir
from os.path import isfile, join

def loadData(mypath = "path/", target=1, onlyTargetData=False):
    # Get list of all files in the specified path
    boatFiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    
    # Define the data structure for storing boat matrices and labels
    boatCounter = 0
    boatMatrixes = []
    boatLabels = []
    totalLength = 0
    
    # Iterate over each boat file
    for boatFile in boatFiles:
        # Check if the file is not a label file
        if "_labels" not in boatFile:
            # Check if there is a corresponding label file
            if isfile(boatFile.replace(".txt", "_labels.pkl")):
                # Load boat data from file and preprocess
                boatData = genfromtxt(boatFile, delimiter=',')
                boatData = boatData[:11]
                boatData = boatData[~np.isnan(boatData[:, [0,5,6]]).any(axis=1)]
                
                # Load labels from corresponding label file
                with open(boatFile.replace(".txt", "_labels.pkl"), 'rb') as handle:
                    labels = pickle.load(handle)
                
                # Create a container to hold boat data along with boat counter
                container = np.zeros((boatData.shape[0], boatData.shape[1]+1))
                container[:, :-1] = boatData
                container[:, -1] = boatCounter
                boatCounter += 1
                
                # Filter data based on target label if required
                if onlyTargetData:
                    if np.sum(labels==target)>0:
                        boatMatrixes.append(container)
                        boatLabels.append(labels)
                    else:
                        boatMatrixes.append(container)
                        boatLabels.append(np.zeros(container.shape[0]))
                else:
                    boatMatrixes.append(container)
                    boatLabels.append(labels)
                totalLength += boatLabels[-1].shape[0]
    
    # Concatenate boat matrices and labels
    flatMatrix = np.concatenate(boatMatrixes, axis=0)
    flatLabels = np.concatenate(boatLabels, axis=0)
    
    # Create labelsAndBoatId array
    labelsAndBoatId = np.zeros((flatLabels.shape[0],2))
    labelsAndBoatId[:, 0] = flatLabels==target
    labelsAndBoatId[:, 1] = flatMatrix[:, -1]
    
    return flatMatrix, labelsAndBoatId

if __name__ == "__main__":
    # Example usage of the function
    loadData()

