import numpy as np
from numpy import genfromtxt
import pickle
from os import listdir
from os.path import isfile, join


def loadData(mypath = "path/", target=1, onlyTargetData=False):
    boatFiles = [join(mypath, f) for f in listdir(mypath) if isfile(join(mypath, f))]
    #rec-timestamp, send-timestamp, message-type, mmsi, nav-status, lon, lat, sog, cog, true-heading, rot

    boatCounter = 0
    boatMatrixes = []
    boatLabels = []
    totalLength = 0
    for boatFile in boatFiles:
        if "_labels" not in boatFile:
            if isfile(boatFile.replace(".txt", "_labels.pkl")):
                boatData = genfromtxt(boatFile, delimiter=',')
                boatData = boatData[:11]
                boatData = boatData[~np.isnan(boatData[:, [0,5,6]]).any(axis=1)]
                with open(boatFile.replace(".txt", "_labels.pkl"), 'rb') as handle:
                    labels = pickle.load(handle)
                container = np.zeros((boatData.shape[0], boatData.shape[1]+1))
                container[:, :-1] = boatData
                container[:, -1] = boatCounter
                boatCounter += 1
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

                
    flatMatrix = np.concatenate(boatMatrixes, axis=0)
    flatLabels = np.concatenate(boatLabels, axis=0)
    labelsAndBoatId = np.zeros((flatLabels.shape[0],2))
    labelsAndBoatId[:, 0] = flatLabels==target
    labelsAndBoatId[:, 1] = flatMatrix[:, -1]
    return flatMatrix, labelsAndBoatId


if __name__ == "__main__":
    loadData()
