from sklearn.base import TransformerMixin
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler



class SelectCols(TransformerMixin):
    def __init__(self, cols):
        self.cols = cols

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("---",X.shape, self.cols)
        res = X[:, self.cols]
        return res


class ToTimeseriesFormat(TransformerMixin):
    def __init__(self, amount, reshape=True):
        self.amount = amount
        self.reshape = reshape

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        print("ToTimeseriesFormat", X.shape)
        extendedData = []
        for boatId in np.unique(X[:, -1]):
            boatData = X[X[:, -1] == boatId, :-1]
            if boatData.shape[0] > self.amount:
                newData = np.zeros((boatData.shape[0]+self.amount, self.amount, boatData.shape[1]))
                for x in range(0, self.amount):
                    newData[x:boatData.shape[0]+x, x] = boatData
                extendedData.append(newData[self.amount-1:-(self.amount-1)].reshape((newData.shape[0]-(2*(self.amount-1)), self.amount, newData.shape[2])))
        extendedData = np.concatenate(extendedData, axis=0)
        print("ToTimeseriesFormat", extendedData.shape)
        if self.reshape:
            return extendedData.reshape((extendedData.shape[0], self.amount, X.shape[1] - 1))
        else:
            return extendedData

        
class AddTimeBetweenPoints(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        res = np.zeros((X.shape[0], X.shape[1] + 1))
        res[:, :-2] = X[:, :-1]
        res[1:, -2] = X[1:, 0] - X[:-1, 0]
        res[:, -1] = X[:, -1]
        return res


# rec-timestamp, send-timestamp, message-type, mmsi, nav-status, lon, lat, sog, cog, true-heading, rot, boatId
class AddDistanceBetweenPoints(TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        R = 6373.0

        lat1 = np.radians(X[:, 6])
        lon1 = np.radians(X[:, 5])
        lat2 = lat1.copy()
        lat2[1:] = lat2[:-1]
        lon2 = lon1.copy()
        lon2[1:] = lon2[:-1]

        dlon = lon2 - lon1
        dlat = lat2 - lat1

        a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

        distance = R * c

        res = np.zeros((X.shape[0], X.shape[1] + 1))
        res[:, :-2] = X[:, :-1]
        res[:, -2] = distance
        res[:, -1] = X[:, -1]
        return res


def distancesToPoint(refLat, refLon, lats, lons):
    refLat = np.radians(refLat.copy())
    refLon = np.radians(refLon.copy())
    lats = np.radians(lats.copy())
    lons = np.radians(lons.copy())

    R = 6373.0

    dlon = lons - refLon
    dlat = lats - refLat

    a = np.sin(dlat / 2) ** 2 + np.cos(refLat) * np.cos(lats) * np.sin(dlon / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))

    distances = R * c
    return distances


def xyDistances(refLat, refLon, positions):
    lats = positions[:, 6]
    lons = positions[:, 5]
    xyDiferences = np.zeros((positions.shape[0], 2))
    xyDiferences[:, 0] = distancesToPoint(refLat, refLon, np.ones(lats.shape[0]) * refLat, lons)
    xyDiferences[:, 1] = distancesToPoint(refLat, refLon, lats, np.ones(lons.shape[0]) * refLon)
    return xyDiferences




