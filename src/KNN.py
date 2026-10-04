__authors__ = ['1710404','1714533','1713947']
__group__ = '74'

import numpy as np
import math
import operator
from scipy.spatial.distance import cdist
import utils


class KNN:
    def __init__(self, train_data, labels):
        self._init_train(train_data)
        self.labels = np.array(labels)
        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

        self.distance = 'euclidean'  # Optional: 'euclidean', 'manhattan', 'cosine'
        self.feature_type = 'raw'    # Optional: 'raw', 'mean_rgb', 'mean_halves', 'grayscale_mean'


    def _init_train(self, train_data):
        """
        initializes the train data
        :param train_data: PxMxNx3 matrix corresponding to P color images
        :return: assigns the train set to the matrix self.train_data shaped as PxD (P points in a D dimensional space)
        """
        if train_data.ndim == 4 and train_data.shape[-1] == 3:
            train_data_gris = utils.rgb2gray(train_data)
        elif train_data.ndim == 3:
            train_data_gris = train_data
        else:
            print("The format of train_data is invalid. It must be (N, 80, 60, 3) or (N, 80, 60)")
            return

        train_data_gris = train_data_gris.astype(float)
        N = train_data_gris.shape[0]
        self.train_data_raw = train_data_gris
        self.train_data = train_data_gris.reshape(N, -1)
        
        
    def get_k_neighbours(self, test_data, k):
        """
        given a test_data matrix calculates de k nearest neighbours at each point (row) of test_data on self.neighbors
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the matrix self.neighbors is created (NxK)
                 the ij-th entry is the j-th nearest train point to the i-th test point
        """
        if test_data.ndim == 4 and test_data.shape[-1] == 3:
            test_data_gray = utils.rgb2gray(test_data)
        elif test_data.ndim == 3:
            test_data_gray = test_data
        else:
            print("The format of train_data is invalid. It must be (N, 80, 60, 3) or (N, 80, 60)")
            return

        test_data_gray = test_data_gray.astype(float)
        N = test_data_gray.shape[0]
        test_data_flat = test_data_gray.reshape(N, -1)

        if self.feature_type == 'raw' and self.distance in ['euclidean', 'manhattan', 'cosine']:
            metric = 'euclidean' if self.distance == 'euclidean' else \
                     'cityblock' if self.distance == 'manhattan' else \
                     'cosine'
            distances_all = cdist(test_data_flat, self.train_data, metric=metric)

        else:
            distances_all = []
            for test_img in test_data_gray:
                distances = [self.compute_distance(test_img, train_img)
                             for train_img in self.train_data_raw]
                distances_all.append(distances)

            distances_all = np.array(distances_all)

        nearest_indices = np.argsort(distances_all, axis=1)[:, :k]
        self.neighbors = self.labels[nearest_indices]

    def get_class(self):
        """
        Get the class by maximum voting
        :return: 1 array of Nx1 elements. For each of the rows in self.neighbors gets the most voted value
                (i.e. the class at which that row belongs)
        """
        return np.array([max(x, key=list(x).count) for x in self.neighbors])

    def predict(self, test_data, k):
        """
        predicts the class at which each element in test_data belongs to
        :param test_data: array that has to be shaped to a NxD matrix (N points in a D dimensional space)
        :param k: the number of neighbors to look at
        :return: the output form get_class a Nx1 vector with the predicted shape for each test image
        """
        self.get_k_neighbours(test_data, k)
        return self.get_class()

    def extract_features(self, img):
        if self.feature_type == 'raw':
            return img.flatten()

        elif self.feature_type == 'mean_rgb':
            return np.mean(img, axis=(0, 1))

        elif self.feature_type == 'mean_halves':
            h = img.shape[0]
            top = img[:h // 2]
            bottom = img[h // 2:]

            # Handles grayscale or RGB without conversion
            mean_top = np.mean(top, axis=(0, 1), keepdims=False)
            mean_bottom = np.mean(bottom, axis=(0, 1), keepdims=False)

            return np.ravel([mean_top, mean_bottom])  # faster than concatenate
        
        elif self.feature_type == 'grayscale_mean':
            # If the image has 3 dimensions, average over the channels to obtain grayscale
            if img.ndim == 3:
                gray = np.mean(img, axis=2)
            else:
                # The image is already 2D, no need to average channels
                gray = img
            return np.array([np.mean(gray)])
        else:
            raise ValueError(f"Unknown feature_type: {self.feature_type}")
            
        

    def compute_distance(self, a, b):
        a = self.extract_features(a)
        b = self.extract_features(b)

        if self.distance == 'euclidean':
            return np.linalg.norm(a - b)
        elif self.distance == 'manhattan':
            return np.sum(np.abs(a - b))
        elif self.distance == 'cosine':
            if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
                return 1
            return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
        else:
            raise ValueError(f"Unknown distance metric: {self.distance}")
