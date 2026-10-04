__authors__ = ['1710404','1714533','1713947']
__group__ = '74'

import numpy as np
import utils


class KMeans:

    def __init__(self, X, K=1, options=None):
        """
         Constructor of KMeans class
             Args:
                 K (int): Number of cluster
                 options (dict): dictionary with options
            """
        self.num_iter = 0
        self.K = K
        self._init_X(X)
        self._init_options(options)  # DICT options

    #############################################################
    ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
    #############################################################

    def _init_X(self, X):
        """Initialization of all pixels, sets X as an array of data in vector form (PxD)
            Args:
                X (list or np.array): list(matrix) of all pixel values
                    if matrix has more than 2 dimensions, the dimensionality of the sample space is the length of
                    the last dimension
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # np.asfarray was removed in NumPy 2.0: keep float dtypes, convert everything else to float64
        X_array = np.asarray(X)
        if not np.issubdtype(X_array.dtype, np.floating):
            X_array = X_array.astype(np.float64)
    
        # Check the input dimensions
        if X_array.ndim > 2:
            # Image case (H × W × 3): reshape to (N × 3)
            X_reshaped = X_array.reshape(-1, 3)
            # (explanation of the change in the partner code) only the option is F x W x 3; we could take the 3 with original_shape[-1], but we are wasting time fetching it when we already know it is 3, so we save a variable
        else:
            # It is already 2D (N × D), no change needed
            X_reshaped = X_array
    
        # Ensure the matrix has exactly 2 dimensions
        if X_reshaped.ndim == 1:
            X_reshaped = X_reshaped.reshape(-1, 1)
        # ".reshape(-1, 1)" converts it to 2 dimensions
   
        self.X = X_reshaped

    def _init_options(self, options=None):
        """
        Initialization of options in case some fields are left undefined
        Args:
            options (dict): dictionary with options
        """
        if options is None:
            options = {}
        if 'km_init' not in options:
            options['km_init'] = 'first'
        if 'verbose' not in options:
            options['verbose'] = False
        if 'tolerance' not in options:
            options['tolerance'] = 0
        if 'max_iter' not in options:
            options['max_iter'] = np.inf
        if 'fitting' not in options:
            options['fitting'] = 'WCD'  # within class distance.

        # If your methods need any other parameter you can add it to the options dictionary
        self.options = options

        #############################################################
        ##  THIS FUNCTION CAN BE MODIFIED FROM THIS POINT, if needed
        #############################################################

    def _init_centroids(self):
        """
        Initialization of centroids
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # (explanation of the change in the partner code) there is no need to call ".lower()" because there is no case difference
        if self.options['km_init'] == 'first':
            # for each row, the unique numbers are ordered and returned in "unique_elements"
            # and since "return_index = true" is used in "index", the indices where the unique values first appear are stored
            unique_elements, index = np.unique(self.X, axis=0, return_index=True)
            # create an np array with the values of self[x] ordered by the indices of the unique values extracted earlier
            self.centroids = np.array([self.X[i] for i in sorted(index)[:self.K]])
            
        elif self.options['km_init'] == 'random':
            # Remove duplicates by creating a temporary dictionary with tuples
            temp_dict = {}
            for point in self.X:
                temp_dict[tuple(point)] = 1

            # Convert to an array of unique points
            unique_points = list(temp_dict.keys())
            unique_points = np.array(unique_points)

            # Shuffle the points
            np.random.shuffle(unique_points)

            # Select the first K points as centroids
            self.centroids = unique_points[:self.K].astype(float)

            # Save a copy to compare convergence
            self.old_centroids = np.copy(self.centroids)

        
        elif self.options['km_init'] == 'custom':
            # Fixed version that works with any dimension
            d = self.X.shape[1]  # Data dimension
            diagonal_points = np.zeros((self.K, d))
            
            # Values scaled between 0 and 1
            values = np.linspace(0, 1, self.K) if self.K > 1 else [0.5]
            
            for i in range(self.K):
                diagonal_points[i] = np.full((d,), values[i])
            
            # Scale to the actual data range
            data_min = np.min(self.X, axis=0)
            data_max = np.max(self.X, axis=0)
            data_range = data_max - data_min
            
            # Avoid division by zero (if the range is zero in any dimension)
            data_range[data_range == 0] = 1.0
            
            self.centroids = diagonal_points * data_range + data_min
               
        elif self.options['km_init'] == 'kmeans++':
            self.centroids = []
            self.centroids.append(self.X[np.random.choice(len(self.X))])

            for _ in range(1, self.K):
                distances = distance(self.X, np.array(self.centroids))
                min_distances = np.min(distances, axis=1)
                probs = (min_distances ** 2) / np.sum(min_distances ** 2)
                next_centroid_idx = np.random.choice(len(self.X), p=probs)
                self.centroids.append(self.X[next_centroid_idx])

            self.centroids = np.array(self.centroids)
        elif self.options['km_init'] == 'pca':
                # Center data
            X_centered = self.X - np.mean(self.X, axis=0)
            
            # Compute the principal component
            U, S, Vt = np.linalg.svd(X_centered, full_matrices=False)
            principal_component = Vt[0]  # First direction
            
            # Project points onto the principal component
            projections = X_centered @ principal_component
            
            # Sort points by projection
            sorted_indices = np.argsort(projections)
            
            # Split into K equal intervals and choose a central point from each
            split_indices = np.array_split(sorted_indices, self.K)
            self.centroids = np.array([self.X[idx[len(idx)//2]] for idx in split_indices])


        self.old_centroids = np.array(self.centroids)

    def get_labels(self):
        """
        Calculates the closest centroid of all points in X and assigns each point to the closest centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # find the closest centroid for EACH POINT (axis=1, each row)
        self.labels = np.argmin((distance(self.X, self.centroids)), axis = 1)

    def get_centroids(self):
        """
        Calculates coordinates of centroids based on the coordinates of all the points assigned to the centroid
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        
        # compute it following the formula in the document "SO_Part 1_Presentation"
        self.old_centroids = self.centroids.copy()
        
        for i in range(self.K):
            assigned_points = self.X[self.labels == i] 
            # keep the points with the same label as centroid "i"
            
            if len(assigned_points) > 0:  # Verify that there are assigned points
                self.centroids[i] = np.sum(assigned_points, axis=0)/len(assigned_points)
            else:
                # If there are no points, keep the previous centroid or reinitialize
                self.centroids[i] = self.old_centroids[i]  # Option 1: Keep previous position

    def converges(self):
        """
        Checks if there is a difference between current and old centroids
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # the hint suggests using the max_iter value
        if self.num_iter >= self.options['max_iter']:
            return True
        
        distances = np.sqrt(np.sum((self.centroids - self.old_centroids)**2, axis=1))

        return np.all(distances <= self.options['tolerance'])
        # we could also use "return np.allclose(self.centroids, self.old_centroids, atol=self.options['tolerance'])"
        # as seen in the example exercises from the slides, using np.allclose returns true if
        # the comparison between variables is equal within a small tolerance

    def fit(self):
        """
        Runs K-Means algorithm until it converges or until the number of iterations is smaller
        than the maximum number of iterations.
        """
        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        # following the pseudocode in the document:
        self._init_centroids()
        
        self.num_iter = 0
        
        while True:
            
            self.get_labels()
            self.get_centroids()
            #self.num_iter += 1
            
            if self.converges() or self.num_iter >= self.options['max_iter']:
                break
            self.num_iter += 1

    def withinClassDistance(self):
        """
         returns the within class distance of the current clustering
        """

        #######################################################
        ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
        ##  AND CHANGE FOR YOUR OWN CODE
        #######################################################
        distances = distance(self.X, self.centroids)
        wcd = np.sum(distances[np.arange(len(self.X)), self.labels] ** 2)
        # sumem les distancies que corresponen als centroides assignats a "distancies"
        # (formula del document)
        return wcd
    
    def betweenClassDistance(self):
        # sum squared distances between centroids
        c = self.centroids
        d = c[:, None, :] - c[None, :, :]
        return np.sum((d**2)) / 2

    def fisher_score(self):
        # Fisher criterion: between/intra
        bcd = self.betweenClassDistance()
        wcd = self.withinClassDistance()
        return (bcd / (wcd + 1e-8)) * 100

    def find_bestK(self, max_K, method='WCD', threshold=20):
        scores = []
        best_K = 2
    
        for K in range(2, max_K + 1):
            model = KMeans(self.X, K=K, options=self.options.copy())
            model.fit()
    
            if method == 'WCD' or method == 'elbow':
                current_wcd = model.withinClassDistance()
                scores.append(current_wcd)
    
                if method == 'WCD':
                    if len(scores) >= 2:
                        drop = 100 * (scores[-2] - scores[-1]) / scores[-2]
                        if drop < threshold:
                            best_K = K - 1
                            break
    
            elif method == 'Fisher':
                score = model.fisher_score()
                scores.append(score)
            elif method == 'BCD':
                score = model.betweenClassDistance()
                scores.append(score)
    
        if method == 'Fisher' or method == 'BCD':
            best_index = np.argmax(scores)
            best_K = best_index + 2  # Because K starts at 2
        elif method == 'elbow':
            # Apply the geometric method to find the "elbow"
            x = np.arange(2, max_K + 1)
            y = np.array(scores)
    
            # Line from the first to the last point
            p1 = np.array([x[0], y[0]])
            p2 = np.array([x[-1], y[-1]])
            line_vec = p2 - p1
    
            distances = []
            for i in range(len(x)):
                p = np.array([x[i], y[i]])
                vec = p - p1
                # Orthogonal projection and distance to the segment
                proj_len = np.dot(vec, line_vec) / np.linalg.norm(line_vec)
                proj_point = p1 + proj_len * line_vec / np.linalg.norm(line_vec)
                distance = np.linalg.norm(p - proj_point)
                distances.append(distance)
    
            best_index = np.argmax(distances)
            best_K = x[best_index]
    
        self.K = best_K
        return best_K

def distance(X, C):
    """
    Calculates the distance between each pixel and each centroid
    Args:
        X (numpy array): PxD 1st set of data points (usually data points)
        C (numpy array): KxD 2nd set of data points (usually cluster centroids points)

    Returns:
        dist: PxK numpy array position ij is the distance between the
        i-th point of the first set an the j-th point of the second set
    """
    """
    # THIS FUNCTION IS CORRECT BUT TAKES 50+ SECONDS TO SOLVE DUE TO PYTHON LOOPS
    N = X.shape[0]
    K = C.shape[0]
    # now we do need to use shape[0] because it is not fixed
    
    dist = np.zeros((N, K))
    # initialize a zero matrix of size based on N and K
    # to iterate over it using the distance formula
    
    for i in range(N):
        for j in range(K):
            dist[i, j] = np.sqrt(np.sum((X[i] - C[j]) ** 2))
            
    return dist
    """
    # optimized distance function
    diferencia = np.expand_dims(X, axis=1) - C  # Calculates the difference between each point of X and each center in C
    distancia = np.linalg.norm(diferencia, axis=2)  # Calculates the Euclidean distance
     
    return distancia

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    #########################################################
    ##  YOU MUST REMOVE THE REST OF THE CODE OF THIS FUNCTION
    ##  AND CHANGE FOR YOUR OWN CODE
    #########################################################
    color_probs = utils.get_color_prob(centroids)
    color_indices = np.argmax(color_probs, axis=1)
    labels = [utils.colors[i] for i in color_indices]
    
    return labels