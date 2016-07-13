import numpy as np
import ctypes
import sys

libpath = sys.prefix + '/' + sys.lib + '/libgaussianmixturemodel.so.1'
gaussianmixturemodel_lib = ctypes.cdll.LoadLibrary(libpath)

class GaussianMixture(object):
    """ Gaussian Mixture Model for function approximation
    
        This model clusters input points and associates an output value
        to each cluster. The value predicted for any point is a weighted
        sum of the values of all the clusters: the closer a cluster is
        to the input point, the more weight it has.
        
        The model tries to be clever with the clusters and detects when
        to add or remove ones. Each cluster has a covariance matrix that
        allows it to span an "oval" region of the input space.
        
        See the documentation of __init__ for details about how the model
        is configured.
    """

    def __init__(self, input_dim, output_dim, initial_variance, max_error):
        """ Create and configure a Gaussian Mixture Model
            
            input_dim, output_dim: size of input/output vectors
            initial_variance: initial (radial) variance of new clusters. A
                              high number produces smooth predictions
                              (high bias, low variance). A lower number leads
                              to sharper predictions.
            max_error: If a training point has an error above this number, a
                       new cluster is added. This represents the maximum error
                       allowed before the model takes action.
        """
        self.obj = gaussianmixturemodel_lib.GaussianMixture_New(
            ctypes.c_int(input_dim),
            ctypes.c_int(output_dim),
            ctypes.c_float(initial_variance),
            ctypes.c_float(max_error))
        self.dout = output_dim

    def __del__(self):
        gaussianmixturemodel_lib.GaussianMixture_Del(self.obj)

    def setValue(self, input, output):
        """ Set the value associated with an input point.
        
            Both parameters are one-dimensional lists, tuples, (N) ndarrays
            or (1, N) ndarrays.
        """
        inT = self._array(input)
        outT = self._array(output)

        gaussianmixturemodel_lib.GaussianMixture_SetValue(
            self.obj,
            inT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inT.shape[1],
            outT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            outT.shape[1]
        )
        
    def value(self, input):
        """ Predict the value of a point
        
            The value is returned as a Numpy column vector of shape (1, output_dim)
        """
        rs = np.empty((1, self.dout), np.float32)
        inT = self._array(input)
        
        gaussianmixturemodel_lib.GaussianMixture_GetValue(
            self.obj,
            inT.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            inT.shape[1],
            rs.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )
        
        return rs
    
    def numClusters(self):
        """ Number of clusters in the model
        
            This allows to measure how complex the model is. If too many clusters
            are created and the model becomes too slow, try increasing initial_variance.
            
            Note that for higher initial variances, the model will make larger
            errors. Don't forget to also increase max_error in order to prevent
            too many clusters from being created just because the bias of the model
            is high.
        """
        return gaussianmixturemodel_lib.GaussianMixture_NumClusters(self.obj)
    
    def _array(self, a):
        """ Map a (ndarray, list, typle, etc) to a column vector of float32
        """
        if type(a) is np.ndarray:
            # Already a Numpy array, check shape
            if len(a.shape) == 1:
                a = a.reshape((1, a.shape[0]))   # Make column vector
            elif len(a.shape) != 2 or a.shape[0] != 1:
                raise ValueError("GaussianMixture can only process row or column vectors")
            
            return a.astype(np.float32)
        else:
            # Try to convert to array
            return np.array(a, dtype=np.float32, ndmin=2)