#Author: Christian Elias Anderssen Dalan, April 2022
import numpy as np
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

def create_network(training_data, fitting_data, kernel = None):
    """
    Function that creates a regressor and fits it to the input data
    
    Parameters:
    
    - Training data: Training features as a numpy ndarray or nested list. 
    - Fitting data: Training labels as a numpy array
    - Kernel: The kernel we want our regressor to use when calculating the covariance matrix. Default is a radial basis function
              of the form: $e^{-\frac{d(x_i, x_j)}{2l^2}}$ where $d(x_i, x_j)$ is the eucledian distance between $x_i$ and $x_j$
              and l is a length scale parameter, which may be tuned to increase or decrease the covariance between $x_i$ and $x_j$
    Returns:
    - Gpr: Our regressor fitted to the input data.
    """
    if kernel == None:
        m = training_data.shape[1]
        kernel = RBF(length_scale=0.5*np.ones(shape=(m), dtype=float))
    
    gpr = GaussianProcessRegressor(kernel = kernel, random_state=42).fit(training_data, fitting_data)
    return gpr

def test_network(test_features, test_labels, regressor):
    """
    Function that evaluates a regressor to some data.
    
    Parameters:
    
    - Test_features: Features that we want our regressor to predict the corresponding labels to
    - Test_labels: True output labels so that our regressor can compare with the predicted labels
    - Regressor: Our gpr 
    
    Returns:
    
    - Score: A fit score using the gprs own score-function.
    """
    score = regressor.score(test_features, test_labels)
    return score