import numpy as np

def sigmoid(z: np.ndarray) -> np.ndarray:
    """The sigmoid function.
    
    Parameters
    ----------
    z : np.ndarray
        The array on which to elementwise compute the sigmoid function

    Returns
    -------
    np.ndarray
        The sigmoid value (1 / (1 + exp(-z)))
    """
    z = np.clip(z, -10, 10)
    return   1 / (1 + np.exp(-z))

def cost(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> float:
    """Computes the cost using theta as the parameters for logistic regression.
    
    Parameters
    ----------
    theta : np.ndarray
        The parameters of the logit regression, of shape (n_features,)
    X : np.ndarray
        The points, of shape (n_sample, n_features)
    y : np.ndarray
        The labels, of shape (n_sample, 1)

    Returns
    -------
    float
        The sum of the cost for each sample
    """
    z = X @ theta
    return -np.sum( y * np.log(sigmoid(z)) + (1 - y)*np.log(1 - sigmoid(z)) ) / X.shape[0]

def costDecentralized(theta: np.ndarray, X: np.ndarray, y: np.ndarray, G : np.ndarray, D : np.ndarray, c : np.ndarray, mu: np.ndarray) -> float:
    """    
    Parameters
    ----------
    theta : np.ndarray
        The parameters of the logit regression, of shape (Nb_models, n_features)
    X : np.ndarray
        The points, of shape (Nb_models, n_sample, n_features)
    y : np.ndarray
        The labels, of shape (Nb_models, n_sample, 1)

    Returns
    -------
    float
        The sum of the cost for each sample
    """
    terme = 0
    Nb_model = len(X)
    for i in range(Nb_model):
        for j in range(i+1, Nb_model):
            terme += G[i,j] * np.linalg.norm(theta[i] - theta[j])**2
    
    terme2 = 0
    for i in range(Nb_model):
        terme2 += D[i] * c[i] * cost(theta[i], X[i], y[i])
    
    return 0.5 * terme + mu * terme2

def compute_grad(theta: np.ndarray, X:np.ndarray, y:np.ndarray) -> np.ndarray:
    """Computes the gradient of the cost with respect to the parameters.
    
    Parameters
    ----------
    theta : np.ndarray
        The parameters of the logit regression, of shape (n_features,)
    X : np.ndarray
        The points, of shape (n_sample, n_features)
    y : np.ndarray
        The labels, of shape (n_sample, 1)

    Returns
    -------
    np.ndarray
        The gradient, of shape (n_features,)
    """
    return np.sum((sigmoid(X @ theta) - y) * X.T, axis=1)