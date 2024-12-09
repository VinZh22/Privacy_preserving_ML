import numpy as np
from sklearn.metrics import accuracy_score


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
    return 1 / (1 + np.exp(-z))


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
    return (
        -np.sum(y * np.log(sigmoid(z)) + (1 - y) * np.log(1 - sigmoid(z))) / X.shape[0]
    )


## cost with L2 regularization forcing a strong convexity of at least lambda_
def costL2(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> float:
    """Computes the cost using theta as the parameters for logistic regression.

    Parameters
    ----------
    theta : np.ndarray
        The parameters of the logit regression, of shape (n_features,)
    X : np.ndarray
        The points, of shape (n_sample, n_features)
    y : np.ndarray
        The labels, of shape (n_sample, 1)
    lambda_ : float
        The regularization parameter

    Returns
    -------
    float
        The sum of the cost for each sample
    """
    z = X @ theta
    return (
        cost(theta, X, y)
        + lambda_ * np.linalg.norm(theta) ** 2
    )

def costDecentralized(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_agents: np.ndarray,
    G: np.ndarray,
    D: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
) -> float:
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
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            terme += G[i, j] * np.linalg.norm(theta[i] - theta[j]) ** 2

    terme2 = 0
    for i in range(num_agents):
        terme2 += D[i] * c[i] * cost(theta[i], X[i], y[i])

    return 0.5 * terme + mu * terme2

def costDecentralizedL2(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_agents: np.ndarray,
    G: np.ndarray,
    D: np.ndarray,
    c: np.ndarray,
    mu: np.ndarray,
    lambda_: float,
) -> float:
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
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            terme += G[i, j] * np.linalg.norm(theta[i] - theta[j]) ** 2

    terme2 = 0
    for i in range(num_agents):
        terme2 += D[i] * c[i] * costL2(theta[i], X[i], y[i], lambda_)

    return 0.5 * terme + mu * terme2

        

def compute_grad(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
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

## gradient with L2 regularization
def compute_gradL2(theta: np.ndarray, X: np.ndarray, y: np.ndarray, lambda_: float) -> np.ndarray:
    """Computes the gradient of the cost with respect to the parameters.

    Parameters
    ----------
    theta : np.ndarray
        The parameters of the logit regression, of shape (n_features,)
    X : np.ndarray
        The points, of shape (n_sample, n_features)
    y : np.ndarray
        The labels, of shape (n_sample, 1)
    lambda_ : float
        The regularization parameter

    Returns
    -------
    np.ndarray
        The gradient, of shape (n_features,)
    """
    return compute_grad(theta, X, y) + 2 * lambda_ * theta


def accuracy(theta: np.ndarray, X: np.ndarray, y: np.ndarray):
    # Compute probabilities
    probabilities = sigmoid(X @ theta)

    # Make predictions (threshold at 0.5)
    predictions = (probabilities >= 0.5).astype(int)

    return accuracy_score(y, predictions)



def accuracyAll(num_agents, theta, X, y) -> float:
    individual_acc = np.zeros((num_agents))
    for i in range(num_agents):
        individual_acc[i] = accuracy(theta[i], X[i], y[i])
    acc = np.mean(individual_acc)
    return acc

def accuracyMajorityVote(num_agents, theta, X, y) -> float:
    """Works only for the testing set (no individual X / y per agent)"""
    predictions = np.zeros((num_agents, X.shape[0]))
    for i in range(num_agents):
        probs = sigmoid(X @ theta[i])
        predictions[i] = (probs >= 0.5).astype(int)
        
    maj_vote = (np.sum(predictions, axis=0) >= (predictions.shape[0] / 2)).astype(int)
    accuracy = np.mean(maj_vote == y)

    return accuracy
