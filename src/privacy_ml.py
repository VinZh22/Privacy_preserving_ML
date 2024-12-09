import numpy as np

import src.logistic_reg as lr


def split_horizontally(X, y, num_subsets=10):
    indices = np.argsort(X[:, 0])

    X = X[indices]
    y = y[indices]

    limiteParAgent = np.random.choice(
        range(1, X.shape[0]), num_subsets - 1, replace=False
    )
    limiteParAgent = np.sort(limiteParAgent)
    print(limiteParAgent)

    X_agent = []
    Y_agent = []

    for i in range(num_subsets):
        if i == 0:
            X_agent.append(X[: limiteParAgent[i]])
            Y_agent.append(y[: limiteParAgent[i]])
        elif i == num_subsets - 1:
            X_agent.append(X[limiteParAgent[i - 1] :])
            Y_agent.append(y[limiteParAgent[i - 1] :])
        else:
            X_agent.append(X[limiteParAgent[i - 1] : limiteParAgent[i]])
            Y_agent.append(y[limiteParAgent[i - 1] : limiteParAgent[i]])

    return X_agent, Y_agent


def split_into_random_subsets(X, y, num_subsets=10, random_state=42):
    # Combine X and y to maintain correspondence
    data = np.column_stack((X, y))

    # Shuffle the combined data
    if random_state is not None:
        np.random.seed(random_state)
    np.random.shuffle(data)

    # Generate random sizes for the subsets
    random_sizes = np.random.rand(num_subsets)  # Random values
    random_sizes /= random_sizes.sum()  # Normalize to sum to 1
    random_sizes = (random_sizes * len(data)).astype(int)  # Scale to dataset size

    # Adjust sizes to ensure they sum exactly to len(data)
    random_sizes[-1] += len(data) - random_sizes.sum()

    # Split the data into subsets
    subsets = []
    start_idx = 0
    for size in random_sizes:
        subsets.append(data[start_idx : start_idx + size])
        start_idx += size

    # Separate X and y in each subset
    subsets_X = [subset[:, :-1] for subset in subsets]
    subsets_y = [subset[:, -1] for subset in subsets]

    return subsets_X, subsets_y


def stepForward(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_agents: int = 10,
    mu: float = 0.05,
    c: np.ndarray = None,
    G: np.ndarray = None,
    D: np.ndarray = None,
) -> np.ndarray:
    for i in range(num_agents):
        L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1) ** 2)
        alpha = 1 / (1 + mu * c[i] * L_i)
        terme = 0
        for j in range(num_agents):
            if G[i, j] > 0:
                terme += (G[i, j] / D[i]) * theta[j]
        terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])

        theta[i] = (1 - alpha) * theta[i] + alpha * terme

    return theta


def stepForwardMono(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    L_i = 0.25 * np.sum(np.linalg.norm(X, axis=1) ** 2)
    theta = theta - (1 / L_i) * lr.compute_grad(theta, X, y)
    return theta


def stepForwardAlone(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_agents: int = 10,
) -> np.ndarray:
    for i in range(num_agents):
        L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1) ** 2)
        theta[i] = theta[i] - (1 / L_i) * lr.compute_grad(theta[i], X[i], y[i])

    return theta


def stepForward_2(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    num_agents: int = 10,
    mu: float = 0.05,
    c: np.ndarray = None,
    G: np.ndarray = None,
    D: np.ndarray = None,
) -> np.ndarray:
    L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1) ** 2)
    alpha = 1 / (1 + mu * c[i] * L_i)
    terme = 0
    for j in range(num_agents):
        if G[i, j] > 0:
            terme += (G[i, j] / D[i]) * theta[j]
    terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])

    theta[i] = (1 - alpha) * theta[i] + alpha * terme

    return theta


def stepForwardPrivate(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    num_agents: int = 10,
    mu: float = 0.05,
    c: np.ndarray = None,
    G: np.ndarray = None,
    D:np.ndarray = None,
    L_0:np.ndarray = None,
    epsilon: float = None,
) -> np.ndarray:
    for i in range(num_agents):
        L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1) ** 2)
        alpha = 1 / (1 + mu * c[i] * L_i)
        terme = 0
        for j in range(num_agents):
            if G[i, j] > 0:
                terme += (G[i, j] / D[i]) * theta[j]
        terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])
        s = 2 * L_0 / (epsilon * X[i].shape[0])
        noise = np.random.laplace(0, s, theta[i].shape[0])

        terme += noise
        theta[i] = (1 - alpha) * theta[i] + alpha * terme

    return theta


def stepForwardPrivate_2(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    num_agents: int = 10,
    mu:float = 0.05,
    c: np.ndarray = None,
    G: np.ndarray = None,
    D: np.ndarray = None,
    L_0: np.ndarray = None,
    epsilon: float = None,
) -> np.ndarray:
    L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1) ** 2)
    alpha = 1 / (1 + mu * c[i] * L_i)
    terme = 0
    for j in range(num_agents):
        if G[i, j] > 0:
            terme += (G[i, j] / D[i]) * theta[j]
    terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])
    s = 2 * L_0 / (epsilon * X[i].shape[0])
    noise = np.random.laplace(0, s, theta[i].shape[0])

    terme += noise
    theta[i] = (1 - alpha) * theta[i] + alpha * terme

    return theta

def stepForwardPrivate_L2(
    theta: np.ndarray,
    X: np.ndarray,
    y: np.ndarray,
    i: int,
    num_agents: int = 10,
    mu:float = 0.05,
    lambda_ = 0.05,
    c: np.ndarray = None,
    G: np.ndarray = None,
    D: np.ndarray = None,
    L_0: np.ndarray = None,
    epsilon: float = None,
) -> np.ndarray:
    L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1) ** 2)
    alpha = 1 / (1 + mu * c[i] * L_i)
    terme = 0
    for j in range(num_agents):
        if G[i, j] > 0:
            terme += (G[i, j] / D[i]) * theta[j]
    terme -= mu * c[i] * lr.compute_gradL2(theta[i], X[i], y[i], lambda_)
    s = 2 * L_0 / (epsilon * X[i].shape[0])
    noise = np.random.laplace(0, s, theta[i].shape[0])

    terme += noise
    theta[i] = (1 - alpha) * theta[i] + alpha * terme

    return theta

def computeSigma(D, c, mu, L_0, nb_agent, lambda_ = 0.05):
    res = 100000
    for i in range(nb_agent):
        res = min(res, mu * c[i] * D[i] * lambda_)
    return res

def computeLMax(D, c, mu, L_0, X_agent):
    L = 0
    for i in range(len(D)):
        L_i = 0.25 * np.sum(np.linalg.norm(X_agent[i], axis=1) ** 2)
        L = max(L, D[i]* (1 + mu*c[i] * L_i))
    return L

def construct_G_D(num_agents):
    G = np.zeros((num_agents, num_agents))
    D = np.zeros(num_agents)
        
    for i in range(num_agents):
        for j in range(i+1, num_agents):
            if j <= i+3 and j >= i-3:
                G[i,j] = np.random.rand()
                G[j,i] = G[i, j]  
        G[i, i] = 0

    for i in range(num_agents):
        D[i] = np.sum(G[i,:])
    
    return G, D