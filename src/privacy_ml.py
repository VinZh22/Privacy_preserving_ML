import numpy as np

import src.logistic_reg as lr

def split_horizontally(X, y, num_subsets=10):   
    indices = np.argsort(X[:, 0])

    X = X[indices]
    y = y[indices]

    limiteParAgent = np.random.choice(range(1, X.shape[0]), num_subsets-1, replace=False)
    limiteParAgent = np.sort(limiteParAgent)
    print(limiteParAgent)

    X_agent = []
    Y_agent = []

    for i in range(num_subsets):
        if i == 0:
            X_agent.append(X[:limiteParAgent[i]])
            Y_agent.append(y[:limiteParAgent[i]])
        elif i == num_subsets-1:
            X_agent.append(X[limiteParAgent[i-1]:])
            Y_agent.append(y[limiteParAgent[i-1]:])
        else:
            X_agent.append(X[limiteParAgent[i-1]:limiteParAgent[i]])
            Y_agent.append(y[limiteParAgent[i-1]:limiteParAgent[i]])

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
        subsets.append(data[start_idx:start_idx + size])
        start_idx += size
    
    # Separate X and y in each subset
    subsets_X = [subset[:, :-1] for subset in subsets]
    subsets_y = [subset[:, -1] for subset in subsets]
    
    return subsets_X, subsets_y

def stepForward(theta: np.ndarray, X: np.ndarray, y: np.ndarray, num_agents: int = 10, mu = 0.05, c = np.ndarray, G = np.ndarray, D = np.ndarray) -> np.ndarray:
    for i in range(num_agents):
        L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1)**2)
        alpha = 1/(1 + mu * c[i] * L_i)
        terme = 0
        for j in range(num_agents):
            if G[i,j] == 1:
                terme += (G[i,j] / D[i]) * theta[j]
        terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])
        
        theta[i] = (1-alpha) * theta[i] + alpha * terme
    
    return theta

def stepForwardMono(theta: np.ndarray, X: np.ndarray, y: np.ndarray) -> np.ndarray:
    L_i = 0.25 * np.sum(np.linalg.norm(X, axis=1)**2)    
    theta = theta - (1/L_i) * lr.compute_grad(theta, X, y)
    return theta

def stepForwardAlone(theta: np.ndarray, X: np.ndarray, y: np.ndarray, num_agents: int = 10, mu = 0.05, c = np.ndarray, G = np.ndarray, D = np.ndarray) -> np.ndarray:
    for i in range(num_agents):
        L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1)**2)        
        theta[i] = theta[i] - (1/L_i) * lr.compute_grad(theta[i], X[i], y[i])
    
    return theta


def stepForward_2(theta: np.ndarray, X: np.ndarray, y: np.ndarray, i: int, num_agents: int = 10, mu = 0.05, c = np.ndarray, G = np.ndarray, D = np.ndarray) -> np.ndarray:
    L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1)**2)
    alpha = 1/(1 + mu * c[i] * L_i)
    terme = 0
    for j in range(num_agents):
        if G[i,j] == 1:
            terme += (G[i,j] / D[i]) * theta[j]
    terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])
    
    theta[i] = (1-alpha) * theta[i] + alpha * terme
    
    return theta

def stepForwardPrivate(theta: np.ndarray, X: np.ndarray, y: np.ndarray, num_agents: int = 10, mu = 0.05, c = np.ndarray, G = np.ndarray, D = np.ndarray, L_0 = np.ndarray, epsilon = np.ndarray) -> np.ndarray:
    for i in range(num_agents):
        L_i = 0.25 * np.sum(np.linalg.norm(X[i], axis=1)**2)
        alpha = 1/(1 + mu * c[i] * L_i)
        terme = 0
        for j in range(num_agents):
            if G[i,j] == 1:
                terme += (G[i,j] / D[i]) * theta[j]
        terme -= mu * c[i] * lr.compute_grad(theta[i], X[i], y[i])
        s = 2 * L_0 / (epsilon * X[i].shape[0])
        noise = np.random.laplace(0, s, theta[i].shape[0])
        
        terme += noise
        theta[i] = (1-alpha) * theta[i] + alpha * terme
    
    return theta