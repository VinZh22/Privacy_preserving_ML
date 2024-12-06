import matplotlib.pyplot as plt
import numpy as np

def plot_curves(num_agents: int, costs_train: np.ndarray, costs_test: np.ndarray, title: str):
    fig, axes = plt.subplots(nrows=2, ncols=int(num_agents/2), figsize=(20, 4))

    for i in range(num_agents):
        row = i // (num_agents // 2)
        col = i % (num_agents // 2)
        axes[row, col].plot(costs_train[:, i], label='Train')
        axes[row, col].plot(costs_test[:, i], label='Test')
        axes[row, col].set_title(f'Agent {i+1}')

    plt.tight_layout()
    plt.legend()
    plt.show()


    plt.plot(np.mean(costs_train, axis=1), label="Avg Train")
    plt.plot(np.mean(costs_test, axis=1), label="Avg Test")
    plt.title(title)
    plt.legend()
    plt.show()

def plot_curves_non_zero(num_agents: int, costs_train: np.ndarray, costs_test: np.ndarray, title: str):
    fig, axes = plt.subplots(nrows=2, ncols=int(num_agents/2), figsize=(20, 4))

    for i in range(num_agents):
        row = i // (num_agents // 2)
        col = i % (num_agents // 2)
        axes[row, col].plot(costs_train[:, i][costs_train[:, i] != 0], label='Train')
        axes[row, col].plot(costs_test[:, i][costs_test[:, i] != 0], label='Test')
        axes[row, col].set_title(f'Agent {i+1}')

    plt.tight_layout()
    plt.legend()
    plt.show()

    train_mean = np.mean(costs_train, axis=1)
    test_mean = np.mean(costs_test, axis=1)

    plt.plot(train_mean[train_mean != 0], label="Avg Train")
    plt.plot(test_mean[test_mean != 0], label="Avg Test")
    plt.title(title)
    plt.legend()
    plt.show()

def reduce_cost_matrix(costs):
    """Reduces the shape of the cost matrix the minimal number of non-zero elements of an agent
    """    
    # Find the count of non-zero elements in each column
    non_zero_counts = [np.sum(costs[:, col] != 0) for col in range(costs.shape[1])]

    # Determine the minimum number of non-zero elements across all columns
    min_non_zero_count = min(non_zero_counts)

    # Create a reduced array with the determined size
    reduced_array = np.zeros((min_non_zero_count, costs.shape[1]))

    # Extract the first `min_non_zero_count` non-zero elements from each column
    for col in range(costs.shape[1]):
        non_zero_elements = costs[:, col][costs[:, col] != 0]
        reduced_array[:, col] = non_zero_elements[:min_non_zero_count]
    
    return reduced_array

def shift_non_zero_costs_to_front(costs): 
    """keeps the original shape of the cost matrix but shifts the non-zero elements to the front
    """
    # Initialize a new array with the same shape, filled with zeros
    shifted_array = np.zeros_like(costs)

    # For each column, extract the non-zero elements and place them at the start
    for col in range(costs.shape[1]):
        non_zero_elements = costs[:, col][costs[:, col] != 0]  # Extract non-zero elements
        shifted_array[:len(non_zero_elements), col] = non_zero_elements  # Place at the start
    return shifted_array