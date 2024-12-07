import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d

def plot_curves(num_agents: int, costs_train: np.ndarray, costs_test: np.ndarray, title: str):
    fig, axes = plt.subplots(nrows=2, ncols=int(num_agents/2), figsize=(20, 4))

    for i in range(num_agents):
        row = i // (num_agents // 2)
        col = i % (num_agents // 2)
        axes[row, col].plot(costs_train[:, i], label='Train')
        axes[row, col].plot(costs_test[:, i], label='Test')
        axes[row, col].set_title(f'Agent {i+1}')

    plt.tight_layout()
    plt.suptitle(title, fontsize="x-large")
    fig.subplots_adjust(top=0.85)
    plt.legend()
    plt.show()


    plt.plot(np.mean(costs_train, axis=1), label="Avg Train")
    plt.plot(np.mean(costs_test, axis=1), label="Avg Test")
    plt.title(title)
    plt.legend()
    plt.show()

def propagate_last_value(arr):
    # Create a copy to propagate values
    propagated = arr.copy()
    for col in range(arr.shape[1]):
        last_value = 0
        for row in range(arr.shape[0]):
            if propagated[row, col] == 0 and last_value != 0:
                propagated[row, col] = last_value
            else:
                last_value = propagated[row, col]
    return propagated

def plot_curves_non_zero(num_agents: int, costs_train: np.ndarray, costs_test: np.ndarray, title: str):
    # Determine the maximum length of non-zero elements for any agent
    max_length = max(np.sum(costs_train != 0, axis=0).max(), np.sum(costs_test != 0, axis=0).max())

    fig, axes = plt.subplots(nrows=2, ncols=int(num_agents/2), figsize=(20, 4))

    for i in range(num_agents):
        row = i // (num_agents // 2)
        col = i % (num_agents // 2)
        train_data = costs_train[:, i][costs_train[:, i] != 0]
        test_data = costs_test[:, i][costs_test[:, i] != 0]
        axes[row, col].plot(np.arange(len(train_data)), train_data, label='Train')
        axes[row, col].plot(np.arange(len(test_data)), test_data, label='Test')
        axes[row, col].set_xlim(0, max_length)
        axes[row, col].set_title(f'Agent {i+1}')

    plt.tight_layout()
    plt.suptitle(title, fontsize="x-large")
    fig.subplots_adjust(top=0.85)
    plt.legend()
    plt.show()

    # # Mask for non-zero (active) values
    # active_mask_train = costs_train != 0
    # active_mask_test = costs_test != 0

    # # Compute the mean over agents at each timestep, ignoring zeros
    # train_mean = np.sum(costs_train, axis=1) / np.sum(active_mask_train, axis=1)
    # test_mean = np.sum(costs_test, axis=1) / np.sum(active_mask_test, axis=1)

    # # Replace NaN values (if any timesteps had no active agents) with 0
    # train_mean = np.nan_to_num(train_mean)
    # test_mean = np.nan_to_num(test_mean)

    train_mean = np.mean(propagate_last_value(costs_train), axis=1)
    test_mean = np.mean(propagate_last_value(costs_test), axis=1)

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
    """
    Reduces the array size to the max number of non-zero elements after shifting
    non-zero elements to the front for each column.
    """
    # Calculate the number of non-zero elements in each column
    non_zero_counts = np.count_nonzero(costs, axis=0)
    
    # Determine the maximum number of non-zero elements across all columns
    max_non_zero = np.max(non_zero_counts)
    
    # Initialize a new array with reduced size
    reduced_array = np.zeros((max_non_zero, costs.shape[1]), dtype=costs.dtype)
    
    # For each column, extract the non-zero elements and place them at the start
    for col in range(costs.shape[1]):
        non_zero_elements = costs[:, col][costs[:, col] != 0]  # Extract non-zero elements
        reduced_array[:len(non_zero_elements), col] = non_zero_elements  # Place at the start
    
    return reduced_array