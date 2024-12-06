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