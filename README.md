# Privacy_preserving_ML
Goals : 
- Plot the relationship between $\epsilon_i$ and $\delta_i$ (in Theorem_1.ipynb)
- Implement the protocol and test: convergence, differential-privacy, practical testing with certain datasets (in multi_agents.ipynb)
- Compare convergence between DP and no DP case (in multi_agents.ipynb)
- Compare the impact of dimension (in diff_analysis.ipynb)
- Compare the impact of $\mu$ in the objective function (in communication_analysis.ipynb)
- Compare the impact of a weight matrix that changes at each iteration (in multi_agent_changingGraph.ipynb)

The basic functions are in the src file. `logistic_reg.py` for the regression (with the cost function, its gradient and how to compute the accuracy), and `privacy_ml.py` used for the training of $\Theta$ with the given protocol.

To install the modules, do : `pip install -r requirements.txt for the modules`