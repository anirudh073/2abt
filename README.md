# 2abt-analysis
Analysis code for 2-armed-bandit task.

### Import data and convert it to a session-wise dataframe:
Use BanditAnalysisFuncitons.py

### Simulate q-learning agent with bi-directional updates and obtain a session-wise dataframe:
Use structured_agent.py.

### Obtain reward-rate and regret plots and compare performance across animals:
Use animal_comparison.ipynb. Uses functions form BanditAnalysisFunctions.py

### Obtain conditional-probabilities and bar-plots:
Use cprobs_functions.py (some functions are from Celia Beron's code in https://github.com/celiaberon/2ABT_behavior_models)
Analysis using these functions is in cprobs.ipynb

### Use logistic regression with a policy of choice (greedy/e-greedy/softmax/stochastic/random) to model animal behaviour:
Use logreg_celiaberon.ipynb. Uses code from https://github.com/celiaberon/2ABT_behavior_models. Download the whole repository save it with all other analysis files. 
