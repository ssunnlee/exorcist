# exorcist
Demon Attack AI

PPO.py: Includes the class PPO, which given the hyperparameters, state space size, and action space size, returns a PPO agent that can be trained and evaluated.

GA.py: Includes the genetic algorithm implementation. Running the main function (GA_main()) starts a GA loop with the given parameters, tunes the optimal hyperparameters, trains a PPO agent using the hyperparameters, and evaluates it. It writes the relevant results in .pkl files.

BO.py: Includes the Bayesian Optimization implementation. Running the main function (BO_main()) starts a BO loop with the given parameters, tunes the optimal hyperparameters, trains a PPO agent using the hyperparameters, and evaluates it. It writes the relevant results in .pkl files.

project.ipynb: Includes scripts for calculating statistics and graphs based on data given by GA and BO from .pkl files. The file names may have to be edited, depending on where it was written to.

project.html: html version of the above Jupyter Notebook with results from our own runthrough.

main.py: Runs both GA_main() and BO_main() for one test run on GA and BO each.

bayesian_eval.pkl, genetic_eval.pkl: pickle files including results for the 100 evaluation runs from our GA and BO.

To run the project, simply run main.py, and then use project.ipynb to show final results. project.ipynb may have to be edited for file names.
