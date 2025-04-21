# Project 2

## Boosting Trees

Implement again from first principles the gradient-boosting tree classification algorithm (with the usual fit-predict interface as in Project 1) as described in Sections 10.9-10.10 of Elements of Statistical Learning (2nd Edition). Answer the questions below as you did for Project 1. In this assignment, you'll be responsible for developing your own test data to ensure that your implementation is satisfactory. (Hint: Use the same directory structure as in Project 1.)

The same "from first principals" rules apply; please don't use SKLearn or any other implementation. Please provide examples in your README that will allow the TAs to run your model code and whatever tests you include. As usual, extra credit may be given for an "above and beyond" effort.

As before, please clone this repo, work on your solution as a fork, and then open a pull request to submit your assignment. *A pull request is required to submit and your project will not be graded without a PR.*

Put your README below. Answer the following questions.

* What does the model you have implemented do and when should it be used?
* How did you test your model to determine if it is working reasonably correctly?
* What parameters have you exposed to users of your implementation in order to tune performance? (Also perhaps provide some basic usage examples.)
* Are there specific inputs that your implementation has trouble with? Given more time, could you work around these or is it fundamental?



----------------------------------------------------------------------------

# Project 2: Gradient Boosting Trees from First Principles

## Overview

This project implements a **Gradient Boosting Tree classifier** from scratch, as described in **Sections 10.9–10.10 of _The Elements of Statistical Learning (2nd Edition)_**. 

The implementation is fully custom — no external ML libraries are used to build the model. Only `numpy` is used for array operations and `matplotlib`/`scikit-learn` are used for **testing and visualization only**.

---

## Project Structure

project2/ ├── models/ │ ├── decision_tree.py # Custom decision tree (for regression) │ └── gradient_boosting.py # Main boosting class and loss functions ├── tests/ │ └── simple_moons_demo.py # Example test using synthetic data ├── requirements.txt # Only necessary dependencies └── README.md #


