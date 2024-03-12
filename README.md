
# AdaBoost Implementation in Python

## Overview
This project implements the AdaBoost (Adaptive Boosting) algorithm from scratch in Python, focusing on binary classification. AdaBoost is a popular ensemble method that combines multiple weak learners to create a strong classifier. The implementation explores the effects of various hyperparameters, including the number of weak learners and their performance margins, on training and validation accuracy.

## Features
- Implementation of the AdaBoost algorithm for binary classification.
- Custom Decision Stump used as a weak learner.
- Analysis of hyperparameters' impact on model performance.
- Utilization of cross-validation for fine-tuning and model evaluation.

## Dependencies
To run this project, you will need:
- Python 3.x
- NumPy
- Optional: Scikit-learn (for dataset handling and additional functionality)

## Quick Start
1. Clone the repository to your local machine.
2. Ensure you have Python 3.x installed along with the required packages.
3. The project contains two main scripts:
   - `AdaBoost_binary.py`: The main AdaBoost implementation.
   - `CrossValidation_binary.py`: Utility script for performing cross-validation and model evaluation.

To run the AdaBoost algorithm:
```bash
python AdaBoost_binary.py
```

To perform cross-validation:
```bash
python CrossValidation_binary.py
```

## How It Works
### AdaBoost
The `AdaBoost` class in `AdaBoost_binary.py` constructs a strong classifier by iteratively adding weak learners, specifically Decision Stumps in this case. It focuses on incorrectly classified instances by adjusting their weights after each iteration, improving the model's accuracy on challenging cases.

### Decision Stump
The `DecisionStump` class serves as the weak learner. It finds the best threshold for a given feature to split the data into two groups, minimizing the weighted classification error.

### Cross-Validation
`CrossValidation_binary.py` is used to evaluate the AdaBoost model's performance. It helps in understanding the effects of different hyperparameters on the model's accuracy and generalization capability.

## Contact
For any queries regarding this project, please contact:

Mehdi Abbassi  
Data Science and Economics  
University of Milan  
Email: mehdi.abbassi@studenti.unimi.it
