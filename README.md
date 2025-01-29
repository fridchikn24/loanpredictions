# Loan Approval Prediction

This project utilizes a stacking classifier to predict loan approval decisions based on applicant data.

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Project Overview

Accurately predicting loan approvals is crucial for financial institutions to manage risk and serve customers effectively. This project employs a stacking classifier approach, combining multiple machine learning models to enhance prediction accuracy.

## Dataset

The dataset used in this project includes various features related to loan applicants, such as:

- **Gender**: Male, Female
- **Marital Status**: Married, Single
- **Education**: Graduate, Not Graduate
- **Dependents**: Number of dependents
- **Applicant Income**: Income of the applicant
- **Coapplicant Income**: Income of the coapplicant
- **Loan Amount**: Loan amount in thousands
- **Loan Amount Term**: Term of the loan in months
- **Credit History**: Credit history meets guidelines (1) or not (0)
- **Property Area**: Urban, Semi-Urban, Rural
- **Loan Status**: Loan approved (Y) or not (N)

## Model Architecture

The stacking classifier is composed of the following base learners:

- **Logistic Regression**
- **Kernel Support Vector Classifier (SVC)**
- **Decision Tree Classifier**

These base learners are combined using a **Random Forest Classifier** as the final estimator to improve predictive performance.

## Installation

To set up the project locally, follow these steps:

1. **Clone the repository:**

   ```bash
   git clone https://github.com/fridchikn24/loanpredictions.git
   cd loanpredictions

    Install dependencies:

    Ensure you have Python installed. Then, install the required packages:

    pip install -r requirements.txt

    Note: If requirements.txt is not provided, you may need to manually install the necessary packages such as pandas, numpy, scikit-learn, etc.

Usage

    Prepare the data:

    Ensure that the dataset (loan_approval_dataset.csv) is present in the project directory. If not, you may need to obtain and preprocess the data accordingly.

    Train the model:

    Run the training script to train the stacking classifier:

    python loans.py

    This script will train the model and may output performance metrics or save the trained model for future use.

    Make predictions:

    After training, you can use the model to make predictions on new applicant data by modifying the script or creating a new script to load the trained model and input data.

Contributing

Contributions are welcome! If you have suggestions for improvements or new features, please open an issue or submit a pull request. Ensure that your contributions align with the project's coding standards and include appropriate tests.
