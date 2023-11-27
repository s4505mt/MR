# Import libraries

import argparse
import glob
import os
import mlflow

import pandas as pd

from sklearn.linear_model import LogisticRegression


# define functions
def main(args):
    # TO DO: enable autologging
    mlflow.autolog()

    print (args.training_data)
    # read data
    df = get_csvs_df(args.training_data)

    # split data
    X_train, X_test, y_train, y_test = split_data(df)

    # train model
    acc_p, auc_p, y_scores_p = train_model(args.reg_rate, X_train, X_test, y_train, y_test)

    #pokaż jakość modelu

    print(acc_p)
    print(auc_p)
    prn(y_test, y_scores_p)

def prn(y_test, y_scores):

    from sklearn.metrics import roc_curve
    import matplotlib.pyplot as plt
    # plot ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_scores[:,1])
    fig = plt.figure(figsize=(6, 4))
    # Plot the diagonal 50% line        
    plt.plot([0, 1], [0, 1], 'k--')
    # Plot the FPR and TPR achieved by our model
    plt.plot(fpr, tpr)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')

def get_csvs_df(path):
   
   # if not os.path.exists(path):
   #     raise RuntimeError(f"Cannot use non-existent path provided: {path}")
    #csv_files = glob.glob(f"{path}/*.csv")
    csv_files = path
    #if not csv_files:
    #    raise RuntimeError(f"No CSV files found in provided data path: {path}")
    return pd.read_csv(path) 
    #return pd.concat((pd.read_csv(f) for f in csv_files), sort=False)

def split_data(df):
    # TO DO: implement data splitting logic
    # You can use train_test_split from scikit-learn or any other method to split your data into training and testing sets.
    # For example:
    from sklearn.model_selection import train_test_split
    #X = df.drop('target_column_name', axis=1)  # Adjust 'target_column_name' to your target column
    #y = df['target_column_name']
    X = df[['Pregnancies','PlasmaGlucose','DiastolicBloodPressure','TricepsThickness','SerumInsulin','BMI','DiabetesPedigree','Age']].values
    y= df['Diabetic'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
    return X_train, X_test, y_train, y_test

def train_model(reg_rate, X_train, X_test, y_train, y_test):
    # train model
    model = LogisticRegression(C=1/reg_rate, solver="liblinear")
    model.fit(X_train, y_train)
    # TO DO: You can add evaluation and saving of the trained model here.

    import numpy as np
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)  

    from sklearn.metrics import roc_auc_score
    y_scores = model.predict_proba(X_test)
    auc = roc_auc_score(y_test,y_scores[:,1])  
    return (acc, auc, y_scores)


def parse_args():
    # setup arg parser
    parser = argparse.ArgumentParser()
     #Users/s4505mt/MR/experimentation/data/diabetes-dev.csv
    # add arguments
    parser.add_argument("--training_data", dest='training_data', type=str)
    parser.add_argument("--reg_rate", dest='reg_rate', type=float, default=0.01)

    # parse args
    args = parser.parse_args()

    # return args
    return args

# run script
if __name__ == "__main__":
    # add space in logs
    print("\n\n")
    print("*" * 60)


    # parse args
    args = parse_args()
    

    args.training_data = '~/cloudfiles/code/Users/s4505mt/MR/experimentation/data/diabetes-dev.csv'
    #df = pd.read_csv('~/cloudfiles/code/Users/s4505mt/MR/experimentation/data/diabetes-dev.csv')
    #args.trainging_data = os.getcwd()
    print(args.training_data)
    # run main function
    main(args)

    # add space in logs
    print("*" * 60)
    print("\n\n")
