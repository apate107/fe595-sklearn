from sklearn.linear_model import LinearRegression
from sklearn.datasets import load_boston
import pandas as pd
import numpy as np


def main():
    # Load data
    boston = load_boston()
    X = pd.DataFrame(data=boston['data'], columns=boston['feature_names'])
    X = X.drop('CHAS', axis=1) # Drop the CHAS feature, which is only comprised of 0s and 1s
    Y = pd.DataFrame(data=boston['target'], columns=['MEDV'])

    # Fit the model
    model = LinearRegression().fit(X, Y)

    # Find the most influential variable (slope farthest from zero)
    coefs = model.coef_[0]
    max_index = np.where(abs(coefs) == max(abs(coefs)))[0][0]

    # Print which one it is
    print('The element with the most influence on home price in Boston is ' + X.columns[max_index] +
          ' with a slope of ' + str(round(coefs[max_index], 4)))


if __name__ == '__main__':
    main()
