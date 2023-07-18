import numpy as np 
import pandas as pd 
import matplotlib as pyplot


def A_mat(x, deg):
    """Create the matrix A part of the least squares problem.
       x: vector of input datas.
       deg: degree of the polynomial fit."""
    A = np.zeros((len(x), deg + 1))  # initialize a matrix full of zeros
    count = deg
    for i in range(deg + 1):
        for j in range(len(x)):
            val = x[j]
            A[j, i] = val ** count
        count -= 1
    return A


def LLS_Solve(x, y, deg):
    """Find the vector w that solves the least squares regression.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       w = (A'A)-1 A'Y """
    A = A_mat(x, deg)
    AT = A.transpose()
    ATA = np.matmul(AT, A)
    ATAInv = np.linalg.inv(ATA)
    ATY = np.matmul(AT, y)
    w = np.matmul(ATAInv, ATY)
    return w


def LLS_ridge(x, y, deg, lam):
    """Find the vector w that solves the ridge regression problem.
       x: vector of input data.
       y: vector of output data.
       deg: degree of the polynomial fit.
       lam: parameter for the ridge regression.
       w = (A'A + LAM*I)-1 A'Y"""
    A = A_mat(x, deg)
    AT = A.transpose()
    ATA = np.matmul(AT, A)
    rows, cols = ATA.shape
    i = np.identity(rows)
    multiplied = i * lam
    ATA_LamI = np.add(ATA, multiplied)
    Inverse = np.linalg.inv(ATA_LamI)
    ATY = np.matmul(AT, y)
    w = np.matmul(Inverse, ATY)
    return w


def poly_func(data, coeffs):
    """ Produce the vector of output data for a polynomial.
        data: x-values of the polynomial.
        coeffs: vector of coefficients for the polynomial.

        what to do:
        data : n x m (rows, columns) --> comes from A matrix
        coeffs : nx1 (data) --> comes from w vector
        y : nx1 (initialize with zeros) --> this is the predicted y values

        for i each row:
            for j in range(m): j should go from 0 to m
                y[i] += data[i,j] * (coeffs[i] raised to columns-j power)"""

    rows, columns = data.shape
    y = np.zeros((rows, 1))
    for i in range(rows):
        for j in range(columns):
            y[i] += data[i, j] * (coeffs[j] ** (columns - j + 1))
    return y


def LLS_func(x, y, w, deg):
    """The linear least squares objective function.
           x: vector of input data.
           y: vector of output data.
           w: vector of weights.
           deg: degree of the polynomial.
           """
    A = A_mat(x, deg)
    AW = np.matmul(A, w)
    norm = np.linalg.norm(np.subtract(AW, y))
    normSq = norm ** 2
    return normSq/len(x)


def RMSE(x, y, w):
    """Compute the root-mean-square error.
       x: vector of input data.
       y: vector of output data.
       w: vector of weights.
       square_root (1/n ((y-AW))^2) """
    A = A_mat(x, len(w) - 1)
    Y_hat = (poly_func(A, w))
    AW = np.matmul(A, w)
    y2 = np.array(y)
    MSE = np.linalg.norm(np.subtract(y, AW)) ** 2
    MSE2 = MSE/len(x)
    RMSE = np.sqrt(MSE2)
    return RMSE


data = pd.read_csv('Desktop/machine learning/athens_ww2_weather.csv')
x = data['MinTemp']
y = data['MaxTemp']

pyplot.scatter(x, y)
deg1 = 1

# A_mat(x, deg) --> to get A
A1 = A_mat(x, deg1)
print("This is A: ")
print(A1)
    
# LLS_Solve(x, y, deg) --> to get w
w1 = LLS_Solve(x, y, deg1)
print("This is w: ")
print(w1)
# poly_func(data, coeffs) --> to get y
# data: x-values of the polynomial.
# coeffs: vector of coefficients for the polynomial.
print("This is y: ")
y1 = poly_func(A1, w1)
print(y1)
# LLS_func(x, y, w, deg) --> to get the function
LLS_function = LLS_func(x, y1, w1, deg1)
print("This is the solution of the objective function for LLS:")
print(LLS_function)
rmse = RMSE(x, y, w1)
print("This is the rmse: ")
print(rmse)
b = w1[1]
m = w1[0]
pyplot.title('Athens Temperature')
pyplot.xlabel('MinTemp', fontsize=15)
pyplot.ylabel('MaxTemp', fontsize=15)
pyplot.plot(x, b + m * x, c='r', linewidth=3, alpha=.5, solid_capstyle='round')
text = "the RMSE is: " + str(rmse)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
pyplot.text(0, 43, text, fontsize=10, verticalalignment='top', bbox=props)
pyplot.show()