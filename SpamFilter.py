import scipy.io as sio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# load data and visualize it
mat_path = "./spamdata.mat"
data = sio.loadmat(mat_path)

# import variables
X = data.get('X')
y = data.get('y')
X2 = data.get('X2')
y2 = data.get('y2')

# make y data 1D
y = np.ravel(y).astype(int)
y2 = np.ravel(y2).astype(int)

# set smooth factor
k = 1 # by default k is 1

# Naive Bayes training function, the function gets input of the training data set and output the conditional probabilities of each feature PXY and the estimate of (y = 1) phat1
def trainNB(X,y,k):

    # get the shape of the training array, N is number of emails (1000) , M is length of emails(200)
    N, M = X.shape


    # Define masks for training data, find the rows that are spam and the rows that are not spam
    one_Row_Idx = (y == 1)
    zero_Row_Idx = (y == 0)

    # Split X to Spam and Non-Spam classes, X_one is spam class and X_zero is non-spam class
    X_one = X[one_Row_Idx]
    X_zero = X[zero_Row_Idx]

    # count the length of each subsets of X
    number_of_ones = X_one.shape[0]
    number_of_zeros = N - number_of_ones

    # Estimate P_hat(y=1) = #(Y=1) / N
    phaty1 = number_of_ones/N

    # create an empty probability tabel with dimension (2,M), PXY[0,j] = P(x_j = 1| y=0), PXY[1,j] = P(x_j = 1| y=1)
    PXY = np.zeros((2,M))

    # Compute the probability table of each element using the formula P_hat(x_j = 1|y=c) = (N_{j= 1,c}+k)/(N_c + 2k)
    for i in range(M):
        PXY[0,i] = (np.sum(X_zero[:,i] == 1) + k) / (number_of_zeros + 2 *k)
        PXY[1,i] = (np.sum(X_one[:,i] == 1) + k) / (number_of_ones + 2 *k)

    return PXY,phaty1


# Evaluation function
def NBEval(PXY,phaty1, X2,y2):
    # get the dimensions of test array
    N2, M = X2.shape

    # create empty prediction array
    ypred = np.ones(N2)

    # loop through all the table and find the likehood
    for i in range(N2):
        # For each email, initialize the class likehood
        p1 = 1
        p2 = 1

        #loop through each index of email, for each index, check whether it is 0 or 1 and assign the class probability to it
        for j in range(M):
            if X2[i,j] == 1:
                p1 = p1 * PXY[0,j]
                p2 = p2 * PXY[1,j]
            else:
                p1 = p1 * (1-PXY[0, j])
                p2 = p2 * (1-PXY[1, j])

        # compute the propotioanl posteriors
        a = p1 * (1-phaty1) # probability that a input is classfied as 0
        b = p2 * phaty1 # probability that an input is classfied as 1

        # update the prediction table
        if a > b:
            ypred[i] = 0
        else:
            ypred[i] = 1

    # calculate the accuracy based on the test label
    accuracy = np.mean(ypred == y2)

    return accuracy

# run the functions
PXY,phaty1 = trainNB(X,y,k)
accuracy = NBEval(PXY,phaty1,X2,y2)
print("the accuracy is", accuracy)
