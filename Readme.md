# Spam Filter Project

Download the below MATLAB file. It includes training data and test data for the spam filter.
Training data : X, y
Test Data: X2, y2
Choose k=1 (Smoothing factor)

Write a code in your preferred language (MATLAB, R, Python, C++, Java, etc.)  for a Bayes classifier as described in the lectures (Please refer to the video recoding below and notes - recorded last semester, so please ignore the mentioned dates). Submit a full report in a single pdf file that includes 
- Your Code
- Documentation and explanation of your Code
- The obtained (test) accuracy.

## Project Description
This is the spam filter project for ECE603. The goal is to construct a method to filter spam emails. 
The repository consist the follow items:
 - `main.py` the spam filter code in python.
 - `spamdata.mat` the data of the spam filter project.
 - `requirements.txt`  the required package in this project.
 - `readme.md` a drscription file.

## Input Data Structure
The input data is a matrix that contains four arrays:
- `X` is a (1000,200) array that represents training input data. Each email has a length of X, and if at index i it contains the word "refinance", it is recorded as 1, if not, it is recorded as 0. 
- `y` is the training labels of shape (1000,1). Each index of y is a binary number of 0 or 1, which represent whether or not the email is spam. 0 means not spam and 1 means spam.
- `X2` is a (1000,200) array that represent text input data, which is 1000 emails for test
- `y2` is a (1000,1) binary label for test data.

## Structure of code
The code is seperated to the following parts. 
### Data import and preprocess
This part imports the data and extract the arrays from `data`.

X,X2,y,y2 is imported. y and y2 is adjusted to 1D shape for process.This is to ensure the compare of y values in the later parts works correctly.


### The Naive Bayes training function
This function gets input of the training data set and output the conditional probabilities of each feature PXY and the estimate of (y = 1) phat1
