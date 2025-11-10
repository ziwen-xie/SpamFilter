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
This function gets input of the training data set and output the conditional probabilities of each feature $PXY$ and the estimate of $\hat{p}(y = 1)$

We assume conditional independence:
 $$P(X_1 = i, X_2 = j | y=1) = P(X_1 = i|y=1) \times P(x_2 =j,|y=1)$$
We can estimate $P(X_1 = i|y=1)$ as:
$$P(x_1 = i|y=1) = \frac{ number \ of  \{x_1 = i\}}{number \ of \{ y=1 \} }$$

So, in the training function, we estimate $P(X_i = 1|y=0)$ and $P(X_i = 1|y = 1)$ for $i = 1,2,3,...,M$

That is:

$$\hat{P}(X_i = 1|y=0) = \frac{ number \ of \{x_i = 1, y=0\} + k }{number \ of \{ y=0 \} + 2k }$$

and that:
$$\hat{P}(X_i = 1|y=1) = \frac{ number \ of \{x_i = 1, y=1\} + k }{number \ of \{ y=1 \} + 2k }$$

With the $\hat{P}(X_i = 1|y=0)$ and $\hat{P}(X_i = 1|y=1)$ we can calculate the probability of $\hat{P}(X_i = 0|y=0)$ and $\hat{P}(X_i = 0|y=1)$ as:

$$\hat{P}(X_i = 0|y=0) = 1- \hat{P}(X_i = 1|y=0)$$
$$\hat{P}(X_i = 0|y=1) = 1- \hat{P}(X_i = 1|y=1)$$

With the above prior probability we can estimate $\hat{P}(y=1)$ as :
$$\hat{P}(y=1) = \frac{ number \ of \{y=1\}  }{N}$$

### Predict and evaluation
The evaluation function gets the probability table we calculated $PXY$ and $\hat{P}(y=1)$ and as well as the test set X2 and y2.

It loops through X2 and predict if it is spam for each email, we then compare the result as the ground truth y2, and output the accuracy as:

$$accuracy = \frac{correct\ prediction}{incorrect \ prediction}$$

For the prediction, for each element in X2, we calculate:

$$P(y=1|x_i) = P(y=1) \times P(x_i|y=1)$$ and we add this up for the entire feature space to get the probability of the email.

For each email, we calculate two numbers a and b,
where a is $$a = P(X_1 = x_1| y=0)P(X_2= x_2| y = 0) ...P(X_m = x_m| y=0)P(y = 0)$$
and $$b = P(X_1 = x_1| y=1)P(X_2= x_2| y = 1) ...P(X_m = x_m| y=1)P(y = 1)$$

this gives the total probability of this email belong to spam or not spam. 
If $a>b$, this email is not spam, otherwise,it is spam

## Result 
The output accuracy is 0.944, which is a hgh accuracy. It shows that the algorithm can give a plausible prediction result. 