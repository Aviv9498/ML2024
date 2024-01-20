import numpy as np
def ID1():
    '''
        Write your personal ID here.
    '''
    # Insert your ID here
    return 209583509
def ID2():
    '''
        Only If you were allowed to work in a pair will you fill this section and place the personal id of your partner otherwise leave it zeros.
    '''
    # Insert your ID here
    return 000000000

def LeastSquares(X,y):
  '''
    Calculates the Least squares solution to the problem
    X*theta=y using the least squares method
    :param X: input matrix
    :param y: input vector
    :return: theta = (Xt*X)^(-1) * Xt * y 
  '''
  theta_star = np.dot(np.matmul(np.linalg.inv(np.matmul(X.T, X)), X.T), y)
  return theta_star

def classification_accuracy(model,X,s):
  '''
    calculate the accuracy for the classification problem
    :param model: the classification model class
    :param X: input matrix
    :param s: input ground truth label
    :return: accuracy of the model
  '''
  s_hat = model.predict(X)
  number_of_total_predictions = len(s_hat)
  number_of_currect_guesses = (s_hat == s).sum()
  accuracy = 100 * (number_of_currect_guesses / number_of_total_predictions)
  
  return accuracy

def linear_regression_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the linear regression problem.  
  '''
  return [-6.44759419e-02,  3.20404588e-02, -3.56042502e-02,  3.65556307e-03,
 -3.36031706e-02, -3.06275227e-02,  6.57175668e-02, -1.35468132e-02,
  1.06974015e-03, -2.70908796e-02,  3.72594075e-02,  2.58074222e-02,
  7.76629491e-02,  1.78721216e-01,  7.45863252e-01,  3.63252667e-02,
  3.23768081e-02, -1.28448521e-02,  2.33140371e-02, -7.05353092e-03,
  3.34700485e-02,  2.13007044e-02,  1.49567105e-02, -4.01833075e-02,
 -1.59628999e-02,  8.67381683e-03, -3.18890984e-04, -2.86520138e-02]

def linear_regression_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return -2.4190973380600387e-16

def classification_coeff_submission():
  '''
    copy the values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of coefficiants for the classification problem.  
  '''
  return [[-0.30670335, -0.25602801,  0.2775661,  -0.26793908, -0.35726201, -0.27267187,
  -0.18585046, -0.0749971,   0.07330806,  0.13731727, -0.43288368, -0.00765622,
  -0.09985522,  0.96836699,  2.93551234, -0.42933245,  0.0251862,  -0.06599892,
  -0.20575145, -0.15599835,  0.01694284,  0.20427903, -0.12162975, -0.05071128,
  -0.45113856,  0.11010529,  0.13145888,  0.23089825]]
def classification_intrcpt_submission():
  '''
    copy the intercept value from your notebook into here.
    :return: the intercept value.  
  '''
  return [0.4190916]

def classification_classes_submission():
  '''
    copy the classes values from your notebook into a list in here. make sure the values
    seperated by commas
    :return: list of classes for the classification problem.  
  '''
  return [-1. , 1.]