from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    N, D = X.shape
    _ , C = W.shape
    diff = np.zeros((N, C))
    z = np.zeros((N, C))
    dW = np.zeros_like(W)
    a = np.zeros((N, C))
    


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    
    for i in range(N):
      for j in range(D):
        for k in range(C):
          z[i,k] += X[i,j] * W[j,k]
      exp = np.exp(z[i])
      sum_ = np.sum(np.exp(z[i]))
      a[i] = exp / sum_
      loss += -np.log(a[i,y[i]])
    loss *= 1 / N
    loss += reg * np.sum(W * W)

    for i in range(N):
      diff = a[i].copy()
      diff[y[i]] -= 1
      for j in range(D):
        for k in range(C):
          dW[j,k] += X[i,j] * diff[k]
    dW *= 1 / N
    dW += 2 * reg * W
          


    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)
    N, D = X.shape
    _ , C = W.shape


    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    z = np.dot(X, W)
    a = np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)
    loss = np.sum(-np.log(a[np.arange(N),y]))
    loss *= 1 / N
    loss += reg * np.sum(W * W)

    diff = a.copy()
    diff[np.arange(N), y] -= 1
    dW = np.dot(X.T ,diff)
    dW *= 1 / N
    dW += 2 * reg * W





    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
