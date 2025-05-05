import matplotlib.pyplot as plt
import numpy as np

class LinearRegression :
 def __init__(self,l_r,max_iter):
  self.l_r = l_r
  self.max_iter = max_iter
  self.w = None
  self.b = None
  
 def fit(self,X,y):
  '''
  Train the model 
  X : variables
  y : target
  '''
  m,n = X.shape
  self.w = np.zeros((n,1))
  self.b =0 
  y = y.reshape(-1, 1)
  for _ in range(self.max_iter):
   y_pred = self.b + np.dot(X,self.w)
   dw = -(1/m)*np.dot(X.T,(y-y_pred))
   db = -(1/m)*np.sum((y_pred-y))
   self.w = self.w - self.l_r*dw
   self.b = self.b - self.l_r*db
 def predict(self,X):
  '''
  predict the target values
  '''
  return self.b + np.dot(X,self.w)
 def plot_line(self,X,y):
   '''
   plot predict line
   '''
   y_pred = self.predict(X)
   plt.scatter(X, y, color='blue', label='Data')
   plt.plot(X, y_pred, color='red', label='Prediction')
   plt.legend()
   plt.show()
 def MSE(self,y,y_pred):
  return np.mean((y - y_pred) ** 2)
  

