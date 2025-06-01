import numpy as np

class UniLinearRegression:
  def __init__(self):
    self.theta_0 = np.random.rand()
    self.theta_1 = np.random.rand()
    print(self.theta_0,self.theta_1)

  def hypothesis(self,x):
    return self.theta_0 + self.theta_1*x

  def grad_theta_0(self,x,y):
    return 2*(self.hypothesis(x)-y)

  def grad_theta_1(self,x,y):
    return 2*x*(self.hypothesis(x)-y)

  def fit(self,X,Y,epochs=1,lr=0.1):
       for epoch in range(epochs):
          grad_theta0_sum = 0
          grad_theta1_sum = 0
          for x,y in zip(X,Y):
             grad_theta0_sum += self.grad_theta_0(x,y)
             grad_theta1_sum += self.grad_theta_1(x,y)
          grad_0 = grad_theta0_sum/len(X)
          grad_1 = grad_theta1_sum/len(X)
          self.theta_0 = self.theta_0 - lr*grad_0
          self.theta_1 = self.theta_1 - lr*grad_1

  def predict(self,X):
    return [self.hypothesis(x) for x in X]


X = np.array([3, 2, 4, 5,0])
Y = np.array([17,12,22,27,2])

model = UniLinearRegression()
model.fit(X, Y, epochs=100, lr=0.01)

test = np.array([1,6])
predictions = model.predict(test)

print("Predictions:", predictions)
print("Theta 0:", model.theta_0)
print("Theta 1:", model.theta_1)
