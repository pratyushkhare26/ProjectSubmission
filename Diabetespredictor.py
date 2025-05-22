import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
data=pd.read_csv(r'D:\Pratyush Khare\ML\Diabetes\diabetes.csv')
x1=data["Pregnancies"].values.reshape(1,-1)
x2=data["Glucose"].values.reshape(1,-1)
x3=data["BloodPressure"].values.reshape(1,-1)
x4=data["SkinThickness"].values.reshape(1,-1)
x5=data["Insulin"].values.reshape(1,-1)
x6=data["BMI"].values.reshape(1,-1)
x7=data["DiabetesPedigreeFunction"].values.reshape(1,-1)
x8=data["Age"].values.reshape(1,-1)
y=data["Outcome"].values.reshape(-1,1)
X=np.vstack([x1,x2,x3,x4,x5,x6,x7,x8])
X=X.T
def sigmoid(z):
  return 1/(1+np.exp(-z))
def Newton(X,Y,iterations):
  n=X.shape
  theta=np.zeros(n)
  for _ in range(iterations):
    h=sigmoid(np.dot(X,theta)).reshape(-1, 1)
    pred=h-Y
    grad = np.dot((X.T),pred).flatten()
    S=np.diag((h*(1-h)).flatten())
    try:
     hess=np.linalg.pinv((X.T)@S@X)
    except np.linalg.LinAlgError:
            print("Hessian is not invertible. Stopping.")
            break
    theta -= np.dot(hess, grad)

  return theta
theta1 = Newton(X,y,20)
x1_f=int(input("Enter pregnancies: "))
x2_f=int(input("Enter Glucose: "))
x3_f=int(input("Enter BloodPressure: "))
x4_f=int(input("Enter SkinThickness: "))
x5_f=int(input("Enter Insulin: "))
x6_f=float(input("Enter BMI: "))
x7_f=float(input("Enter DiabetesPedigreeFunction: "))
x8_f=int(input("Enter Age: "))

X=[x1_f,x2_f,x3_f,x4_f,x5_f,x6_f,x7_f,x8_f]
outcome=sigmoid(np.dot(theta1.T,X))
prob=outcome*100
print(f"You have a {prob}% chance of developing Diabetes")


