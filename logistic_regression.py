import numpy as np
import matplotlib.pyplot as plt
from logistic import logistic
x_train=np.array([10,5,7,1,3,4,6,7.5,9.5,2,2.5,3.5,1.5,12])
y_train=np.array([1,1,1,0,0,0,1,1,1,0,0,0,0,1])
regressor = logistic(lr=0.0001,n_iters=1000)
regressor.fit(x_train,y_train)
print("\t\t\t\t\t\t\t\t\t\tLOGISTIC REGRESSION ")
t='y'
while(t=='y'):

    print("\n\t\t\t1.Logistic Regression Example\n\t\t\t2.Logistic Regression Curve")
    c = int(input("Enter u want to see:"))
    if (c == 1):
        a = int(input("\tEnter no. of hours a student will spend for study: "))
        pred = regressor.predict(a)
        if (pred == 1):
            print("\tStudent will pass if he study", a, 'hour')
        else:
            print("\tStudent will fail if he study", a, 'hour')
    elif (c == 2):
        def sigmoid(x):
            l = []
            for item in x:
                l.append(1 / (1 + np.exp(-item)))
            return l

        x = np.linspace(-10, 10)
        sig = sigmoid(x)
        plt.plot(x, sig)
        plt.show()
    t=input("Do you want to continue(y/n):")