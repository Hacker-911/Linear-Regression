import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

data = pd.read_csv('data.csv')

data.shape
(47, 3)

data.head()


x=data['Size']
y=data['Price']

x.head()


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.3, shuffle = False)

x_train.shape,y_train.shape,x_test.shape,y_test.shape
((32,), (32,), (15,), (15,))


X=x_train.values.reshape(-1,1)
Y=y_train.values.reshape(-1,1)
x_test=x_test.values.reshape(-1,1)
y_test=y_test.values.reshape(-1,1)

X.shape
(32, 1)

model = LinearRegression()

model.fit(X,Y)
LinearRegression(copy_X=True, fit_intercept=True, n_jobs=None,normalize=False)

plt.scatter(X,Y)
y_pred=model.predict(x_test)
plt.plot(x_test,y_pred,color='red')
plt.show()
