
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('economic_data.csv')
print(data)

print(data.describe())

print(data.head())
plt.scatter(data['Year'],data['GDP'])
plt.show()


print(data.head())

x=data.iloc[:,:1]  
y=data.iloc[:,1]

print(x)
print(y)

from sklearn.linear_model import LinearRegression
model =LinearRegression()
model.fit(x,y)

print(model.coef_)

print(model.intercept_)

plt.scatter(x,y)
plt.plot(x,model.predict(x),'r')

model.predict([[2]])

model.predict([[24]])
model.predict([[12]])
model.predict([[100]])







#valdation
model.score(x,y)
