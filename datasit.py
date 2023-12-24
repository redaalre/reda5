import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv("economic_data.csv")


plt.scatter(data["Year"],data['GDP'])

x=data.iloc[:,:1]
y=data.iloc[:,1]
print(data.describe())
print(x,y)

from sklearn.linear_model import LinearRegression
module=LinearRegression()
module.fit(x,y)
print(module.coef_)
print(module.intercept_)
plt.scatter(x,y)

plt.plot(x,module.predict(x),'r')
module.predict([[24]])
module.predict([[2]])
module.score(x, y)
