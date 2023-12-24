
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


data=pd.read_csv('economic_data.csv')

x=data['Year']
y=data['GDP']

plt.plot(x,y,marker='.')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title("GDP vs. Year")
plt.show()

****************************
plt.bar(data['Year'],data['GDP'])
plt.show()

*****************************
import seaborn as sns 
sns.lineplot(data=data,x='Year',y='GDP',marker='.' )
plt.title("GDP vs. Year")
plt.show()

*********************

x=data.iloc[:,0]  
y=data.iloc[:,1]
plt.plot(x,y,marker='.')
plt.xlabel('Year')
plt.ylabel('GDP')
plt.title("GDP vs. Year")
plt.show()
