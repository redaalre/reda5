
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression 
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import seaborn as sns
import matplotlib.pyplot as plt

data=pd.read_csv("cars.csv")
print(data.isnull().sum())
x=data.iloc[:,:-1]
y=data.iloc[:,-1]

x_train,x_test,y_train,y_test=train_test_split(x,y,train_size=0.8)

model=LogisticRegression()
model.fit(x_train, y_train)   
y_pred=model.predict(x_test)

print("train_Acc is ", model.score(x_train,y_train))
print("test_Acc  is ", model.score(x_test,y_test))
print(accuracy_score(y_test,y_pred))



colors = np.where(y_test == 1, 'red', 'blue')

  
plt.scatter(x_test.iloc[:, 0], y_test, c=colors, label='Actual Data', marker='o')
plt.xlabel('Feature 1')
plt.ylabel('DEATH_EVENT')
plt.title('Scatter Plot of Data')
 
plt.legend()
plt.show()
plt.scatter(x_test.iloc[:, 0], y_pred, c='green', label='Predicted Data', marker='x')
plt.xlabel('Feature 1')
plt.ylabel('DEATH_EVENT')
plt.title('Scatter Plot of Data') 
plt.legend()
plt.show()

\\
plt.pie([np.sum(y_test == 0), np.sum(y_test == 1)], labels=['Class 0', 'Class 1'], colors=['blue', 'red'], autopct='%1.1f%%')
plt.title('Distribution of Classes in Actual Data')
  
plt.show()
    
plt.pie([np.sum(y_pred == 0), np.sum(y_pred == 1)], labels=['Class 0', 'Class 1'], colors=['blue', 'red'], autopct='%1.1f%%')
plt.title('Distribution of Classes in Predicted Data')

  
plt.show()


    
plt.bar(['Class 0', 'Class 1'], [np.sum(y_test == 0), np.sum(y_test == 1)], color=colors)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Classes in Actual Data')

  
plt.show()


plt.bar(['Class 0', 'Class 1'], [np.sum(y_pred == 0), np.sum(y_pred == 1)], color=colors)
plt.xlabel('Class')
plt.ylabel('Count')
plt.title('Distribution of Classes  in Predicted Data')
  
plt.show()


conf_matrix = confusion_matrix(y_test, y_pred)

     
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=['Class 0', 'Class 1'], yticklabels=['Class 0', 'Class 1'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()


