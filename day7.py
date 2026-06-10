from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score
data=load_iris()
x,y=data.data,data.target
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
scaler=StandardScaler()
x_train=scaler.fit_transform(x_train)
x_test=scaler.transform(x_test)
log_model=LogisticRegression()
dt_model=DecisionTreeClassifier()
knn_model=KNeighborsClassifier()

log_model.fit(x_train,y_train)
dt_model.fit(x_train,y_train)
knn_model.fit(x_train,y_train)


ensemble_model=VotingClassifier(estimators=[('lr',log_model),('dt',dt_model),('knn',knn_model)],voting='hard')
ensemble_model.fit(x_train,y_train)
y_pred=ensemble_model.predict(x_test)
accuracy=accuracy_score(y_test,y_pred)
 

#Evaluating individual models
log_pred=log_model.predict(x_test)
dt_pred=dt_model.predict(x_test)
knn_pred=knn_model.predict(x_test)

print("logistic regression accuracy:",accuracy_score(y_test,log_pred))
print("decision tree accuracy:",accuracy_score(y_test,dt_pred))
print("k-nearest neighbors accuracy:",accuracy_score(y_test,knn_pred))  
print("Ensemble model accuracy:",accuracy)
