#Hyper Parameter Tuning To improve Accuracy

import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

import numpy as np


with open('newdata.pickle','rb') as f:
    data_dict=pickle.load(f)
    

X=np.asarray(data_dict['data'])
y=np.asarray(data_dict['label'])

X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.25,shuffle=True,stratify=y)

#number of trees
n_estimators=[int(x) for x in np.linspace(start=100,stop=500,num=5)]

'''
#number of features to consider at each split
max_features=['auto','sqrt']

#max levels in the tree
max_depth=[2,4]

#min number of samples required at each leaf node
min_samples_leaf=[1,2]
'''

#Creating param grid
param_grid={'n_estimators':n_estimators}
'''
            'max_features':max_features,
            'max_depth':max_depth,
            'min_samples_leaf':min_samples_leaf,
'''
            

print('Parameter Grid:\n',param_grid)

rfmodel=RandomForestClassifier()

from sklearn.model_selection import GridSearchCV

rf_grid=GridSearchCV(estimator=rfmodel, param_grid=param_grid,cv=3,verbose=1,n_jobs=4)


rf_grid.fit(X_train,y_train)


print('Best parameters:',rf_grid.best_params_)

best_rf = rf_grid.best_estimator_
print(f'Accuracy={best_rf.score(X_test,y_test):.3f}')

with open('bestmodel.p','wb') as f:
    pickle.dump({'model':best_rf},f)
f.close()
