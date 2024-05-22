import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

import numpy as np

# Unpickling the dataset
with open('newdata.pickle','rb') as f:
    data_dict=pickle.load(f)

'''
#Only Run if problem occurs
rows=[]
for i in range(len(data_dict['data'])):
    l=len(data_dict['data'][i])
    if l!=42:
        print(i,l)
        rows.append(i)
    
#print(rows)

for i in rows:
    data_dict['data'][i]=data_dict['data'][i][:42]
'''


X=np.asarray(data_dict['data'])
y=np.asarray(data_dict['label'])


# Now we will split into training and testing dataset
X_train,X_test, y_train,y_test= train_test_split(X,y,test_size=0.25,shuffle=True,stratify=y)

# RandomForestClassifier used to train our model
rfmodel=RandomForestClassifier()


rfmodel.fit(X_train,y_train)

# Checking accuracy of our model
score1=rfmodel.score(X_test, y_test)
print("Model Accuracy=",score1)


# Saving/Pickling our model
with open('newmodel.p','wb') as f:
    pickle.dump({'model':rfmodel},f)
f.close()


'''
# Create a confusion matrix
#cm=confusion_matrix(y_test, y_predict)

# Create a heatmap
#fig, ax = plt.subplots()

sns.heatmap(cm,cmap="PuBuGn",annot=True,fmt="d",cbar_kws={"label":"Scale"},xticklabels=[0,1,2,3,4,5],yticklabels=[0,1,2,3,4,5])
plt.title("Confusion Matrix")
plt.xlabel("Predicted Values")
plt.ylabel("Actual Values")
plt.show()

# Saving plot as high quality image
image_format = 'png' # e.g .png, .svg, etc.
image_name = 'Heatmap.png'

fig.savefig(image_name, format=image_format, dpi=1200)

'''