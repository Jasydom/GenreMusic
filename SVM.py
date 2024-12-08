import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne
import matplotlib.pyplot as plt # Pour la visualisation
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve, RandomizedSearchCV # Split de dataset et optimisation des hyperparamètres

from sklearn.svm import SVC # SVM
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, zero_one_loss, classification_report # Métriques pour la mesure de performances

import seaborn as sns

genres = ['blues', 'classical', 'country']

df = pd.read_csv("music.csv")

X = df[['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
           'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
           'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
           'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']]

y = df['label']

#On prend une portion de 20% du total des données, et on les sépare entre "train" et "test"
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

##### ENTRAINEMENT DU MODELE ########

model_svm = SVC()
model_svm.fit(X_train, y_train)

print('Train score : ', model_svm.score(X_train,y_train))
print('Test score : ', model_svm.score(X_test,y_test))

k_3 = np.arange(1,31)
tr_score_3, val_score_3 = validation_curve(model_svm, X_train, y_train, param_name='C', param_range=k_3, cv = 5)

#5 splits sets de cross validation, on fait la moyenne des scores obtenus sur chacun des 5 splits

train = model_svm.predict(X_train)
predictions = model_svm.predict(X_test)

plt.plot(k_3, val_score_3.mean(axis = 1), label = 'validation')
plt.plot(k_3, tr_score_3.mean(axis = 1), label = 'train')

plt.ylabel('score')
plt.xlabel('C')
plt.legend()
plt.show()

####### MESURE DES PERFORMANCES #######

np.max(val_score_3.mean(axis = 1))
np.argmax(val_score_3.mean(axis = 1)) + 1

#+1 pour avoir le nombre de voisins optimal (indice commençant à 0)
#on trouve entre 5 et 10 voisins pour environ 30-35% de réussite

sns.set()
mat = confusion_matrix(y_test, model_svm.predict(X_test))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=genres, yticklabels=genres)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# on remarque que les résultats sont peu pertinents, précision de 40%