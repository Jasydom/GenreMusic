import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne
import matplotlib.pyplot as plt # Pour la visualisation

import librosa.display # Pour récupérer les spectrogrammes des audio
import librosa.feature

from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split, GridSearchCV, validation_curve# Split de dataset et optimisation des hyperparamètres
from sklearn.ensemble import RandomForestClassifier # Random forest
from sklearn.ensemble import GradientBoostingClassifier # XGBoost
from sklearn.neighbors import KNeighborsClassifier # k-NN
from sklearn.svm import SVC # SVM
from sklearn.metrics import accuracy_score, confusion_matrix, recall_score, f1_score, zero_one_loss, classification_report # Métriques pour la mesure de performances
from sklearn.preprocessing import normalize, StandardScaler

import tensorflow as tf # Pour le reseau de neurones simple et pour le CNN

import seaborn as sns

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.optimizers import Adam
from keras.utils import to_categorical

from xgboost import XGBClassifier
from pprint import pprint

genres = ['blues', 'classical', 'country']

df = pd.read_csv("music.csv")

X = df[['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
           'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
           'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
           'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']]

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

#### ENTRAINEMENT DU MODELE ######

model_1 = KNeighborsClassifier(n_neighbors = 5)

model_1.fit(X_train, y_train)
print('Train score :', model_1.score(X_train,y_train))
print('Test score :', model_1.score(X_test,y_test))

k_1 = np.arange(1, 31)
train_score_1, val_score_1 = validation_curve(model_1, X_train, y_train, param_name='n_neighbors', param_range=k_1, cv = 5)

#5 splits sets de cross validation, puis on fait la moyenne des scores obtenus sur chacun des 5 splits

plt.plot(k_1, val_score_1.mean(axis = 1), label = 'validation')
plt.plot(k_1, train_score_1.mean(axis = 1), label = 'train')

plt.ylabel('score')
plt.xlabel('n_neighbors')
plt.legend()
plt.show()

####### MESURE DES PERFORMANCES #######

np.max(val_score_1.mean(axis = 1))
np.argmax(val_score_1.mean(axis = 1)) + 1

#+1 pour avoir le nombre de voisins optimal (indice commençant à 0)
#on trouve entre 5 et 10 voisins pour environ 30-35% de réussite

sns.set()
mat = confusion_matrix(y_test, model_1.predict(X_test))
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=genres, yticklabels=genres)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

# on remarque que les résultats sont peu pertinents (51.7% de précision)