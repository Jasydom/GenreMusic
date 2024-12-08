import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne
import matplotlib.pyplot as plt # Pour la visualisation

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV # Split de dataset et optimisation des hyperparamètres
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, zero_one_loss, classification_report # Métriques pour la mesure de performances
from sklearn.preprocessing import StandardScaler

import seaborn as sns

from pprint import pprint

genres = ['blues', 'classical', 'country']
df = pd.read_csv("music.csv")

X = df[['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
           'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
           'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
           'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20']]

y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

features = df
# valeurs à prédire
labels = np.array(features['label'])
# supprime les labels des données
features = features.drop('label', axis = 1)
# sauvegarde le nom de features
feature_list = list(features.columns)
# conversion en numpy array
features = np.array(features)

# séparer les données en training et testing sets
train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25, random_state = 0)
print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)

sc = StandardScaler()
train_features = sc.fit_transform(train_features)
test_features = sc.transform(test_features)

rf = RandomForestClassifier(random_state = 0)
print('Paramètres actuels:\n')
pprint(rf.get_params())
#trop de paramètres, on se focalise uniquement sur (nombre d'arbres; profonder max de l'arbre; le min d'échantillons par noeur et par feuille)

# nombre d'arbres
n_estimators = [500, 1000, 2000, 3000, 4000, 5000]
# profondeur max de l'arbre
max_depth = [20]
max_depth.append(None)
# nombre d'échantillon min nécessaire par noeuds
min_samples_split = [2, 4]#[2]
# nombre d'échantillon min nécessaire par feuilles
min_samples_leaf = [1, 2]#[1]

# création de la grille
random_grid = {'n_estimators': n_estimators,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
              }
print('Nouveaux paramètres:\n')
pprint(random_grid)

# création du modèle
rf = RandomForestClassifier(random_state = 0, max_features = 'sqrt', bootstrap = True)

# random search
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 10, cv = 3, verbose=2, random_state=0, n_jobs = -1)

# fit le modèle
rf_random.fit(train_features, train_labels)

pd_res = pd.concat([pd.DataFrame(rf_random.cv_results_["params"]),pd.DataFrame(rf_random.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
pd_res = pd_res.sort_values('Accuracy', ascending=False)
print(rf_random.best_params_)
print(pd_res.head(5))

# création du modèle en se focalisant sur la profondeur max et le nb d'arbres
rf = RandomForestClassifier(random_state = 0, bootstrap=True)

param_grid = {
    'max_depth': [20, None],
    'min_samples_split': [2],
    'n_estimators': [2000, 4000]
}
pprint(param_grid)

# grid search
grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, cv = 3, n_jobs = -1, verbose = 2)

grid_search.fit(train_features, train_labels)

pd_res = pd.concat([pd.DataFrame(grid_search.cv_results_["params"]),pd.DataFrame(grid_search.cv_results_["mean_test_score"], columns=["Accuracy"])],axis=1)
pd_res = pd_res.sort_values('Accuracy', ascending=False)
print(pd_res.head(5))
#On choisit donc 4000 arbres d'une profondeur max de 20 niveaux

# création du modèle
rf = RandomForestClassifier(n_estimators=4000, max_features='sqrt', max_depth=20, min_samples_split=2, min_samples_leaf=1, bootstrap=True, criterion='gini' ,random_state=0)

# fit le modèle
rf.fit(train_features, train_labels)

# prédictions
predictions = rf.predict(test_features)

# Zero_one_loss error
errors = zero_one_loss(test_labels, predictions, normalize=False)
print('zero_one_loss error :', errors)

# Accuracy Score
accuracy_test = accuracy_score(test_labels, predictions)
print('accuracy_score on test dataset :', accuracy_test)

print(classification_report(predictions, test_labels))
#on obtient donc une précision de 84%

##### MATRICE DE CONFUSION #####

sns.set()
mat = confusion_matrix(test_labels, predictions)
sns.heatmap(mat.T, square=True, annot=True, fmt='d', cbar=False, xticklabels=genres, yticklabels=genres)
plt.xlabel('true label')
plt.ylabel('predicted label')
plt.show()

####### EVALUATION DE L'IMPORTANCE DE CHAQUE VARIABLES #######

plt.style.use('fivethirtyeight')

importances = list(rf.feature_importances_)

x_values = list(range(len(importances)))
plt.bar(x_values, importances, orientation = 'vertical')
plt.xticks(x_values, feature_list, rotation='vertical')
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances')
plt.show()
#on remarque que le paramètres mfcc1 est le plus important