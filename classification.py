import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne
import matplotlib.pyplot as plt # Pour la visualisation

from sklearn.feature_selection import VarianceThreshold


genres = ['blues', 'classical', 'country']

df = pd.read_csv("music.csv")

print(df.head()) #pour voir à quoi ressemble le tableau

selector = VarianceThreshold(threshold=(0.2)) #on retire les variances en dessous de 0.2

selected_features = selector.fit_transform(df[['zcr', 'spectral_c', 'rolloff', 
                                                    'mfcc1', 'mfcc2', 'mfcc3',
                                                    'mfcc4','mfcc5', 'mfcc6', 
                                                    'mfcc7', 'mfcc8', 'mfcc9',
                                                    'mfcc10','mfcc11', 'mfcc12',
                                                    'mfcc13', 'mfcc14', 'mfcc15',
                                                    'mfcc16', 'mfcc17', 'mfcc18', 
                                                    'mfcc19', 'mfcc20']])

pd.DataFrame(selected_features)

#### CORRELATIONS DES VARIABLES ####

f = plt.figure()

plt.matshow(df.corr(), fignum=f.number)
plt.xticks(range(df.shape[1]), df.columns, fontsize=14, rotation=45)
plt.yticks(range(df.shape[1]), df.columns, fontsize=14)

cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Matrice de corrélation', fontsize=16, y=-0.15)
plt.show()
# plus la couleur est proche du jaune, plus les variables sont corrélées

#### REPARTITIONS DES CLASSES #####

y = df['label']
values = np.unique(y,return_counts=True)[1]

labels = genres
sizes = values

# Choix des couleurs

colors =['#0aaff1','#edf259','#a79674']
 
# Construction du diagramme et affichage des genres et leurs fréquences en pourcentage

fig1, ax1 = plt.subplots()
ax1.pie(sizes, colors = colors, labels=labels, autopct='%1.1f%%', startangle=90)

# Tracé du cercle au milieu

centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

# Affichage du diagramme

ax1.axis('equal')  
plt.tight_layout()
plt.show()



