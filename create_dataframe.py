###### CE FICHIER EST A EXECUTER UNE SEULE FOIS POUR CREER LE DATAFRAME "music.csv" ##########

import pandas as pd # Pour le dataframe
import numpy as np # Pour la normalisation et calculs de moyenne

import librosa # Pour l'extraction des parametres et la lecture des fichiers wav
import librosa.display # Pour récupérer les spectrogrammes des audio
import librosa.feature

import os # C'est ce qui va nous permettre d'itérer sur les fichiers de l'environnement de travail

########### EXTRACTION DES parametres ##########

# Liste avec les genres musicaux

genres = ['blues', 'classical', 'country']

# Dictionnaire avec les genres

fichiers = {}


for g in genres:
  fichiers[g] = []

# Remplissage du dictionnaire grace à Librosa

for g in genres:
  for audio in os.listdir(f'./Data/genres_original/{g}'):
    fichiers[g].append(librosa.load(f'./Data/genres_original/{g}/{audio}')[0])

def data_audio(audio): #cette fonction renvoie une liste des 23 variables les plus pertinentes pour l'analyse musicale
    
  parametres = []

  # Calcul du ZCR

  zcr = librosa.zero_crossings(audio)
  parametres.append(sum(zcr))

  # Calcul de la moyenne du Spectral centroid

  spectral_centroids = librosa.feature.spectral_centroid(audio)[0]
  parametres.append(np.mean(spectral_centroids))
  
  # Calcul du spectral rolloff point

  rolloff = librosa.feature.spectral_rolloff(audio)
  parametres.append(np.mean(rolloff))

  # Calcul des moyennes des MFCC

  mfcc = librosa.feature.mfcc(audio)

  for x in mfcc:
    parametres.append(np.mean(x))

  return parametres

# Définition du nom des colonnes

column_names = ['zcr', 'spectral_c', 'rolloff', 'mfcc1', 'mfcc2', 'mfcc3',
                'mfcc4', 'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9',
                'mfcc10', 'mfcc11', 'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15',
                'mfcc16', 'mfcc17', 'mfcc18', 'mfcc19', 'mfcc20', 'label']

# Création d'un dataframe vide

df = pd.DataFrame(columns = column_names)

# On itère sur les audios pour remplir le dataframe

i = 0
for g in genres:
  for music in fichiers[g]:
    df.loc[i] = data_audio(music)+[g]
    i+=1

df.to_csv('music.csv', index = False)
#on enregistre dans un fichier pour ne pas avoir à importer tous les audios à chaque fois

