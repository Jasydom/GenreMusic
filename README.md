### README - Projet de Classification Musicale

#### Description  
Ce projet utilise des techniques de **Machine Learning** pour classifier des morceaux de musique selon leurs genres. Les genres choisis sont **blues**, **classical** et **country**, et le dataset provient de **GTZAN**.

---

#### Structure  
1. **create_dataframe.py**  
   - Extrait les caractéristiques audio (MFCC, ZCR, etc.) à l'aide de **Librosa** et crée un fichier `music.csv`.  

2. **classification.py**  
   - Analyse les données, réduit la dimensionnalité et visualise la répartition des genres.

3. **knn.py**  
   - Implémente un modèle **K-Nearest Neighbors** avec optimisation d'hyperparamètres.  

4. **random_forest.py**  
   - Utilise **Random Forest** pour entraîner un modèle avec une précision optimale (~84%).

5. **SVM.py**  
   - Implémente un **SVM** pour la classification avec analyse des performances.

---

#### Prérequis  
- Python 3.x  
- Libraries : Pandas, NumPy, Librosa, Scikit-learn, Matplotlib, Seaborn  

---

#### Exécution  
1. Exécutez `create_dataframe.py` pour générer `music.csv`.  
2. Lancez les scripts de classification pour tester différents modèles :  
   - `python knn.py`  
   - `python random_forest.py`  
   - `python SVM.py`  

---

#### Résultats  
- **Random Forest** : Meilleure précision (~84%).  
- **KNN** et **SVM** : Précision plus faible autour de **30-50%**.
