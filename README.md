# AI Diabetes Detection — détection du diabète par machine learning

Application web Streamlit qui explore un jeu de données médical et entraîne deux modèles de classification pour prédire si une personne est diabétique.

> Les prédictions n'ont aucune valeur médicale et ne remplacent pas un diagnostic.

## Ce que fait l'application

L'application comporte quatre pages, accessibles depuis la barre latérale :

| Page | Contenu |
|---|---|
| **Home** | Aperçu des données nettoyées et statistiques descriptives |
| **Data Visualizations** | Histogramme des âges, nuage de points glucose / IMC, carte de corrélation, distribution de chaque mesure selon le diagnostic |
| **ML Model 1** | Régression logistique : choix des variables et de la taille du jeu de test, entraînement, métriques, puis prédiction sur des valeurs saisies |
| **ML Model 2** | Même parcours avec un arbre de décision |

Pour chaque modèle, l'application affiche l'exactitude, la matrice de confusion et le rapport de classification (précision, rappel, F1).

Avec les variables par défaut (glucose et IMC) et 20 % des données en test, la régression logistique atteint une exactitude de 0,76 et l'arbre de décision 0,63.

## Les données

`diabetes.csv` est le jeu de données « Pima Indians Diabetes », collecté par le National Institute of Diabetes and Digestive and Kidney Diseases (États-Unis) : des mesures relevées sur des femmes d'origine pima âgées de 21 ans et plus.

| Colonne | Signification |
|---|---|
| `Pregnancies` | Nombre de grossesses |
| `Glucose` | Concentration de glucose dans le plasma |
| `BloodPressure` | Pression artérielle diastolique (mm Hg) |
| `SkinThickness` | Épaisseur du pli cutané du triceps (mm) |
| `Insulin` | Insuline sérique |
| `BMI` | Indice de masse corporelle |
| `DiabetesPedigreeFunction` | Score d'antécédents familiaux de diabète |
| `Age` | Âge en années |
| `Outcome` | 1 = diabétique, 0 = non diabétique |

Dans ce fichier, un zéro signifie « valeur manquante », sauf pour `Pregnancies` et `Outcome`. L'application supprime donc les lignes où le glucose, la pression artérielle, l'IMC, le score d'antécédents ou l'âge valent zéro. Les colonnes `Insulin` et `SkinThickness` sont conservées telles quelles, car elles contiennent trop de zéros pour être filtrées.

## Installation

Il faut Python 3.10 ou plus récent.

```bash
git clone git@github.com:SteveTetchoup26/ai-diabetes-detection.git
cd ai-diabetes-detection

python3 -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt
```

## Lancement

```bash
streamlit run app.py
```

L'application s'ouvre sur http://localhost:8501.

## Contenu du dépôt

| Fichier | Rôle |
|---|---|
| `app.py` | L'application, commentée en français étape par étape |
| `diabetes.csv` | Le jeu de données |
| `requirements.txt` | Les dépendances Python |

## Technologies

Python, Streamlit, pandas, NumPy, Matplotlib, Seaborn, scikit-learn.

## Limites connues

- Les textes d'aide du rapport de classification citent des chiffres fixes qui ne correspondent pas toujours aux résultats affichés.
- Les deux pages de modèles partagent la même liste de variables en mémoire : après avoir entraîné un modèle, il faut réentraîner l'autre avant de lui demander une prédiction.
- Le jeu de données est petit et ne concerne qu'une population précise ; les modèles ne se généralisent pas au-delà.
