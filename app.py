 

# imports
import streamlit as st  # pour la création de l'interface de l'application web
import pandas as pd # pour l'organisation des données en tableaux
import matplotlib.pyplot as plt # pour la création de graphiques
import seaborn as sns # pour des visualisations de données plus avancées
import numpy as np # pour les opérations numériques rapides

# Cleaning up our data
df = pd.read_csv("diabetes.csv") # chargement du fichier CSV avec les données sur le diabète dans un DataFrame pandas
df_temp = df.drop(columns=['Outcome','Pregnancies','Insulin','SkinThickness']) # création d'un DataFrame temporaire en supprimant certaines colonnes
df_temp = df_temp.replace(0,np.nan) # remplacement des zéros par NaN pour indiquer les valeurs manquantes
df = pd.concat([df['Pregnancies'],df['Insulin'],df['SkinThickness'],df_temp,df['Outcome']],axis=1) # concaténation des colonnes d'origine avec le DataFrame temporaire
df = df.dropna().reset_index(drop=True) # suppression des lignes avec des valeurs manquantes et réinitialisation des index

# rename the column names so they have spaces
df = df.rename(columns={'BloodPressure': 'Diastolic Blood Pressure','DiabetesPedigreeFunction': 'Diabetes Pedigree Function'}) # renommage des colonnes pour une meilleure lisibilité

# Setting up the sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Visualizations", "ML Model 1", "ML Model 2"])

# Homepage
if page == "Home":
    st.title("Welcome to the Diabetes App!")
    st.write("Use the sidebar to navigate to different sections.")
    st.write("Here's a quick look at the data:")
    st.dataframe(df.head()) # Retourner les N premières lignes du DataFrame par défaut N=5
    st.dataframe(df.describe()) # Afficher les statistiques descriptives du DataFrame

    with st.expander("Read more about this dataset!"): # Créer une section pliable/dépliable
        st.write("""This is data collected from a group of Pima Indian women, aged 21 and above. 
        Zeros indicate missing values, except in the case of outcome, where they indicate a non-diabetic person. 
        There are 768 rows of data, and 8 features, not including Outcome.""")

# Data Viz Page
elif page == "Data Visualizations":
    st.title("Let's take a look, shall we?")
    st.write("Choose a figure you're curious about!")

    # Graphiques prédéfinis
    if st.button("Display Age Histogram"):
        st.title("Age Histogram")
        fig, ax = plt.subplots()
        #Fonction : plt.subplots()

        # Rôle : Créer une figure (conteneur) et des axes (zone de dessin)
        # Retour : Tuple de 2 éléments
        # fig : Figure (la toile)
        # ax : Axes (l'espace de dessin sur la toile)

        ax.hist(df["Age"], bins=20)
        # Fonction : .hist()
        # Rôle : Créer un histogramme
        # Argument 1 : df["Age"] = données à représenter (une colonne)
        # Argument 2 : bins=20 = nombre de barres dans l'histogramme

        st.pyplot(fig)
        # Rôle : Afficher une figure matplotlib dans Streamlit
        # Argument : L'objet figure créé précédemment

    if st.button("Display Glucose vs. BMI Scatterplot"):
        st.title("Glucose vs BMI")
        fig, ax = plt.subplots()
        ax.scatter(df["Glucose"], df["BMI"])
        # Rôle : Créer un nuage de points
        # Argument 1 : df["Glucose"] = valeurs pour l'axe X
        # Argument 2 : df["BMI"] = valeurs pour l'axe Y
        st.pyplot(fig)
    if st.button("Display Correlation Heatmap"):
        corr = df.corr()
        # Rôle : Calculer la matrice de corrélation
        st.title("Correlation Heatmap")
        fig, ax = plt.subplots()
        sns.heatmap(data=corr, ax=ax,cmap='coolwarm', annot=True)
        # Fonction : sns.heatmap()
        # Rôle : Créer une carte thermique (heatmap) pour visualiser les corrélations
        # Argument 1 : data=corr = matrice de corrélation calculée précéd
        # Argument 2 : ax=ax = axes où dessiner la heatmap
        # Argument 3 : cmap='coolwarm' = palette de couleurs utilisée
        # Argument 4 : annot=True = afficher les valeurs numériques sur la heatmap
        st.pyplot(fig)

        # Correlation Heatmap Explanation
        with st.expander("What is this plot showing?"):
            st.write("""This is a correlation heatmap. Warmer colors indicate positive correlations and cooler colors indicate negative correlations. 
            The closer the absolute value of the correlation is to 1, the more strongly correlated.""")
        
    # Buttons for distribution by outomce
    st.write("Choose a characteristic to see its distribution by outcome.")
    for feature in df.columns:
        if feature == 'Outcome':
            continue
        if st.button(feature):
            fig, ax = plt.subplots()
            sns.kdeplot(data = df[df['Outcome']==1], x = feature, label = 'Diabetic', ax=ax)

            sns.kdeplot(data = df[df['Outcome']==0], x = feature, label = 'Non-diabetic',ax=ax)
            ax.set_title(f'{feature} Distribution by Outcome')
            ax.legend()
            st.pyplot(fig)
        
    
# Logistic Regression Page
elif page == "ML Model 1":
    if 'scaler' not in st.session_state:
        scaler = None
    # imports
    from sklearn.model_selection import train_test_split # pour diviser les données en ensembles d'entraînement et de test
    from sklearn.linear_model import LogisticRegression # pour créer et entraîner un modèle de régression logistique
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report # pour évaluer les performances du modèle

    st.title("Logistic Regression Model")
    # Explain the model
    with st.expander("How does this model work?"):
            st.write("""This is a logistic regression model, often used for classification tasks, such as 'spam' vs. 'not spam' or our scenario, 
            'diabetic' vs. 'non-diabetic'. It multiplies each value for each feature in the information passed to it and sums all of them;  
            it then passes this sum through a sigmoid function, which squeezes this to ensure all values are between 0 and 1.  For larger sums, 
            the output is closer to one, and smaller sums produce an output closer to 0. If the final output is over a certain threshold, usually ~0.5, 
            the model returns '1.' Otherwise, it returns '0.'
            """)

    # Choosing features
    features = st.multiselect("Choose features:", df.columns[:-1], default=["Glucose", "BMI"]) # Créer une liste déroulante à sélection multiple
    target = "Outcome"
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100 # Créer un curseur pour sélectionner la taille de l'ensemble de test
    # role: Permettre à l'utilisateur de choisir la taille de l'ensemble de test entre 10% et 50%, avec une valeur par défaut de 20%

    # Training the model
    if st.button("Train model"):
        X = df[features]
        y = df[target]

       
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        # Doing the scaling
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler() # Standardiser les features (moyenne = 0, écart-type = 1)
        # Formule: z = (x - μ) / σ
        # z : valeur standardisée
        # x : valeur originale
        # μ : moyenne des données
        # σ : écart-type des données

        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # .fit(): Calculer les paramètres de mise à l'échelle (moyenne et écart-type) à partir des données d'entraînement
        # .transform(): Appliquer la mise à l'échelle aux données en utilisant les paramètres calculés
        
        model = LogisticRegression() # Créer une instance du modèle de régression logistique
        model.fit(X_train_scaled, y_train) # Entraîner le modèle sur les données d'entraînement mises à l'échelle

        y_pred = model.predict(X_test_scaled) # Faire des prédictions sur les données de test mises à l'échelle

        acc = accuracy_score(y_test, y_pred) # Calculer la précision des prédictions
        # y_pred : valeurs prédites par le modèle
        # y_test : valeurs réelles des données de test
        # formule : Accuracy = (Nombre de prédictions correctes) / (Nombre total de prédictions)

        cm = confusion_matrix(y_test, y_pred) # Calculer la matrice de confusion 2x2 montrant les types d'erreurs de classification
        report = classification_report(y_test,y_pred) # Générer un rapport de classification avec des métriques détaillées

        st.success(f"Accuracy: {acc:.2f}")
        st.write("Confusion matrix:")
        st.write(cm)

        with st.expander("What is this showing?"):
            st.write("""This is a confusion matrix, a way to get a quick look at how our model performed on the testing set. The top indicates the 
            actual correct answer - 0 or 1. The first row includes all the model's false predictions; for example, the top left square indicates how 
            many false cases were assigned 'false' by the model. Similarly, the botton row indicates all the model's true predictions, so the bottom 
            left square indicates how many negative cases were assigned 'positive' by the model.""")

       
        st.write("Classification Report:")
        st.text(report)  # st.text preserves the formatting from sklearn's output
        with st.expander('What is this showing?'):
            st.write("""The first number, 0.81, tells us the portion of 0 predictions that actually were 0. The second number, 0.80, tells us the
            portion of true 0s, or negative outcomes, that the model predicted to be 0. Similarly, in the bottom row, the precision is the portion 
            of 1 predictions that actually were 1. Essentially, if the model predicted 1, there’s a 65% chance that you actually do have diabetes. 
            On the other hand, the recall is telling us that 67% of the true diabetics in the test set were actually caught by the 
            model - that is, 0.67 is the portion of true 1s that the model predicted to be 1.""") 

        #Saving the model
        st.session_state['model'] = model
        st.session_state['features'] = features
        st.session_state['scaler'] = scaler

    # Show form for prediction if model exists
    if 'model' in st.session_state:
        st.subheader("Make a Prediction")

        #Reload the model
        model = st.session_state['model']
        features = st.session_state['features']
        scaler = st.session_state['scaler']
        # Allow users to enter data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in features:
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0) # créer un champ de saisie numérique pour chaque feature
            submitted = st.form_submit_button("Predict!")
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                prediction = model.predict(scaler.transform(input_df))
                st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")
                

elif page == "ML Model 2":
    # Imports
    from sklearn.model_selection import train_test_split
    from sklearn.tree import DecisionTreeClassifier # pour créer et entraîner un modèle de classification par arbre de décision
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    st.title("Decision Tree Classifier")
    with st.expander("How does this model work?"):
            st.write("""A decision tree creates a series of splits, or branches, using layers of condtions such as 'Age < 30' and 'Age > 30' to classify data. When making a prediction, it simply follows the path of the tree, given a certain piece of data, and outputs the result of the last branch it reaches, which in our case is either a 0 or a 1.""")
            
    # Allow user to select features
    features = st.multiselect("Choose features:", df.columns[:-1], default=["Glucose", "BMI"])
    target = "Outcome"
    test_size = st.slider("Test set size (%)", 10, 50, 20) / 100

    # Train model
    if st.button("Train model"):
        X = df[features]
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

        model1 = DecisionTreeClassifier()
        model1.fit(X_train, y_train)

        preds = model1.predict(X_test)
        acc = accuracy_score(y_test, preds)
        cm = confusion_matrix(y_test, preds)

        y_pred = model1.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test,y_pred)

        st.success(f"Accuracy: {acc:.2f}")
        st.write("Confusion matrix:")
        st.write(cm)

        with st.expander("What is this showing?"):
            st.write("""This is a confusion matrix, a way to get a quick look at how our model performed on the testing set. The top indicates the 
            actual correct answer - 0 or 1. The first row includes all the model's false predictions; for example, the top left square indicates how 
            many false cases were assigned 'false' by the model. Similarly, the botton row indicates all the model's true predictions, so the bottom 
            left square indicates how many negative cases were assigned 'positive' by the model.""")

       
        st.write("Classification Report:")
        st.text(report)  # st.text preserves the formatting from sklearn's output
        with st.expander('What is this showing?'):
            st.write("""The first number, 0.81, tells us the portion of 0 predictions that actually were 0. The second number, 0.80, tells us the
            portion of true 0s, or negative outcomes, that the model predicted to be 0. Similarly, in the bottom row, the precision is the portion 
            of 1 predictions that actually were 1. Essentially, if the model predicted 1, there’s a 65% chance that you actually do have diabetes. 
            On the other hand, the recall is telling us that 67% of the true diabetics in the test set were actually caught by the 
            model - that is, 0.67 is the portion of true 1s that the model predicted to be 1.""") 
        

        # Save model
        st.session_state['model1'] = model1
        st.session_state['features'] = features

    # Show form for prediction if model exists
    if 'model1' in st.session_state:
        st.subheader("Make a Prediction")

        # Reload model
        model = st.session_state['model1']
        features = st.session_state['features']

        # Allow user to input data
        with st.form("prediction_form"):
            feature_dict = {}
            for feature in features:
                feature_dict[feature] = st.number_input(f"Enter your {feature}:", value=0.0)
            submitted = st.form_submit_button("Predict!")

            # Make a prediction
            if submitted:
                input_df = pd.DataFrame([feature_dict])
                prediction = model.predict(input_df)
                st.success(f"Prediction: {'Diabetic' if prediction[0] == 1 else 'Not Diabetic'}")