import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

# 1. Charger les données que tu viens de créer
X = np.load("features.npy")
y = np.load("labels.npy")

# 2. Diviser : 80% pour apprendre, 20% pour l'examen final
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Préparer les 3 modèles demandés par le prof
modeles = {
    "Arbre de Décision": DecisionTreeClassifier(),
    "Random Forest": RandomForestClassifier(n_estimators=100),
    "SVM": SVC(probability=True)
}

# 4. Lancer le tournoi
print("Début de l'entraînement des modèles...\n")
meilleur_score = 0
meilleur_modele = None
nom_gagnant = ""

for nom, model in modeles.items():
    model.fit(X_train, y_train) # L'apprentissage
    predictions = model.predict(X_test) # L'examen
    score = accuracy_score(y_test, predictions)
    
    print(f"Résultat pour {nom} : {score:.2%} d'accuracy")
    
    # On garde le champion en mémoire
    if score > meilleur_score:
        meilleur_score = score
        meilleur_modele = model
        nom_gagnant = nom

# 5. Sauvegarder le champion pour ton application finale (Streamlit)
joblib.dump(meilleur_modele, "mon_modele_ia.pkl")
print(f"\nFélicitations ! Le gagnant est '{nom_gagnant}' avec {meilleur_score:.2%}.")
print("Le fichier 'mon_modele_ia.pkl' a été créé.") 