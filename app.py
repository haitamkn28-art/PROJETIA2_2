import streamlit as st
import numpy as np
import joblib
import os
from cbir_utils import distance_euclidienne, distance_canberra, distance_cosinus
from extraction_features import extract_features 

# 1. Titre et chargement du modèle
st.title("Système de Recherche d'Images (CBIR)")
model = joblib.load("mon_modele_ia.pkl")

# 2. Upload de l'image
uploaded_file = st.file_uploader("Choisissez une image d'animal...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Sauvegarde temporaire
    with open("temp.jpg", "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    # Extraction des caractéristiques de l'image envoyée
    feat = extract_features("temp.jpg")
    
    if feat is not None:
        # Prédiction de la classe
        classe_predite = model.predict([feat])[0]
        st.write(f"Modèle prédit : **{classe_predite}**")
        
        # Choix de la distance
        dist_type = st.selectbox("Choisir la méthode de calcul de distance", ["Euclidienne", "Canberra", "Cosinus"])
        
        if st.button("Rechercher des images similaires"):
            dataset_path = f"dataset/{classe_predite}"
            resultats = []
            
            # Comparaison avec toutes les images de la même classe
            if os.path.exists(dataset_path):
                for img_name in os.listdir(dataset_path):
                    img_path = os.path.join(dataset_path, img_name)
                    feat_img = extract_features(img_path)
                    
                    if feat_img is not None:
                        if dist_type == "Euclidienne":
                            d = distance_euclidienne(feat, feat_img)
                        elif dist_type == "Canberra":
                            d = distance_canberra(feat, feat_img)
                        else:
                            d = distance_cosinus(feat, feat_img)
                        
                        resultats.append((img_path, d))
                
                # Trier par distance (la plus petite est la plus proche)
                resultats.sort(key=lambda x: x[1])
                
                # Affichage des 5 meilleurs résultats
                st.write("Résultats les plus proches :")
                cols = st.columns(5)
                for i, (path, score) in enumerate(resultats[:5]):
                    cols[i].image(path, caption=f"Dist: {score:.2f}")
            else:
                st.error(f"Le dossier {dataset_path} n'a pas été trouvé.") 