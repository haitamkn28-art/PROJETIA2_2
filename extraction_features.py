import os
import cv2
import numpy as np
from skimage.feature import graycomatrix, graycoprops

def extract_features(image_path):
    # 1. Chargement et Prétraitement
    img = cv2.imread(image_path)
    if img is None: 
        return None
    
    # Redimensionnement (pour que toutes les images aient la même taille)
    img = cv2.resize(img, (128, 128))
    # Conversion en gris pour les calculs de texture (GLCM)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # 2. Extraction GLCM (Texture)
    # On calcule la matrice pour 4 angles afin d'être très précis
    glcm = graycomatrix(gray, distances=[1], angles=[0, np.pi/4, np.pi/2, 3*np.pi/4], 
                        levels=256, symmetric=True, normed=True)
    
    # On extrait les propriétés demandées (Remplace Haralick par ces 6 mesures clés)
    properties = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
    glcm_feat = [graycoprops(glcm, prop).mean() for prop in properties]
    
    # 3. BiT / Intensité (Moyenne et Écart-type)
    bit_feat = [np.mean(gray), np.std(gray)]
    
    # 4. Concaténation des caractéristiques [cite: 14, 32]
    return np.hstack([glcm_feat, bit_feat])

# --- Boucle principale sur votre dossier 'dataset' ---
dataset_path = "dataset/"
data = []
labels = []

# Parcourt chaque dossier d'animal (bear, bee, butterfly, etc.)
for category in os.listdir(dataset_path):
    cat_path = os.path.join(dataset_path, category)
    if os.path.isdir(cat_path):
        print(f"Extraction en cours : {category}...")
        for img_name in os.listdir(cat_path):
            img_full_path = os.path.join(cat_path, img_name)
            feat = extract_features(img_full_path)
            if feat is not None:
                data.append(feat)
                labels.append(category)

# --- Sauvegarde des caractéristiques extraites [cite: 34] ---
np.save("features.npy", np.array(data))
np.save("labels.npy", np.array(labels))

print(f"\nTerminé ! {len(data)} images traitées.")
print("Fichiers 'features.npy' et 'labels.npy' sauvegardés dans votre dossier.") 
