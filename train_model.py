import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

# === Données d'entraînement ===
train_data = {
    "ballon_dor": [
        "Qui a gagné le ballon d'or 2025 ?",
        "Quel joueur a remporté le ballon d'or cette année ?",
        "Qui a reçu le trophée du meilleur footballeur ?",
        "Ballon d'or masculin 2025"
    ],
    "ballon_dor_feminin": [
        "Qui a gagné le ballon d'or féminin ?",
        "Quelle joueuse a remporté le ballon d'or ?",
        "Ballon d'or féminin 2025"
    ],
    "lieu": [
        "Où s'est déroulée la cérémonie ?",
        "Dans quelle ville s'est passé le ballon d'or ?",
        "Lieu de la cérémonie"
    ],
    "date": [
        "Quand a eu lieu la cérémonie du ballon d'or ?",
        "Date du ballon d'or 2025",
        "C'était quand le ballon d'or ?"
    ],
    "trophee_kopa": [
        "Qui a remporté le trophée kopa ?",
        "Quel jeune joueur a gagné le kopa ?",
        "Gagnant du trophée kopa 2025"
    ]
}

# === Préparation des données ===
X, y = [], []
for label, texts in train_data.items():
    for text in texts:
        X.append(text)
        y.append(label)

# === Vectorisation et modèle ===
vectorizer = TfidfVectorizer()
X_vec = vectorizer.fit_transform(X)

model = LogisticRegression()
model.fit(X_vec, y)

# === Sauvegarde ===
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("vectorizer.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print("✅ Modèle et vectorizer sauvegardés avec succès !")
