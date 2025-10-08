import streamlit as st
import pickle

# Charger les objets
model = pickle.load(open("model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Config de la page
st.set_page_config(page_title="Chatbot Ballon d'Or 2025", page_icon="🏆", layout="centered")

st.title("🏆 Chatbot – Ballon d'Or 2025")
st.write("Posez une question sur la cérémonie du Ballon d'Or 2025 🎤")

# Réponses
responses = {
    "ballon_dor": "🏆 Le Ballon d'Or 2025 a été remporté par **Kylian Mbappé** 🇫🇷.",
    "ballon_dor_feminin": "👑 Le Ballon d'Or Féminin 2025 a été remporté par **Alexia Putellas** 🇪🇸.",
    "lieu": "📍 La cérémonie s'est tenue à **Paris, France**.",
    "date": "🗓️ La cérémonie a eu lieu le **27 octobre 2025**.",
    "trophee_kopa": "🌟 Le Trophée Kopa 2025 a été remporté par **Lamine Yamal** 🇪🇸."
}

# Interface utilisateur
question = st.text_input("💬 Votre question :")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if question:
    q_vec = vectorizer.transform([question])
    pred = model.predict(q_vec)[0]
    answer = responses[pred]

    st.session_state.chat_history.append(("👤 Vous", question))
    st.session_state.chat_history.append(("🤖 Bot", answer))

# Affichage de la conversation
for sender, msg in st.session_state.chat_history:
    if sender == "👤 Vous":
        st.markdown(f"**{sender}** : {msg}")
    else:
        st.success(f"**{sender}** : {msg}")
