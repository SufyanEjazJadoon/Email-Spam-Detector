import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# Page Configuration
st.set_page_config(page_title="AI Spam Detector", page_icon="📩")

# Training Data
data = {
    'text': [
        'Hey Sufyan, are we still meeting for the software engineering project today?',
        'WINNER!! You have been selected to receive a £900 cash prize reward! Call now.',
        'Idrees, please send me the class timetable for BSSE 4D.',
        'URGENT! Your account balance is low. Claim your £2000 bonus cash immediately.',
        'Free entry in a weekly competition to win national football match tickets.',
        'I am running a bit late for the AUST university lecture, text you soon.',
        # New generic spam patterns added here:
        'You won 20000 dollar lottery cash prize click here to claim now',
        'Congratulations you won free cash rewards and dollar bonus',
        'Dear student, your assignment submission date has been extended.'
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'spam', 'ham']
}
df = pd.DataFrame(data)

cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# Interface Design
st.title("📩 Intelligent Email Spam Detection System")
st.subheader("BSSE-4D - Sufyan Ejaz & Muhammad Idrees (AUST)")

user_input = st.text_area("Enter Email Text Here:", placeholder="Type your email content...")

if st.button("Classify Email"):
    if user_input.strip() == "":
        st.warning("Please type something first!")
    else:
        input_vector = cv.transform([user_input])
        prediction = model.predict(input_vector)[0]
        if prediction == 'spam':
            st.error("🚨 SPAM DETECTED!")
        else:
            st.success("✅ HAM (NORMAL EMAIL)")