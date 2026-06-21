import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 🚀 1. Advanced Page Setup
st.set_page_config(
    page_title="AI Email Spam Detector | BSSE-4D",
    page_icon="📩",
    layout="wide"
)

# Custom CSS for Premium Dark Theme UI
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background: linear-gradient(135deg, #111111, #252526);
        color: #ffffff !important;
    }
    
    /* Header Box Styling */
    .header-box {
        background-color: rgba(255, 255, 255, 0.05);
        padding: 35px;
        border-radius: 15px;
        text-align: center;
        border-bottom: 4px solid #ff4b4b;
        margin-bottom: 25px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.3);
    }
    
    /* Typography Fixes */
    .header-box h1 {
        color: #ffffff !important;
        font-size: 2.6em !important;
        font-weight: 700 !important;
        margin-bottom: 10px;
    }
    .header-box h3 {
        color: #cccccc !important;
        font-size: 1.1em !important;
        font-weight: 400 !important;
        margin: 5px 0;
    }
    
    /* Input Text Area Customization */
    .stTextArea textarea {
        background-color: #1e1e1e !important;
        color: #ffffff !important;
        border: 1px solid #ff4b4b !important;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True) # <-- Fixed the typo here!

# 💻 2. Dataset & AI Model Setup
data = {
    'text': [
        'Hey Sufyan, are we still meeting for the software engineering project today?',
        'WINNER!! You have been selected to receive a £900 cash prize reward! Call now.',
        'Idrees, please send me the class timetable for BSSE 4D.',
        'URGENT! Your account balance is low. Claim your £2000 bonus cash immediately.',
        'Free entry in a weekly competition to win national football match tickets.',
        'I am running a bit late for the AUST university lecture, text you soon.',
        'You won 20000 dollar lottery cash prize click here to claim now',
        'Congratulations you won free cash rewards and dollar bonus',
    ],
    'label': ['ham', 'spam', 'ham', 'spam', 'spam', 'ham', 'spam', 'spam']
}
df = pd.DataFrame(data)

cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# 🌐 3. Visual Layout Structure
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Header Banner
    st.markdown("""
    <div class="header-box">
        <h1>Intelligent Email Spam Detection System</h1>
        <h3>Department of Software Engineering, AUST</h3>
        <h3>A Project by: Sufyan Ejaz Jadoon & Muhammad Idrees (BSSE-4D)</h3>
    </div>
    """, unsafe_allow_html=True)

    st.write(" ") # Whitespace

    # Text Input from User
    user_input = st.text_area("Paste / Type Your Email Text Here:", placeholder="Example: WINNER!! You have been selected...", height=150)

    st.write(" ")

    # Centered Button Action
    b_col1, b_col2, b_col3 = st.columns([2, 1, 2])
    with b_col2:
        classify_btn = st.button("Classify Email", use_container_width=True)

    # Result Evaluation
    if classify_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter email text to test.")
        else:
            input_vector = cv.transform([user_input])
            prediction = model.predict(input_vector)[0]

            st.write("---") # Section Divider

            # Custom styled outcome banners
            if prediction == 'spam':
                st.error("🚨 **AI DECISION:** This is **SPAM DETECTED!**")
            else:
                st.success("✅ **AI DECISION:** This is **HAM (NORMAL EMAIL)**")

# Visual Footer
st.write("\n\n---\n")
st.caption("AI Email Classifier v1.1 | CS-402 Final Project Presentation (AUST 2026)")