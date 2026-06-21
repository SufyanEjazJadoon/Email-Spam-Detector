import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 🚀 1. Advanced Styling with Custom CSS & Dark Theme Neural Network Art
st.set_page_config(
    page_title="AI Email Spam Detector | BSSE-4D",
    page_icon="📩",
    layout="wide"
)

# Custom CSS for UI Enhancement
st.markdown("""
<style>
    /* Main Background & Fonts */
    .stApp {
        background: linear-gradient(135deg, #1a1a1d, #4e4e50);
        color: #fff !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    
    /* Header Container Styling */
    .header-box {
        background-color: rgba(0, 0, 0, 0.4);
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        border: 2px solid #ed1c24; /* Red border for emphasis */
        margin-bottom: 20px;
    }
    
    /* Main Title Styling */
    h1 {
        color: #fff !important;
        font-size: 3em !important;
        font-weight: 700 !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
        margin: 0;
    }
    
    /* Subheader Styling */
    h3 {
        color: #c9c9c9 !important;
        font-size: 1.2em !important;
        font-weight: 400 !important;
        margin-top: 10px;
    }
    
    /* Text Area Styling */
    .stTextArea textarea {
        background-color: #2b2b2b !important;
        color: #fff !important;
        border: 1px solid #ed1c24 !important;
        border-radius: 10px;
    }
    
    /* Button Styling */
    .stButton>button {
        background: linear-gradient(45deg, #ed1c24, #9a1a1f) !important;
        color: white !important;
        font-size: 18px !important;
        font-weight: bold !important;
        padding: 10px 30px !important;
        border-radius: 50px !important;
        border: none !important;
        transition: 0.3s;
    }
    .stButton>button:hover {
        box-shadow: 0 0 15px rgba(237, 28, 36, 0.6);
        transform: scale(1.05);
    }
</style>
""", unsafe_allow_stdio=True)

# 💻 2. Updated AI Data & Model Setup
data = {
    'text': [
        'Hey Sufyan, are we still meeting for the software engineering project today?',
        'WINNER!! You have been selected to receive a £900 cash prize reward! Call now.',
        'Idrees, please send me the class timetable for BSSE 4D.',
        'URGENT! Your account balance is low. Claim your £2000 bonus cash immediately.',
        'Free entry in a weekly competition to win national football match tickets.',
        'I am running a bit late for the AUST university lecture, text you soon.',
        # Testing fix for broader vocabulary
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

# 🌐 3. Professional Visual Layout Design
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Title & Subheader Container
    st.markdown("""
    <div class="header-box">
        <h1>Intelligent Email Spam Detection System</h1>
        <h3>Department of Software Engineering, AUST</h3>
        <h3>A Project by: Sufyan Ejaz Jadoon & Muhammad Idrees (BSSE-4D)</h3>
    </div>
    """, unsafe_allow_stdio=True)

    st.write("---") # Visual divider

    # User Input Section
    user_input = st.text_area("Paste/Type Email Text Here:", placeholder="Example: WINNER!! You have been selected...", height=150)

    # Classification Area
    classify_col1, classify_col2, classify_col3 = st.columns([2, 1, 2])

    with classify_col2:
        classify_btn = st.button("Classify Email")

    # Output Handling
    if classify_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please enter email text to test.")
        else:
            with st.spinner('Analysing with AI Neural Engine...'):
                input_vector = cv.transform([user_input])
                prediction = model.predict(input_vector)[0]

                st.write("\n") # Line break

                # Visualized Results
                if prediction == 'spam':
                    st.error("🚨 **AI DECISION:** This is **SPAM DETECTED!**")
                else:
                    st.success("✅ **AI DECISION:** This is **HAM (NORMAL EMAIL)**")

# Visual Footer
st.write("\n\n---\n")
st.caption("AI Email Classifier v1.0 | CS-402 Final Project Presentation (AUST 2026)")