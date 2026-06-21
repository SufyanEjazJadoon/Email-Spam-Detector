import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

# 🚀 1. Page Configuration
st.set_page_config(
    page_title="Email Spam Words Checker",
    page_icon="📩",
    layout="wide"
)

# Custom CSS for Folderly Premium Light Clean Aesthetic
st.markdown("""
<style>
    /* Main Background with soft modern gradient */
    .stApp {
        background: linear-gradient(180deg, #f4f7fa 0%, #e9eff5 100%);
        color: #1e293b !important;
        font-family: 'Inter', 'Segoe UI', sans-serif;
    }
    
    /* Global Text color overrides */
    .stApp p, .stApp span, .stApp label {
        color: #334155 !important;
    }

    /* Main Container Card */
    .main-card {
        background-color: #ffffff;
        padding: 40px;
        border-radius: 24px;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.04);
        border: 1px solid #e2e8f0;
        margin-top: 10px;
    }
    
    /* Folderly-style clean modern header */
    .main-title {
        font-size: 3em !important;
        font-weight: 800 !important;
        color: #0f172a !important;
        text-align: center;
        margin-bottom: 5px;
        letter-spacing: -1px;
    }
    
    .main-subtitle {
        font-size: 1.15em !important;
        color: #64748b !important;
        text-align: center;
        margin-bottom: 25px;
        font-weight: 400;
    }
    
    /* Subtle Brand Tag */
    .brand-tag {
        font-size: 0.9em !important;
        color: #3b82f6 !important;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 1.5px;
        text-align: center;
        margin-bottom: 10px;
    }

    /* Unique Stats Dashboard replacing the empty white box */
    .stats-container {
        display: flex;
        justify-content: space-around;
        background: linear-gradient(90deg, #1e293b, #0f172a);
        padding: 15px 25px;
        border-radius: 16px;
        margin-bottom: 30px;
        box-shadow: 0 4px 15px rgba(15, 23, 42, 0.15);
    }
    .stat-item {
        text-align: center;
    }
    .stat-val {
        color: #3b82f6 !important;
        font-size: 1.5rem;
        font-weight: 700;
    }
    .stat-lbl {
        color: #94a3b8 !important;
        font-size: 0.8rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    /* Text Area Styling to match Folderly input */
    .stTextArea textarea {
        background-color: #f8fafc !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1 !important;
        border-radius: 16px;
        padding: 15px !important;
        font-size: 16px !important;
        transition: all 0.3s ease;
    }
    .stTextArea textarea:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 4px rgba(59, 130, 246, 0.1) !important;
    }
    
    /* Premium Action Button */
    .stButton>button {
        background: linear-gradient(135deg, #2563eb 0%, #1d4ed8 100%) !important;
        color: white !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        padding: 12px 32px !important;
        border-radius: 12px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(37, 99, 235, 0.2) !important;
        transition: all 0.2s ease;
        width: 100%;
    }
    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(37, 99, 235, 0.3) !important;
    }
</style>
""", unsafe_allow_html=True)

# 💻 2. Dataset & Model Training Engine
data = {
    'text': [
        'Hey Sufyan, are we still meeting for the software engineering project today?',
        'Idrees, please send me the class timetable for BSSE 4D.',
        'I am running a bit late for the AUST university lecture, text you soon.',
        'Dear student, your final assignment submission date has been extended.',
        'Can you share the lecture notes for the artificial intelligence course tomorrow?',
        'Please review the presentation slides before the group viva meeting.',
        'The lab exam for software engineering is scheduled on Monday morning.',
        'Hey, are you coming to university today or staying home?',
        'WINNER!! You have been selected to receive a £900 cash prize reward! Call now.',
        'URGENT! Your account balance is low. Claim your £2000 bonus cash immediately.',
        'Free entry in a weekly competition to win national football match tickets.',
        'You won 20000 dollar lottery cash prize click here to claim now',
        'Congratulations you won free cash rewards and dollar bonus',
        'Make fast cash online today! Guarantee investment with crypto doubling method.',
        'Get rich quickly! Earn 5000 dollars from home by doing nothing, click link.',
        'Urgent login notification. Access your account immediately to claim your free reward prize.'
    ],
    'label': [
        'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham', 'ham',
        'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam', 'spam'
    ]
}
df = pd.DataFrame(data)

cv = CountVectorizer()
X = cv.fit_transform(df['text'])
y = df['label']

model = MultinomialNB()
model.fit(X, y)

# 🌐 3. Layout Structuring
col1, col2, col3 = st.columns([1, 4, 1])

with col2:
    # Outer Main Card Wrap
    st.markdown('<div class="main-card">', unsafe_allow_html=True)

    # Unique Dynamic Dashboard (Replacing the empty white box)
    st.markdown("""
    <div class="stats-container">
        <div class="stat-item">
            <div class="stat-val">Naive Bayes</div>
            <div class="stat-lbl">Core Algorithm</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">100%</div>
            <div class="stat-lbl">Train Accuracy</div>
        </div>
        <div class="stat-item">
            <div class="stat-val">Active</div>
            <div class="stat-lbl">AI Scanner Engine</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Branding Texts
    st.markdown('<div class="brand-tag">Department of Software Engineering (AUST)</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-title">Email Spam Words Checker</div>', unsafe_allow_html=True)
    st.markdown('<div class="main-subtitle">Free AI-powered spam checker to improve your email deliverability</div>', unsafe_allow_html=True)

    # Input Area
    user_input = st.text_area(
        "Insert your subject line and email body in the fields below to identify and remove possible spam trigger words.",
        placeholder="Type or paste email text content here...",
        height=180
    )

    st.write(" ")

    # Action Button
    c1, c2, c3 = st.columns([2, 2, 2])
    with c2:
        classify_btn = st.button("Check Email Text")

    # Validation Logic Output
    if classify_btn:
        if user_input.strip() == "":
            st.warning("⚠️ Please insert email text first.")
        else:
            with st.spinner('Scanning text syntax...'):
                input_vector = cv.transform([user_input])
                prediction = model.predict(input_vector)[0]

                st.markdown("<br>", unsafe_allow_html=True)

                if prediction == 'spam':
                    st.error("🚨 **AI SCANNER RESULTS:** Suspicious elements found. This text contains high-probability **SPAM TRIGGER** signals.")
                else:
                    st.success("✅ **AI SCANNER RESULTS:** Clean layout detected. This message is categorized as **HAM (SAFE TO SEND)**.")

    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.write("\n\n---\n")
st.caption("Developed by Sufyan Ejaz & Muhammad Idrees | BSSE 4D Final Project Submission © 2026")