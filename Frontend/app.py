#Run using: streamlit run Frontend/app.py

import streamlit as st
import pandas as pd
from PIL import Image
from datetime import time
import sqlite3
import base64
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from chatbot_agent import run_chatbot_interface
from email_sender import send_gmail

# --- Page Configuration --- This must be at the TOP of the code (Don't change order !!)
icon_path = "Frontend/assets/SAL_Agent.png" 

# Check if SAL Agent logo exists
if os.path.exists(icon_path):
    custom_logo = Image.open(icon_path)
else:
    custom_logo = "🤖" 

st.set_page_config(
    layout="wide",
    page_title="SAL Agent",
    page_icon=custom_logo
)

# --- Sidebar Content ---
with st.sidebar:
    st.image(icon_path, use_container_width=True)
    st.header("SAL Agent")
    
    # Check if a user is logged in and has a name stored in memory
    if st.session_state.get('logged_in') and 'user_name' in st.session_state:
        first_name = st.session_state['user_name'].split()[0]
        st.write(f"Welcome, **{first_name}** 👋")
    else:
        st.write("Welcome to SAL Agent")

# --- Page Functions ---
def home_page():
    st.title("Welcome to SAL Agent")
    st.markdown("#### *Optimizing the B2B Sales Automation Lifecycle with Agentic AI*")

    st.markdown(
        """
        <style>
        .sales-card {
            background-color: #F1F5F9; 
            padding: 20px;
            border-radius: 20px;
            
            border-top: 6px solid #bedfdb; 
            text-align: center;
            margin-bottom: 20px;
            min-height: 380px;
            transition: transform 0.2s;
        }
        
        /* Subtle interactive lift when hovering over a team card */
        .sales-card:hover {
            transform: translateY(-5px);
        }

        .card-name {
            font-size: 22px;
            font-weight: bold;
            margin-top: 15px;
            margin-bottom: 8px;
            color: #0F172A; /* Matches textarrow color */
            line-height: 1.2;
        }
        .card-role {
            font-style: italic;
            color: #475569;
            margin-bottom: 10px;
            font-size: 15px;
        }
        .card-caption {
            font-size: 13px;
            color: #64748B;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    st.image(
        "Frontend/assets/Agent.png", 
        use_container_width=True
    )
    
    st.subheader("Meet the Engineering & Development Team")
    
    def get_image_base64(path):
        if os.path.exists(path):
            with open(path, "rb") as img_file:
                encoded = base64.b64encode(img_file.read()).decode()
            return f"data:image/png;base64,{encoded}"
        return "https://static.vecteezy.com/system/resources/previews/024/098/521/large_2x/female-developer-with-desktop-free-png.png" # Backup icon if missing

    shomokh_src = get_image_base64("Frontend/assets/Shomokh.png")
    layan_src = get_image_base64("Frontend/assets/Layan.png")
    aqilah_src = get_image_base64("Frontend/assets/Aqilah.png")

    col1, col2, col3 = st.columns(3)

    # --- MEMBER 1: SHOMOKH ---
    with col1:
        st.markdown(
            f"""
            <div class="sales-card">
                <img src="{shomokh_src}" width="110" style="display: block; margin: 0 auto;">
                <div class="card-name">Shomokh<br>Althagafi</div>
                <div class="card-role">Software Engineer</div>
                <div class="card-caption">Focus: Database Architecture & AI Core Logic</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- MEMBER 2: LAYAN ---
    with col2:
        st.markdown(
            f"""
            <div class="sales-card">
                <img src="{layan_src}" width="110" style="display: block; margin: 0 auto;">
                <div class="card-name">Layan<br>AlMutairi</div>
                <div class="card-role">Software Engineer</div>
                <div class="card-caption">Focus: Backend (SQLAlchemy) & API Integrations</div>
            </div>
            """, 
            unsafe_allow_html=True
        )

    # --- MEMBER 3: AQILAH ---
    with col3:
        st.markdown(
            f"""
            <div class="sales-card">
                <img src="{aqilah_src}" width="110" style="display: block; margin: 0 auto;">
                <div class="card-name">Aqilah<br>Alkhalaf</div>
                <div class="card-role">Electrical & Software Engineer</div>
                <div class="card-caption">Focus: Database Architecture & Frontend (Streamlit/UI Interface)</div>
            </div>
            """, 
            unsafe_allow_html=True
        )
        

def about_page():
    st.subheader("About")
    st.write("**SAL Agent** is an AI-powered sales agent built for Beam Data, a Toronto-based AI and data consulting group trusted by Samsung, EY, Mars, and 30+ enterprise clients. It answers customer questions in real time using a knowledge base grounded in Beam Data's AI Hub product, services, and past project case studies, naturally captures lead information during conversation, and automatically sends a personalised follow-up email — turning every website visitor into a qualified, CRM-logged lead without manual sales effort.")
    st.subheader("Key features:")
    st.markdown("""
    -  Grounded answers from a RAG knowledge base (AI Hub, services & 50 past projects)
    -  Conversation memory — remembers context across the full chat 
    -  Automatic lead capture — name, email, company 
    -  AI-generated personalized follow-up emails 
    -  Full conversation and lead history logged to CRM database
    """)
    st.markdown("We chose the name **SAL Agent** because it carries a powerful dual meaning. In Arabic, **SAL** is a verb that captures the smooth fluidity of motion—which is exactly how we want our agent to streamline sales operations. In English, it stands perfectly for what it achieves: our **S**ales **A**utomation **L**ifecycle Agent. ")


# Create pages using Material icons for navigation
pages = [
    st.Page(home_page, title="Home", icon=":material/home:", default=True),
    st.Page(about_page, title="About", icon=":material/info:"),
    st.Page(run_chatbot_interface, title="AI Assistant", icon=":material/smart_toy:"),
]

# Set up top navigation
current_page = st.navigation(pages, position="top")

# --Navbar Color--
st.html("""
<style>
.stAppHeader {
    background-color: #9d7fe3;
}
</style>
""")

# Run the selected page
current_page.run()