# SAL Agent (Sales Automation Lifecycle Agent)

AI-powered B2B Sales &amp; CRM Agent designed to support outbound sales workflows through CRM logging, lead tracking, follow-up reminder generation, and pipeline summarization. Leverages LLMs to extract structured CRM data, and enhance sales engagement efficiency.

---

## Project Summary
The **SAL Agent** serves as an intelligent, real-time interface that handles incoming customer prospects. Utilizing an advanced Retrieval-Augmented Generation (RAG) pipeline powered by dense vector search indexes (FAISS), it delivers hyper-accurate answers directly grounded in Beam Data’s proprietary technical offerings, product documentation, and past project portfolios. 

Beyond conversational support, the system utilizes natural language parsing to dynamically extract critical customer details—such as names, company profiles, and email addresses—directly from ongoing chats, logging them into a local CRM database and triggering automated, structured communication workflows.

* **The Arabic Meaning:** The word **"SAL"** functions as an evocative verb capturing the smooth, uninterrupted fluidity of motion, symbolizing the frictionless manner in which the agent accelerates corporate workflows.
* **The English Meaning:** It stands for **Sales Automation Lifecycle Agent**, highlighting its capability to orchestrate the entire client journey from initial query to automated lead capture.

---

## Requirements & Dependencies
To run this project locally, ensure you have **Python 3.10+** (Recommended: Python 3.11 or 3.13) installed. Below is the complete breakdown of all foundational libraries and ecosystem dependencies required by the application:

### 1. Framework & User Interface
* `streamlit` — Powering the reactive web dashboard, chat container interface, and evaluation layout grids.

### 2. AI & LLM Orchestration (LangChain Ecosystem)
* `langchain` — Core architecture framework used to build the autonomous agent logic, memory management, and execution chains.
* `langchain-community` — Community-driven integrations used to connect local data systems and utilities.
* `langchain-openai` — Specialized connection wrapper linking the agent pipelines directly to official OpenAI backend endpoints.

### 3. Core Machine Learning & Vector Data Store
* `openai` — Official low-level Python client library utilized for generating text completions and dense embedding matrices.
* `faiss-cpu` — Facebook AI Similarity Search library used to store high-dimensional text vectors locally and calculate lightning-fast cosine similarity rankings for the RAG dataset.
* `tiktoken` — Fast BPE tokeniser optimized for OpenAI models, used to accurately count context window capacities and manage pricing costs.

### 4. Database, Environment, & Communication Utilities
* `sqlite3` — Native Python relational engine used to manage database schemas, track ongoing chat histories, and save captured B2B client leads.
* `python-dotenv` — Security dependency used to safely parse configuration keys out of your hidden local `.env` profile.
* `regex (re)` — Built-in processing engine used to cleanly match pattern structures for immediate client name and email parsing extraction.

---

## Installation
Follow these steps to set up the environment and install all necessary packages locally:

1. **Clone the Repository:**
   ```bash
   git clone "https://github.com/alkhalafaqilah-prog/SAL-Agent"
   cd SAL_Agent

---

2. **Install related libraries:**

   ```bash 
   pip install streamlit langchain langchain-openai faiss-cpu sentence-transformers sqlalchemy python-dotenv numpy

---
3. **Run Your Code:**
   ```bash
   streamlit run Frontend/app.py

---
4. **Create your own .env file:**

include the following keys: 

   ```bash
   OPENAI_API_KEY=
   OPENROUTER_API_KEY=
   SENDER_NAME=
   SENDER_TITLE=
   SENDER_COMPANT=
   GMAIL_USER=
   GMAIL_APP_PASSWORD=