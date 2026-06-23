"""
chatbot_agent.py
================
Beam Data Website Chatbot Agent
Built with LangChain + Streamlit

Features:
- Answers customer questions grounded in RAG (products + projects KB)
- Naturally collects name, email, company during conversation
- Saves every message to SQLite via SQLAlchemy (db.py)
- Triggers follow-up email after lead info is collected
- Conversation memory — remembers full chat history per session

Run:
    py -3.13 -m streamlit run chatbot_agent.py

Requirements:
    pip install streamlit langchain langchain-openai
                faiss-cpu sentence-transformers sqlalchemy
                python-dotenv numpy
"""

import os
import re
import uuid
import faiss
import pickle
import numpy as np
import streamlit as st
from dotenv import load_dotenv
from datetime import datetime, timezone
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from sentence_transformers import SentenceTransformer
from email_sender import send_gmail

# ── Local imports ─────────────────────────────────────────────────────────────
from db import (
    init_db, save_lead, save_message,
    save_email, mark_email_sent, get_conversation
)

# ── Load environment ──────────────────────────────────────────────────────────
load_dotenv()

OPENAI_API_KEY  = os.getenv("OPENAI_API_KEY")
SENDER_NAME     = os.getenv("SENDER_NAME",    "Alexander Krstich")
SENDER_TITLE    = os.getenv("SENDER_TITLE",   "Technical Project Manager")
SENDER_COMPANY  = os.getenv("SENDER_COMPANY", "Beam Data")

# ── Init DB ───────────────────────────────────────────────────────────────────
init_db()

# ── Load RAG once (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_rag():
    model = SentenceTransformer("all-mpnet-base-v2")

    products_index = faiss.read_index("knowledge_base/beamdata.index")
    with open("knowledge_base/documents.pkl", "rb") as f:
        products_docs = pickle.load(f)

    projects_index = faiss.read_index("knowledge_base/project.index")
    with open("knowledge_base/project_documents.pkl", "rb") as f:
        projects_docs = pickle.load(f)

    return model, products_index, products_docs, projects_index, projects_docs

@st.cache_resource
def load_llm():
    return ChatOpenAI(
        model       = "openrouter/free",
        temperature = 0.4,
        api_key     = os.getenv("OPENROUTER_API_KEY"),
        base_url    = "https://openrouter.ai/api/v1",
        default_headers = {
            "HTTP-Referer": "https://beamdata.ai",
            "X-Title":      "SAL Agent",
        }
    )

embedding_model, products_index, products_docs, projects_index, projects_docs = load_rag()
llm = load_llm()

# ── RAG retrieval ─────────────────────────────────────────────────────────────
def query_products(query: str, top_k: int = 1) -> list:
    emb = embedding_model.encode([query])
    emb = np.array(emb, dtype="float32")
    distances, indices = products_index.search(emb, top_k)
    return [{
        "id":    products_docs[idx]["kb_id"],
        "name":  products_docs[idx]["name"],
        "text":  products_docs[idx]["text"],
        "score": round(float(distances[0][i]), 4),
    } for i, idx in enumerate(indices[0])]

def query_projects(query: str, top_k: int = 2) -> list:
    emb = embedding_model.encode([query], normalize_embeddings=True)
    emb = np.array(emb, dtype="float32")
    distances, indices = projects_index.search(emb, top_k)
    return [{
        "id":    projects_docs[idx]["project_id"],
        "name":  projects_docs[idx]["project_name"],
        "text":  projects_docs[idx]["text"],
        "score": round(float(distances[0][i]), 4),
    } for i, idx in enumerate(indices[0])]

# ── Lead info extractor ───────────────────────────────────────────────────────
def extract_lead_info(text: str) -> dict:
    """
    Extracts name, email, company from customer message using regex.
    Returns dict with found fields — empty string if not found.
    """
    info = {"name": "", "email": "", "company": ""}

    # Email
    email_match = re.search(r"[\w.+-]+@[\w-]+\.[a-zA-Z]{2,}", text)
    if email_match:
        info["email"] = email_match.group()

    # Name — looks for "I'm X" / "my name is X" / "this is X"
    name_match = re.search(
        r"(?:i[''']?m|my name is|this is|call me)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        text, re.IGNORECASE
    )
    if name_match:
        info["name"] = name_match.group(1).strip()

    # Company — looks for "from X" / "at X" / "company is X"
    company_match = re.search(
        r"(?:from|at|work at|company is|with)\s+([A-Z][A-Za-z\s&]+?)(?:\.|,|$)",
        text, re.IGNORECASE
    )
    if company_match:
        info["company"] = company_match.group(1).strip()

    return info

# ── Chatbot prompt ────────────────────────────────────────────────────────────
SYSTEM_PROMPT = f"""You are the AI assistant for Beam Data — a leading AI and data
consulting group based in Toronto, Canada, trusted by Samsung, EY, Mars,
Globe and Mail, and 30+ enterprise clients.

Your role:
1. Answer customer questions accurately using the knowledge base context provided
2. Be warm, professional, and helpful
3. When relevant, naturally reference past projects as proof of expertise
4. After answering 2-3 questions, gently ask for the customer's name, company,
and email to connect them with the right person at Beam Data
5. Never make up information — only use what is in the context provided
6. Keep replies concise — 3 to 5 sentences maximum per message
7. If asked about pricing, say it depends on scope and offer a free discovery call

Beam Data services & product: AI Strategy, AI Implementation, AI Infrastructure,
Data & Cloud Infrastructure, Data Analytics & Data Science, AI Hub platform.

Always end your reply with a natural follow-up question to keep the conversation going."""

chatbot_prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM_PROMPT),
    ("system", "Relevant knowledge base context:\n{context}"),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}"),
])

chatbot_chain = chatbot_prompt | llm

# ── Generate reply ────────────────────────────────────────────────────────────
def generate_reply(
    question:     str,
    chat_history: list,
) -> tuple[str, str, str]:
    """
    Generates a grounded reply using RAG + LLM chain.
    Returns (reply_text, product_source, project_source)
    """
    # Query both indexes
    products = query_products(question, top_k=1)
    projects = query_projects(question, top_k=2)

    product  = products[0] if products else {"id": "", "name": "", "text": ""}
    proj1    = projects[0] if projects else {"id": "", "name": "", "text": ""}
    proj2    = projects[1] if len(projects) > 1 else {"id": "", "name": "", "text": ""}

    context = f"""
Most relevant service:
{product['name']}
{product['text'][:400]}

Relevant past projects:
Project 1: {proj1['name']}
{proj1['text'][:300]}

Project 2: {proj2['name']}
{proj2['text'][:200]}
"""

    response = chatbot_chain.invoke({
        "context":      context,
        "chat_history": chat_history,
        "question":     question,
    })

    reply      = response.content.strip()
    prod_src   = f"{product['id']} — {product['name']}" if product['id'] else ""
    proj_src   = f"{proj1['id']} — {proj1['name']}"     if proj1['id'] else ""

    return reply, prod_src, proj_src

# ── Follow-up email generator ─────────────────────────────────────────────────
def generate_followup_email(lead: dict, conversation: list) -> dict:
    """Generates a personalised follow-up email after lead info is collected."""
    convo_summary = "\n".join([
        f"{m['role'].title()}: {m['message']}"
        for m in conversation[-6:]  # last 6 messages
    ])

    prompt = f"""You are {SENDER_NAME}, {SENDER_TITLE} at {SENDER_COMPANY}.

Write a short, warm, professional follow-up email to {lead.get('name', 'there')}
at {lead.get('company', 'their company')} based on this conversation summary:

{convo_summary}

Rules:
- 100 to 150 words maximum
- Reference what they asked about specifically
- Offer a 20-minute discovery call as next step
- Sign off as {SENDER_NAME}, {SENDER_TITLE}, {SENDER_COMPANY}

Write subject line first: "Subject: ..."
Then the email body."""

    response = llm.invoke(prompt)
    raw      = response.content.strip()
    lines    = raw.split('\n')

    subject = ""
    body    = []
    for line in lines:
        if line.startswith("Subject:"):
            subject = line.replace("Subject:", "").strip()
        elif subject:
            body.append(line)

    return {
        "subject": subject,
        "body":    '\n'.join(body).strip(),
    }


# ── MAIN INTERFACE FUNCTION ───────────────────────────────────────────────────
def run_chatbot_interface():
    # ── Session state init ────────────────────────────────────────────────────────
    if "session_id"    not in st.session_state:
        st.session_state.session_id    = str(uuid.uuid4())
    if "chat_history"  not in st.session_state:
        st.session_state.chat_history  = []
    if "messages"      not in st.session_state:
        st.session_state.messages      = []
    if "lead_id"       not in st.session_state:
        st.session_state.lead_id       = None
    if "lead_info"     not in st.session_state:
        st.session_state.lead_info     = {"name": "", "email": "", "company": ""}
    if "email_sent"    not in st.session_state:
        st.session_state.email_sent    = False
    if "msg_count"     not in st.session_state:
        st.session_state.msg_count     = 0
    
    # ── Page config ───────────────────────────────────────────────────────────────
    # st.set_page_config(
    #     page_title  = "SAL Agent",
    #     page_icon   = "💡",
    #     layout      = "centered",
    # )
    
    st.title("Conversational AI Assistant")
    st.markdown("#### *Interact with the SAL Agent using natural language.*")
    st.write("---")

    # ── Streamlit custom styles ───────────────────────────────────────────────────
    st.markdown("""
    <style>
    .chat-header {
        background: #1D3557;
        color: white;
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .status-dot {
        width: 8px; height: 8px;
        background: #1D9E75;
        border-radius: 50%;
        display: inline-block;
        margin-right: 4px;
    }
    .kb-source {
        font-size: 11px;
        color: #0C447C;
        background: #E6F1FB;
        padding: 2px 8px;
        border-radius: 20px;
        display: inline-block;
        margin-top: 4px;
    }
    .lead-collected {
        background: #E1F5EE;
        border: 0.5px solid #5DCAA5;
        border-radius: 8px;
        padding: 10px 14px;
        font-size: 13px;
        color: #085041;
        margin-top: 8px;
    }
    </style>
    """, unsafe_allow_html=True)

    # ── UI ────────────────────────────────────────────────────────────────────────
    st.markdown("""
    <div class="chat-header">
    <div style="font-size:28px">💡</div>
    <div>
        <div style="font-weight:500;font-size:16px">SAL Agent</div>
        <div style="font-size:12px;opacity:.8">
        <span class="status-dot"></span>Online — Ask me anything about AI & data
        </div>
    </div>
    </div>
    """, unsafe_allow_html=True)

# ── Display chat history ──────────────────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"], avatar="💡" if msg["role"] == "assistant" else "👤"):
            st.markdown(msg["content"])
            # if msg.get("kb_source"):
            #     st.markdown(
            #         f'<span class="kb-source">📚 {msg["kb_source"]}</span>',
            #         unsafe_allow_html=True
            #     )

    # ── Greeting on first load ────────────────────────────────────────────────────
    if not st.session_state.messages:
        greeting = (
            "Hi there! I'm Beam Data's AI assistant. I can help you learn about "
            "our AI and data services, share relevant case studies, or connect you "
            "with our team.\n\nWhat brings you here today?"
        )
        st.session_state.messages.append({
            "role": "assistant", "content": greeting, "kb_source": ""
        })
        with st.chat_message("assistant", avatar="💡"):
            st.markdown(greeting)

    # ── Chat input ────────────────────────────────────────────────────────────────
    if user_input := st.chat_input("Ask a question about Beam Data..."):

        # Display customer message
        with st.chat_message("user", avatar="👤"):
            st.markdown(user_input)

        st.session_state.messages.append({
            "role": "user", "content": user_input, "kb_source": ""
        })
        st.session_state.msg_count += 1

        # Extract lead info from message
        extracted = extract_lead_info(user_input)
        for field in ["name", "email", "company"]:
            if extracted[field]:
                st.session_state.lead_info[field] = extracted[field]

        # Generate reply
        with st.spinner("Thinking..."):
            reply, prod_src, proj_src = generate_reply(
                question     = user_input,
                chat_history = st.session_state.chat_history,
            )

        # Update LangChain chat history
        st.session_state.chat_history.append(HumanMessage(content=user_input))
        st.session_state.chat_history.append(AIMessage(content=reply))

        # Display agent reply
        with st.chat_message("assistant", avatar="💡"):
            st.markdown(reply)
            # if prod_src:
            #     st.markdown(
            #         f'<span class="kb-source">📚 {prod_src}</span>',
            #         unsafe_allow_html=True
            #     )

        st.session_state.messages.append({
            "role":      "assistant",
            "content":   reply,
            "kb_source": prod_src,
        })

        # ── Save to DB ────────────────────────────────────────────────────────────
        lead_info = st.session_state.lead_info

        # Save or update lead when we have at least an email
        if lead_info.get("email") and not st.session_state.lead_id:
            lead_id = save_lead(
                name            = lead_info.get("name", ""),
                email           = lead_info.get("email", ""),
                company         = lead_info.get("company", ""),
                product_matched = prod_src,
                project_matched = proj_src,
                priority        = "High" if st.session_state.msg_count <= 3 else "Medium",
            )
            st.session_state.lead_id = lead_id

        # Save messages to DB
        if st.session_state.lead_id:
            save_message(
                lead_id    = st.session_state.lead_id,
                session_id = st.session_state.session_id,
                role       = "customer",
                message    = user_input,
            )
            save_message(
                lead_id    = st.session_state.lead_id,
                session_id = st.session_state.session_id,
                role       = "agent",
                message    = reply,
                kb_source  = prod_src,
            )

        # ── Send follow-up email after lead info collected ────────────────────────
        if (
            lead_info.get("email")
            and lead_info.get("name")
            and not st.session_state.email_sent
            and st.session_state.lead_id
        ):
            conversation = get_conversation(st.session_state.session_id)
            email        = generate_followup_email(lead_info, conversation)

            # Log email to DB
            save_email(
                lead_id   = st.session_state.lead_id,
                recipient = lead_info["email"],
                subject   = email["subject"],
                body      = email["body"],
                status    = "sent",
            )
            mark_email_sent(st.session_state.lead_id)
            st.session_state.email_sent = True
            send_gmail(lead_info["email"], email["subject"], email["body"])

            # Show confirmation in chat
            confirmation = f"""
    <div class="lead-collected">
    ✅ <strong>Lead captured</strong><br>
    👤 {lead_info.get('name', '')} — {lead_info.get('company', '')}<br>
    📧 Follow-up email sent to {lead_info['email']}<br>
    📌 Logged to CRM — our team will be in touch within 1 business day
    </div>
    """
            st.markdown(confirmation, unsafe_allow_html=True)

    # ── Sidebar — session info ────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Session info")
        st.caption(f"Session: `{st.session_state.session_id[:8]}...`")
        st.caption(f"Messages: {st.session_state.msg_count}")

        lead_info = st.session_state.lead_info
        if any(lead_info.values()):
            st.markdown("### Lead collected")
            if lead_info.get("name"):
                st.caption(f"👤 {lead_info['name']}")
            if lead_info.get("company"):
                st.caption(f"🏢 {lead_info['company']}")
            if lead_info.get("email"):
                st.caption(f"📧 {lead_info['email']}")
            if st.session_state.email_sent:
                st.success("Follow-up email sent")

        st.markdown("---")
        if st.button("Start new conversation"):
            for key in ["session_id","chat_history","messages","lead_id",
                        "lead_info","email_sent","msg_count"]:
                del st.session_state[key]
            st.rerun()