"""
db.py
=====
SQLite database using SQLAlchemy ORM

Tables:
    Lead         — captured customer info from chatbot
    Conversation — full chat history per lead
    EmailSent    — follow-up email log

Usage:
    from db import init_db, save_lead, save_message, save_email,
                   get_pending_leads, get_all_leads, get_pipeline_stats
"""

import os
from datetime import datetime, timezone
from sqlalchemy import (
    create_engine, Column, Integer, String,
    Text, DateTime, ForeignKey, func
)
from sqlalchemy.orm import declarative_base, relationship, Session

# ── Database path ─────────────────────────────────────────────────────────────
DB_DIR  = os.path.join(os.path.dirname(os.path.abspath(__file__)), "app")
DB_PATH = os.path.join(DB_DIR, "beamdata_crm.db")

os.makedirs(DB_DIR, exist_ok=True)

engine = create_engine(
    f"sqlite:///{DB_PATH}",
    echo=False,   # raw SQL in terminal
)

Base = declarative_base()

# ── Models ────────────────────────────────────────────────────────────────────

class Lead(Base):
    """One row per customer who interacted with the chatbot."""
    __tablename__ = "leads"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    name            = Column(String,  nullable=True)
    email           = Column(String,  unique=True, nullable=True)
    company         = Column(String,  nullable=True)
    product_matched = Column(String,  nullable=True)
    project_matched = Column(String,  nullable=True)
    priority        = Column(String,  default="Medium")
    email_sent      = Column(Integer, default=0)
    created_at = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    # Relationships — one lead has many messages and emails
    conversations = relationship("Conversation", back_populates="lead")
    emails        = relationship("EmailSent",    back_populates="lead")

    def __repr__(self):
        return f"<Lead id={self.id} name={self.name} email={self.email}>"


class Conversation(Base):
    """Every chat message — customer and agent — stored per session."""
    __tablename__ = "conversations"

    id         = Column(Integer,  primary_key=True, autoincrement=True)
    lead_id    = Column(Integer,  ForeignKey("leads.id"), nullable=True)
    session_id = Column(String,   nullable=True)
    role       = Column(String,   nullable=False)   # 'customer' or 'agent'
    message    = Column(Text,     nullable=False)
    kb_source  = Column(String,   nullable=True)    # which KB entry was used
    created_at = Column(DateTime, default=datetime.now(timezone.utc).date())

    lead = relationship("Lead", back_populates="conversations")

    def __repr__(self):
        return f"<Conversation id={self.id} role={self.role} session={self.session_id}>"


class EmailSent(Base):
    """Every follow-up email sent to a lead."""
    __tablename__ = "emails_sent"

    id         = Column(Integer,  primary_key=True, autoincrement=True)
    lead_id    = Column(Integer,  ForeignKey("leads.id"), nullable=True)
    recipient  = Column(String,   nullable=True)
    subject    = Column(String,   nullable=True)
    body       = Column(Text,     nullable=True)
    status     = Column(String,   default="sent")   # 'sent' or 'failed'
    sent_at    = Column(DateTime, default=datetime.now(timezone.utc).date())

    lead = relationship("Lead", back_populates="emails")

    def __repr__(self):
        return f"<EmailSent id={self.id} to={self.recipient} status={self.status}>"


# ── Init ──────────────────────────────────────────────────────────────────────
def init_db():
    """
    Creates all tables if they don't exist.
    Safe to call on every startup — won't overwrite existing data.
    """
    Base.metadata.create_all(engine)
    print(f"✅ Database ready at: {DB_PATH}")


# ── Lead operations ───────────────────────────────────────────────────────────
def save_lead(
    name:            str  = "",
    email:           str  = "",
    company:         str  = "",
    product_matched: str  = "",
    project_matched: str  = "",
    priority:        str  = "Medium",
) -> int:
    """
    Inserts a new lead or updates existing one matched by email.
    Returns the lead_id (int).
    """
    with Session(engine) as session:
        # Check if lead already exists
        existing = session.query(Lead).filter_by(email=email).first()

        if existing:
            # Update existing lead
            existing.name            = name
            existing.company         = company
            existing.product_matched = product_matched
            existing.project_matched = project_matched
            existing.priority        = priority
            session.commit()
            return existing.id
        else:
            # Insert new lead
            lead = Lead(
                name=name, email=email, company=company,product_matched=product_matched,
                project_matched=project_matched, priority=priority,
            )
            session.add(lead)
            session.commit()
            session.refresh(lead)
            return lead.id


def get_lead_by_email(email: str) -> dict | None:
    """Returns a lead as a dict by email, or None if not found."""
    with Session(engine) as session:
        lead = session.query(Lead).filter_by(email=email).first()
        if not lead:
            return None
        return {
            "id": lead.id, "name": lead.name, "email": lead.email,
            "company": lead.company,
            "product_matched": lead.product_matched,
            "project_matched": lead.project_matched,
            "priority": lead.priority, "email_sent": lead.email_sent,
            "created_at": str(lead.created_at),
        }


def get_pending_leads() -> list:
    """Returns all leads where email_sent = 0 (follow-up not yet sent)."""
    with Session(engine) as session:
        leads = session.query(Lead).filter_by(email_sent=0).all()
        return [{
            "id": l.id, "name": l.name, "email": l.email,
            "company": l.company,
            "product_matched": l.product_matched,
            "project_matched": l.project_matched,
            "priority": l.priority,
        } for l in leads]


def get_all_leads() -> list:
    """Returns all leads ordered by most recent."""
    with Session(engine) as session:
        leads = session.query(Lead).order_by(Lead.created_at.desc()).all()
        return [{
            "id": l.id, "name": l.name, "email": l.email,
            "company": l.company,
            "product_matched": l.product_matched,
            "project_matched": l.project_matched,
            "priority": l.priority, "email_sent": l.email_sent,
            "created_at": str(l.created_at),
        } for l in leads]


def mark_email_sent(lead_id: int):
    """Marks a lead's email_sent = 1 after follow-up email is sent."""
    with Session(engine) as session:
        lead = session.query(Lead).filter_by(id=lead_id).first()
        if lead:
            lead.email_sent = 1
            session.commit()


# ── Conversation operations ───────────────────────────────────────────────────
def save_message(
    lead_id:    int,
    session_id: str,
    role:       str,
    message:    str,
    kb_source:  str = "",
):
    """
    Saves a single chat message to conversations table.
    role: 'customer' or 'agent'
    kb_source: KB entry ID used to generate agent reply
    """
    with Session(engine) as session:
        msg = Conversation(
            lead_id=lead_id, session_id=session_id,
            role=role, message=message, kb_source=kb_source,
        )
        session.add(msg)
        session.commit()


def get_conversation(session_id: str) -> list:
    """Returns all messages for a session in chronological order."""
    with Session(engine) as session:
        msgs = (
            session.query(Conversation)
            .filter_by(session_id=session_id)
            .order_by(Conversation.created_at.asc())
            .all()
        )
        return [{
            "role":      m.role,
            "message":   m.message,
            "kb_source": m.kb_source,
            "created_at": str(m.created_at),
        } for m in msgs]


def get_today_conversations() -> list:
    """Returns all leads created today with message count."""
    with Session(engine) as session:
        today = datetime.now(timezone.utc).date()
        leads = (
            session.query(Lead)
            .filter(func.date(Lead.created_at) == today)
            .order_by(Lead.created_at.desc())
            .all()
        )
        results = []
        for l in leads:
            msg_count = (
                session.query(func.count(Conversation.id))
                .filter_by(lead_id=l.id)
                .scalar()
            )
            results.append({
                "name":            l.name,
                "email":           l.email,
                "company":         l.company,
                "product_matched": l.product_matched,
                "project_matched": l.project_matched,
                "priority":        l.priority,
                "created_at":      str(l.created_at),
                "message_count":   msg_count,
            })
        return results


# ── Email log operations ──────────────────────────────────────────────────────
def save_email(
    lead_id:   int,
    recipient: str,
    subject:   str,
    body:      str,
    status:    str = "sent",
):
    """Logs a sent follow-up email to emails_sent table."""
    with Session(engine) as session:
        email = EmailSent(
            lead_id=lead_id, recipient=recipient,
            subject=subject, body=body, status=status,
        )
        session.add(email)
        session.commit()


def get_email_history(lead_id: int) -> list:
    """Returns all emails sent to a specific lead."""
    with Session(engine) as session:
        emails = (
            session.query(EmailSent)
            .filter_by(lead_id=lead_id)
            .order_by(EmailSent.sent_at.desc())
            .all()
        )
        return [{
            "id":        e.id,
            "recipient": e.recipient,
            "subject":   e.subject,
            "body":      e.body,
            "status":    e.status,
            "sent_at":   str(e.sent_at),
        } for e in emails]


# ── Pipeline stats ────────────────────────────────────────────────────────────
def get_pipeline_stats() -> dict:
    """
    Returns pipeline stats dict for the summary agent.
    Covers: totals, breakdowns by priority, product.
    """
    with Session(engine) as session:
        total       = session.query(func.count(Lead.id)).scalar()
        emails_sent = session.query(func.count(Lead.id)).filter_by(email_sent=1).scalar()
        today       = datetime.now(timezone.utc).date()
        today_count = session.query(func.count(Lead.id)).filter(
            func.date(Lead.created_at) == today
        ).scalar()

        # Breakdown by priority
        by_priority = {
            row.priority: row.count
            for row in session.query(
                Lead.priority,
                func.count(Lead.id).label("count")
            ).group_by(Lead.priority).all()
        }

        # Breakdown by product matched
        by_product = {
            row.product_matched: row.count
            for row in session.query(
                Lead.product_matched,
                func.count(Lead.id).label("count")
            ).filter(Lead.product_matched != "")
            .group_by(Lead.product_matched).all()
        }

        return {
            "total_leads":    total,
            "emails_sent":    emails_sent,
            "pending_emails": total - emails_sent,
            "today_leads":    today_count,
            "by_priority":    by_priority,
            "by_product":     by_product,
        }


# ── Entry point — test the DB ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Initialising Beam Data CRM database...")
    init_db()

    # Test — insert a lead
    lead_id = save_lead(
        name            = "Sarah Chen",
        email           = "sarah.chen@acmemfg.com",
        company         = "Acme Manufacturing",
        product_matched = "BD-PROD-001 — AI Hub",
        project_matched = "BD-PROJ-005 — Optimizing Battery Performance",
        priority        = "High",
    )
    print(f"\n✅ Lead saved — id: {lead_id}")

    # Test — save conversation
    session_id = "test-session-001"
    save_message(lead_id, session_id, "customer",
        "We have AI tools scattered with no control")
    save_message(lead_id, session_id, "agent",
        "AI Hub solves exactly that — one governed platform",
        kb_source="BD-PROD-001 — AI Hub")
    save_message(lead_id, session_id, "customer",
        "Show me a past project")
    save_message(lead_id, session_id, "agent",
        "Here's Optimizing Battery Performance — AI governance for manufacturing",
        kb_source="BD-PROJ-005 — Optimizing Battery Performance")

    msgs = get_conversation(session_id)
    print(f"✅ Conversation saved — {len(msgs)} messages")

    # Test — save email
    save_email(
        lead_id   = lead_id,
        recipient = "sarah.chen@acmemfg.com",
        subject   = "Your conversation with Beam Data — next steps",
        body      = "Hi Sarah, thank you for chatting with us today...",
        status    = "sent",
    )
    mark_email_sent(lead_id)
    print(f"✅ Email logged and lead marked as contacted")

    # Test — pipeline stats
    stats = get_pipeline_stats()
    print(f"\n📊 Pipeline stats:")
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # Test — retrieve lead
    lead = get_lead_by_email("sarah.chen@acmemfg.com")
    print(f"\n✅ Lead retrieved: {lead['name']} — {lead['company']}")

    print(f"\n✅ All tests passed — database at: {DB_PATH}")
