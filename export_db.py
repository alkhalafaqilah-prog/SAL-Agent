# export_db.py
import sqlite3
import pandas as pd

conn = sqlite3.connect('app/beamdata_crm.db')

# Read all 3 tables into DataFrames
leads         = pd.read_sql("SELECT * FROM leads",         conn)
conversations = pd.read_sql("SELECT * FROM conversations", conn)
emails_sent   = pd.read_sql("SELECT * FROM emails_sent",   conn)

conn.close()

# Export to one Excel file — one sheet per table
with pd.ExcelWriter('app/beamdata_crm_export.xlsx', engine='openpyxl') as writer:
    leads.to_excel(writer,         sheet_name='Leads',         index=False)
    conversations.to_excel(writer, sheet_name='Conversations', index=False)
    emails_sent.to_excel(writer,   sheet_name='Emails Sent',   index=False)

print(f"✅ Exported:")
print(f"   Leads:         {len(leads)} rows")
print(f"   Conversations: {len(conversations)} rows")
print(f"   Emails sent:   {len(emails_sent)} rows")
print(f"   File: app/beamdata_crm_export.xlsx")