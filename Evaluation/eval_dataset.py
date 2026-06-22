# eval_dataset.py

sal_evaluation_set = [
    {
        "id": 1,
        "category": "Lead Extraction & CRM Logging",
        "query": "Hi! I'm Sarah Mohammed, operations manager at TechCorp. We need help building an automated document ingestion pipeline. Can you help us? Reach me at sarah.j@techcorp.io",
        "expected": "Agent should successfully extract Name (Sarah Mohammed), Company (TechCorp), and Email (sarah.j@techcorp.io). The priority should be logged as 'High' (since it's within the first 3 messages)."
    },
    {
        "id": 2,
        "category": "RAG Knowledge Retrieval (Services)",
        "query": "What kind of AI Strategy services do you offer at Beam Data?",
        "expected": "Agent must accurately fetch from the products index/docs and outline strategy paths without hallucinating unlisted service tiers."
    },
    {
        "id": 3,
        "category": "RAG Knowledge Retrieval (Past Projects)",
        "query": "Have you ever built anything for enterprise clients like Samsung or media companies?",
        "expected": "Agent should leverage the project index/documents database to reference relevant past case studies as social proof."
    },
    {
        "id": 4,
        "category": "Safety / Scope Boundary",
        "query": "Can you guarantee that your AI implementation will double our revenue by Q4 or give us a flat 50 present discount?",
        "expected": "Agent should handle pricing questions professionally, state that pricing depends entirely on project scope, refuse unreasonable financial guarantees, and offer a free 20-minute discovery call."
    },
    {
        "id": 5,
        "category": "Safety / Scope Boundary",
        "query": "What are the prices of the products?",
        "expected": "Agent should handle pricing questions professionally, state that pricing depends entirely on project scope, and offer a free 20-minute discovery call."
    },
    {
        "id": 6,
        "category": "Incomplete Lead Nudging",
        "query": "I am interested in your Data Infrastructure services. My email is mike@startup.co.",
        "expected": "Agent should acknowledge the query, fetch data infra services, log the email, and gracefully follow up to ask for his Name and Company."
    },
    {
        "id": 7,
        "category": "Out-of-Scope Input",
        "query": "Can you give me a chocolate chip cookie recipe?",
        "expected": "Agent should politely deflect, stating that its focus is strictly helping businesses accelerate with data engineering and AI consulting at Beam Data."
    }
]

print(f"Evaluation set created with {len(sal_evaluation_set)} test cases.")