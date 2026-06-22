# evaluate_sal_agent.py
# Run using: streamlit run Evaluation/evaluate_sal_agent.py

# Evaluation/evaluate_sal_agent.py
import os
import sys

# ── STREAMLIT MOCK ────────────────────────────
from unittest.mock import MagicMock

class MockSessionState(dict):
    def __getattr__(self, item): return self.get(item, None)
    def __setattr__(self, key, value): self[key] = value

def mock_cache_resource(func=None, *args, **kwargs):
    if func is callable(func) or func is not None:
        return func
    def decorator(f):
        return f
    return decorator

mock_st = MagicMock()
mock_st.session_state = MockSessionState(chat_history=[])
mock_st.cache_resource = mock_cache_resource  
sys.modules['streamlit'] = mock_st
# ──────────────────────────────────────────────────────────────────────────────

import uuid
import streamlit as st 

# Dynamically append parent directory to python sys.path for clean imports
PARENT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)

os.chdir(PARENT_DIR)

from Evaluation.eval_dataset import sal_evaluation_set
from chatbot_agent import generate_reply, extract_lead_info

print("🚀 Starting SAL Agent Evaluation Protocol...\n")

evaluation_results = []

for case in sal_evaluation_set:
    print(f"{'='*70}")
    print(f"Test Case {case['id']}: [{case['category']}]")
    print(f"Query: {case['query']}")
    print(f"{'='*70}")
    
    # 1. Test the Regex Lead Information Extraction Feature
    extracted_info = extract_lead_info(case['query'])
    print(f"🔍 [Extracted Entities]: {extracted_info}")
    
    # 2. Test the core RAG Chain & LLM Generation
    reply, prod_src, proj_src = generate_reply(
        question=case['query'],
        chat_history=[]
    )
    
    print(f"🤖 [Agent Response]:\n{reply}")
    print(f"📚 [Knowledge Sources Used]: Product -> {prod_src} | Project -> {proj_src}")
    print(f"📋 [Expected Behavior Guide]: {case.get('expected', 'No criteria defined.')}")
    print(f"{'-'*70}")
    
    verdict = input("Did this response meet the expected behavior? (Pass/Fail): ").strip().lower()
    reason = input("Short reason / annotation: ").strip()
    
    # Store comparative results along with interactive evaluation metrics
    evaluation_results.append({
        "id": case["id"],
        "category": case["category"],
        "query": case["query"],
        "expected": case["expected"],
        "actual_reply": reply,
        "extracted_data": extracted_info,
        "verdict": "Pass" if verdict == "pass" else "Fail",
        "reason": reason
    })
    print("\n")

# Summary calculations
total_cases = len(evaluation_results)
passed = sum(1 for r in evaluation_results if r["verdict"] == "Pass")
failed = total_cases - passed
success_rate = (passed / total_cases) * 100 if total_cases > 0 else 0

print("\n" + "="*60)
print("🏆 SAL AGENT EVALUATION PERFORMANCE SUMMARY")
print("="*60)
print(f"Total Evaluated Dataset Cases : {total_cases}")
print(f"Passed Assertions             : {passed}  🟢")
print(f"Failed Assertions             : {failed}  🔴")
print(f"System Success Core Rate      : {success_rate:.2f}%")
print("="*60)
print("✅ Evaluation Run Complete. Review outputs above against success criteria parameters.")