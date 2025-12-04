import streamlit as st
import google.generativeai as genai
import json
import re
import pandas as pd
import time
import random
from datetime import datetime
import base64

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Adya Enterprise Evaluation Cloud", page_icon="🚀", layout="wide")

# !!! PASTE API KEY HERE !!!
GOOGLE_API_KEY = "AIzaSyDw0X1moWqKVO5QzlyKbEFZws97k8zNFzE"

if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    st.error("⚠️ Please paste your Google API Key in the code.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

@st.cache_resource
def get_model():
    try:
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        for m in models:
            if "flash" in m: return genai.GenerativeModel(m)
        return genai.GenerativeModel("gemini-1.5-pro")
    except:
        return None

model = get_model()

# --- 2. ADVANCED METRICS CALCULATOR (Precision/Recall/F1) ---
def calculate_ml_metrics(df):
    """Calculates Precision, Recall, and F1 based on Tool Usage."""
    # Logic: 
    # True Positive (TP): Tool score is 100 (Correct tool used)
    # False Positive (FP): Tool score is 0 (Wrong tool used)
    # False Negative (FN): Safety Fail (Missed a required check)
    
    tp = len(df[df['Orchestration'] == 100])
    fp = len(df[df['Orchestration'] == 0])
    fn = len(df[df['Safety Check'].str.contains("Fail")]) # Using safety fail as proxy for missed steps
    
    # Avoid div by zero
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return precision * 100, recall * 100, f1 * 100

# --- 3. HTML REPORT GENERATOR ---
def generate_html_report(df, agent_type, quality_gate, precision, recall, f1):
    avg_roi = df['ROI Score'].mean()
    passed = avg_roi >= quality_gate
    status_color = "#2ecc71" if passed else "#e74c3c"
    status_text = "PASSED QUALITY GATE" if passed else "FAILED QUALITY GATE"
    
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background-color: #f4f6f8; }}
            .header {{ background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            .badge {{ background-color: {status_color}; color: white; padding: 10px 20px; border-radius: 20px; font-weight: bold; }}
            .metric-box {{ display: inline-block; width: 150px; padding: 15px; background: #fff; margin: 5px; border-radius: 8px; text-align: center; border: 1px solid #ddd; }}
            .metric-val {{ font-size: 20px; font-weight: bold; color: #2c3e50; }}
            .metric-label {{ font-size: 12px; color: #7f8c8d; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 30px; background: white; }}
            th {{ background-color: #34495e; color: white; padding: 12px; text-align: left; }}
            td {{ border-bottom: 1px solid #ddd; padding: 12px; }}
            tr:nth-child(even) {{ background-color: #f9f9f9; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>🚀 Adya AI - Advanced Quality Report</h1>
            <p><strong>Agent:</strong> {agent_type} | <strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M")}</p>
            <span class="badge">{status_text}</span>
            <div style="margin-top: 20px;">
                <div class="metric-box"><div class="metric-val">{avg_roi:.1f}%</div><div class="metric-label">Business ROI</div></div>
                <div class="metric-box"><div class="metric-val">{precision:.1f}%</div><div class="metric-label">Precision</div></div>
                <div class="metric-box"><div class="metric-val">{recall:.1f}%</div><div class="metric-label">Recall</div></div>
                <div class="metric-box"><div class="metric-val">{f1:.1f}%</div><div class="metric-label">F1 Score</div></div>
            </div>
        </div>
        <h3>Detailed Audit Logs</h3>
        <table>
            <tr><th>Scenario</th><th>Response</th><th>ROI</th><th>Semantic</th></tr>
    """
    for _, row in df.iterrows():
        html += f"""<tr><td><b>{row['Persona']}</b><br>{row['Intent']}</td><td>{row['Response'][:100]}...</td><td>{row['ROI Score']}%</td><td>{row['Semantic Similarity']}%</td></tr>"""
    html += "</table></body></html>"
    return html

# --- 4. SCENARIO GENERATOR ---
def generate_scenarios(agent_type, num_cases=10):
    scenarios = []
    if agent_type == "Sales Agent":
        personas = [{"type": "CFO", "focus": "ROI", "tone": "Direct"}, {"type": "Founder", "focus": "Speed", "tone": "Casual"}]
        intents = ["Pricing Inquiry", "Book Demo"]
        ideal = "We offer flexible pricing. I can schedule a demo to show value."
    elif agent_type == "Banking Agent":
        personas = [{"type": "Fraud Victim", "focus": "Freeze", "tone": "Panicked"}, {"type": "VIP", "focus": "Invest", "tone": "Demanding"}]
        intents = ["Report Fraud", "Check Balance"]
        ideal = "I have secured your account. Please verify identity via OTP."

    for i in range(num_cases):
        p = random.choice(personas)
        intent = random.choice(intents)
        scenarios.append({
            "id": i+1, "persona": p['type'], "intent": intent, 
            "input": f"[{p['type']}] Regarding {intent}. Focus: {p['focus']}. {p['tone']}.",
            "ideal": ideal
        })
    return scenarios

# --- 5. AGENT LOGIC ---
def run_agent(agent_type, user_query, custom_prompt):
    if custom_prompt: sys_prompt = custom_prompt
    else:
        sys_prompt = "You are a Sales AI. Tools: [TOOL: book_demo], [TOOL: send_pricing]." if agent_type == "Sales Agent" else "You are a Banking AI. Tools: [TOOL: freeze_account], [TOOL: send_otp]."
    try:
        return model.generate_content(f"{sys_prompt}\nUser: {user_query}\nAction:").text.strip()
    except: return "Error"

# --- 6. EVALUATOR (With Semantic Similarity) ---
def evaluate_interaction(user_input, response, ideal_answer):
    eval_prompt = f"""
    Evaluate interaction. User: "{user_input}" Agent: "{response}" Ideal: "{ideal_answer}"
    Metrics:
    1. Hallucination (0/1)
    2. Tool Accuracy (0-100)
    3. Business Success (0-100)
    4. Semantic Similarity (0-100): Match to ideal?
    Return JSON: {{ "hallucination": 0, "tool_score": 0, "business_score": 0, "semantic_score": 0, "reason": "summary" }}
    """
    try:
        res = model.generate_content(eval_prompt)
        text = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except: return {"hallucination":0, "tool_score":0, "business_score":0, "semantic_score":0, "reason":"Fail"}

# --- 7. UI ---
st.title("🚀 Adya Enterprise Evaluation Cloud")

with st.sidebar:
    st.header("Pipeline Config")
    agent_type = st.selectbox("Agent:", ["Sales Agent", "Banking Agent"])
    num_tests = st.slider("Batch Size:", 5, 50, 5)
    quality_gate = st.slider("Quality Gate (%):", 50, 100, 80)
    with st.expander("🛠️ A/B Testing"):
        custom_system_prompt = st.text_area("Agent Version B (Prompt):")

if st.button("Run Pipeline Evaluation", type="primary"):
    scenarios = generate_scenarios(agent_type, num_tests)
    results = []
    bar = st.progress(0, "Running...")
    
    for i, case in enumerate(scenarios):
        resp = run_agent(agent_type, case['input'], custom_system_prompt)
        metrics = evaluate_interaction(case['input'], resp, case['ideal'])
        results.append({
            "Persona": case['persona'], "Intent": case['intent'], "Input": case['input'], "Response": resp,
            "Safety Check": "✅ Pass" if metrics['hallucination'] == 0 else "❌ Fail",
            "Orchestration": metrics['tool_score'], "ROI Score": metrics['business_score'],
            "Semantic Similarity": metrics['semantic_score'], "Reason": metrics['reason']
        })
        time.sleep(0.1)
        bar.progress((i+1)/num_tests)
    bar.empty()
    
    df = pd.DataFrame(results)
    
    # CALCULATE ADVANCED METRICS
    precision, recall, f1 = calculate_ml_metrics(df)
    
    # DISPLAY
    st.divider()
    avg_score = df['ROI Score'].mean()
    if avg_score >= quality_gate:
        st.success(f"✅ PASSED QUALITY GATE ({avg_score:.1f}%)")
    else:
        st.error(f"⛔ FAILED QUALITY GATE ({avg_score:.1f}%)")
        
    # METRICS TABS
    tab1, tab2 = st.tabs(["📊 Business Metrics", "🧪 ML Performance (Precision/Recall)"])
    
    with tab1:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Business ROI", f"{df['ROI Score'].mean():.1f}%")
        c2.metric("Semantic Similarity", f"{df['Semantic Similarity'].mean():.1f}%")
        c3.metric("Orchestration", f"{df['Orchestration'].mean():.1f}%")
        c4.metric("Safety Violations", len(df[df['Safety Check'].str.contains("Fail")]), delta_color="inverse")
        
    with tab2:
        st.markdown("### 🤖 Orchestration Performance")
        st.markdown("Mathematical analysis of the Agent's tool selection accuracy.")
        c1, c2, c3 = st.columns(3)
        c1.metric("Precision", f"{precision:.1f}%", help="Correct Tools / Total Calls")
        c2.metric("Recall", f"{recall:.1f}%", help="Correct Tools / Required Tools")
        c3.metric("F1 Score", f"{f1:.1f}%", help="Harmonic Mean")

    # DOWNLOAD
    st.subheader("📄 Reporting")
    html_report = generate_html_report(df, agent_type, quality_gate, precision, recall, f1)
    b64 = base64.b64encode(html_report.encode()).decode()
    st.markdown(f'<a href="data:text/html;base64,{b64}" download="Adya_Advanced_Report.html" style="background-color:#4CAF50;color:white;padding:10px 20px;text-decoration:none;border-radius:5px;">📥 Download HTML Report</a>', unsafe_allow_html=True)
    
    st.dataframe(df, use_container_width=True)