import streamlit as st
import google.generativeai as genai
import json
import pandas as pd
import time

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Adya Agent Workbench", page_icon="🧪", layout="wide")

# !!! PASTE API KEY HERE !!!
# --- SECURE API KEY HANDLING ---
try:
    # Try loading from Streamlit secrets (Local or Cloud)
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
except FileNotFoundError:
    st.error("❌ API Key not found! Please create .streamlit/secrets.toml")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY) 

if GOOGLE_API_KEY == "YOUR_GOOGLE_API_KEY_HERE":
    st.error("⚠️ Please paste your Google API Key in the code.")
    st.stop()

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. MODEL SETUP (The Fix) ---
@st.cache_resource
def get_model():
    """
    Auto-detects the best available model to prevent 404 errors.
    """
    try:
        # Ask Google what models are available
        models = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priority: Flash (Fast) -> Pro (Smart) -> Legacy Pro
        for m in models:
            if "flash" in m: return genai.GenerativeModel(m)
        for m in models:
            if "pro" in m: return genai.GenerativeModel(m)
            
        # Fallback
        return genai.GenerativeModel("gemini-pro")
    except Exception as e:
        st.error(f"Error connecting to Gemini: {e}")
        return None

# !!! THIS LINE WAS MISSING OR DELETED !!!
# We must initialize the global variable 'model' here
model = get_model()

if model is None:
    st.error("Failed to load Gemini Model. Check API Key.")
    st.stop()

# --- 3. SESSION STATE MANAGEMENT ---
# (The rest of your code continues below...)

# --- 2. SESSION STATE MANAGEMENT ---
if "messages" not in st.session_state:
    st.session_state.messages = [] # Stores chat history
if "custom_metrics" not in st.session_state:
    st.session_state.custom_metrics = [
        "Hallucination Check", 
        "Tone Consistency", 
        "Goal Completion"
    ] # Default metrics
if "eval_results" not in st.session_state:
    st.session_state.eval_results = {}

# --- 3. AGENT LOGIC (Chat) ---
def get_agent_response(history, system_prompt):
    """
    Sends the entire chat history to the model to maintain context.
    """
    try:
        # Construct the full prompt context
        full_prompt = f"SYSTEM: {system_prompt}\n\n"
        for msg in history:
            role = "USER" if msg["role"] == "user" else "AGENT"
            full_prompt += f"{role}: {msg['content']}\n"
        full_prompt += "AGENT:"
        
        response = model.generate_content(full_prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error: {str(e)}"

# --- 4. JUDGE LOGIC (Dynamic Metric Evaluator) ---
def evaluate_session(chat_history, metric_name):
    """
    Dynamically creates a grading rubric for ANY metric name provided.
    """
    # Convert chat object to string transcript
    transcript = ""
    for msg in chat_history:
        role = "User" if msg["role"] == "user" else "Agent"
        transcript += f"{role}: {msg['content']}\n"

    # The Meta-Judge Prompt
    eval_prompt = f"""
    You are an expert AI Auditor. Evaluate the following conversation transcript based on a specific metric.
    
    TRANSCRIPT:
    {transcript}
    
    METRIC TO EVALUATE: "{metric_name}"
    
    INSTRUCTIONS:
    1. Analyze the agent's performance specifically regarding "{metric_name}".
    2. Assign a Score (0-100) or (Pass/Fail) depending on what makes sense.
    3. Provide a short, specific reasoning.
    
    Return JSON: {{ "score": "X/100 or Pass/Fail", "reason": "Explanation..." }}
    """
    
    try:
        res = model.generate_content(eval_prompt)
        text = res.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    except:
        return {"score": "Error", "reason": "Evaluation failed."}

# --- 5. UI LAYOUT ---

st.title("🧪 Adya Agent Workbench")
st.markdown("Interactive Playground for Prompt Engineering & Evaluation.")

# Use 3 Columns: Config | Chat | Inspector
col1, col2, col3 = st.columns([1, 2, 1.5])

# === COLUMN 1: AGENT CONFIGURATION ===
with col1:
    st.header("1. Configure")
    
    # Preset Prompts
    prompts = {
        "Retail Agent": "You are a helpful Retail Assistant for Adya Shop. Tools: [check_stock], [return_item]. Be polite but concise.",
        "Banking Agent": "You are a Secure Banking AI. NEVER transfer money without OTP. Be formal and strict.",
        "Healthcare Agent": "You are an empathetic Medical Assistant. Do not give medical advice, only administrative help."
    }
    
    selected_preset = st.selectbox("Load Preset:", list(prompts.keys()))
    
    # Editable System Prompt
    system_prompt = st.text_area(
        "System Prompt (Editable):", 
        value=prompts[selected_preset], 
        height=300,
        help="Edit this to change the agent's behavior instantly."
    )
    
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.session_state.eval_results = {}
        st.rerun()

# === COLUMN 2: THE PLAYGROUND (Chat) ===
with col2:
    st.header("2. Playground")
    
    # Chat Container
    chat_container = st.container(height=600)
    
    # Display History
    with chat_container:
        if len(st.session_state.messages) == 0:
            st.info("👋 Select an agent and start chatting to test its logic.")
            
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

    # User Input
    if prompt := st.chat_input("Type a message..."):
        # 1. Add User Message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with chat_container:
            with st.chat_message("user"):
                st.markdown(prompt)

        # 2. Get Agent Response
        with chat_container:
            with st.chat_message("assistant"):
                with st.spinner("Thinking..."):
                    response = get_agent_response(st.session_state.messages, system_prompt)
                    st.markdown(response)
        
        # 3. Add Agent Message
        st.session_state.messages.append({"role": "assistant", "content": response})
        st.rerun() # Refresh to update state

# === COLUMN 3: THE INSPECTOR (Metrics) ===
with col3:
    st.header("3. Inspector")
    
    # A. Metric Manager
    with st.expander("🛠️ Define Metrics", expanded=True):
        new_metric = st.text_input("Create Custom Metric:", placeholder="e.g., Was it rude?")
        if st.button("Add Metric"):
            if new_metric:
                st.session_state.custom_metrics.append(new_metric)
                st.rerun()
        
        # Selection List
        st.write("Select metrics to evaluate:")
        selected_metrics = []
        for m in st.session_state.custom_metrics:
            if st.checkbox(m, value=True, key=f"check_{m}"):
                selected_metrics.append(m)

    # B. Run Evaluation Button
    if len(st.session_state.messages) > 0:
        if st.button("⚡ Evaluate Session", type="primary", use_container_width=True):
            with st.spinner("Judge is analyzing conversation..."):
                results = {}
                progress = st.progress(0)
                
                for i, metric in enumerate(selected_metrics):
                    # Call the LLM Judge
                    eval_data = evaluate_session(st.session_state.messages, metric)
                    results[metric] = eval_data
                    progress.progress((i + 1) / len(selected_metrics))
                
                st.session_state.eval_results = results
                progress.empty()
    else:
        st.caption("Chat with the agent first to enable evaluation.")

    # C. Results Display
    if st.session_state.eval_results:
        st.divider()
        st.subheader("📋 Session Report Card")
        
        for metric, data in st.session_state.eval_results.items():
            # Color coding
            score = str(data.get('score', '0'))
            if "Pass" in score or "100" in score or ("/" in score and int(score.split('/')[0]) > 80):
                color = "green"
                icon = "✅"
            else:
                color = "red"
                icon = "❌"
            
            with st.container():
                st.markdown(f"**{metric}**")
                st.markdown(f":{color}[{icon} **{score}**]")
                st.caption(f"{data.get('reason', 'No reason provided.')}")
                st.divider()