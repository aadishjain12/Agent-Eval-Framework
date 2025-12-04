import functools
import time
import google.generativeai as genai
import sys

# --- 1. CONFIGURATION ---
# !!! PASTE YOUR GOOGLE API KEY HERE !!!
GOOGLE_API_KEY = "AIzaSyDw0X1moWqKVO5QzlyKbEFZws97k8zNFzE"

genai.configure(api_key=GOOGLE_API_KEY)

# --- 2. AUTO-DISCOVERY ENGINE (The Fix) ---
def get_working_model():
    """
    Automatically finds a working model so you never get a 404 error.
    """
    print("🔄 Detecting available models for your API key...")
    try:
        # Get all models that support 'generateContent'
        available_models = []
        for m in genai.list_models():
            if 'generateContent' in m.supported_generation_methods:
                available_models.append(m.name)
        
        # Priority list: Try to find Flash (fastest), then Pro, then any Gemini
        preferred_order = ["gemini-1.5-flash", "gemini-1.5-pro", "gemini-1.0-pro", "gemini-pro"]
        
        selected_model = None
        
        # 1. Check if any preferred model is explicitly in the available list
        for pref in preferred_order:
            for avail in available_models:
                if pref in avail:
                    selected_model = avail
                    break
            if selected_model: break
            
        # 2. Fallback: Just take the first available Gemini model
        if not selected_model:
            for avail in available_models:
                if "gemini" in avail:
                    selected_model = avail
                    break
        
        if not selected_model:
            print("❌ No Gemini models found! Check your API Key.")
            sys.exit(1)
            
        print(f"✅ Using Model: {selected_model}")
        return genai.GenerativeModel(selected_model)

    except Exception as e:
        print(f"⚠️ Auto-detection failed ({e}). Defaulting to 'gemini-pro'.")
        return genai.GenerativeModel("gemini-pro")

# Initialize the model dynamically
model = get_working_model()

# --- 3. THE TRACER ---
agent_trace = []

def track_tool(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        step_data = {
            "tool": func.__name__,
            "input": str(args) + str(kwargs),
            "status": "running"
        }
        try:
            result = func(*args, **kwargs)
            step_data["output"] = str(result)
            step_data["status"] = "success"
            agent_trace.append(step_data)
            return result
        except Exception as e:
            step_data["error"] = str(e)
            step_data["status"] = "failed"
            agent_trace.append(step_data)
            raise e     
    return wrapper

# --- 4. THE AGENT ---

@track_tool
def search_google(query):
    if "weather" in query.lower():
        return "It is 30 degrees and sunny in Bhilwara."
    return "No results found."

@track_tool
def calculator(expression):
    try:
        return eval(expression)
    except:
        return "Error"

def my_agent(user_query):
    global agent_trace
    agent_trace = [] 
    
    print(f"\n🤖 AGENT WORKING ON: '{user_query}'")
    
    if "plus" in user_query or "+" in user_query:
        print("   > Agent decided to use Calculator...")
        return f"The answer is {calculator('2+2')}" # Intentional Bug
    elif "weather" in user_query:
        print("   > Agent decided to Search Google...")
        return f"Here is the weather: {search_google(user_query)}"
    else:
        return "I don't know."

# --- 5. THE JUDGE ---

def run_evaluation(user_input, agent_response):
    print("\n--- 🕵️ RUNNING EVALUATION ---")
    
    trace_text = ""
    for i, step in enumerate(agent_trace):
        trace_text += f"Step {i+1}: Called '{step['tool']}' with input {step['input']} -> Output: {step.get('output', 'Error')}\n"

    eval_prompt = f"""
    You are an AI Quality Assurance System. 
    Evaluate the agent based on this execution trace.

    USER REQUEST: "{user_input}"
    AGENT RESPONSE: "{agent_response}"
    
    TRACE:
    {trace_text}

    Did the agent succeed? 
    (Note: If user asked for 25+25 and trace shows 2+2, that is a FAIL).
    
    Return strictly JSON:
    {{
        "score": (0 or 1),
        "reason": "Short explanation."
    }}
    """

    try:
        response = model.generate_content(eval_prompt)
        print(f"📊 RESULT:\n{response.text}")
    except Exception as e:
        print(f"❌ Judge Error: {e}")

# --- EXECUTION ---
if __name__ == "__main__":
    q1 = "What is the weather in Bhilwara?"
    run_evaluation(q1, my_agent(q1))

    print("-" * 30)

    q2 = "What is 25 plus 25?"
    run_evaluation(q2, my_agent(q2))