import os
import json
import pip
import pandas as pd
import streamlit as st
from pydantic import BaseModel
from typing import Optional
from langchain_groq import ChatGroq

# =====================================================================
# 1. PAGE CONFIGURATION & SETUP
# =====================================================================
st.set_page_config(
    page_title="AI Credit Decisioning Agent",
    page_icon="💳",
    layout="wide"
)

st.title("💳 AI Credit Decisioning Platform")
st.markdown("---")

# Securely retrieve API Key from sidebar or environment variable
if "GROQ_API_KEY" not in os.environ:
    api_key_input = st.sidebar.text_input("Enter Groq API Key:", type="password")
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input
    else:
        st.sidebar.warning("Please enter your Groq API Key in the sidebar to proceed.")
        st.warning("Please enter your Groq API Key in the sidebar to proceed.")
        st.stop()

# Initialize LLM
@st.cache_resource
def get_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        temperature=0
    )

llm = get_llm()

# =====================================================================
# 2. DEFINITIONS & CREDIT POLICY
# =====================================================================
class Applicant(BaseModel):
    applicant_id: str
    age: int
    income: float  # monthly
    employment_type: str  # salaried, self-employed, gig
    employment_tenure: float  # in years
    credit_score: Optional[int]
    existing_emi: float
    loan_amount: float
    loan_tenure: int  # months
    loan_purpose: str

CREDIT_POLICY = """
Strong Applicant:
- Credit score > 750
- DTI (EMI/income) < 35%
- Stable employment (>2 years salaried)
- Low existing obligations

Moderate:
- Credit score 650–750
- DTI 35–50%
- Some instability or thin file

Weak:
- Credit score < 650
- DTI > 50%
- Irregular or unverifiable income

Automatic Reject:
- Credit score < 550
- Missing income
- EMI > income

Refer to Human:
- No credit history
- Gig workers with high income variability
- Borderline DTI (45–55%)
- Inconsistent profile

Scoring Guide (0–100):
- Credit score: 40%
- DTI: 30%
- Employment stability: 20%
- Profile quality: 10%
"""

# =====================================================================
# 3. AGENT CORE LOGIC FUNCTIONS
# =====================================================================
def data_assessment(applicant_dict):
    applicant_json = json.dumps(applicant_dict, indent=2)
    prompt = f"""
    You are a credit pre-screening agent.

    Analyze this applicant:
    {applicant_json}

    Identify:
    - Missing fields among only the defined set of attributes under each application.
    Don't hallucinate and choose only from the defined set of attributes.
    - Red flags if any, among only the defined set of attributes under each application.
    - Clarifying questions, if any, among only the defined set of attributes under each application.

    Output strictly a valid JSON matching this format:
    {{
      "missing_fields": "...",
      "red_flags": "...",
      "questions": "..."
    }}
    """
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip().replace("```json", "").replace("```", ""))
    except Exception:
        return {"missing_fields": "N/A", "red_flags": "Failed to parse JSON response", "questions": "Error processing"}

def scoring_agent(applicant_dict, policy):
    applicant_json = json.dumps(applicant_dict, indent=2)
    prompt = f"""
    You are a credit assessment agent.

    Follow policy:
    {policy}

    Assess each Applicant, basis the policy defined above:
    {applicant_json}

    IMPORTANT:
    - Return ONLY valid JSON
    - Do NOT add explanations outside JSON
    - Do NOT use markdown
    - score cannot be zero, it should have a value
    - restrict the risk tier between Low, Medium, High
    - recommendation can be Approve, Reject, or Refer to Human

    Output JSON:
    {{
      "applicant_id": "{applicant_dict.get('applicant_id')}",
      "score": 0,
      "risk_tier": "",
      "recommendation": "",
      "reasoning_trace": "",
      "explanation": ""
    }}
    """
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip().replace("```json", "").replace("```", ""))
    except Exception:
        return None

def reviewer_agent(decision_output):
    prompt = f"""
    You are a credit committee reviewer.

    Review this decision:
    {json.dumps(decision_output, indent=2)}

    Challenge:
    - Is the decision justified?
    - Any overlooked risks?
    - Should it be escalated?

    IMPORTANT:
    - Return ONLY valid JSON matching this layout.
    - Do NOT wrap outside text or notes.

    Output:
    {{
      "agreement": "Yes/No",
      "concerns": "Any overlooked risk notes or justification issues found.",
      "final_decision": "Approve / Reject / Refer to Human"
    }}
    """
    try:
        response = llm.invoke(prompt)
        return json.loads(response.content.strip().replace("```json", "").replace("```", ""))
    except Exception:
        return {"agreement": "Error", "concerns": "Review parsing failed.", "final_decision": decision_output.get("recommendation") if decision_output else "Refer to Human"}

# =====================================================================
# 4. STREAMLIT INTERFACE / TWO INPUT METHODS
# =====================================================================
with st.expander("📄 View Active System Credit Policy Matrix", expanded=False):
    st.text(CREDIT_POLICY)

tab_single, tab_bulk = st.tabs(["📝 Process Single Applicant", "📁 Process Bulk Excel / CSV Upload"])

# ----------------- TAB 1: SINGLE APPLICANT FORM PROCESSING -----------------
with tab_single:
    sample_profiles = {
        "Select custom or sample profile...": None,
        "Profile 1: Strong Applicant (John Doe)": {
            "applicant_id": "APP-001", "age": 35, "income": 8500.0, "employment_type": "salaried",
            "employment_tenure": 5.5, "credit_score": 780, "existing_emi": 500.0,
            "loan_amount": 25000.0, "loan_tenure": 36, "loan_purpose": "Home Renovation"
        },
        "Profile 2: Borderline / Gig Worker (Jane Smith)": {
            "applicant_id": "APP-002", "age": 28, "income": 4200.0, "employment_type": "gig",
            "employment_tenure": 1.2, "credit_score": 660, "existing_emi": 1200.0,
            "loan_amount": 15000.0, "loan_tenure": 24, "loan_purpose": "Debt Consolidation"
        }
    }

    selected_sample = st.selectbox("Quick-Load Test Profiles:", list(sample_profiles.keys()))
    loaded_data = sample_profiles[selected_sample] if sample_profiles[selected_sample] else {}

    st.subheader("📝 Manual Entry Form Details")
    with st.form("loan_form"):
        col1, col2 = st.columns(2)
        with col1:
            app_id = st.text_input("Applicant ID", value=loaded_data.get("applicant_id", "APP-100"))
            age = st.number_input("Age", min_value=18, max_value=100, value=loaded_data.get("age", 30))
            income = st.number_input("Monthly Income ($)", min_value=0.0, value=loaded_data.get("income", 5000.0))
            emp_type_val = loaded_data.get("employment_type", "salaried")
            employment_type = st.selectbox("Employment Type", ["salaried", "self-employed", "gig"], 
                                           index=["salaried", "self-employed", "gig"].index(emp_type_val))
            employment_tenure = st.number_input("Employment Tenure (Years)", min_value=0.0, value=loaded_data.get("employment_tenure", 3.0))
        
        with col2:
            credit_score = st.number_input("Credit Score (0 for missing)", min_value=0, max_value=850, value=loaded_data.get("credit_score", 700))
            existing_emi = st.number_input("Existing Monthly EMIs ($)", min_value=0.0, value=loaded_data.get("existing_emi", 400.0))
            loan_amount = st.number_input("Requested Loan Amount ($)", min_value=0.0, value=loaded_data.get("loan_amount", 10000.0))
            loan_tenure = st.number_input("Loan Tenure (Months)", min_value=1, value=loaded_data.get("loan_tenure", 12))
            loan_purpose = st.text_input("Loan Purpose", value=loaded_data.get("loan_purpose", "Car Purchase"))
            
        submit_btn = st.form_submit_button("🚀 Run Credit Risk Evaluation")

    if submit_btn:
        applicant_data = {
            "applicant_id": app_id, "age": age, "income": income, "employment_type": employment_type,
            "employment_tenure": employment_tenure, "credit_score": None if credit_score == 0 else credit_score,
            "existing_emi": existing_emi, "loan_amount": loan_amount, "loan_tenure": loan_tenure, "loan_purpose": loan_purpose
        }
        
        st.subheader("🔮 Agent Processing Pipeline")
        with st.spinner("Agent 1 executing Pre-Screening Data Check..."):
            assessment_res = data_assessment(applicant_data)
        
        st.success("Step 1 Assessment Completed!")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("⚠️ Missing Fields", assessment_res.get("missing_fields", "None"))
        col_b.error(f"🚩 Red Flags: {assessment_res.get('red_flags', 'None')}")
        col_c.info(f"❓ Clarifying Questions: {assessment_res.get('questions', 'None')}")
        
        with st.spinner("Agent 2 scoring risk values..."):
            decision_res = scoring_agent(applicant_data, CREDIT_POLICY)
            
        if decision_res:
            st.success("Step 2 Core Underwriting Completed!")
            
            with st.spinner("Agent 3 running Credit Committee Review Audit..."):
                review_res = reviewer_agent(decision_res)
            
            rec = review_res.get("final_decision", decision_res.get("recommendation"))
            
            if rec == "Approve":
                st.balloons()
                st.success(f"### Final Committee Approved Recommendation: **{rec}**")
            elif rec == "Reject":
                st.error(f"### Final Committee Approved Recommendation: **{rec}**")
            else:
                st.warning(f"### Final Committee Approved Recommendation: **{rec}**")
                
            col_m1, col_m2 = st.columns(2)
            col_m1.metric(label="Risk Score (0-100)", value=decision_res.get("score"))
            col_m2.metric(label="Assigned Risk Tier", value=decision_res.get("risk_tier"))
            
            t1, t2, t3 = st.tabs(["🧠 Agent Underwriting Trace", "🛡️ Committee Audit Review Logs", "💬 Customer Explanation"])
            with t1:
                st.code(decision_res.get("reasoning_trace"), language="text")
            with t2:
                col_x, col_y = st.columns([1, 3])
                col_x.metric("Committee Agreement?", review_res.get("agreement"))
                col_y.info(f"**Reviewer Critique Comments:** {review_res.get('concerns')}")
            with t3:
                st.write(decision_res.get("explanation"))

# ----------------- TAB 2: BULK EXCEL SPREADSHEET UPLOAD -----------------
with tab_bulk:
    st.subheader("📁 Mass Upload Pipeline")
    st.markdown("Upload a `.xlsx` or `.csv` spreadsheet containing your loan file matrix.")
    
    uploaded_file = st.file_uploader("Drop spreadsheet here:", type=["xlsx", "csv"])
    
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            st.success(f"Successfully loaded file: **{uploaded_file.name}**")
            st.dataframe(df, use_container_width=True)
            
            process_bulk = st.button("🏁 Initiate Bulk AI Risk Batch Run")
            
            if process_bulk:
                rows = []
                progress_bar = st.progress(0)
                total_rows = len(df)
                
                for index, row in df.iterrows():
                    # Parse inputs matching your case-sensitive headers
                    bulk_applicant = {
                        "applicant_id": str(row.get("Applicant ID", f"ROW-{index+1}")),
                        "age": int(row.get("Age", 30)),
                        "income": float(row.get("Income", 0.0)),
                        "employment_type": str(row.get("Employment_type", "salaried")),
                        "employment_tenure": float(row.get("Employment_tenure", 0.0)),
                        "credit_score": None if pd.isna(row.get("Credit_score")) or row.get("Credit_score") == 0 else int(row.get("Credit_score")),
                        "existing_emi": float(row.get("Existing_emi", 0.0)),
                        "loan_amount": float(row.get("Loan_amount", 0.0)),
                        "loan_tenure": int(row.get("Loan_tenure", 12)),
                        "loan_purpose": str(row.get("Loan_purpose", "Business Expansion"))
                    }
                    
                    with st.spinner(f"Processing Row {index+1}/{total_rows} ({bulk_applicant['applicant_id']})..."):
                        # Execute sequential multi-agent stack
                        assess = data_assessment(bulk_applicant)
                        decision = scoring_agent(bulk_applicant, CREDIT_POLICY)
                        review = reviewer_agent(decision) if decision else None
                        
                    # Build custom matrix row matching your template preference
                    rows.append({
                        "applicant_id": bulk_applicant["applicant_id"],
                        "income": bulk_applicant["income"],
                        "employment_type": bulk_applicant["employment_type"],
                        "credit_score": bulk_applicant["credit_score"],
                        "existing_emi": bulk_applicant["existing_emi"],
                        "loan_amount": bulk_applicant["loan_amount"],
                        
                        "missing_fields": assess.get("missing_fields") if assess else "N/A",
                        "red_flags": assess.get("red_flags") if assess else "N/A",
                        
                        "score": decision.get("score") if decision else 0,
                        "risk_tier": decision.get("risk_tier") if decision else "High",
                        "recommendation": decision.get("recommendation") if decision else "Reject",
                        "reasoning_trace": decision.get("reasoning_trace") if decision else "",
                        "explanation": decision.get("explanation") if decision else "",
                        
                        "review_agreement": review.get("agreement") if review else "No",
                        "review_concerns": review.get("concerns") if review else "Failed execution",
                        "final_decision": review.get("final_decision") if review else "Reject",
                    })
                    
                    progress_bar.progress((index + 1) / total_rows)
                
                st.success("🎉 Bulk Review & Audit Processing Completed!")
                
                # Convert list to DataFrame and display in the UI
                output_df = pd.DataFrame(rows)
                st.subheader("📊 Combined Input/Evaluation Output Matrix")
                st.dataframe(output_df, use_container_width=True)
                
                # =====================================================================
                # 📥 STREAMLIT SECURE FILE DOWNLOAD EXPORT COMPONENT
                # =====================================================================
                # Convert the output dataframe to CSV format in memory
                csv_data = output_df.to_csv(index=False).encode('utf-8')
                
                st.markdown("---")
                st.subheader("💾 Export Report Options")
                st.download_button(
                    label="📥 Download Final Evaluation Excel Report (CSV)",
                    data=csv_data,
                    file_name="credit_decision_output.csv",
                    mime="text/csv",
                    key="download-csv"
                )
                
        except Exception as e:
            st.error(f"Error parsing file: {e}. Please ensure your file columns align exactly with your case-sensitive headers.")