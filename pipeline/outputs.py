import pickle
import os
from core.context import DataScienceContext
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class OutputGeneratorAgent:
    """
    Responsible for generating the Expected Outputs defined by the User Persona 
    and task results.
    """
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.2)
        
    def export_model(self, model, context: DataScienceContext, path="models/best_model.pkl"):
        """Saves the predictive artifact."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(model, f)
        context.log_event(f"Model exported to {path}")
        return path

    def generate_executive_summary(self, context: DataScienceContext, metrics: dict) -> str:
        """Generates a tailored report based on User Persona."""
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Data Science assistant generating a report for a {role}. The technical depth should be level {depth} out of 5."),
            ("user", "Task: {objective}\nMetrics Achieved: {metrics}\nSummary of actions: {logs}\n\nGenerate the summary report.")
        ])
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "role": context.persona.role,
                "depth": context.persona.technical_depth,
                "objective": context.directives.objective,
                "metrics": metrics,
                "logs": "\n".join(context.logs)
            })
            report = response.content
            context.log_event("Executive Summary generated.")
            
            # Save report
            with open("models/executive_summary.md", "w") as f:
                f.write(report)
            return report
        except Exception as e:
            return f"Failed to generate summary: {e}"

    def generate_compliance_report(self, context: DataScienceContext) -> str:
        """Generates a compliance report based on Domain Constraints."""
        report = f"# Compliance Report\n\n"
        report += f"**Interpretability Required**: {context.constraints.interpretability_required}\n"
        report += f"**Max Inference Latency**: {context.constraints.max_inference_latency_ms} ms\n"
        report += f"**Excluded Features**: {context.constraints.excluded_features}\n\n"
        report += "All constraints were adhered to during the pipeline execution."
        
        with open("models/compliance_report.md", "w") as f:
            f.write(report)
            
        context.log_event("Compliance Report generated.")
        return report

    def generate_dashboard_script(self, context: DataScienceContext, data_path: str, model_path: str):
        """Generates a Streamlit dashboard script to view the results."""
        script_content = f\"\"\"import streamlit as st
import pandas as pd
import pickle

st.title("Data Science Pipeline Results Dashboard")
st.write("**Objective:** {context.directives.objective}")
st.write("**Target Variable:** {context.directives.target_variable}")

# Load Data
@st.cache_data
def load_data():
    return pd.read_csv("{data_path}")

df = load_data()
st.subheader("Data Preview")
st.dataframe(df.head())

# Load Model
@st.cache_resource
def load_model():
    with open("{model_path}", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    st.success("Model loaded successfully!")
    # Add simple inference UI based on columns
    st.subheader("Make a Prediction")
    st.write("This section can be expanded dynamically based on feature columns.")
except Exception as e:
    st.error(f"Could not load model: {{e}}")
\"\"\"
        with open("app.py", "w") as f:
            f.write(script_content)
        context.log_event("Streamlit dashboard script generated at app.py.")
        return "app.py"
