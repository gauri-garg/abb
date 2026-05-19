import pandas as pd
import os
from sklearn.datasets import make_classification
from core.context import DataScienceContext, Metadata, TaskDirectives, DomainConstraints, UserPersona
from core.orchestrator import PipelineOrchestrator
from pipeline.outputs import OutputGeneratorAgent

def create_dummy_data():
    """Creates a sample dataset for testing."""
    X, y = make_classification(n_samples=1000, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f"feature_{i}" for i in range(5)])
    df["target"] = y
    
    os.makedirs("data", exist_ok=True)
    data_path = "data/sample_classification.csv"
    df.to_csv(data_path, index=False)
    return df, data_path

def main():
    print("Initializing Adaptive Data Science LLM Pipeline...")
    
    # 1. Setup Dummy Data
    df, data_path = create_dummy_data()
    
    # 2. Define the Four Context Pillars
    metadata = Metadata(
        dataset_name="Sample Classification Dataset",
        num_rows=len(df),
        num_columns=len(df.columns),
        column_types={col: str(df[col].dtype) for col in df.columns},
        missing_values={col: 0.0 for col in df.columns},
        sample_data=df.head(2).to_dict(orient="records")
    )
    
    directives = TaskDirectives(
        objective="Predict the binary target variable based on features.",
        target_variable="target",
        metrics=["accuracy", "f1"]
    )
    
    constraints = DomainConstraints(
        interpretability_required=True, # Forces Logistic Regression
        max_inference_latency_ms=100,
        excluded_features=["feature_4"] # We'll pretend feature_4 is biased
    )
    
    persona = UserPersona(
        role="Business Executive",
        technical_depth=2 # Keeps the summary high-level
    )
    
    context = DataScienceContext(
        metadata=metadata,
        directives=directives,
        constraints=constraints,
        persona=persona
    )
    
    # 3. Run Pipeline
    orchestrator = PipelineOrchestrator()
    try:
        results = orchestrator.run_pipeline(context, df)
    except Exception as e:
        print(f"Pipeline failed: {e}")
        return
        
    # 4. Generate Outputs
    print("Generating Expected Outputs...")
    output_agent = OutputGeneratorAgent()
    
    model_path = output_agent.export_model(results['model'], context)
    output_agent.generate_compliance_report(context)
    output_agent.generate_dashboard_script(context, data_path, model_path)
    
    # Optional: If you have OPENAI_API_KEY set, uncomment below to test LLM generation
    # if os.getenv("OPENAI_API_KEY"):
    #     output_agent.generate_executive_summary(context, results['metrics'])
    
    print("Pipeline Execution Finished.")
    print("Execution Logs:")
    for log in context.logs:
        print(f" - {log}")

if __name__ == "__main__":
    main()
