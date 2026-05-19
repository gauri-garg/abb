import pandas as pd
from core.context import DataScienceContext
from pipeline.discovery import SemanticDiscoveryAgent, TaskRouterAgent
from pipeline.data_prep import EDAAgent, FeatureEngineeringAgent
from pipeline.modeling import ModelTrainingAgent, EvaluationAgent

class PipelineOrchestrator:
    """
    Main orchestrator that runs the agents sequentially.
    In a full LangGraph implementation, this would be a compiled StateGraph.
    """
    def __init__(self):
        self.discovery_agent = SemanticDiscoveryAgent()
        self.router_agent = TaskRouterAgent()
        self.eda_agent = EDAAgent()
        self.feature_agent = FeatureEngineeringAgent()
        self.training_agent = ModelTrainingAgent()
        self.eval_agent = EvaluationAgent()
        
    def run_pipeline(self, context: DataScienceContext, df: pd.DataFrame):
        context.log_event("--- Starting Adaptive Data Science Pipeline ---")
        context.pipeline_status = "Running"
        
        try:
            # Phase 1: Discovery & Task Classification
            context.current_phase = "Discovery"
            context = self.discovery_agent.discover(context)
            context = self.router_agent.route(context)
            
            # Phase 2: Data Preparation
            context.current_phase = "Data Preparation"
            eda_results = self.eda_agent.perform_eda(context, df)
            df_engineered = self.feature_agent.engineer_features(context, df)
            
            # Phase 3: Modeling & Evaluation
            context.current_phase = "Modeling"
            model, X_test, y_test = self.training_agent.train(context, df_engineered)
            
            context.current_phase = "Evaluation"
            eval_metrics = self.eval_agent.evaluate(context, model, X_test, y_test)
            
            # Pack results
            context.current_phase = "Completed"
            context.pipeline_status = "Success"
            context.log_event("--- Pipeline Completed Successfully ---")
            
            return {
                "context": context,
                "model": model,
                "eda": eda_results,
                "metrics": eval_metrics,
                "processed_data": df_engineered
            }
            
        except Exception as e:
            context.pipeline_status = "Failed"
            context.log_event(f"Pipeline Failed at phase {context.current_phase}: {e}")
            raise e
