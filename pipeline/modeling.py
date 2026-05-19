from core.context import DataScienceContext
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import pickle
import os

class ModelTrainingAgent:
    """
    Selects and trains a model based on the Task Directive and Domain Constraints.
    """
    def train(self, context: DataScienceContext, df: pd.DataFrame):
        context.log_event("Starting Model Training...")
        target = context.directives.target_variable
        
        if target not in df.columns:
            raise ValueError(f"Target variable '{target}' not found in DataFrame.")
            
        X = df.drop(columns=[target])
        y = df[target]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        task_type = context.directives.task_type
        interpretability = context.constraints.interpretability_required
        
        model = None
        if task_type == 'classification':
            if interpretability:
                model = LogisticRegression(max_iter=1000)
                context.log_event("Selected Logistic Regression (interpretable classification).")
            else:
                model = RandomForestClassifier(n_estimators=100, random_state=42)
                context.log_event("Selected Random Forest (complex classification).")
        elif task_type == 'regression':
            if interpretability:
                model = LinearRegression()
                context.log_event("Selected Linear Regression (interpretable regression).")
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42)
                context.log_event("Selected Random Forest (complex regression).")
        else:
            raise NotImplementedError(f"Task type '{task_type}' training not yet implemented in this MVP.")
            
        model.fit(X_train, y_train)
        context.log_event("Model training completed.")
        
        return model, X_test, y_test

class EvaluationAgent:
    """
    Evaluates the trained model against the specified metrics in Task Directives.
    """
    def evaluate(self, context: DataScienceContext, model, X_test, y_test) -> dict:
        context.log_event("Starting Model Evaluation...")
        predictions = model.predict(X_test)
        
        results = {}
        task_type = context.directives.task_type
        metrics_requested = context.directives.metrics
        
        if task_type == 'classification':
            results['accuracy'] = accuracy_score(y_test, predictions)
            if 'f1' in metrics_requested:
                results['f1'] = f1_score(y_test, predictions, average='weighted')
        elif task_type == 'regression':
            results['r2'] = r2_score(y_test, predictions)
            if 'rmse' in metrics_requested:
                results['rmse'] = mean_squared_error(y_test, predictions, squared=False)
                
        context.log_event(f"Evaluation metrics: {results}")
        return results

class ValidationSelfHealingAgent:
    """
    Acts as a critique agent. If training/evaluation fails or metrics are too poor, 
    it tries to diagnose and adjust the approach using the LLM.
    """
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o", temperature=0.2)
        
    def diagnose_and_heal(self, context: DataScienceContext, error_message: str) -> str:
        context.log_event(f"Self-Healing Agent invoked due to error: {error_message}")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert MLOps Self-Healing Agent. Given an error message from a data pipeline, suggest a single, actionable Python code fix or strategy to resolve it."),
            ("user", "Context Objective: {objective}\nError: {error}")
        ])
        
        chain = prompt | self.llm
        try:
            response = chain.invoke({
                "objective": context.directives.objective,
                "error": error_message
            })
            suggestion = response.content
            context.log_event(f"Self-Healing Suggestion: {suggestion}")
            return suggestion
        except Exception as e:
            return f"Failed to generate healing strategy: {e}"
