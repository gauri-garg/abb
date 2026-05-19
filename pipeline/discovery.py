from core.context import DataScienceContext
import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

class SemanticDiscoveryAgent:
    """
    Analyzes the dataset metadata to automatically infer semantic relationships 
    and meanings of columns.
    """
    def __init__(self, llm=None):
        # Default to gpt-4o-mini if not provided
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def discover(self, context: DataScienceContext) -> DataScienceContext:
        context.log_event("Starting Semantic Discovery...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an expert Data Scientist. Analyze the dataset metadata and provide semantic meaning for each column. Keep it brief."),
            ("user", "Dataset: {dataset_name}\nColumns: {columns}\nSample: {sample}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "dataset_name": context.metadata.dataset_name,
                "columns": context.metadata.column_types,
                "sample": context.metadata.sample_data
            })
            context.log_event(f"Semantic Discovery Complete: {response.content}")
        except Exception as e:
            context.log_event(f"Semantic Discovery failed (possibly missing API key). Error: {e}")
            
        return context

class TaskRouterAgent:
    """
    Looks at the objective and metadata to classify the problem (e.g., classification, regression)
    and route to the appropriate next steps.
    """
    def __init__(self, llm=None):
        self.llm = llm or ChatOpenAI(model="gpt-4o-mini", temperature=0)

    def route(self, context: DataScienceContext) -> DataScienceContext:
        context.log_event("Starting Task Routing...")
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI router. Given the objective and target variable, determine the machine learning task type. Respond ONLY with one of: 'classification', 'regression', 'clustering', 'time-series'."),
            ("user", "Objective: {objective}\nTarget Variable: {target_variable}")
        ])
        
        chain = prompt | self.llm
        
        try:
            response = chain.invoke({
                "objective": context.directives.objective,
                "target_variable": context.directives.target_variable or "None"
            })
            task_type = response.content.strip().lower()
            context.directives.task_type = task_type
            context.log_event(f"Task classified as: {task_type}")
        except Exception as e:
            # Fallback heuristic
            context.log_event(f"Task Routing failed, applying fallback heuristic. Error: {e}")
            if context.directives.target_variable:
                # Naive fallback
                context.directives.task_type = "classification"
            else:
                context.directives.task_type = "clustering"
            
        return context
