from core.context import DataScienceContext
import pandas as pd
import numpy as np

class EDAAgent:
    """
    Automated statistical profiling of the dataset based on Context.
    """
    def perform_eda(self, context: DataScienceContext, df: pd.DataFrame) -> dict:
        context.log_event("Starting Exploratory Data Analysis...")
        eda_results = {}
        
        # Basic stats
        eda_results["description"] = df.describe(include='all').to_dict()
        eda_results["missing_values"] = df.isnull().sum().to_dict()
        eda_results["correlation_matrix"] = df.corr(numeric_only=True).to_dict() if len(df.select_dtypes(include=np.number).columns) > 1 else {}
        
        context.log_event("EDA completed successfully.")
        return eda_results

class FeatureEngineeringAgent:
    """
    Generates and applies feature engineering based on Domain Constraints and Task Directives.
    In a full LLM application, this would write Python code to transform the DataFrame.
    For this implementation, it applies standard transforms based on heuristics.
    """
    def engineer_features(self, context: DataScienceContext, df: pd.DataFrame) -> pd.DataFrame:
        context.log_event("Starting Feature Engineering...")
        
        df_transformed = df.copy()
        
        # 1. Drop excluded features
        excluded = context.constraints.excluded_features
        for col in excluded:
            if col in df_transformed.columns:
                df_transformed.drop(columns=[col], inplace=True)
                context.log_event(f"Dropped excluded feature: {col}")
                
        # 2. Impute missing values (simple heuristic)
        for col in df_transformed.columns:
            if df_transformed[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df_transformed[col]):
                    df_transformed[col] = df_transformed[col].fillna(df_transformed[col].median())
                    context.log_event(f"Imputed missing numeric values in {col} with median.")
                else:
                    df_transformed[col] = df_transformed[col].fillna(df_transformed[col].mode()[0])
                    context.log_event(f"Imputed missing categorical values in {col} with mode.")
                    
        # 3. Encoding (One-Hot for simple categoricals)
        categorical_cols = df_transformed.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            df_transformed = pd.get_dummies(df_transformed, columns=categorical_cols, drop_first=True)
            context.log_event(f"Applied One-Hot Encoding to categorical columns.")
            
        context.log_event("Feature Engineering completed.")
        return df_transformed
