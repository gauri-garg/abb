from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class Metadata(BaseModel):
    """
    Pillar 1: Metadata
    Represents the schema and statistics of the dataset.
    """
    dataset_name: str
    num_rows: int
    num_columns: int
    column_types: Dict[str, str] = Field(description="Mapping of column names to data types")
    missing_values: Dict[str, float] = Field(description="Percentage of missing values per column")
    sample_data: List[Dict[str, Any]] = Field(description="A few rows of sample data")

class TaskDirectives(BaseModel):
    """
    Pillar 2: Task Directives
    The specific goal of the data science task.
    """
    objective: str = Field(description="E.g., 'Predict customer churn'")
    task_type: Optional[str] = Field(None, description="E.g., 'classification', 'regression', 'clustering', 'time-series'")
    target_variable: Optional[str] = Field(None, description="The column to predict, if supervised")
    metrics: List[str] = Field(default_factory=list, description="Metrics to optimize, e.g., 'f1', 'rmse'")

class DomainConstraints(BaseModel):
    """
    Pillar 3: Domain Constraints
    Business rules, latency requirements, or regulatory constraints.
    """
    interpretability_required: bool = Field(default=False, description="If True, favor simpler models like Logistic Regression or Trees")
    max_inference_latency_ms: Optional[int] = Field(None, description="Maximum allowed inference time")
    excluded_features: List[str] = Field(default_factory=list, description="Features that cannot be used (e.g., for fairness/compliance)")

class UserPersona(BaseModel):
    """
    Pillar 4: User Persona
    Dictates the depth and style of the output artifacts.
    """
    role: str = Field(description="E.g., 'Data Scientist', 'Business Executive', 'Compliance Officer'")
    technical_depth: int = Field(ge=1, le=5, description="1 is high-level summary, 5 is raw code and complex metrics")

class DataScienceContext(BaseModel):
    """
    The unifying Context Engine state that holds all pillars.
    """
    metadata: Metadata
    directives: TaskDirectives
    constraints: DomainConstraints
    persona: UserPersona
    
    # Execution state tracking
    current_phase: str = "Initialization"
    pipeline_status: str = "Pending"
    logs: List[str] = Field(default_factory=list)

    def log_event(self, message: str):
        self.logs.append(message)
        print(f"[{self.current_phase}] {message}")
