# Adaptive Data Science LLM System

This system is an intelligent, agentic workflow designed to automate data science tasks such as exploratory data analysis (EDA), feature engineering, modeling, and deployment.

It operates on four core context pillars to understand the problem space:
1. **Metadata**: The structure and statistics of the dataset.
2. **Task Directives**: The specific goal (e.g., classification, regression).
3. **Domain Constraints**: Business and regulatory rules.
4. **User Persona**: Output tailoring (e.g., Data Scientist vs. Executive).

## Directory Structure
- `core/`: Core data structures and state management (e.g., Context Engine).
- `pipeline/`: Agentic workflows (Discovery, EDA, Modeling, Validation).
- `models/`: Exported predictive artifacts.
- `api/`: REST APIs for interaction.
- `utils/`: Helper functions.
- `data/`: Sample datasets for testing.

## Setup
1. Create a virtual environment: `python3 -m venv venv`
2. Activate it: `source venv/bin/activate`
3. Install dependencies: `pip install -r requirements.txt` (or install manually: `langchain pydantic pandas scikit-learn fastapi streamlit`)
4. Set your `OPENAI_API_KEY` in a `.env` file.
