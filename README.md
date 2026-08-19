Azure MLOps Local Churn Pipeline

A lightweight, modular MLOps workflow for training, evaluating, and testing customer churn models locally before deploying to Azure ML. Built with Python 3.10, Ruff, Pytest, and GitHub Actions.

Repository Structure

.
├── .github/workflows/ci.yml   # GitHub Actions CI workflow
├── mlops/pipelines/
│   ├── train_model.py         # Local model training & logging
│   └── evaluate_model.py      # Evaluation against test metrics
├── src/
│   ├── features/              # Data ingestion & feature processing
│   └── models/                # Model logic, saving, & metrics
├── tests/                     # Pytest unit test suite
├── pyproject.toml             # Ruff linter configuration
└── requirements.txt           # Core Python dependencies
