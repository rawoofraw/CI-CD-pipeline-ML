# MLOps: Automated CI/CD Pipeline for Machine Learning 🤖🚀

This project showcases a lightweight Continuous Integration and Continuous Deployment (CI/CD) pipeline tailored for Machine Learning models, treating ML models as production-grade software.

## 🏗️ Architecture & Workflow
Whenever a developer submits a Pull Request modifying the model architecture or hyperparameters:
1. **GitHub Actions** automatically triggers a cloud runner.
2. Dependencies (`scikit-learn`, `matplotlib`) are provisioned with optimized caching.
3. The training script executes, evaluating data and generating performance artifacts.
4. **Continuous Machine Learning (CML)** automatically posts a live performance report (Accuracy metrics & visual Confusion Matrix) directly into the PR comments for reviewer sign-off.

## 🛠️ Tech Stack
- **Modeling:** Python, Scikit-Learn
- **Automation:** GitHub Actions, CML (Iterative.ai)
