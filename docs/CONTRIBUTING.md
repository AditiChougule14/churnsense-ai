
# Contributing to ChurnSense AI

> **Predict Churn. Understand Why. Drive Retention.**

Thank you for your interest in contributing to **ChurnSense AI**! This document provides guidelines and instructions for contributing to the project.

**Author / Maintainer:** Aditi Chougule  
**Repository:** [churnsense-ai](https://github.com/aditichougule/churnsense-ai)

---

## Code of Conduct

- Be respectful and inclusive of all contributors.
- Welcome people of all backgrounds and experience levels.
- Focus on constructive collaboration and code quality.
- Report issues and ask questions constructively.

---

## How to Contribute

### 1. Report Issues

Found a bug or have a suggestion? Please create an issue with:

- A clear, descriptive title.
- Detailed description of the problem or feature request.
- Steps to reproduce (for bugs).
- Expected vs. actual behavior.
- Environment details (OS, Python version, dependencies).

### 2. Fork and Clone

Fork the repository on GitHub, then clone your fork:

```bash
git clone https://github.com/aditichougule/churnsense-ai.git
cd churnsense-ai
```

### 3. Create a Feature Branch

```bash
git checkout -b feature/your-feature-name

# Or for bug fixes:
git checkout -b fix/bug-description
```

### 4. Make Your Changes

#### Code Style & Standards

- Follow PEP 8 guidelines.
- Use 4 spaces for indentation.
- Keep line lengths clean and readable.
- Add descriptive variable and function names.
- Provide type hints and docstrings for functions and classes.
- Preserve existing pipelines, XAI logic, and data validation guardrails.

### 5. Test Your Changes

```bash
# Run unit and integration tests
pytest -q

# Run specific test modules
pytest tests/test_explain.py -v
pytest tests/test_automl.py -v
pytest tests/test_app_integration.py -v
```

### 6. Commit and Push

```bash
git add .
git commit -m "feat: describe your change clearly"
git push origin feature/your-feature-name
```

### 7. Create a Pull Request

1. Navigate to the GitHub repository and open a Pull Request.
2. Provide a clear summary of changes, motivation, and test coverage.
3. Verify that all automated tests and syntax checks pass.

---

## Areas for Contribution

- **Models & Algorithms:** Evaluate gradient boosting (LightGBM, XGBoost, CatBoost).
- **Explainable AI (XAI):** Add additional SHAP summary visualizations or counterfactual explanations.
- **AutoML Enhancements:** Support multi-class targets and advanced automated feature generation.
- **Documentation:** Expand guides, add case studies, or improve code walkthroughs.

---

## Questions & Contact

For questions or feedback, please open an issue on the repository:

[https://github.com/aditichougule/churnsense-ai/issues](https://github.com/aditichougule/churnsense-ai/issues)

Thank you for contributing to **ChurnSense AI**!
