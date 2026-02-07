# Contributing to Geocentric Orbit Prediction

Thank you for your interest in contributing to this project! This guide will help you get started.

## Development Setup

### Prerequisites
- Python 3.8 or higher
- Git

### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/Adidushi/stars.git
   cd stars
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Verify installation by running tests:
   ```bash
   cd geocentric_project
   python -m unittest discover tests
   ```

## Project Structure

```
geocentric_project/
├── stars_utils.py          # Core physics engine and utilities
├── config.py              # Configuration and constants
├── training/              # Training pipelines
│   └── train_inner_planets.py
├── analysis/              # Visualization and reporting
│   ├── solar_system_viz.py
│   └── generate_pdf_report.py
├── tests/                 # Unit tests
│   └── test_geocentric.py
├── data/                  # Generated datasets (CSV)
├── models/                # Trained models (.keras)
└── visualizations/        # Output plots (HTML)
```

## Code Style Guidelines

### Python Style
- Follow PEP 8 conventions
- Use type hints for function signatures
- Add docstrings to all public functions
- Maximum line length: 100 characters

### Documentation
- Update docstrings when modifying functions
- Keep README.md up-to-date with new features
- Add comments for complex physics calculations

### Testing
- Write tests for new features
- Ensure all tests pass before submitting PR
- Aim for high test coverage of core functionality

## Making Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them thoroughly

3. Run the test suite:
   ```bash
   python -m unittest discover tests
   ```

4. Commit with a descriptive message:
   ```bash
   git commit -m "Add feature: description of what you did"
   ```

5. Push to your fork and submit a pull request

## Pull Request Guidelines

- Provide a clear description of the changes
- Reference any related issues
- Include test coverage for new features
- Ensure all tests pass
- Keep PRs focused on a single feature or fix

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce the issue
- Expected vs. actual behavior
- Relevant error messages or logs

## Questions?

Feel free to open an issue for questions or discussions about the project.

---
*Thank you for contributing!*
