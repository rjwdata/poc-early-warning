# 🚀 UV Package Manager Guide

This guide explains how to use [UV](https://github.com/astral-sh/uv) with the POC Early Warning System project.

## What is UV?

UV is an extremely fast Python package installer and resolver written in Rust by Astral (creators of Ruff). It's designed as a drop-in replacement for pip and pip-tools, offering:

- ⚡ **10-100x faster** installation than pip
- 🔒 **Deterministic** dependency resolution
- 🎯 **Better conflict resolution** and error messages
- 📦 **Cross-platform** consistency
- 🚀 **Standalone binary** (no Python required for installation)

## Installation

### macOS/Linux

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Windows

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

### Alternative: Using pip

```bash
pip install uv
```

## Quick Start

### 1. Create Virtual Environment

```bash
# Create a virtual environment
uv venv

# Activate it
# On Windows:
.venv\Scripts\activate
# On macOS/Linux:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
# Install project in editable mode
uv pip install -e .

# Or install with development dependencies
uv pip install -e ".[dev,test]"
```

### 3. Run Commands

```bash
# Run Python scripts
uv run python script.py

# Run installed commands
uv run streamlit run app_pred.py
```

## Common UV Commands

### Package Installation

```bash
# Install a single package
uv pip install pandas

# Install multiple packages
uv pip install pandas numpy scikit-learn

# Install from requirements.txt
uv pip install -r requirements.txt

# Install from pyproject.toml
uv pip install -e .
uv pip install -e ".[dev]"  # with optional dependencies
```

### Package Management

```bash
# List installed packages
uv pip list

# Show package details
uv pip show pandas

# Uninstall a package
uv pip uninstall pandas

# Update a package
uv pip install --upgrade pandas

# Update all packages
uv pip install --upgrade -r requirements.txt
```

### Virtual Environment

```bash
# Create virtual environment
uv venv

# Create with specific Python version
uv venv --python 3.10

# Create in custom directory
uv venv my-venv
```

### Running Commands

```bash
# Run command in virtual environment (no activation needed)
uv run python script.py
uv run pytest
uv run streamlit run app.py

# Run with specific Python version
uv run --python 3.10 python script.py
```

## Using Make Commands

We've created convenient Make commands that use UV under the hood:

```bash
# Show all available commands
make help

# Setup and installation
make setup-uv       # Install UV
make install        # Install project dependencies
make install-dev    # Install with dev dependencies
make venv           # Create virtual environment

# Running applications
make run-app        # Run prediction app
make run-reports    # Run reports viewer
make train          # Train the model

# Development
make test           # Run tests
make lint           # Run linters
make format         # Format code
make clean          # Clean build artifacts

# Utilities
make update         # Update dependencies
make requirements   # Generate requirements.txt
make info           # Show project info
```

## Project Configuration

### pyproject.toml

Modern Python projects use `pyproject.toml` for configuration. Our project includes:

```toml
[project]
name = "poc-early-warning"
version = "0.1.0"
dependencies = [
    "streamlit==1.39.0",
    "xgboost==2.1.1",
    # ... other dependencies
]

[project.optional-dependencies]
dev = [
    "black>=24.0.0",
    "pytest>=8.0.0",
    # ... dev tools
]
```

### .python-version

UV automatically respects `.python-version` files:

```
3.10
```

This ensures everyone uses the same Python version.

## Workflow Examples

### Starting a New Development Session

```bash
# 1. Clone the repository
git clone https://github.com/rjwdata/poc-early-warning.git
cd poc-early-warning

# 2. Install UV (if needed)
make setup-uv

# 3. Install everything
make install-dev

# 4. Run the application
make run-app
```

### Daily Development

```bash
# Activate virtual environment
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Make code changes...

# Format and lint
make format
make lint

# Run tests
make test

# Run application
make run-app
```

### Adding New Dependencies

```bash
# Add to pyproject.toml under [project.dependencies]
# Then install:
uv pip install -e .

# Or install directly:
uv pip install new-package

# Update pyproject.toml to reflect the change
```

### Updating Dependencies

```bash
# Update all dependencies to latest compatible versions
make update

# Or manually:
uv pip install --upgrade -e ".[dev,test]"
```

## Troubleshooting

### UV Command Not Found

```bash
# Add UV to your PATH
# Add to ~/.bashrc, ~/.zshrc, or equivalent:
export PATH="$HOME/.cargo/bin:$PATH"

# On Windows, the installer usually handles this automatically
```

### Virtual Environment Issues

```bash
# Remove and recreate virtual environment
rm -rf .venv
uv venv
uv pip install -e ".[dev]"
```

### Dependency Conflicts

```bash
# UV provides better error messages than pip
# Read the output carefully - it usually suggests solutions

# Clear UV cache if needed
uv cache clean
```

### Python Version Issues

```bash
# Check Python version
python --version

# UV can manage Python versions
uv python install 3.10
uv venv --python 3.10
```

## Performance Comparison

### Installation Speed

```bash
# pip (traditional)
time pip install -r requirements.txt
# ~60 seconds

# UV (modern)
time uv pip install -r requirements.txt
# ~3 seconds (20x faster!)
```

### Dependency Resolution

```bash
# pip often gives cryptic error messages
# UV provides clear, actionable errors with suggestions
```

## Migration from pip

### Converting requirements.txt to pyproject.toml

UV works with both! But pyproject.toml is more modern:

```bash
# Keep using requirements.txt
uv pip install -r requirements.txt

# Or use pyproject.toml
uv pip install -e .

# Generate requirements.txt from pyproject.toml
make requirements
```

### Existing Virtual Environments

You can use UV with existing venvs:

```bash
# Activate existing venv
source venv/bin/activate

# Use UV for installations
uv pip install pandas
```

## Best Practices

### 1. Use pyproject.toml

Modern Python projects should use `pyproject.toml` for configuration.

### 2. Pin Python Version

Use `.python-version` file to ensure consistency.

### 3. Use Optional Dependencies

Organize dependencies into groups:

```toml
[project.optional-dependencies]
dev = ["black", "pytest"]
test = ["pytest", "pytest-cov"]
docs = ["mkdocs"]
```

Install with: `uv pip install -e ".[dev,test]"`

### 4. Use Make Commands

Abstract common commands in Makefile for consistency.

### 5. Don't Commit Virtual Environments

Always add to `.gitignore`:

```
.venv/
venv/
.uv/
```

## Advanced Features

### Lock Files

UV can generate lock files for reproducible builds:

```bash
# Generate lock file
uv pip compile pyproject.toml -o requirements.lock

# Install from lock file
uv pip install -r requirements.lock
```

### Pip Compatibility

UV is designed as a drop-in replacement:

```bash
# These work the same:
pip install pandas
uv pip install pandas

pip install -r requirements.txt
uv pip install -r requirements.txt
```

### Integration with CI/CD

UV is perfect for CI/CD due to speed:

```yaml
# GitHub Actions example
- name: Install UV
  run: curl -LsSf https://astral.sh/uv/install.sh | sh

- name: Install dependencies
  run: uv pip install -e ".[test]"

- name: Run tests
  run: uv run pytest
```

## Resources

- **UV Documentation**: https://github.com/astral-sh/uv
- **Astral (makers of UV)**: https://astral.sh
- **Python Packaging**: https://packaging.python.org
- **pyproject.toml spec**: https://peps.python.org/pep-0621/

## Getting Help

```bash
# Show UV help
uv --help
uv pip --help

# Show project help
make help

# Check project info
make info
```

## Summary

UV makes Python package management **faster**, **more reliable**, and **easier**. Key benefits:

1. ⚡ **Speed**: 10-100x faster than pip
2. 🔒 **Reliability**: Better dependency resolution
3. 🎯 **User-friendly**: Clear error messages
4. 📦 **Modern**: Works with pyproject.toml
5. 🚀 **Easy**: Drop-in pip replacement

Start using UV today:

```bash
make setup-uv
make install-dev
make run-app
```

Happy coding! 🎉
