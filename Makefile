.PHONY: help install install-dev sync update clean test lint format run-app run-reports train

# Default target
help:
	@echo "POC Early Warning System - Make Commands"
	@echo ""
	@echo "Available commands:"
	@echo "  make install       - Install project dependencies with uv"
	@echo "  make install-dev   - Install project with development dependencies"
	@echo "  make sync          - Sync dependencies to match lock file"
	@echo "  make update        - Update dependencies to latest versions"
	@echo "  make clean         - Remove build artifacts and cache"
	@echo "  make test          - Run test suite"
	@echo "  make lint          - Run linters (ruff, black check, mypy)"
	@echo "  make format        - Format code with black and ruff"
	@echo "  make run-app       - Run prediction Streamlit app"
	@echo "  make run-reports   - Run reports viewer Streamlit app"
	@echo "  make train         - Train the ML model"
	@echo "  make setup-uv      - Install uv package manager"

# Install uv if not already installed
setup-uv:
	@echo "Checking for uv installation..."
	@command -v uv >/dev/null 2>&1 || { \
		echo "Installing uv..."; \
		curl -LsSf https://astral.sh/uv/install.sh | sh; \
	}
	@echo "uv is installed!"

# Install project dependencies
install:
	@echo "Installing dependencies with uv..."
	uv pip install -e .
	@echo "Installation complete!"

# Install with development dependencies
install-dev:
	@echo "Installing development dependencies with uv..."
	uv pip install -e ".[dev,test]"
	@echo "Development installation complete!"

# Sync dependencies
sync:
	@echo "Syncing dependencies..."
	uv pip sync
	@echo "Dependencies synced!"

# Update dependencies
update:
	@echo "Updating dependencies..."
	uv pip install --upgrade -e ".[dev,test]"
	@echo "Dependencies updated!"

# Clean build artifacts
clean:
	@echo "Cleaning build artifacts..."
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info
	rm -rf .pytest_cache/
	rm -rf .mypy_cache/
	rm -rf .ruff_cache/
	rm -rf htmlcov/
	rm -rf .coverage
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	find . -type f -name "*.pyo" -delete
	@echo "Clean complete!"

# Run tests
test:
	@echo "Running tests..."
	uv run pytest tests/
	@echo "Tests complete!"

# Run linters
lint:
	@echo "Running linters..."
	uv run ruff check src/
	uv run black --check src/
	uv run mypy src/
	@echo "Linting complete!"

# Format code
format:
	@echo "Formatting code..."
	uv run black src/
	uv run ruff check --fix src/
	@echo "Formatting complete!"

# Run prediction app
run-app:
	@echo "Starting prediction application..."
	uv run streamlit run app_pred.py

# Run reports viewer
run-reports:
	@echo "Starting reports viewer..."
	uv run streamlit run app.py

# Train model
train:
	@echo "Training model..."
	uv run python src/components/data_ingestion_eda.py
	@echo "Training complete!"

# Create virtual environment
venv:
	@echo "Creating virtual environment with uv..."
	uv venv
	@echo "Virtual environment created! Activate with: source .venv/bin/activate (Unix) or .venv\\Scripts\\activate (Windows)"

# Install pre-commit hooks
setup-hooks:
	@echo "Installing pre-commit hooks..."
	uv run pre-commit install
	@echo "Pre-commit hooks installed!"

# Run pre-commit on all files
pre-commit:
	@echo "Running pre-commit on all files..."
	uv run pre-commit run --all-files
	@echo "Pre-commit checks complete!"

# Generate requirements.txt from pyproject.toml (for compatibility)
requirements:
	@echo "Generating requirements.txt..."
	uv pip compile pyproject.toml -o requirements.txt
	@echo "requirements.txt generated!"

# Development server with auto-reload
dev:
	@echo "Starting development server with auto-reload..."
	uv run streamlit run app_pred.py --server.runOnSave=true

# Show project info
info:
	@echo "Project Information:"
	@echo "  Name: poc-early-warning"
	@echo "  Version: 0.1.0"
	@echo "  Python: $$(python --version)"
	@echo "  UV Version: $$(uv --version 2>/dev/null || echo 'Not installed')"
	@echo ""
	@echo "Installed packages:"
	@uv pip list
