# ⚡ Quick Start Guide

Get up and running with the POC Early Warning System in under 5 minutes!

## 🚀 For the Impatient

```bash
# Install UV (ultra-fast package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh  # macOS/Linux
# OR
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"  # Windows

# Clone and setup
git clone https://github.com/rjwdata/poc-early-warning.git
cd poc-early-warning
make install

# Run the app
make run-app
```

🎉 Done! The app will open at http://localhost:8501

---

## 📋 Step-by-Step Guide

### 1️⃣ Install UV (One-time setup)

**macOS / Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**Verify installation:**
```bash
uv --version
```

### 2️⃣ Clone the Repository

```bash
git clone https://github.com/rjwdata/poc-early-warning.git
cd poc-early-warning
```

### 3️⃣ Install Dependencies

**Option A: Using Make (Easiest)**
```bash
make install-dev
```

**Option B: Using UV directly**
```bash
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```

**Option C: Traditional pip**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -e .
```

### 4️⃣ Run the Application

**Prediction App:**
```bash
make run-app
# OR
uv run streamlit run app_pred.py
```

**Reports Viewer:**
```bash
make run-reports
# OR
uv run streamlit run app.py
```

---

## 🎯 Common Tasks

### Make a Prediction

1. Open http://localhost:8501
2. Enter student information in the sidebar
3. Click "🚀 Run Prediction Model"
4. View results and recommendations

### Train the Model

```bash
make train
# OR
uv run python src/components/data_ingestion_eda.py
```

### Run Tests

```bash
make test
# OR
uv run pytest tests/
```

### Format Code

```bash
make format
# OR
uv run black src/
uv run ruff check --fix src/
```

### View All Commands

```bash
make help
```

---

## 🆘 Troubleshooting

### UV not found

**Fix:** Make sure UV is in your PATH
```bash
# Add to ~/.bashrc or ~/.zshrc
export PATH="$HOME/.cargo/bin:$PATH"
```

### Module not found errors

**Fix:** Ensure virtual environment is activated
```bash
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### Port 8501 already in use

**Fix:** Kill the process or use a different port
```bash
# Kill existing process
lsof -ti:8501 | xargs kill -9  # macOS/Linux
# OR run on different port
uv run streamlit run app_pred.py --server.port 8502
```

### Dependencies out of sync

**Fix:** Reinstall dependencies
```bash
make clean
make install-dev
```

---

## 📚 Next Steps

- Read the full [README](README.md)
- Learn about [UV in detail](UV_GUIDE.md)
- Check the [Usage examples](README.md#usage)
- Explore the [API documentation](README.md#usage)

---

## 🎓 Learning Resources

### Project Documentation
- [README.md](README.md) - Complete project documentation
- [UV_GUIDE.md](UV_GUIDE.md) - Detailed UV usage guide
- [pyproject.toml](pyproject.toml) - Project configuration

### External Resources
- [UV Documentation](https://github.com/astral-sh/uv)
- [Streamlit Docs](https://docs.streamlit.io)
- [XGBoost Guide](https://xgboost.readthedocs.io)

---

## 💡 Pro Tips

### Use Make Commands

```bash
make run-app      # Instead of: uv run streamlit run app_pred.py
make test         # Instead of: uv run pytest tests/
make format       # Instead of: uv run black src/ && uv run ruff...
```

### Enable Auto-reload

```bash
make dev  # Streamlit auto-reloads on file changes
```

### Check Project Info

```bash
make info  # Shows Python version, packages, etc.
```

### Generate requirements.txt

```bash
make requirements  # Creates requirements.txt from pyproject.toml
```

---

## 🔥 One-Liners

**Complete setup:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh && \
git clone https://github.com/rjwdata/poc-early-warning.git && \
cd poc-early-warning && make install-dev && make run-app
```

**Quick development session:**
```bash
git pull && make install-dev && make format && make test && make run-app
```

**Clean slate:**
```bash
make clean && rm -rf .venv && uv venv && make install-dev
```

---

## ⌨️ Keyboard Shortcuts

When running Streamlit app:

- `Ctrl/Cmd + R` - Rerun the app
- `Ctrl/Cmd + C` - Stop the server
- `Ctrl/Cmd + K` - Clear cache

---

## 🚦 Status Indicators

✅ **Green** - Low risk / Recommended
🟡 **Yellow** - Medium risk / Optional
🔴 **Red** - High risk / Required action

---

Need help? Open an [issue](https://github.com/rjwdata/poc-early-warning/issues) or check the [discussions](https://github.com/rjwdata/poc-early-warning/discussions)!

**Happy predicting! 🎉**
