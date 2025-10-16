# HOW TO INSTALL AND RUN

## Installation Steps

### 1. Change directory to the project
```bash
cd path/to/your/project
```

### 2. Create virtual environment
```bash
# Use Python 3
python3 -m venv venv

# Or use Python 3.12
python3.12 -m venv venv
```

### 3. Activate virtual environment
**On Windows:**
```bash
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
source venv/bin/activate
```

### 4. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 5. Run the app
```bash
python App.py
```

### 6. Access the app
Open your browser and go to:
```
http://127.0.0.1:7860
```

---

## Troubleshooting

### Check Python Version
```bash
python --version
```
If it shows 3.14, try to use `python3.12` or `python3.11` instead.

```bash
python3 App.py # instead of python App.py
```

### First Run Notes
- **First run will be slow** - the app downloads:
  - TinyLlama model (~2.2GB)
  - EasyOCR models for Swedish and English (~200MB total)
- **GPU recommended** for faster performance
- **Memory requirements**: At least 4-6GB free RAM
