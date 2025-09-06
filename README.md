# Auto Insights  Excel/CSV

This Streamlit app lets you upload an Excel/CSV dataset and automatically:
- Profile columns (types, missingness, basic stats)
- Generate informative charts (matplotlib)
- Produce natural-language **insights**
- Run **clustering** and **anomaly detection**
- Optionally train a supervised model (classification/regression) if you select a target

## Quick Start

1) Create and activate a virtual environment (optional but recommended)
```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

2) Install requirements
```bash
pip install -r requirements.txt
```

3) Run the app
```bash
streamlit run streamlit_app.py
```

4) Open the local URL shown (usually http://localhost:8501) and upload your Excel/CSV.
