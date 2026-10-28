# Patient Admission Analytics System
**Academic Project | SDG 3: Good Health and Well-being**

An interactive healthcare analytics dashboard built with Python, pandas, Plotly and Streamlit. It explores admission trends, outcomes, age distribution, length of stay, clinical indicators, missing data and duplicate rows.

## Features
- Interactive filters for date range, age, gender, outcome and admission type
- KPI cards for admission count, average age, average length of stay and recorded expiry outcomes
- Monthly admission trend, outcome composition, age distribution and length-of-stay charts
- Demographic and outcome comparisons
- Descriptive clinical indicator statistics and correlation heatmap
- Data-quality summary, missing-value chart and filtered CSV export
- Optional pollution-data trend if the local pollution CSV is present
- Synthetic demo mode so the app opens even before local CSV files are added

## Important data and privacy note
The app is designed to read local CSVs from the `data/` folder. The raw CSV files are intentionally excluded from Git tracking. **Do not commit patient-level records or identifiers to a public repository.** Keep the dataset on your own computer and add it to `data/` locally. The dashboard hides common record identifier columns in its table preview, but raw files must still remain private.

## Dataset setup
Download the HDHI dataset from the dataset source you used. Place these files inside `data/`:
- `HDHI Admission data.csv` (required for real-data mode)
- `HDHI Pollution Data.csv` (optional)
- `HDHI Mortality Data.csv` (optional)

The `table_headings.csv` file can be kept locally for reference but is not required to run the dashboard.

## Run locally on Windows
1. Install Python 3.12 and VS Code.
2. Open this folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:
   ```powershell
   py -3.12 -m venv .venv
   ```
5. Activate it:
   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```
6. Install dependencies:
   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```
7. Put your CSV files in `data/`.
8. Start the dashboard:
   ```powershell
   streamlit run app.py
   ```

If PowerShell blocks environment activation, run this instead:
```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```

## GitHub upload
Before pushing:
1. Confirm `.gitignore` exists.
2. Run `git status` and check that no `.csv` dataset is listed.
3. Upload only source code, documentation and requirements. Do not publish raw patient-level files.

Suggested repository description:
> Interactive patient admission analytics dashboard using Python, Streamlit, pandas and Plotly, aligned with SDG 3.

## Deployment
Streamlit Community Cloud can deploy this repository when the code and `requirements.txt` are on GitHub. For a public deployment, use only data you are permitted to publish. Since the actual dataset is excluded from the repository, the deployed app will show synthetic demo data unless a safe, authorized dataset is configured for deployment.

## Limitations
- This is an educational analytics prototype, not a clinical decision-support system.
- Date parsing can be ambiguous in source files; verify date quality before making time-based claims.
- The app does not infer causal relationships between pollution and outcomes.
- Outcome percentages are descriptive and should not be used to evaluate clinical quality without additional context.
