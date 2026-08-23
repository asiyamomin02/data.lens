# DataLens

A Flask-based analytics dashboard for exploring uploaded CSV and Excel data through KPIs, charts, filters, statistics, and natural-language tools.

## Overview

DataLens lets users upload `.csv`, `.xlsx`, or `.xls` files and explore the active dataset in a browser dashboard. It automatically detects numeric, categorical, and date-like fields to generate suitable KPIs, charts, column statistics, data-quality indicators, and quick insights.

When a `GROQ_API_KEY` is configured, Groq is used to generate a short upload summary and as a fallback interpreter for questions and chart requests. The core calculations, filtering, and chart aggregation are performed locally with Pandas against the current uploaded or filtered dataset.

## Key Features

- Upload CSV and Excel files (`.csv`, `.xlsx`, `.xls`) by drag-and-drop or file picker.
- Automatically detect useful numeric, categorical, status, grouping, and date fields.
- Generate dataset-aware KPIs such as record count, total/average/minimum/maximum value, status win rate, group counts, and top groups when compatible columns exist.
- Show automatic insights based on the detected fields and current data.
- Apply categorical and date-range filters; KPIs, charts, statistics, table data, insights, and Data Quality Score refresh for the filtered dataset.
- View animated Chart.js visualisations, including bar, line, doughnut, and scatter charts when the dataset supports them.
- Use the **AI Chart Generator** to describe a bar, line, pie/doughnut, or scatter chart in natural language. It uses a deterministic local parser first and an optional Groq fallback for unsupported requests.
- Compare categorical values such as `Electronics vs Clothing` or compare compatible numeric fields.
- Ask dataset questions through **Ask Anything**. Common requests such as counts, missing values, value breakdowns, top/bottom results, aggregates, comparisons, and filtered record previews are calculated from the full active dataset.
- Review a read-only **Data Quality Score** that reports completeness, blank text cells, columns with missing values, and exact duplicate rows.
- Browse a searchable, sortable, paginated table preview of up to 300 rows returned by the dashboard.
- Review per-column numeric statistics: mean, median, minimum, maximum, total, and missing values.
- Export the current filtered dataset as CSV and export a summary report as PDF or TXT.
- Download rendered charts as PNG images.

## Tech Stack

| Area | Technologies |
| --- | --- |
| Backend | Python, Flask, Pandas, NumPy |
| AI/API integration | Groq Chat Completions API via `requests` (optional) |
| Reports and configuration | FPDF2, python-dotenv |
| Frontend | HTML, CSS, vanilla JavaScript |
| Charts | Chart.js 4.4.0 (loaded from CDN) |
| WSGI server dependency | Gunicorn |

## How It Works

1. Upload a CSV or Excel file.
2. The Flask backend stores the dataset for the browser session and identifies compatible numeric, categorical, and date fields.
3. DataLens generates KPIs, charts, insights, a Data Quality Score, a table preview, and column statistics.
4. Apply category or date filters to refresh the dashboard for the active subset of data.
5. Use **Ask Anything** for supported dataset questions or describe a chart in the **AI Chart Generator**.
6. Download the active filtered data, reports, or chart images when needed.

## Installation

git clone <repository-url>
cd csv-analyzer-web
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
# source .venv/bin/activate

pip install -r requirements.txt
python app.py

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SECRET_KEY=your_flask_session_secret

Then add a new heading before the folder tree:

```markdown
## Project Structure

csv-analyzer-web/
├── app.py                 # Flask app, analysis logic, API routes, and exports
├── requirements.txt       # Python dependencies
├── README.md
├── .gitignore
├── .env                   # Local environment variables; ignored by Git
└── templates/
    └── index.html         # Dashboard UI, styles, and browser-side logic
