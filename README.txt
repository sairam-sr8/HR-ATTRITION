Employee Attrition Predictor - Installation & Usage Guide
========================================================

This guide helps you set up and run the Employee Attrition Predictor app on any Windows PC.

REQUIREMENTS:
- Python 3.8 or above (https://www.python.org/downloads/)
- All files in this folder: streamlit_app.py, HR-Employee-Attrition.csv, requirements.txt

STEP-BY-STEP INSTALLATION:
1. Copy all files (including HR-Employee-Attrition.csv and streamlit_app.py) to a folder on your new PC.
2. Open Windows Command Prompt (cmd) or PowerShell.
3. Navigate to your project folder. Example:
   cd "C:\path\to\your\folder"
4. (Optional but recommended) Create a virtual environment:
   python -m venv venv
   venv\Scripts\activate
5. Install required Python packages:
   pip install -r requirements.txt
6. Run the Streamlit app:
   streamlit run streamlit_app.py
7. The app will open in your browser (usually at http://localhost:8501)

TROUBLESHOOTING:
- If you get errors about missing modules, make sure you installed requirements.txt.
- If Python is not recognized, ensure it is added to your PATH during installation.

That's it! You can now use the Employee Attrition Predictor web app on your new PC.
