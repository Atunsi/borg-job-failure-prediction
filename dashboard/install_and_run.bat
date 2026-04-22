@echo off
echo Installing all required dependencies for the dashboard...
python -m pip install -r requirements.txt
echo.
echo Installation complete! Starting the dashboard...
python -m streamlit run streamlitdash.py
pause
