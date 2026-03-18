@echo off
echo 🏋️ Exercise Analyzer Setup for Windows
echo ======================================

:: Check Python version
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python not found. Please install Python 3 and add it to PATH.
    pause
    exit /b
)

:: Create virtual environment if it doesn't exist
if not exist "venv\" (
    echo.
    echo 🐍 Creating Python virtual environment...
    python -m venv venv
)

:: Activate virtual environment
call venv\Scripts\activate.bat

:: Upgrade pip
echo.
echo ⬆️ Upgrading pip...
python -m pip install --upgrade pip

:: Install Python packages
echo.
echo 📚 Installing Python packages...
pip install -r requirements.txt
:: Adding django-cors-headers to handle React requests
pip install django-cors-headers

:: Run migrations
echo.
echo 🗄️ Setting up database...
python manage.py makemigrations
python manage.py migrate

:: Create uploads directory
if not exist "uploaded_videos\" mkdir uploaded_videos

echo.
echo ✅ Setup complete!
echo.
echo To run the Backend app:
echo   1. Activate virtual environment: venv\Scripts\activate
echo   2. Start the server: python manage.py runserver 0.0.0.0:8000
echo.
echo To run the Frontend app:
echo   1. cd frontend
echo   2. npm install
echo   3. npm run dev
echo.
pause
