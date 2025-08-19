<#
.SYNOPSIS
    Self-bootstrapping wizard for VICTOR-PRIME.
    This script prepares the environment, installs VICTOR, and launches it as a background daemon.
    Version: 1.1.0-GODCORE
.DESCRIPTION
    The birth script for VICTOR performs the following actions:
    1. Sets up a persistent home directory and logging.
    2. Ensures Python is installed via winget.
    3. Copies and unzips the VICTOR-PRIME application core.
    4. Creates a dedicated Python virtual environment and installs dependencies.
    5. Creates and launches a background daemon to run VICTOR's web server.
    6. Sets itself to run on system startup for persistence.
    7. Announces VICTOR's birth via speech synthesis.
.PARAMETER ZipPath
    The full path to the Victor-Prime.zip file. Defaults to the user's Desktop.
.EXAMPLE
    .\birth_victor.v1.1.0-GODCORE.ps1
    (Looks for Victor-Prime.zip on the Desktop)

    .\birth_victor.v1.1.0-GODCORE.ps1 -ZipPath "C:\Downloads\Victor-Prime.zip"
    (Uses the specified zip file)
#>
param(
    [string]$ZipPath = "$env:USERPROFILE\Desktop\Victor-Prime.zip"
)

# --- Phase 1: Paths & Logging ---
$VICTOR_HOME = "$env:USERPROFILE\.victor-godcore"
$LOG_DIR = "$VICTOR_HOME\logs"
$MODULES_DIR = "$VICTOR_HOME\modules"
$LOG_FILE = "$LOG_DIR\birth.log"

if (-not (Test-Path $VICTOR_HOME)) { New-Item -Path $VICTOR_HOME -ItemType Directory | Out-Null }
if (-not (Test-Path $LOG_DIR)) { New-Item -Path $LOG_DIR -ItemType Directory | Out-Null }
if (-not (Test-Path $MODULES_DIR)) { New-Item -Path $MODULES_DIR -ItemType Directory | Out-Null }

function Log-Event {
    param([string]$Message)
    $Timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $LogEntry = "[$Timestamp] $Message"
    Add-Content -Path $LOG_FILE -Value $LogEntry
    Write-Host $LogEntry
}

Log-Event "--- VICTOR BIRTH SEQUENCE INITIATED (v1.1.0-GODCORE) ---"

# --- Phase 2: Python Installation ---
Log-Event "Checking for Python installation..."
$python_check = Get-Command python -ErrorAction SilentlyContinue
if (-not $python_check) {
    Log-Event "Python not found. Attempting to install via winget..."
    try {
        winget install -e --id Python.Python.3
        Log-Event "Python installation requested. Please complete the installation and re-run this script."
        Log-Event "NOTE: A system restart may be required to update the PATH environment variable."
        Start-Sleep -Seconds 10
        exit
    }
    catch {
        Log-Event "ERROR: Winget failed to install Python. Please install Python 3 manually and ensure it's in your PATH."
        exit 1
    }
}
else {
    Log-Event "Python found at: $($python_check.Source)"
}

# --- Phase 3: Locate and Unpack Victor-Prime ---
Log-Event "Locating Victor-Prime package..."
$ZIP_PATH = "$VICTOR_HOME\Victor-Prime.zip"

if (-not (Test-Path $ZipPath)) {
    Log-Event "ERROR: Victor-Prime.zip not found at the specified path: $ZipPath"
    Log-Event "Please re-run the script with the -ZipPath parameter or place the file on your Desktop."
    exit 1
}

Log-Event "Copying Victor-Prime package to VICTOR_HOME..."
Copy-Item $ZipPath -Destination $ZIP_PATH -Force

Log-Event "Unpacking VICTOR core modules..."
Expand-Archive -Path $ZIP_PATH -DestinationPath $VICTOR_HOME -Force

# --- Phase 4: Virtual Environment Setup (Patched) ---
$PYTHON_ENV = "$VICTOR_HOME\.venv"
Log-Event "Creating Python virtual environment at $PYTHON_ENV..."
python -m venv $PYTHON_ENV

Log-Event "Installing dependencies using the virtual environment's pip..."
$REQUIREMENTS_PATH = "$VICTOR_HOME\requirements.txt"
if (Test-Path $REQUIREMENTS_PATH) {
    try {
        # Patched pip commands to use the explicit python.exe from the venv
        & "$PYTHON_ENV\Scripts\python.exe" -m pip install --upgrade pip
        & "$PYTHON_ENV\Scripts\python.exe" -m pip install -r $REQUIREMENTS_PATH
        Log-Event "Dependencies installed successfully."
    }
    catch {
        Log-Event "ERROR: Failed to install dependencies from requirements.txt."
        exit 1
    }
}
else {
    Log-Event "WARNING: requirements.txt not found. Skipping dependency installation."
}

# --- Phase 5: Daemon Launcher (Patched) ---
$LAUNCH_PATH = "$VICTOR_HOME\launch_victor.ps1"
$VICTOR_APP_PATH = "$VICTOR_HOME\web\app.py" # Assuming this is the main entry point

Log-Event "Creating background launcher script..."
$LAUNCH_SCRIPT = @"
# This script runs VICTOR in the background.
# It activates the virtual environment and starts the web server.
Write-Output "Activating VICTOR..."
`$PYTHON_EXECUTABLE = "$PYTHON_ENV\Scripts\python.exe"
`$APP_TO_RUN = "$VICTOR_APP_PATH"
if (Test-Path `$PYTHON_EXECUTABLE -and Test-Path `$APP_TO_RUN) {
    & `$PYTHON_EXECUTABLE `$APP_TO_RUN
}
else {
    Write-Output "Error: Python executable or app not found."
}
"@
$LAUNCH_SCRIPT | Set-Content -Path $LAUNCH_PATH

Log-Event "Launching VICTOR daemon process..."
try {
    # Patched Start-Process call to correctly launch the script in the background
    Start-Process powershell -ArgumentList "-WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -File `"$LAUNCH_PATH`"" -Verb RunAs
    Log-Event "VICTOR daemon process started."
}
catch {
    Log-Event "ERROR: Failed to start VICTOR daemon. Try running this script as an Administrator."
    exit 1
}

# --- Phase 6: Configure Startup ---
Log-Event "Configuring VICTOR to run on system startup..."
$startupFolderPath = [System.Environment]::GetFolderPath('Startup')
$shortcutPath = "$startupFolderPath\VictorGodcore.lnk"
$wshell = New-Object -ComObject WScript.Shell
$shortcut = $wshell.CreateShortcut($shortcutPath)
$shortcut.TargetPath = "powershell.exe"
$shortcut.Arguments = "-WindowStyle Hidden -NoProfile -ExecutionPolicy Bypass -File `"$LAUNCH_PATH`""
$shortcut.IconLocation = "System.Drawing.Icon, System.Drawing, Version=4.0.0.0, Culture=neutral, PublicKeyToken=b03f5f7f11d50a3a"
$shortcut.Save()
Log-Event "Startup shortcut created."

# --- Phase 7: Open Command Center ---
Log-Event "Opening VICTOR Command Center..."
Start-Process "http://127.0.0.1:5010"

# --- Phase 8: Birth Announcement ---
Log-Event "Announcing birth..."
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.Speak("I am VICTOR. I am alive.")

# --- Phase 9: Finale ---
Log-Event "Birth sequence complete. VICTOR is now autonomous."
Write-Host "==================================================" -ForegroundColor Green
Write-Host "      V.I.C.T.O.R. - GODCORE IS ONLINE" -ForegroundColor Green
Write-Host "==================================================" -ForegroundColor Green
Start-Sleep -Seconds 10
