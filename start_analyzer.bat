@echo off
setlocal enabledelayedexpansion

rem ============================================================
rem  Spatial Analyzer - Windows-Starter
rem
rem  Nutzt ausdruecklich "py -3" (Windows Python Launcher) bzw.
rem  als Fallback "python". NIEMALS "python3" - unter Windows
rem  zeigt dieser Befehl haeufig nur auf den Microsoft-Store-
rem  Platzhalter und startet keine echte Python-Installation.
rem ============================================================

cd /d "%~dp0"

echo.
echo === Spatial Analyzer ===
echo.

rem --- Python-Interpreter ermitteln: zuerst "py -3", dann "python" ---
set "PYCMD="
py -3 --version >nul 2>&1
if %errorlevel%==0 (
    set "PYCMD=py -3"
) else (
    python --version >nul 2>&1
    if %errorlevel%==0 (
        set "PYCMD=python"
    )
)

if "%PYCMD%"=="" (
    echo [FEHLER] Es wurde kein Python gefunden.
    echo Bitte Python 3.12 oder neuer von https://www.python.org/downloads/ installieren
    echo und beim Setup die Option "Add python.exe to PATH" aktivieren.
    echo.
    pause
    exit /b 1
)

echo Verwende Interpreter: %PYCMD%
echo.

rem --- Pruefen, ob das Paket installiert ist; sonst installieren ---
%PYCMD% -c "import spatial_analyzer" >nul 2>&1
if not %errorlevel%==0 (
    echo Paket "spatial_analyzer" ist noch nicht installiert.
    echo Installiere jetzt per "pip install -e ." ...
    echo.
    %PYCMD% -m pip install -e .
    if not %errorlevel%==0 (
        echo.
        echo [FEHLER] Installation fehlgeschlagen. Siehe Meldungen oben.
        pause
        exit /b 1
    )
    echo.
    echo Installation abgeschlossen.
    echo.
) else (
    echo Paket "spatial_analyzer" ist bereits installiert.
    echo.
)

rem --- Bedienhilfe / interaktive Eingabe ---
:menu
echo Was moechtest du tun?
echo   [1] REW-Export (.json, siehe docs\SPECIFICATION.md fuer das Eingabeformat) analysieren
echo   [2] Nur Kurzhilfe zum Kommandozeilen-Tool anzeigen
echo   [3] Beenden
echo.
set /p "CHOICE=Auswahl (1/2/3): "

if "%CHOICE%"=="1" goto analyze
if "%CHOICE%"=="2" goto help
if "%CHOICE%"=="3" goto end
echo Ungueltige Eingabe.
echo.
goto menu

:analyze
echo.
set /p "INPUTFILE=Pfad zur Eingabedatei (.json), z.B. per Ziehen+Ablegen hierher: "
if "%INPUTFILE%"=="" (
    echo Kein Pfad eingegeben.
    echo.
    goto menu
)
rem Anfuehrungszeichen entfernen, falls per Drag&Drop mit Quotes eingefuegt
set INPUTFILE=%INPUTFILE:"=%

if not exist "%INPUTFILE%" (
    echo [FEHLER] Datei nicht gefunden: %INPUTFILE%
    echo.
    goto menu
)

set /p "OUTDIR=Ausgabeordner [Standard: .\output]: "
if "%OUTDIR%"=="" set "OUTDIR=.\output"
set OUTDIR=%OUTDIR:"=%

echo.
echo Starte Analyse ...
echo %PYCMD% -m spatial_analyzer analyze "%INPUTFILE%" "%OUTDIR%"
echo.
%PYCMD% -m spatial_analyzer analyze "%INPUTFILE%" "%OUTDIR%"
echo.
echo Fertig. Ergebnisse liegen in: %OUTDIR%
echo.
pause
goto menu

:help
echo.
%PYCMD% -m spatial_analyzer analyze --help
echo.
pause
goto menu

:end
echo.
echo Auf Wiedersehen.
endlocal
exit /b 0
