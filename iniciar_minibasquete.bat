@echo off
rem Abre a Plataforma de Avaliacao do Minibasquete com duplo clique (Windows).
cd /d "%~dp0"
title Plataforma de Avaliacao do Minibasquete

set PY=
where python >nul 2>nul && set PY=python
if "%PY%"=="" where py >nul 2>nul && set PY=py -3
if "%PY%"=="" (
  echo.
  echo O Python nao esta instalado neste computador.
  echo Instale-o em https://www.python.org/downloads/ e marque "Add Python to PATH".
  echo Depois volte a abrir este ficheiro.
  echo.
  pause
  exit /b 1
)

echo A preparar a plataforma (so demora na primeira vez)...
%PY% -m pip install --quiet --disable-pip-version-check streamlit plotly pandas reportlab pypdf
if errorlevel 1 (
  echo.
  echo Nao foi possivel instalar os componentes. Verifique a ligacao a Internet.
  pause
  exit /b 1
)

echo.
echo A abrir a plataforma no browser. NAO feche esta janela enquanto a usar.
echo Para terminar, feche esta janela ou carregue em Ctrl+C.
echo.
%PY% -m streamlit run minibasket/app.py --server.headless false
pause
