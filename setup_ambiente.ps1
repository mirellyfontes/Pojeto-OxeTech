# setup_ambiente.ps1
# Roda uma vez na raiz do projeto (Pojeto OxeTech) para deixar tudo pronto
# no VSCode: venv com Python 3.12, dependencias instaladas e kernel do
# Jupyter registrado com nome fixo "triagem-ocular".
#
# Como rodar:
#   1. Abra o terminal do VSCode na pasta raiz do projeto.
#   2. Se der erro de "execution policy", rode antes:
#        Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
#   3. Rode:  .\setup_ambiente.ps1

$ErrorActionPreference = "Stop"

Write-Host "== 1. Verificando Python 3.12 instalado ==" -ForegroundColor Cyan
$pyCheck = & py -0 2>$null
if ($pyCheck -notmatch "3\.12") {
    Write-Host "Python 3.12 nao encontrado. Baixe em:" -ForegroundColor Red
    Write-Host "https://www.python.org/downloads/release/python-3128/"
    exit 1
}

Write-Host "== 2. Removendo venv antiga (se existir) ==" -ForegroundColor Cyan
if (Test-Path ".venv") {
    Remove-Item -Recurse -Force ".venv"
}

Write-Host "== 3. Criando venv nova com Python 3.12 ==" -ForegroundColor Cyan
py -3.12 -m venv .venv

Write-Host "== 4. Instalando dependencias dentro da venv ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip install ipykernel

Write-Host "== 5. Registrando o kernel Jupyter 'triagem-ocular' ==" -ForegroundColor Cyan
.\.venv\Scripts\python.exe -m ipykernel install --user --name=triagem-ocular --display-name="Python (triagem-ocular)"

Write-Host ""
Write-Host "==================================================" -ForegroundColor Green
Write-Host " Pronto! Agora, no VSCode:" -ForegroundColor Green
Write-Host " 1. Abra 01_eda_baseline.ipynb ou 02_treino_modelo.ipynb"
Write-Host " 2. Clique no seletor de kernel (canto superior direito)"
Write-Host " 3. Escolha 'Python (triagem-ocular)'"
Write-Host " 4. Rode as celulas normalmente"
Write-Host "==================================================" -ForegroundColor Green
