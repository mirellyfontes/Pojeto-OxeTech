"""
API de inferência para o modelo de predição de sinistros a partir da frota veicular.

Rodar localmente:
    uvicorn api.main:app --reload

Documentação automática (OpenAPI):
    http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel, Field
import joblib
import pandas as pd
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent / "modelo_final.pkl"

app = FastAPI(
    title="API — Predição de Sinistros de Trânsito em Alagoas",
    description=(
        "Prevê o número esperado de sinistros de trânsito a partir de dados "
        "de frota veicular por bairro/mês. Projeto Final Integrador (AV2) — "
        "OxeTech Academy."
    ),
    version="1.0.0",
)

modelo = joblib.load(MODEL_PATH)


class EntradaPredicao(BaseModel):
    frota_total: int = Field(..., description="Frota total de veículos no bairro")
    motos: int = Field(..., description="Número de motocicletas registradas")
    automoveis: int = Field(..., description="Número de automóveis registrados")
    mes: int = Field(..., ge=1, le=12, description="Mês de referência (1-12)")
    ano: int = Field(..., description="Ano de referência")


class SaidaPredicao(BaseModel):
    sinistros_previstos: float


@app.get("/")
def raiz():
    return {"status": "ok", "mensagem": "API rodando. Veja /docs para a documentação."}


@app.post("/prever", response_model=SaidaPredicao)
def prever(entrada: EntradaPredicao):
    dados = pd.DataFrame([entrada.dict()])
    pred = modelo.predict(dados)[0]
    return SaidaPredicao(sinistros_previstos=round(float(pred), 2))
