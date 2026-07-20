"""
API de inferência para triagem de doenças oculares a partir de imagens de retina.

IMPORTANTE: esta API é uma ferramenta de APOIO À TRIAGEM, não um diagnóstico
médico. O resultado deve sempre ser confirmado por um oftalmologista — ver
docs/notas_eticas.md.

Rodar localmente:
    uvicorn api.main:app --reload

Documentação automática (OpenAPI):
    http://localhost:8000/docs
"""

import io
import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image
from pydantic import BaseModel

MODEL_PATH = Path(__file__).resolve().parent / "modelo_final.keras"
CLASSES_PATH = Path(__file__).resolve().parent / "class_names.json"
IMG_SIZE = (224, 224)

DESCRICAO_CLASSES = {
    "normal": "Nenhum sinal das condições avaliadas detectado.",
    "cataract": "Sinais compatíveis com catarata.",
    "glaucoma": "Sinais compatíveis com glaucoma.",
    "diabetic_retinopathy": "Sinais compatíveis com retinopatia diabética.",
}

app = FastAPI(
    title="API — Triagem de Doenças Oculares em Imagens de Retina",
    description=(
        "Classifica imagens de fundo de retina em 4 categorias (normal, "
        "catarata, glaucoma, retinopatia diabética) como AUXÍLIO à triagem. "
        "Não substitui avaliação de um oftalmologista. Projeto Final "
        "Integrador (AV2) — OxeTech Academy."
    ),
    version="2.0.0",
)

modelo = tf.keras.models.load_model(MODEL_PATH)
with open(CLASSES_PATH, "r", encoding="utf-8") as f:
    class_names = json.load(f)


class SaidaPredicao(BaseModel):
    classe_prevista: str
    descricao: str
    confianca: float
    probabilidades: dict
    aviso: str = (
        "Resultado gerado por um modelo de IA para fins de TRIAGEM. Não é um "
        "diagnóstico médico. Procure um oftalmologista para avaliação."
    )


def preprocessar_imagem(bytes_imagem: bytes) -> np.ndarray:
    try:
        imagem = Image.open(io.BytesIO(bytes_imagem)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Arquivo enviado não é uma imagem válida.")

    imagem = imagem.resize(IMG_SIZE)
    array = tf.keras.utils.img_to_array(imagem)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    return np.expand_dims(array, axis=0)


@app.get("/")
def raiz():
    return {"status": "ok", "mensagem": "API rodando. Veja /docs para a documentação."}


@app.post("/prever", response_model=SaidaPredicao)
async def prever(arquivo: UploadFile = File(..., description="Imagem de fundo de retina (jpg/png)")):
    if arquivo.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Envie um arquivo de imagem (jpg ou png).")

    bytes_imagem = await arquivo.read()
    entrada = preprocessar_imagem(bytes_imagem)

    probs = modelo.predict(entrada, verbose=0)[0]
    indice_previsto = int(np.argmax(probs))
    classe_prevista = class_names[indice_previsto]

    return SaidaPredicao(
        classe_prevista=classe_prevista,
        descricao=DESCRICAO_CLASSES.get(classe_prevista, ""),
        confianca=round(float(probs[indice_previsto]), 4),
        probabilidades={c: round(float(p), 4) for c, p in zip(class_names, probs)},
    )
