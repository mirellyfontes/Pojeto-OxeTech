#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convertido a partir de notebooks/01_eda_baseline.ipynb
# Rodar: python notebooks/01_eda_baseline.py

# # EDA — Triagem de Doenças Oculares em Imagens de Retina
#
# Antes de rodar: baixe o dataset do Kaggle e organize em `data/raw/dataset/<classe>/` (ver README).
#
# Importante (VSCode): no canto superior direito do notebook, selecione o kernel `Python 3` do `venv` do projeto # (não um kernel de SQL ou de outra linguagem). Se o VSCode não oferecer o kernel Python, rode `pip install ipykernel` dentro do `venv` ativado e reabra o notebook.

from pathlib import Path
import sys


def encontrar_raiz_projeto(start: Path) -> Path:
    """Sobe nos diretórios a partir de onde o notebook está rodando até 
    achar a raiz do projeto (a pasta que contém src/load_data.py). Assim o 
    notebook funciona independente de ser aberto direto da pasta notebooks/ 
    ou da raiz do projeto."""
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "load_data.py").exists():
            return candidate
    raise FileNotFoundError(
        f"Não foi possível localizar a raiz do projeto a partir de: {start}"
    )


ROOT = encontrar_raiz_projeto(Path.cwd().resolve())
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import matplotlib.pyplot as plt

try:
    import tensorflow as tf
except ModuleNotFoundError:
    tf = None
    print("TensorFlow não está instalado neste ambiente.")
    print("Instale as dependências com: pip install -r requirements.txt")

from load_data import contar_imagens_por_classe, carregar_datasets, DATASET_DIR

print("Raiz do projeto:", ROOT)
print("Dataset:", DATASET_DIR)

# ## 1. Quantas imagens por classe?
#
# O dataset é anunciado como praticamente balanceado (~1000 por classe) — vamos confirmar.

contagem = contar_imagens_por_classe()
contagem

plt.figure(figsize=(6,4))
plt.bar(contagem.keys(), contagem.values()) # type: ignore
plt.title("Imagens por classe")
plt.ylabel("Quantidade")
plt.xticks(rotation=20)
plt.tight_layout()
plt.show()

# ## 2. Ver exemplos de cada classe
#
# Sempre vale olhar as imagens de verdade antes de treinar qualquer coisa — ajuda a notar diferenças de qualidade/iluminação entre classes que podem virar viés (ver docs/notas_eticas.md).

import random

classes = sorted(contagem.keys())
fig, axs = plt.subplots(len(classes), 4, figsize=(12, 3 * len(classes)))

for i, classe in enumerate(classes):
    pasta = DATASET_DIR / classe
    arquivos = [f for f in pasta.iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    amostras = random.sample(arquivos, min(4, len(arquivos)))
    for j, caminho in enumerate(amostras):
        img = plt.imread(caminho)
        axs[i, j].imshow(img)
        axs[i, j].axis("off")
        if j == 0:
            axs[i, j].set_ylabel(classe)
    axs[i, 0].set_title(classe, loc="left", fontsize=10)

plt.tight_layout()
plt.show()

# ## 3. Carregar os datasets (treino/validação/teste)
#
# Usa o mesmo split que o `src/model.py` vai usar no treino — conferimos aqui só para ver o tamanho de cada batch e a ordem das classes.

if tf is None:
    print("Não foi possível carregar os datasets porque o TensorFlow não está instalado.")
else:
    treino_ds, val_ds, teste_ds, class_names = carregar_datasets()

    print("Classes:", class_names)
    print("Batches -> treino:", tf.data.experimental.cardinality(treino_ds).numpy())
    print("Batches -> validação:", tf.data.experimental.cardinality(val_ds).numpy())
    print("Batches -> teste:", tf.data.experimental.cardinality(teste_ds).numpy())

# ## 4. Conferir um batch de imagens já carregado

if tf is None:
    print("Não foi possível mostrar o batch porque o TensorFlow não está instalado.")
else:
    for imagens, rotulos in treino_ds.take(1):
        fig, axs = plt.subplots(1, 6, figsize=(15, 3))
        for i in range(6):
            axs[i].imshow(imagens[i].numpy().astype("uint8"))
            axs[i].set_title(class_names[rotulos[i]])
            axs[i].axis("off")
        plt.tight_layout()
        plt.show()

# ## 5. Próximos passos
#
# O treino de fato (transfer learning com MobileNetV2 + fine-tuning) está em `src/model.py`, para rodar fora do notebook:
#
# ```bash
# python src/model.py
# ```
