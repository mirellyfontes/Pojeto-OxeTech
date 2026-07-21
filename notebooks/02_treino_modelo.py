#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Convertido a partir de notebooks/02_treino_modelo.ipynb
# Rodar: python notebooks/02_treino_modelo.py

# # Treino do modelo — Triagem de Doenças Oculares em Imagens de Retina
#
# > ## ⚠️ AVISO IMPORTANTE
# > Este notebook treina um modelo de **apoio à triagem**, não um sistema de diagnóstico médico. O modelo resultante **não substitui, em nenhuma hipótese, a avaliação de um oftalmologista**. Ele foi treinado com um dataset acadêmico público, sem validação clínica, e deve ser usado apenas para fins de estudo neste projeto.
# >
# > Qualquer predição gerada aqui (ou pela API em `api/main.py`) é probabilística e sujeita a erro — inclusive falsos negativos, que em um cenário real poderiam atrasar o tratamento de uma condição progressiva. Ver `docs/notas_eticas.md` para a discussão completa.
#
# **Antes de rodar:** baixe o dataset do Kaggle ([eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification)) e organize em `data/raw/dataset/<classe>/` (ver README).
#
# **O que este notebook faz:**
# 1. Carrega as imagens (treino/validação/teste)
# 2. Monta o modelo (MobileNetV2 pré-treinada + cabeça nova)
# 3. Treina em 2 etapas: cabeça nova (base congelada) → fine-tuning das últimas camadas
# 4. Avalia no conjunto de teste (acurácia, F1, matriz de confusão)
# 5. Salva o modelo para a API usar (`api/modelo_final.keras` + `api/class_names.json`)
# 6. Testa uma predição de exemplo, já com o aviso de não-diagnóstico

# ## 0. Configuração e imports

from pathlib import Path
import sys


def encontrar_raiz_projeto(start: Path) -> Path:
    for candidate in [start, *start.parents]:
        if (candidate / "src" / "load_data.py").exists():
            return candidate
    raise FileNotFoundError(f"Não foi possível localizar a raiz do projeto a partir de: {start}")


ROOT = encontrar_raiz_projeto(Path.cwd().resolve())
SRC_DIR = ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

print("Raiz do projeto:", ROOT)

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from load_data import carregar_datasets, IMG_SIZE

print("TensorFlow:", tf.__version__)
print("GPUs disponíveis:", tf.config.list_physical_devices("GPU") or "nenhuma (rodando em CPU)")

# Hiperparâmetros — os mesmos usados em src/model.py.
# Reduza as épocas aqui só se estiver testando o notebook rapidamente;
# para o resultado que vai pro relatório, use os valores completos.
EPOCHS_CABECA = 10          # etapa 1: só a cabeça nova, base congelada
EPOCHS_FINE_TUNING = 5      # etapa 2: descongelando as últimas camadas
CAMADAS_DESCONGELADAS = 30  # quantas camadas finais da MobileNetV2 liberar
LR_CABECA = 1e-3
LR_FINE_TUNING = 1e-5       # bem menor, para não "destruir" os pesos da ImageNet

MODELS_DIR = ROOT / "api"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ## 1. Carregar os dados
#
# Usa o mesmo pipeline de `src/load_data.py`: split 70% treino / 15% validação / 15% teste, estratificado por classe.

treino_ds, val_ds, teste_ds, class_names = carregar_datasets()

print("Classes:", class_names)
print("Batches -> treino:", tf.data.experimental.cardinality(treino_ds).numpy())
print("Batches -> validação:", tf.data.experimental.cardinality(val_ds).numpy())
print("Batches -> teste:", tf.data.experimental.cardinality(teste_ds).numpy())

# ### 1.1 Pré-processamento e data augmentation
#
# Normalização (`preprocess_input` da MobileNetV2) em todos os conjuntos; data augmentation leve (flip, rotação, zoom, contraste) só no treino, para o modelo generalizar melhor e não decorar as imagens de treino.

def preparar_pipeline(ds, treino: bool):
    normalizacao = tf.keras.applications.mobilenet_v2.preprocess_input

    aumento = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    ds = ds.map(lambda x, y: (normalizacao(x), y), num_parallel_calls=tf.data.AUTOTUNE)
    if treino:
        ds = ds.map(lambda x, y: (aumento(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.shuffle(500)
    return ds.prefetch(tf.data.AUTOTUNE)


treino_prep = preparar_pipeline(treino_ds, treino=True)
val_prep = preparar_pipeline(val_ds, treino=False)
teste_prep = preparar_pipeline(teste_ds, treino=False)

# ## 2. Construir o modelo
#
# Base MobileNetV2 pré-treinada na ImageNet (congelada inicialmente) + cabeça de classificação nova.

def construir_modelo(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False  # etapa 1: base congelada

    entradas = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(entradas, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.3)(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    saidas = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    modelo = tf.keras.Model(entradas, saidas)
    return modelo, base


modelo, base = construir_modelo(num_classes=len(class_names))
modelo.summary()

# ## 3. Etapa 1 — Treinar a cabeça (base congelada)

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(LR_CABECA),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

historico_etapa1 = modelo.fit(
    treino_prep,
    validation_data=val_prep,
    epochs=EPOCHS_CABECA,
)

# ### 3.1 Curva de treino — etapa 1

def plotar_historico(historico, titulo):
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    axs[0].plot(historico.history["accuracy"], label="treino")
    if "val_accuracy" in historico.history:
        axs[0].plot(historico.history["val_accuracy"], label="validação")
    else:
        print("Aviso: sem métrica de validação nesta etapa (conjunto de validação " 
              "ficou vazio — normal só se o dataset usado for muito pequeno/parcial).")
    axs[0].set_title(f"Acurácia — {titulo}")
    axs[0].set_xlabel("Época")
    axs[0].legend()

    axs[1].plot(historico.history["loss"], label="treino")
    if "val_loss" in historico.history:
        axs[1].plot(historico.history["val_loss"], label="validação")
    axs[1].set_title(f"Loss — {titulo}")
    axs[1].set_xlabel("Época")
    axs[1].legend()

    plt.tight_layout()
    plt.savefig(ROOT / "docs" / f"curva_{titulo.lower().replace(chr(32), chr(95))}.png", dpi=150)
    plt.show()


plotar_historico(historico_etapa1, "Etapa 1")

# ## 4. Etapa 2 — Fine-tuning das últimas camadas
#
# Descongela as últimas camadas da MobileNetV2 e re-treina tudo com uma taxa de aprendizado bem menor, para adaptar os filtros mais específicos da rede ao domínio de imagens de retina sem apagar o conhecimento genérico da ImageNet.

base.trainable = True
for camada in base.layers[:-CAMADAS_DESCONGELADAS]:
    camada.trainable = False

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(LR_FINE_TUNING),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

historico_etapa2 = modelo.fit(
    treino_prep,
    validation_data=val_prep,
    epochs=EPOCHS_FINE_TUNING,
)

# ### 4.1 Curva de treino — etapa 2 (fine-tuning)

plotar_historico(historico_etapa2, "Etapa 2 Fine Tuning")

# ## 5. Avaliação no conjunto de teste
#
# Métricas calculadas **uma única vez**, sobre o conjunto de teste (nunca usado até aqui) — é esse resultado que deve ir para a seção 4 do relatório técnico.

y_true, y_pred = [], []
for imagens, rotulos in teste_prep:
    probs = modelo.predict(imagens, verbose=0)
    y_pred.extend(np.argmax(probs, axis=1))
    y_true.extend(rotulos.numpy())

rotulos_possiveis = list(range(len(class_names)))

print("=== Relatório de classificação (conjunto de teste) ===")
relatorio_txt = classification_report(
    y_true, y_pred, labels=rotulos_possiveis, target_names=class_names,
    digits=3, zero_division=0,
)
print(relatorio_txt)

# Também como dicionário, útil para montar a tabela do relatório em Word
relatorio_dict = classification_report(
    y_true, y_pred, labels=rotulos_possiveis, target_names=class_names,
    digits=3, zero_division=0, output_dict=True,
)
relatorio_dict

# ### 5.1 Matriz de confusão

matriz = confusion_matrix(y_true, y_pred, labels=rotulos_possiveis)

plt.figure(figsize=(6, 5))
sns.heatmap(matriz, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel("Classe prevista")
plt.ylabel("Classe real")
plt.title("Matriz de confusão — conjunto de teste")
plt.tight_layout()
plt.savefig(ROOT / "docs" / "matriz_confusao.png", dpi=150)
plt.show()

print(matriz)

# ## 6. Salvar o modelo treinado
#
# Salva em `api/modelo_final.keras` e `api/class_names.json` — é isso que a API (`api/main.py`) carrega para servir as predições.

caminho_modelo = MODELS_DIR / "modelo_final.keras"
modelo.save(caminho_modelo)

with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
    json.dump(class_names, f, ensure_ascii=False, indent=2)

print(f"Modelo salvo em: {caminho_modelo}")
print(f"Classes salvas em: {MODELS_DIR / 'class_names.json'}")

# ## 7. Testar uma predição de exemplo
#
# > ## ⚠️ Lembrete
# > O resultado abaixo é gerado por um modelo de IA para fins de **triagem**, treinado em um dataset acadêmico sem validação clínica. **Não é um diagnóstico médico.** Qualquer suspeita real de doença ocular deve ser avaliada por um oftalmologista.
#
# Troque `caminho_imagem_teste` por uma imagem qualquer de `data/raw/dataset/<classe>/` para ver como o modelo se comporta.

def prever_imagem(caminho_imagem):
    """Mesma lógica de pré-processamento usada em api/main.py, para
    garantir que o teste aqui reflita o que a API vai fazer de verdade."""
    imagem = tf.keras.utils.load_img(caminho_imagem, target_size=IMG_SIZE)
    array = tf.keras.utils.img_to_array(imagem)
    array = tf.keras.applications.mobilenet_v2.preprocess_input(array)
    entrada = np.expand_dims(array, axis=0)

    probs = modelo.predict(entrada, verbose=0)[0]
    indice = int(np.argmax(probs))

    print(f"Classe prevista: {class_names[indice]}  (confiança: {probs[indice]:.2%})")
    print("Probabilidades por classe:")
    for nome, p in zip(class_names, probs):
        print(f"  {nome}: {p:.2%}")
    print("\n⚠️  Resultado de triagem por IA — NÃO é um diagnóstico médico. "
          "Consulte um oftalmologista para avaliação profissional.")

    plt.figure(figsize=(4, 4))
    plt.imshow(imagem)
    plt.axis("off")
    plt.title(f"Previsto: {class_names[indice]} ({probs[indice]:.1%})")
    plt.show()


# Exemplo — troque pelo caminho de uma imagem real do dataset:
caminho_imagem_teste = DATASET_DIR = ROOT / "data" / "raw" / "dataset"
exemplos = [f for f in (caminho_imagem_teste / class_names[0]).iterdir() if f.suffix.lower() in {".jpg", ".jpeg", ".png"}]
if exemplos:
    prever_imagem(exemplos[0])
else:
    print("Nenhuma imagem de exemplo encontrada — confira se o dataset foi baixado.")

# ## 8. Próximos passos
#
# - Copiar o `classification_report` e a matriz de confusão acima para a seção 4 do relatório técnico (`docs/relatorio_tecnico.md` ou o .docx entregue).
# - As imagens `docs/curva_etapa_1.png`, `docs/curva_etapa_2_fine_tuning.png` e `docs/matriz_confusao.png` já foram salvas automaticamente e podem ser inseridas direto no relatório.
# - Subir a API (`uvicorn api.main:app --reload`) para testar o modelo treinado via `POST /prever`, com upload de imagem — a resposta da API já inclui o aviso de não-diagnóstico embutido.
