"""
Treino do modelo de triagem de doenças oculares por imagem de retina.

Estratégia: transfer learning com MobileNetV2 pré-treinada na ImageNet
(mesma abordagem usada na penúltima atividade do curso) — não treinamos uma
CNN do zero porque o dataset (~4.200 imagens, 4 classes) é pequeno demais
para isso convergir bem no tempo que temos.

Duas etapas:
  1) "Feature extraction": a MobileNetV2 fica congelada (pesos da ImageNet)
     e só treinamos uma cabeça de classificação nova em cima dela.
  2) "Fine-tuning": descongelamos as últimas camadas da MobileNetV2 e
     re-treinamos tudo com uma taxa de aprendizado bem baixa, para adaptar
     os filtros mais "específicos" da rede ao domínio de imagens de retina.

Rodar depois de baixar o dataset em data/raw/dataset/ (ver src/load_data.py):
    python src/model.py
"""

import json
from pathlib import Path

import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src.load_data import carregar_datasets, IMG_SIZE

MODELS_DIR = Path(__file__).resolve().parent.parent / "api"

EPOCHS_CABECA = 10          # etapa 1: só a cabeça nova, base congelada
EPOCHS_FINE_TUNING = 5      # etapa 2: descongelando as últimas camadas
CAMADAS_DESCONGELADAS = 30  # quantas camadas finais da MobileNetV2 liberar
LR_CABECA = 1e-3
LR_FINE_TUNING = 1e-5       # bem menor, para não "destruir" os pesos da ImageNet

def criar_callbacks(paciencia: int = 3):
    return [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=paciencia,
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.2,
            patience=2,
            min_lr=1e-7,
            verbose=1,
        ),
    ]

def preparar_pipeline(ds, treino: bool):
    """Aplica normalização (preprocess_input da MobileNetV2) e, só no
    treino, data augmentation leve — para o modelo não decorar as ~1000
    imagens de cada classe e generalizar melhor para fotos de retina
    tiradas com outros aparelhos/condições de luz."""
    normalizacao = tf.keras.applications.mobilenet_v2.preprocess_input

    aumento = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    if treino:
        ds = ds.shuffle(500, seed=42, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda x, y: (aumento(x, training=True), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
    ds = ds.map(
        lambda x, y: (normalizacao(x), y),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return ds.prefetch(tf.data.AUTOTUNE)


def construir_modelo(num_classes: int) -> tuple[tf.keras.Model, tf.keras.Model]:
    """Monta o modelo completo (base MobileNetV2 congelada + cabeça nova).
    Retorna o modelo e a referência à base, para descongelar depois no
    fine-tuning."""
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


def avaliar(modelo, ds_teste, class_names):
    perda, acuracia = modelo.evaluate(ds_teste, verbose=0)

    y_true, y_pred = [], []

    for imagens, rotulos in ds_teste:
        probs = modelo.predict(imagens, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(rotulos.numpy().tolist())

    rotulos = list(range(len(class_names)))  # garante todas as classes no relatório,
    # mesmo que alguma não apareça no conjunto de teste (dataset pequeno/desbalanceado)

    relatorio = classification_report(
        y_true,
        y_pred,
        labels=rotulos,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )

    matriz = confusion_matrix(
        y_true,
        y_pred,
        labels=rotulos,
    )

    resultados = {
        "test_loss": float(perda),
        "test_accuracy": float(acuracia),
        "test_f1_macro": float(relatorio["macro avg"]["f1-score"]),
        "classification_report": relatorio,
        "confusion_matrix": matriz.tolist(),
    }

    print("\n=== Resultados no conjunto de teste ===")
    print(f"Loss: {resultados['test_loss']:.4f}")
    print(f"Acurácia: {resultados['test_accuracy']:.4f}")
    print(f"F1-macro: {resultados['test_f1_macro']:.4f}")

    print("\n=== Relatório de classificação ===")
    print(classification_report(
        y_true,
        y_pred,
        labels=rotulos,
        target_names=class_names,
        digits=3,
        zero_division=0,
    ))

    print("=== Matriz de confusão ===")
    print(matriz)

    return resultados


def main():
    treino_ds, val_ds, teste_ds, class_names = carregar_datasets()
    print(f"Classes: {class_names}")

    treino_ds = preparar_pipeline(treino_ds, treino=True)
    val_ds = preparar_pipeline(val_ds, treino=False)
    teste_ds = preparar_pipeline(teste_ds, treino=False)

    modelo, base = construir_modelo(num_classes=len(class_names))

    # --- Etapa 1: treinar só a cabeça (base congelada) ---
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR_CABECA),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n--- Etapa 1: treinando a cabeça de classificação (base congelada) ---")

    modelo.fit(
        treino_ds,
        validation_data=val_ds,
        epochs=EPOCHS_CABECA,
        callbacks=criar_callbacks(),
    )

    # --- Etapa 2: fine-tuning das últimas camadas da MobileNetV2 ---
    base.trainable = True

    for camada in base.layers[:-CAMADAS_DESCONGELADAS]:
        camada.trainable = False

    # BatchNormalization permanece congelada para preservar
    # as estatísticas aprendidas na ImageNet.
    for camada in base.layers[-CAMADAS_DESCONGELADAS:]:
        if isinstance(camada, tf.keras.layers.BatchNormalization):
            camada.trainable = False

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(LR_FINE_TUNING),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n--- Etapa 2: fine-tuning das últimas {CAMADAS_DESCONGELADAS} camadas ---")

    modelo.fit(
        treino_ds,
        validation_data=val_ds,
        epochs=EPOCHS_FINE_TUNING,
        callbacks=criar_callbacks(paciencia=2),
    )

    # --- Avaliação final no conjunto de teste (nunca visto até aqui) ---
    resultados = avaliar(modelo, teste_ds, class_names)

    # --- Salvar modelo + nomes das classes para a API usar ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo = MODELS_DIR / "modelo_final.keras"
    modelo.save(caminho_modelo)

    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)

    with open(MODELS_DIR / "metricas_teste.json", "w", encoding="utf-8") as f:
        json.dump(resultados, f, ensure_ascii=False, indent=2)

    print(f"\nModelo salvo em: {caminho_modelo}")
    print(f"Classes salvas em: {MODELS_DIR / 'class_names.json'}")
    print(f"Métricas salvas em: {MODELS_DIR / 'metricas_teste.json'}")


if __name__ == "__main__":
    main()
