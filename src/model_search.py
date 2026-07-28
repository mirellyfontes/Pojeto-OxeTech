"""
Busca de hiperparâmetros para o modelo de triagem de doenças oculares.

Ideia: treinar VÁRIAS configurações diferentes (variando tamanho da cabeça de
classificação, dropout, taxa de aprendizado e quantas camadas da MobileNetV2
são liberadas no fine-tuning), comparar todas usando o conjunto de VALIDAÇÃO,
e só então escolher a melhor.

Por que comparar na validação e não no teste?
    O conjunto de teste só deve ser usado UMA VEZ, no final, para reportar a
    métrica oficial do modelo já escolhido. Se usássemos o teste para decidir
    qual configuração é "a melhor", estaríamos indiretamente ajustando o
    modelo ao teste (overfitting na escolha), o que invalidaria a métrica
    final como estimativa de generalização — problema clássico em ML
    conhecido como "test set leakage" via seleção de modelo.

Métrica de seleção: F1-macro na validação (mais robusta que acurácia para
dataset multiclasse, e trata todas as classes com o mesmo peso — importante
no contexto clínico deste projeto, onde confundir retinopatia diabética com
normal é um erro muito mais grave que confundir catarata com glaucoma).

Segue as mesmas práticas do src/model.py atual:
  - EarlyStopping + ReduceLROnPlateau em cada etapa de treino
  - BatchNormalization da MobileNetV2 permanece congelada no fine-tuning
    (preserva as estatísticas aprendidas na ImageNet)
  - shuffle -> augmentation -> normalização, na mesma ordem do pipeline
    de treino usado em model.py

Como usar (rodar a partir da raiz do projeto, "Pojeto OxeTech/"):
    python src/model_search.py

Saídas:
    api/modelo_final.keras                      -> melhor modelo (mesmo caminho da API)
    api/class_names.json                        -> nomes das classes, na ordem do modelo
    api/metricas_teste.json                     -> métricas oficiais (teste) do modelo escolhido
    data/processed/model_search_resultados.csv  -> tabela comparando todas as configurações
    data/processed/experimentos/<nome>.keras    -> modelo de cada configuração testada
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix, f1_score

# Adiciona a própria pasta (src/) ao sys.path, assim o script funciona tanto
# rodando "python src/model_search.py" quanto "python -m src.model_search",
# sem depender de como o interpretador resolve o pacote "src".
sys.path.insert(0, str(Path(__file__).resolve().parent))
from load_data import carregar_datasets, IMG_SIZE  # noqa: E402

ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "api"
PROCESSED_DIR = ROOT_DIR / "data" / "processed"
EXPERIMENTOS_DIR = PROCESSED_DIR / "experimentos"

SEED = 42

# ---------------------------------------------------------------------------
# Espaço de busca: cada dicionário é uma configuração completa e independente.
# Adicionar uma configuração nova é só copiar um bloco e mudar os valores.
# Evitem uma grade cartesiana gigante (todas combinações possíveis): com
# poucas imagens e treino em CPU, um punhado de configurações bem escolhidas
# conta uma história melhor no relatório do que uma busca exaustiva que
# ninguém consegue interpretar depois.
# ---------------------------------------------------------------------------
CONFIGURACOES = [
    {
        "nome": "baseline",
        "unidades_densas": 128,
        "dropout1": 0.3,
        "dropout2": 0.2,
        "lr_cabeca": 1e-3,
        "lr_fine_tuning": 1e-5,
        "camadas_descongeladas": 30,
        "epochs_cabeca": 10,
        "epochs_fine_tuning": 5,
    },
    {
        "nome": "mais_regularizacao",
        "unidades_densas": 128,
        "dropout1": 0.5,
        "dropout2": 0.3,
        "lr_cabeca": 1e-3,
        "lr_fine_tuning": 1e-5,
        "camadas_descongeladas": 30,
        "epochs_cabeca": 10,
        "epochs_fine_tuning": 5,
    },
    {
        "nome": "cabeca_maior_lr_menor",
        "unidades_densas": 256,
        "dropout1": 0.3,
        "dropout2": 0.2,
        "lr_cabeca": 5e-4,
        "lr_fine_tuning": 1e-5,
        "camadas_descongeladas": 30,
        "epochs_cabeca": 10,
        "epochs_fine_tuning": 5,
    },
    {
        "nome": "fine_tuning_mais_profundo",
        "unidades_densas": 128,
        "dropout1": 0.3,
        "dropout2": 0.2,
        "lr_cabeca": 1e-3,
        "lr_fine_tuning": 1e-5,
        "camadas_descongeladas": 60,
        "epochs_cabeca": 10,
        "epochs_fine_tuning": 8,
    },
]


def criar_callbacks(paciencia: int = 3):
    """Mesmos callbacks do src/model.py: para de treinar quando a validação
    para de melhorar (restaurando os melhores pesos) e reduz a taxa de
    aprendizado quando a val_loss estagna."""
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
    """Mesma ordem do src/model.py: shuffle -> augmentation -> normalização.
    Augmentation só no treino; normalização (preprocess_input da MobileNetV2)
    em todos os conjuntos."""
    normalizacao = tf.keras.applications.mobilenet_v2.preprocess_input

    aumento = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])

    if treino:
        ds = ds.shuffle(500, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(lambda x, y: (aumento(x, training=True), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(lambda x, y: (normalizacao(x), y), num_parallel_calls=tf.data.AUTOTUNE)

    return ds.prefetch(tf.data.AUTOTUNE)


def construir_modelo(num_classes: int, config: dict):
    """Monta o modelo variando os hiperparâmetros da configuração recebida."""
    base = tf.keras.applications.MobileNetV2(
        input_shape=IMG_SIZE + (3,),
        include_top=False,
        weights="imagenet",
    )
    base.trainable = False

    entradas = tf.keras.Input(shape=IMG_SIZE + (3,))
    x = base(entradas, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(config["dropout1"])(x)
    x = tf.keras.layers.Dense(config["unidades_densas"], activation="relu")(x)
    x = tf.keras.layers.Dropout(config["dropout2"])(x)
    saidas = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

    modelo = tf.keras.Model(entradas, saidas)
    return modelo, base


def treinar_uma_configuracao(config, treino_ds, val_ds, num_classes):
    """Treina (cabeça + fine-tuning) uma configuração e devolve o modelo já
    treinado junto com as métricas na validação."""
    print(f"\n{'=' * 70}")
    print(f"Configuração: {config['nome']}")
    print(f"{'=' * 70}")
    for chave, valor in config.items():
        if chave != "nome":
            print(f"  {chave}: {valor}")

    modelo, base = construir_modelo(num_classes, config)

    # --- Etapa 1: só a cabeça, base congelada ---
    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(config["lr_cabeca"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print("\n--- Etapa 1: treinando a cabeça (base congelada) ---")
    modelo.fit(
        treino_ds,
        validation_data=val_ds,
        epochs=config["epochs_cabeca"],
        callbacks=criar_callbacks(),
        verbose=2,
    )

    # --- Etapa 2: fine-tuning das últimas N camadas ---
    base.trainable = True
    n_descongelar = config["camadas_descongeladas"]
    for camada in base.layers[:-n_descongelar]:
        camada.trainable = False

    # BatchNormalization permanece congelada para preservar as estatísticas
    # aprendidas na ImageNet, mesmo dentro das camadas "descongeladas".
    for camada in base.layers[-n_descongelar:]:
        if isinstance(camada, tf.keras.layers.BatchNormalization):
            camada.trainable = False

    modelo.compile(
        optimizer=tf.keras.optimizers.Adam(config["lr_fine_tuning"]),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    print(f"\n--- Etapa 2: fine-tuning das últimas {n_descongelar} camadas ---")
    modelo.fit(
        treino_ds,
        validation_data=val_ds,
        epochs=config["epochs_fine_tuning"],
        callbacks=criar_callbacks(paciencia=2),
        verbose=2,
    )

    # --- Avaliação na validação (usada só para ESCOLHER a configuração) ---
    y_true, y_pred = [], []
    for imagens, rotulos in val_ds:
        probs = modelo.predict(imagens, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1))
        y_true.extend(rotulos.numpy())

    val_accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    val_f1_macro = float(f1_score(y_true, y_pred, average="macro"))

    print(f"\nValidação -> accuracy: {val_accuracy:.4f} | f1_macro: {val_f1_macro:.4f}")

    metricas = {
        "nome": config["nome"],
        "val_accuracy": val_accuracy,
        "val_f1_macro": val_f1_macro,
        "epochs_cabeca": config["epochs_cabeca"],
        "epochs_fine_tuning": config["epochs_fine_tuning"],
        "unidades_densas": config["unidades_densas"],
        "dropout1": config["dropout1"],
        "dropout2": config["dropout2"],
        "lr_cabeca": config["lr_cabeca"],
        "lr_fine_tuning": config["lr_fine_tuning"],
        "camadas_descongeladas": config["camadas_descongeladas"],
    }

    return metricas, modelo


def avaliar_no_teste(modelo, ds_teste, class_names):
    """Avaliação final, oficial, no conjunto de teste — só deve ser chamada
    UMA VEZ, com o modelo já escolhido. Segue o mesmo formato de
    src/model.py (inclusive salvando em metricas_teste.json)."""
    perda, acuracia = modelo.evaluate(ds_teste, verbose=0)

    y_true, y_pred = [], []
    for imagens, rotulos in ds_teste:
        probs = modelo.predict(imagens, verbose=0)
        y_pred.extend(np.argmax(probs, axis=1).tolist())
        y_true.extend(rotulos.numpy().tolist())

    rotulos_possiveis = list(range(len(class_names)))  # garante todas as classes
    # no relatório, mesmo que alguma não apareça no teste (dataset pequeno/desbalanceado)

    relatorio = classification_report(
        y_true,
        y_pred,
        labels=rotulos_possiveis,
        target_names=class_names,
        digits=3,
        zero_division=0,
        output_dict=True,
    )

    matriz = confusion_matrix(y_true, y_pred, labels=rotulos_possiveis)

    resultados = {
        "test_loss": float(perda),
        "test_accuracy": float(acuracia),
        "test_f1_macro": float(relatorio["macro avg"]["f1-score"]), # pyright: ignore[reportArgumentType]
        "classification_report": relatorio,
        "confusion_matrix": matriz.tolist(),
    }

    print("\n=== Resultados no conjunto de teste (modelo final) ===")
    print(f"Loss: {resultados['test_loss']:.4f}")
    print(f"Acurácia: {resultados['test_accuracy']:.4f}")
    print(f"F1-macro: {resultados['test_f1_macro']:.4f}")
    print("\n=== Relatório de classificação ===")
    print(classification_report(
        y_true, y_pred, labels=rotulos_possiveis, target_names=class_names,
        digits=3, zero_division=0,
    ))
    print("=== Matriz de confusão ===")
    print(matriz)

    return resultados


def main():
    treino_ds, val_ds, teste_ds, class_names = carregar_datasets()
    num_classes = len(class_names)
    print(f"Classes: {class_names}")
    print(f"Testando {len(CONFIGURACOES)} configurações diferentes...")

    treino_prep = preparar_pipeline(treino_ds, treino=True)
    val_prep = preparar_pipeline(val_ds, treino=False)
    teste_prep = preparar_pipeline(teste_ds, treino=False)

    EXPERIMENTOS_DIR.mkdir(parents=True, exist_ok=True)

    resultados = []

    for config in CONFIGURACOES:
        metricas, modelo = treinar_uma_configuracao(config, treino_prep, val_prep, num_classes)
        resultados.append(metricas)

        # Salva cada modelo testado (bom para o relatório / apêndice) antes
        # de liberar a memória — evita ter que retreinar depois.
        caminho_experimento = EXPERIMENTOS_DIR / f"{config['nome']}.keras"
        modelo.save(caminho_experimento)
        print(f"Modelo desta configuração salvo em: {caminho_experimento}")

        tf.keras.backend.clear_session()
        del modelo

    # --- Tabela comparativa de todas as configurações ---
    df_resultados = pd.DataFrame(resultados).sort_values("val_f1_macro", ascending=False)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    caminho_csv = PROCESSED_DIR / "model_search_resultados.csv"
    df_resultados.to_csv(caminho_csv, index=False)

    print(f"\n{'=' * 70}")
    print("RESUMO — todas as configurações (ordenadas por f1_macro na validação)")
    print(f"{'=' * 70}")
    print(df_resultados[["nome", "val_accuracy", "val_f1_macro"]].to_string(index=False))
    print(f"\nTabela completa salva em: {caminho_csv}")

    melhor_nome = df_resultados.iloc[0]["nome"]
    print(f"\n>>> Melhor configuração: {melhor_nome} <<<")

    # Recarrega do disco o modelo vencedor — como cada sessão do Keras foi
    # limpa (clear_session) para economizar memória entre configurações,
    # isso garante que o modelo final avaliado é exatamente o que foi salvo,
    # sem precisar retreinar do zero.
    caminho_melhor = EXPERIMENTOS_DIR / f"{melhor_nome}.keras"
    melhor_modelo = tf.keras.models.load_model(caminho_melhor)

    # --- Avaliação final no teste, feita UMA ÚNICA VEZ, com o modelo escolhido ---
    resultados_teste = avaliar_no_teste(melhor_modelo, teste_prep, class_names)

    # --- Salva o modelo vencedor no lugar que a API espera ---
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    caminho_modelo_final = MODELS_DIR / "modelo_final.keras"
    melhor_modelo.save(caminho_modelo_final)
    with open(MODELS_DIR / "class_names.json", "w", encoding="utf-8") as f:
        json.dump(class_names, f, ensure_ascii=False, indent=2)
    with open(MODELS_DIR / "metricas_teste.json", "w", encoding="utf-8") as f:
        json.dump(resultados_teste, f, ensure_ascii=False, indent=2)

    print(f"\nModelo final (melhor configuração) salvo em: {caminho_modelo_final}")
    print(f"Classes salvas em: {MODELS_DIR / 'class_names.json'}")
    print(f"Métricas salvas em: {MODELS_DIR / 'metricas_teste.json'}")


if __name__ == "__main__":
    main()
