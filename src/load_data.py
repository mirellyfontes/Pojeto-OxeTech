"""
Carga dos dados de imagens de retina para triagem de doenças oculares.

Dataset esperado (baixar do Kaggle e extrair em data/raw/dataset/):
    https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification

Estrutura esperada depois de extrair:
    data/raw/dataset/
        cataract/
        diabetic_retinopathy/
        glaucoma/
        normal/

Cada subpasta é uma classe (nome da pasta = rótulo), ~1000 imagens cada,
dataset praticamente balanceado.

Como usar:
    from src.load_data import carregar_datasets
    train_ds, val_ds, test_ds, class_names = carregar_datasets()
"""

from pathlib import Path
import tensorflow as tf

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_DIR = RAW_DIR / "dataset"

# MobileNetV2 foi treinada originalmente com entrada 224x224
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Sementes fixas para os splits serem reprodutíveis entre quem rodar o código
SEED = 42

# Proporção usada para separar validação+teste do treino (o restante, 80%,
# fica para treino). A metade de "validation_split" vira validação e a
# outra metade vira teste (ver `carregar_datasets`).
VALIDATION_SPLIT = 0.30


def checar_dataset_baixado():
    if not DATASET_DIR.exists() or not any(DATASET_DIR.glob("*/*.*")):
        raise FileNotFoundError(
            f"Dataset não encontrado em {DATASET_DIR}.\n"
            "Baixe em https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification, "
            "extraia e organize em subpastas por classe:\n"
            f"  {DATASET_DIR}/cataract/\n"
            f"  {DATASET_DIR}/diabetic_retinopathy/\n"
            f"  {DATASET_DIR}/glaucoma/\n"
            f"  {DATASET_DIR}/normal/"
        )


def contar_imagens_por_classe() -> dict:
    """Conta quantas imagens existem em cada subpasta (classe) — útil para EDA
    e para checar se o dataset está mesmo balanceado como a documentação do
    Kaggle promete."""
    checar_dataset_baixado()
    extensoes = {".jpg", ".jpeg", ".png"}
    contagem = {}
    for pasta_classe in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        n = sum(1 for f in pasta_classe.iterdir() if f.suffix.lower() in extensoes)
        contagem[pasta_classe.name] = n
    return contagem


def carregar_datasets(
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    validation_split=VALIDATION_SPLIT,
    seed=SEED,
):
    """
    Carrega o dataset de imagens organizado em subpastas por classe e devolve
    três `tf.data.Dataset` já em batches: treino (70%), validação (15%) e
    teste (15%).

    Não aplicamos augmentation nem normalização aqui dentro — isso fica a
    cargo de `model.py`, para deixar claro no pipeline de treino o que é
    "carregar dado" (aqui) e o que é "preparar para o modelo" (lá).
    """
    checar_dataset_baixado()

    treino = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=validation_split,
        subset="training",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    # A parte "validation" do Keras aqui ainda tem 30% dos dados; dividimos
    # ela ao meio, por batch, para virar validação (15%) e teste (15%) —
    # o teste fica reservado e só deve ser usado uma vez, no final, para
    # reportar a métrica oficial no relatório.
    resto = tf.keras.utils.image_dataset_from_directory(
        DATASET_DIR,
        validation_split=validation_split,
        subset="validation",
        seed=seed,
        image_size=img_size,
        batch_size=batch_size,
        label_mode="int",
    )

    class_names = treino.class_names  # ordem alfabética das subpastas

    n_batches_resto = tf.data.experimental.cardinality(resto).numpy()
    metade = n_batches_resto // 2
    validacao = resto.take(metade)
    teste = resto.skip(metade)

    return treino, validacao, teste, class_names


def main():
    contagem = contar_imagens_por_classe()
    print("Imagens por classe em data/raw/dataset/:")
    for classe, n in contagem.items():
        print(f"  {classe}: {n}")
    print(f"  TOTAL: {sum(contagem.values())}")

    treino, validacao, teste, class_names = carregar_datasets()
    print(f"\nClasses (na ordem usada pelo modelo): {class_names}")
    print(f"Batches -> treino: {tf.data.experimental.cardinality(treino).numpy()}, "
          f"validação: {tf.data.experimental.cardinality(validacao).numpy()}, "
          f"teste: {tf.data.experimental.cardinality(teste).numpy()}")


if __name__ == "__main__":
    main()
