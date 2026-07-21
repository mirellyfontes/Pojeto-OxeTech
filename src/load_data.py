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
from sklearn.model_selection import train_test_split


RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
DATASET_DIR = RAW_DIR / "dataset"

# MobileNetV2 foi treinada originalmente com entrada 224x224
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Sementes fixas para os splits serem reprodutíveis entre quem rodar o código
SEED = 42

# Divisão reprodutível: 70% treino, 15% validação e 15% teste
TRAIN_SIZE = 0.70
EXTENSOES = {".jpg", ".jpeg", ".png"}


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
    contagem = {}
    for pasta_classe in sorted(p for p in DATASET_DIR.iterdir() if p.is_dir()):
        n = sum(1 for f in pasta_classe.iterdir() if f.is_file() and f.suffix.lower() in EXTENSOES)
        contagem[pasta_classe.name] = n
    return contagem

def listar_imagens_e_rotulos():
    """Lista os caminhos das imagens e cria os rótulos pelas subpastas."""
    checar_dataset_baixado()

    pastas_classes = sorted(
        pasta for pasta in DATASET_DIR.iterdir() if pasta.is_dir()
    )
    class_names = [pasta.name for pasta in pastas_classes]

    caminhos = []
    rotulos = []

    for indice, pasta_classe in enumerate(pastas_classes):
        arquivos = sorted(
            arquivo
            for arquivo in pasta_classe.iterdir()
            if arquivo.is_file() and arquivo.suffix.lower() in EXTENSOES
        )
        caminhos.extend(str(arquivo) for arquivo in arquivos)
        rotulos.extend([indice] * len(arquivos))

    return caminhos, rotulos, class_names


def carregar_imagem(caminho, rotulo, img_size):
    """Lê uma imagem, converte para RGB e redimensiona."""
    conteudo = tf.io.read_file(caminho)
    imagem = tf.io.decode_image(
        conteudo,
        channels=3,
        expand_animations=False,
    )
    imagem.set_shape([None, None, 3])
    imagem = tf.image.resize(imagem, img_size)
    imagem = tf.cast(imagem, tf.float32)

    return imagem, rotulo


def criar_dataset(
    caminhos,
    rotulos,
    img_size,
    batch_size,
    treino=False,
    seed=SEED,
):
    """Cria um tf.data.Dataset a partir dos caminhos e rótulos."""
    dataset = tf.data.Dataset.from_tensor_slices((caminhos, rotulos))

    if treino:
        dataset = dataset.shuffle(
            buffer_size=len(caminhos),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    dataset = dataset.map(
        lambda caminho, rotulo: carregar_imagem(caminho, rotulo, img_size),
        num_parallel_calls=tf.data.AUTOTUNE,
    )

    return dataset.batch(batch_size)


def carregar_datasets(
    img_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
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
    caminhos, rotulos, class_names = listar_imagens_e_rotulos()

    caminhos_treino, caminhos_resto, rotulos_treino, rotulos_resto = (
        train_test_split(
            caminhos,
            rotulos,
            test_size=0.30,
            random_state=seed,
            stratify=rotulos,
        )
    )

    # A parte "validation" do Keras aqui ainda tem 30% dos dados; dividimos
    # ela ao meio, por batch, para virar validação (15%) e teste (15%) —
    # o teste fica reservado e só deve ser usado uma vez, no final, para
    # reportar a métrica oficial no relatório.
    caminhos_validacao, caminhos_teste, rotulos_validacao, rotulos_teste = (
        train_test_split(
            caminhos_resto,
            rotulos_resto,
            test_size=0.50,
            random_state=seed,
            stratify=rotulos_resto,
        )
    )

    treino = criar_dataset(
        caminhos_treino,
        rotulos_treino,
        img_size,
        batch_size,
        treino=True,
        seed=seed,
    )
    validacao = criar_dataset(
        caminhos_validacao,
        rotulos_validacao,
        img_size,
        batch_size,
    )
    teste = criar_dataset(
        caminhos_teste,
        rotulos_teste,
        img_size,
        batch_size,
    )

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
