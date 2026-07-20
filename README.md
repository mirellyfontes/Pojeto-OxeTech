# Triagem de Doenças Oculares em Imagens de Retina

Projeto Final Integrador (AV2) — OxeTech Academy | Curso Avançado de IA e Aprendizado de Máquina
Instrutor: Prof. Me. Derek Nielsen Araújo Alves

## 1. Problema

É possível classificar automaticamente uma imagem de fundo de retina entre
**normal**, **catarata**, **glaucoma** ou **retinopatia diabética**, como
ferramenta de **apoio à triagem** — priorizando quem deveria ser encaminhado
com mais urgência a um oftalmologista?

**Tipo de problema:** Classificação de imagens (multiclasse, 4 categorias).

**Abordagem:** Transfer learning com **MobileNetV2** pré-treinada na ImageNet
(mesma rede usada na penúltima atividade do curso) — congelamos a base e
treinamos uma cabeça de classificação nova, depois fazemos fine-tuning das
últimas camadas. Não treinamos uma CNN do zero: o dataset (~4.200 imagens) é
pequeno demais para isso convergir bem no tempo disponível, e uma rede
pré-treinada já traz um extrator de características genérico muito forte.

**Métrica principal:** Acurácia e F1-score (macro, por ser multiclasse) no
conjunto de teste, além de matriz de confusão — importante em contexto
clínico entender em quais classes o modelo mais erra (ex: confundir
retinopatia diabética com normal é um erro muito mais grave que confundir
catarata com glaucoma).

**Importante:** este projeto é um exercício acadêmico de triagem assistida
por IA, **não é um dispositivo médico e não substitui diagnóstico
profissional**. Ver `docs/notas_eticas.md`.

## 2. Fonte de dados

| Dataset | Status | Formato | Fonte |
|---|---|---|---|
| Eye Diseases Classification (retinal fundus images) — 4 classes (normal, cataract, glaucoma, diabetic_retinopathy), ~1.000 imagens por classe | ⏳ falta baixar (ver abaixo) | Pastas de imagens (.jpg/.png) | [Kaggle — gunavenkatdoddi/eye-diseases-classification](https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification) |

O dataset já vem organizado em uma pasta por classe (o nome da pasta é o
próprio rótulo), o que facilita bastante usar direto com
`tf.keras.utils.image_dataset_from_directory`. As imagens vêm de fontes
públicas de imagem de retina (IDRiD, HRF, entre outras) — isso importa para
a discussão de viés, ver `docs/notas_eticas.md`.

### Como baixar

1. Baixe o dataset em
   https://www.kaggle.com/datasets/gunavenkatdoddi/eye-diseases-classification
   (é preciso login no Kaggle; dá pra baixar pelo site ou via
   `kaggle datasets download -d gunavenkatdoddi/eye-diseases-classification`
   se tiverem a API do Kaggle configurada).
2. Extraia o `.zip` e organizem as imagens em `data/raw/dataset/` assim:
   ```
   data/raw/dataset/
   ├── cataract/
   ├── diabetic_retinopathy/
   ├── glaucoma/
   └── normal/
   ```
   (o dataset original do Kaggle já vem quase nesse formato — só confiram os
   nomes das pastas depois de extrair, alguns downloads vêm com uma pasta
   `dataset/` a mais dentro, é só mover o conteúdo um nível acima).
3. Rode `python src/load_data.py` para conferir a contagem de imagens por
   classe e ver se os splits de treino/validação/teste estão sendo montados
   certo.

## 3. Estrutura do repositório

```
Pojeto OxeTech/
├── README.md
├── CRONOGRAMA.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/
│   │   └── dataset/    # imagens baixadas do Kaggle (NÃO versionar no git)
│   └── processed/      # eventuais artefatos derivados (ex: csv de contagem)
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── src/
│   ├── load_data.py    # carrega as imagens em tf.data.Dataset (treino/val/teste)
│   └── model.py         # transfer learning com MobileNetV2 + fine-tuning
├── api/
│   ├── main.py          # API FastAPI (upload de imagem -> predição)
│   ├── Dockerfile
│   └── requirements.txt
└── docs/
    ├── relatorio_tecnico.md   # rascunho do relatório (10-15 páginas depois em PDF/Word)
    └── notas_eticas.md        # pontos levantados sobre viés, limitações e uso responsável
```

## 4. Como rodar localmente (VSCode)

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Baixar e organizar o dataset (ver seção 2 acima)

# 4. Conferir o carregamento dos dados
python src/load_data.py

# 5. Treinar o modelo (salva em api/modelo_final.keras + api/class_names.json)
python src/model.py

# 6. Rodar a API localmente
uvicorn api.main:app --reload
# depois abra http://localhost:8000/docs e teste o endpoint /prever
# subindo uma imagem de retina
```

### Rodar via Docker (API)

```bash
docker build -f api/Dockerfile -t triagem-ocular-api .
docker run -p 8000:8000 triagem-ocular-api
```

## 5. Autoria

Grupo: Mirelly Fontes, Wanessa Costa.
Curso: OxeTech Academy — IA e Aprendizado de Máquina
