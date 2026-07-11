# Frota Veicular x Sinistros de Trânsito em Alagoas

Projeto Final Integrador (AV2) — OxeTech Academy | Curso Avançado de IA e Aprendizado de Máquina
Instrutor: Prof. Me. Derek Nielsen Araújo Alves

## 1. Problema

O crescimento da frota veicular em Alagoas (por bairro/município) está relacionado ao
aumento no número de sinistros (acidentes) de trânsito registrados no estado?

**Tipo de problema:** Regressão (predição de nº de sinistros / vítimas a partir de
variáveis de frota e tempo).

**Métrica principal:** RMSE / MAE (a definir com base na distribuição do alvo — se
muito assimétrica, considerar log-transform e reportar também R²).

## 2. Fontes de dados

| Dataset | Status | Formato | Fonte |
|---|---|---|---|
| Sinistros de trânsito em Alagoas (microdados, 2022-2025, 10.922 registros) | ✅ já está em `data/raw/` | CSV | Dashboard Sinistros — DETRAN-AL/RENAEST |
| Frota de veículos em Alagoas (série histórica) | ⏳ falta baixar | CSV / Google Sheets | https://docs.google.com/spreadsheets/d/1Ek3kg9_TCBTJLsC1zMHKARabqwwd9F-nHufVk-U3504/edit?usp=sharing |
| Frota por bairro 2025 (opcional, enriquecimento futuro) | ⏳ opcional | CSV | https://indicadores.detran.al.gov.br/wp-content/uploads/2025/09/Frota-por-bairro-2025.csv |

O dataset de sinistros já é o microdado oficial: um sinistro por linha, com
ano, mês, município, bairro, tipo de sinistro, óbitos, condições da via, clima
e geolocalização. O `src/load_data.py` já sabe agregá-lo para uma série
mensal (`agregar_sinistros_mensal`).

### Como baixar o que falta

1. Baixe a planilha de frota histórica (link acima) como CSV (`Arquivo >
   Fazer download > CSV` no Google Sheets) e salve em `data/raw/` — não
   precisa de nome exato, o script procura por qualquer `.csv` que contenha
   "frota" no nome (e não contenha "bairro").
2. Rode `python src/load_data.py`. Ele já carrega e agrega os sinistros; na
   etapa `cruzar_com_frota`, ele imprime as colunas reais da planilha de
   frota — usem esse print para ajustar o `merge` (a função já vem com um
   exemplo comentado, é só descomentar e trocar os nomes das colunas).

## 3. Estrutura do repositório

```
projeto-transito-al/
├── README.md
├── CRONOGRAMA.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── raw/            # arquivos originais baixados (NÃO versionar no git)
│   └── processed/      # arquivos tratados (pode versionar se forem pequenos)
├── notebooks/
│   └── 01_eda_baseline.ipynb
├── src/
│   ├── load_data.py    # leitura e limpeza dos dados
│   └── model.py        # treino e avaliação do(s) modelo(s)
├── api/
│   ├── main.py          # API FastAPI
│   ├── Dockerfile
│   └── requirements.txt
└── docs/
    ├── relatorio_tecnico.md   # rascunho do relatório (10-15 páginas depois em PDF/Word)
    └── notas_eticas.md        # pontos levantados sobre viés e limitações dos dados
```

## 4. Como rodar localmente (VSCode)

```bash
# 1. Criar e ativar ambiente virtual
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Abrir o notebook no VSCode
#    (extensão "Jupyter" do VSCode precisa estar instalada)
code notebooks/01_eda_baseline.ipynb
```

## 5. Autoria

Grupo: [nome 1], [nome 2]
Curso: OxeTech Academy — IA e Aprendizado de Máquina
