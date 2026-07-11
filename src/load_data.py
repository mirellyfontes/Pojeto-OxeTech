"""
Carga e tratamento dos dados de frota e sinistros de trânsito em Alagoas.

Datasets esperados em data/raw/ (nomes flexíveis — veja `encontrar_arquivo`):
    - Sinistros: microdados do RENAEST/DETRAN-AL (um sinistro por linha),
      exportado do Dashboard Sinistros do DETRAN-AL.
    - Frota: série histórica de frota de veículos em Alagoas (Google Sheets
      "Frota Geral Alagoas - 1990-2024").

Como usar:
    python src/load_data.py
O resultado tratado (agregado por mês) é salvo em data/processed/frota_sinistros_al.csv
"""

import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

MESES_PT = {
    "janeiro": 1, "fevereiro": 2, "março": 3, "abril": 4,
    "maio": 5, "junho": 6, "julho": 7, "agosto": 8,
    "setembro": 9, "outubro": 10, "novembro": 11, "dezembro": 12,
}


def encontrar_arquivo(deve_conter: str, nao_conter: str = None) -> Path:
    """
    Procura em data/raw/ um .csv cujo nome contenha `deve_conter` (e, se
    informado, NÃO contenha `nao_conter`) — sem diferenciar maiúsculas de
    minúsculas. Evita quebrar o script quando o nome do arquivo baixado vem
    diferente do esperado.
    """
    candidatos = [f for f in RAW_DIR.glob("*.csv") if deve_conter.lower() in f.name.lower()]
    if nao_conter:
        candidatos = [f for f in candidatos if nao_conter.lower() not in f.name.lower()]
    if not candidatos:
        existentes = [f.name for f in RAW_DIR.glob("*.csv")]
        raise FileNotFoundError(
            f"Nenhum .csv em data/raw/ contém '{deve_conter}' no nome "
            f"(excluindo '{nao_conter}'). Arquivos existentes: {existentes}"
        )
    if len(candidatos) > 1:
        print(f"[aviso] mais de um arquivo bate com '{deve_conter}', usando: {candidatos[0].name}")
    return candidatos[0]


def carregar_sinistros() -> pd.DataFrame:
    """
    Lê o microdado de sinistros do RENAEST/DETRAN-AL (um sinistro por linha).
    Colunas relevantes: Ano, Mês, Município, Bairro, Tipo Sinistro, Óbito, etc.
    """
    arquivo = encontrar_arquivo("sinistro")
    print(f"Carregando sinistros de: {arquivo.name}")
    df = pd.read_csv(arquivo, encoding="utf-8")
    df.columns = [c.strip() for c in df.columns]
    return df


def carregar_frota_historica() -> pd.DataFrame:
    """
    Lê a série histórica de frota de veículos em Alagoas (1990-2024).
    Os números vêm no formato brasileiro (ponto como separador de milhar,
    ex: "1.036.837"), por isso o `thousands="."` abaixo. A primeira coluna
    (sem nome) e a linha em branco depois do cabeçalho vêm do jeito que o
    Google Sheets exporta e são descartadas aqui.
    """
    arquivo = encontrar_arquivo("frota", nao_conter="bairro")
    print(f"Carregando frota histórica de: {arquivo.name}")
    df = pd.read_csv(arquivo, thousands=".")
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    df = df.dropna(subset=["Ano"])
    df["Ano"] = df["Ano"].astype(int)
    df.columns = [c.strip().lower().replace(" ", "_").replace(".", "") for c in df.columns]
    return df


def agregar_sinistros_mensal(sinistros: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega o microdado de sinistros (um por linha) para uma série mensal:
    total de sinistros e total de óbitos por Ano/Mês, em todo o estado.

    Quem quiser detalhar por município depois, basta incluir 'Município' no
    groupby abaixo.
    """
    df = sinistros.copy()
    df["mes_num"] = df["Mês"].str.lower().map(MESES_PT)
    df["obito_flag"] = df["Óbito"].fillna(0) > 0

    agregado = (
        df.groupby(["Ano", "mes_num"])
        .agg(
            total_sinistros=("SIN", "count"),
            total_obitos=("Óbito", lambda s: s.fillna(0).sum()),
            sinistros_com_obito=("obito_flag", "sum"),
        )
        .reset_index()
        .sort_values(["Ano", "mes_num"])
    )
    return agregado


def cruzar_com_frota(sinistros_mensal: pd.DataFrame, frota: pd.DataFrame) -> pd.DataFrame:
    """
    Cruza a série mensal de sinistros com a frota anual.

    A frota só é registrada uma vez por ano (não por mês), então o valor do
    ano é repetido em todos os meses daquele ano — isso é uma SIMPLIFICAÇÃO
    que deve ser citada como limitação no relatório (ver docs/notas_eticas.md):
    o modelo vai enxergar a frota como "constante" dentro de cada ano, então
    ele aprende a variação sazonal dos sinistros e o efeito do crescimento
    ano a ano da frota, mas não consegue captar uma variação de frota
    dentro do mesmo ano.
    """
    colunas_frota = [
        "ano", "total_do_estado", "total_de_carros", "total_de_motos",
        "motocicleta", "automovel",
    ]
    colunas_frota = [c for c in colunas_frota if c in frota.columns]

    base = sinistros_mensal.merge(
        frota[colunas_frota], left_on="Ano", right_on="ano", how="left"
    ).drop(columns=["ano"])

    sem_frota = base["total_do_estado"].isna().sum()
    if sem_frota:
        anos_sem_frota = sorted(base.loc[base["total_do_estado"].isna(), "Ano"].unique())
        print(
            f"[aviso] {sem_frota} meses sem dado de frota correspondente "
            f"(anos: {anos_sem_frota}) — a planilha de frota só vai até 2024, "
            f"então 2025 fica sem esse cruzamento até sair um dado mais recente."
        )

    return base


def main():
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    sinistros = carregar_sinistros()
    print("Sinistros (microdados):", sinistros.shape)

    sinistros_mensal = agregar_sinistros_mensal(sinistros)
    print("Sinistros agregados por mês:", sinistros_mensal.shape)
    print(sinistros_mensal.head(12))

    frota = carregar_frota_historica()
    base = cruzar_com_frota(sinistros_mensal, frota)

    saida = PROCESSED_DIR / "frota_sinistros_al.csv"
    base.to_csv(saida, index=False)
    print(f"Base tratada salva em: {saida}")


if __name__ == "__main__":
    main()
