"""
Treino e avaliação dos modelos: baseline (regressão linear) e final (Random Forest).

Rodar depois de gerar data/processed/frota_sinistros_al.csv com load_data.py

IMPORTANTE sobre vazamento de dados (um dos critérios da rubrica):
Como esses dados são uma SÉRIE TEMPORAL (um valor por mês), não dá pra usar
train_test_split embaralhando aleatoriamente — isso deixaria meses do futuro
no treino e meses do passado no teste, o que é vazamento de dados. Por isso
aqui separamos treino/teste em ordem cronológica: treina com os meses mais
antigos, testa com os mais recentes.
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODELS_DIR = Path(__file__).resolve().parent.parent / "api"

COLUNA_ALVO = "total_sinistros"
COLUNAS_FEATURES = ["mes_num", "total_do_estado", "total_de_motos", "automovel"]

# Proporção dos meses mais recentes usada como teste (ex: 0.2 = últimos 20%)
PROPORCAO_TESTE = 0.2


def avaliar(y_true, y_pred, nome_modelo: str):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    print(f"[{nome_modelo}] MAE={mae:.2f} | RMSE={rmse:.2f} | R2={r2:.3f}")
    return {"modelo": nome_modelo, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = pd.read_csv(PROCESSED_DIR / "frota_sinistros_al.csv")
    df = df.sort_values(["Ano", "mes_num"]).dropna(subset=[COLUNA_ALVO] + COLUNAS_FEATURES)

    print(f"Total de meses com dado completo (frota + sinistros): {len(df)}")
    if len(df) < 20:
        print(
            "[aviso] amostra pequena (menos de 20 meses) — os resultados servem "
            "para o exercício, mas comentem essa limitação no relatório."
        )

    corte = int(len(df) * (1 - PROPORCAO_TESTE))
    treino, teste = df.iloc[:corte], df.iloc[corte:]
    print(
        f"Treino: {treino['Ano'].min()}-{treino['mes_num'].min():02d} até "
        f"{treino['Ano'].max()}-{treino['mes_num'].max():02d} ({len(treino)} meses)"
    )
    print(
        f"Teste:  {teste['Ano'].min()}-{teste['mes_num'].min():02d} até "
        f"{teste['Ano'].max()}-{teste['mes_num'].max():02d} ({len(teste)} meses)"
    )

    X_train, y_train = treino[COLUNAS_FEATURES], treino[COLUNA_ALVO]
    X_test, y_test = teste[COLUNAS_FEATURES], teste[COLUNA_ALVO]

    # --- Baseline ---
    baseline = LinearRegression()
    baseline.fit(X_train, y_train)
    avaliar(y_test, baseline.predict(X_test), "Baseline (Regressão Linear)")

    # --- Modelo final ---
    modelo_final = RandomForestRegressor(n_estimators=300, random_state=42)
    modelo_final.fit(X_train, y_train)
    avaliar(y_test, modelo_final.predict(X_test), "Modelo Final (Random Forest)")

    # Salvar o modelo final para a API usar
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo_final, MODELS_DIR / "modelo_final.pkl")
    print(f"Modelo salvo em: {MODELS_DIR / 'modelo_final.pkl'}")


if __name__ == "__main__":
    main()
