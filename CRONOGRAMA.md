# Cronograma — 11 a 20/07 (10 dias, grupo de 2 pessoas)

Premissa: nenhum dos dois tem muita experiência com análise de dados ainda. Por isso
o cronograma é bem passo a passo, com uma tarefa "principal" por dia — evitem tentar
fazer tudo em paralelo, é mais fácil terminar uma coisa de cada vez.

| Dia | Data | Foco do dia | Entregável do curso |
|---|---|---|---|
| 1 | Sáb 11/07 | Criar repositório no GitHub, configurar ambiente (venv, VSCode, extensão Jupyter), baixar os datasets do DETRAN-AL para `data/raw/` | — |
| 2 | Dom 12/07 | Abrir os arquivos no notebook, entender as colunas de cada planilha (frota e sinistros), anotar dúvidas/inconsistências | — |
| 3 | Seg 13/07 | Cruzar frota + sinistros por bairro/mês, definir a métrica final, escrever a proposta (problema, dataset, métrica) | **Proposta** |
| 4 | Ter 14/07 | EDA completo (gráficos, distribuição do alvo, valores faltantes) + treinar um modelo baseline simples (regressão linear) | **Baseline** |
| 5 | Qua 15/07 | Treinar modelo mais robusto (Random Forest ou Gradient Boosting) e comparar com o baseline usando a métrica definida | — |
| 6 | Qui 16/07 | Interpretabilidade (SHAP ou LIME) + escrever a discussão de vieses (sub-registro em bairros periféricos, cobertura policial desigual, dados "desconhecido") | — |
| 7 | Sex 17/07 | Construir a API em FastAPI (endpoint de predição) + Dockerfile, testar localmente | — |
| 8 | Sáb 18/07 | Empacotar tudo (código final modular em `src/`), subir pro GitHub com README explicativo, começar o relatório técnico | **Código final + API com deploy** |
| 9 | Dom 19/07 | Finalizar o relatório técnico (10-15 páginas): fundamentação, metodologia, resultados, auditoria ética | **Relatório técnico** |
| 10 | Seg 20/07 | Revisar tudo, ensaiar a apresentação de 15 min com demonstração ao vivo da API, ajustes finais | **Defesa oral** |

## Dicas para dividir o trabalho a dois

- Uma pessoa foca em **dados + modelo** (dias 2 a 6), a outra foca em **API + Docker +
  redação do relatório** em paralelo a partir do dia 5, usando resultados parciais.
- Façam commits pequenos e frequentes no Git (a cada etapa concluída), assim o
  professor consegue acompanhar a evolução real ao longo dos dias, não só no fim.
- Se atrasar em algum dia, priorizem nesta ordem: **Pipeline (35%) > Deploy (20%) =
  Ética (20%) > Relatório/Defesa (25%)** — ou seja, não abram mão de ter um modelo
  funcionando e comparado com baseline, isso pesa mais que polir a apresentação.
