# Relatório Técnico — Frota Veicular x Sinistros de Trânsito em Alagoas

> Rascunho em Markdown. Depois de finalizado, exportem para Word/PDF (10-15
> páginas) para a entrega final.

## 1. Introdução
- Contexto do problema em Alagoas
- Motivação (segurança viária, políticas públicas — PNATRANS)
- Objetivo do trabalho

## 2. Fundamentação teórica
- Regressão aplicada a dados de trânsito
- Trabalhos relacionados (anuários RENAEST, estudos DETRAN-AL)

## 3. Metodologia
### 3.1 Fonte e coleta dos dados
### 3.2 Pré-processamento e engenharia de features
### 3.3 Divisão treino/teste (sem vazamento de dados)
### 3.4 Modelos utilizados (baseline vs. final)
### 3.5 Métricas de avaliação

## 4. Resultados
### 4.1 Comparação baseline vs. modelo final
### 4.2 Interpretabilidade (SHAP/LIME)

## 5. Auditoria ética e de vieses
(usar `docs/notas_eticas.md` como ponto de partida)

## 6. Deploy
- Arquitetura da API (FastAPI + Docker)
- Como reproduzir o ambiente
- Endpoint documentado (OpenAPI/Swagger)

## 7. Conclusão
- Limitações
- Trabalhos futuros

## Referências
