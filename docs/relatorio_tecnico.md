# Relatório Técnico — Triagem de Doenças Oculares em Imagens de Retina

> Rascunho em Markdown. Depois de finalizado, exportem para Word/PDF (10-15
> páginas) para a entrega final.

## 1. Introdução
- Contexto: importância da detecção precoce de doenças oculares (catarata,
  glaucoma, retinopatia diabética) e da falta de acesso a oftalmologistas
  em muitas regiões
- Motivação (triagem assistida por IA como forma de priorizar
  encaminhamentos, não de substituir diagnóstico)
- Objetivo do trabalho

## 2. Fundamentação teórica
- Classificação de imagens com redes neurais convolucionais (CNNs)
- Transfer learning: por que reaproveitar uma rede pré-treinada (MobileNetV2 /
  ImageNet) em vez de treinar do zero, especialmente com dataset pequeno
- Trabalhos relacionados (estudos que usaram este mesmo dataset ou datasets
  similares de retina — citar ao menos 1-2 artigos)

## 3. Metodologia
### 3.1 Fonte e coleta dos dados
- Dataset Kaggle `eye-diseases-classification`, 4 classes, ~1.000 imagens/classe
### 3.2 Pré-processamento e engenharia de features
- Redimensionamento para 224x224, normalização (`preprocess_input` da
  MobileNetV2), data augmentation (flip, rotação, zoom, contraste) só no treino
### 3.3 Divisão treino/validação/teste
- 70% treino / 15% validação / 15% teste, split estratificado por classe
  (via `image_dataset_from_directory`)
### 3.4 Arquitetura do modelo
- Base MobileNetV2 (congelada, pesos ImageNet) + `GlobalAveragePooling2D` +
  `Dense` + `Dropout` + camada de saída softmax (4 classes)
- Etapa 1: treino só da cabeça nova (base congelada)
- Etapa 2: fine-tuning das últimas N camadas da base, com learning rate bem
  menor
### 3.5 Métricas de avaliação
- Acurácia, F1-score (macro), matriz de confusão no conjunto de teste

## 4. Resultados
### 4.1 Curvas de treino (loss/accuracy por época, treino vs. validação)
### 4.2 Métricas finais no conjunto de teste + matriz de confusão
### 4.3 Interpretabilidade (Grad-CAM em exemplos de cada classe)
- Discutir se as regiões destacadas fazem sentido clínico ou se há sinal de
  "atalho"/viés (ver `docs/notas_eticas.md`)

## 5. Auditoria ética e de vieses
(usar `docs/notas_eticas.md` como ponto de partida — dado sensível de saúde,
ausência de metadados demográficos, classes artificialmente balanceadas,
risco de uso indevido como diagnóstico definitivo)

## 6. Deploy
- Arquitetura da API (FastAPI + Docker)
- Endpoint `/prever`: recebe upload de imagem, devolve classe prevista,
  confiança e probabilidades de todas as classes + aviso de que não é
  diagnóstico médico
- Como reproduzir o ambiente
- Endpoint documentado (OpenAPI/Swagger em `/docs`)

## 7. Conclusão
- Limitações (dataset pequeno, sem validação clínica, sem metadados
  demográficos)
- Trabalhos futuros (mais dados, validação com especialistas, calibração de
  probabilidade, explicabilidade mais robusta)

## Referências
