# Notas éticas e de vieses (rascunho para o relatório)

## 1. Natureza sensível do dado

Imagens de retina são **dado de saúde**, categoria de dado sensível pela LGPD
(art. 5º, II). Mesmo o dataset público usado aqui sendo anonimizado, se este
projeto fosse além do exercício acadêmico (uso com pacientes reais), seria
necessário: consentimento explícito, base legal específica para tratamento
de dado sensível, e cuidado redobrado com armazenamento/descarte das
imagens enviadas à API.

## 2. Composição e origem do dataset

O dataset (Kaggle, `gunavenkatdoddi/eye-diseases-classification`) reúne
imagens de fontes públicas diferentes (ex: IDRiD, HRF, entre outras),
compiladas por terceiros para fins de pesquisa. Isso traz alguns pontos de
atenção que devem entrar no relatório:

1. **Sem metadados demográficos.** Não há informação de idade, sexo, etnia,
   país de origem ou tipo de câmera/aparelho usado para capturar cada
   imagem. Isso impede qualquer auditoria de viés demográfico — não dá pra
   saber se o modelo funciona igualmente bem para diferentes perfis de
   população, e é importante deixar essa limitação explícita.

2. **Fontes heterogêneas misturadas.** Como as imagens vêm de bases
   diferentes, pode haver diferenças sistemáticas de equipamento,
   iluminação, resolução e enquadramento entre classes — o que cria risco
   de o modelo aprender um "atalho" (ex: aprender a reconhecer o dataset de
   origem em vez da doença em si) em vez do padrão clínico real. Vale usar
   Grad-CAM (ou técnica equivalente) para verificar se o modelo está
   "olhando" para a região da retina relevante (disco óptico, mácula, vasos)
   e não para bordas/artefatos da imagem.

3. **Classes artificialmente balanceadas.** O dataset tem ~1.000 imagens por
   classe — um balanceamento que **não reflete a prevalência real** dessas
   condições na população. Isso é bom para o treino (evita viés de classe
   majoritária), mas significa que a probabilidade que o modelo retorna
   **não pode ser lida como uma probabilidade clínica real** de o paciente
   ter a doença; ela reflete a distribuição do dataset, não a epidemiologia.

4. **Amostra pequena.** ~4.200 imagens no total é pouco para um problema
   médico de alta estrada — reforça por que usamos transfer learning (a
   MobileNetV2 já vem com um extrator de características forte da ImageNet)
   em vez de treinar do zero, e por que os resultados devem ser tratados como
   prova de conceito, não como validação clínica.

## 3. Risco de uso indevido / more-harm-than-good

Este é o ponto mais importante a deixar claro no relatório e na
apresentação:

- **O modelo é uma ferramenta de apoio à triagem, não um diagnóstico.** Um
  resultado "normal" nunca deve ser usado para dispensar uma consulta com
  oftalmologista — um falso negativo aqui pode atrasar o tratamento de uma
  doença progressiva (ex: glaucoma, que causa dano irreversível se não
  tratado a tempo).
- **Falsos positivos** também têm custo: geram ansiedade e exames
  desnecessários. A API já retorna a probabilidade de cada classe (não só a
  classe vencedora) exatamente para permitir que quem interpretar o
  resultado veja o grau de incerteza, em vez de tratar a saída como
  binária/definitiva.
- **Nunca deve ser usado sozinho para negar/aprovar** algo com impacto real
  na vida de alguém (ex: elegibilidade a plano de saúde, seguro, ou decisão
  de tratamento) — isso seria discriminação/automação indevida de decisão
  em cima de um modelo que nem foi validado clinicamente.
- O aviso de "isto não é diagnóstico médico" está embutido diretamente na
  resposta da API (`aviso` no `SaidaPredicao`), para não depender de alguém
  lembrar de avisar o usuário final.

## 4. Como usar isso na interpretabilidade

Depois de treinar o modelo, gerar Grad-CAM (ou similar) para algumas imagens
de cada classe e verificar visualmente se as regiões destacadas fazem
sentido clínico (ex: para retinopatia diabética, o modelo deveria reagir a
microaneurismas/hemorragias nos vasos, não ao fundo da imagem) — isso é
exatamente o tipo de evidência que a dimensão "Ética e interpretabilidade"
da rubrica está pedindo, e também ajuda a detectar o risco descrito no item
2.2 acima (o modelo "trapaceando" olhando para artefato em vez do sinal
clínico real).
