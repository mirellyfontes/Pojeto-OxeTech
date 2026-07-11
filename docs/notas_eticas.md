# Notas éticas e de vieses (rascunho para o relatório)

## Achados no dataset real (microdados RENAEST/DETRAN-AL, 2022-2025, 10.922 registros)

1. **Concentração geográfica.** Maceió (4.429 registros) e Arapiraca (2.442)
   somam mais de 60% de todos os sinistros registrados, enquanto Alagoas tem
   mais de 100 municípios. Isso pode refletir tanto maior trânsito real nessas
   cidades quanto maior capacidade de registro/fiscalização — importante
   deixar claro que o modelo será mais confiável para os municípios com mais
   dados, e generaliza pouco para o interior.

2. **Sub-registro geográfico.** Cerca de 35% dos registros não têm
   coordenadas (Lat/Long) preenchidas — mais de um terço dos sinistros não
   pode ser mapeado espacialmente. Se o modelo usar localização como
   variável, essa parcela fica de fora ou precisa de tratamento à parte.

3. **Campo "Comunicante" praticamente vazio** (quase 100% nulo) e "Fonte"
   com ~30% de nulos — sinal de que parte da coleta em campo não preenche
   todos os metadados, o que limita análises sobre quem/como reporta.

4. **Valor discrepante em "Óbito".** A maioria dos registros tem 0 ou 1
   óbito, mas há um valor "19" isolado — provavelmente erro de digitação ou
   caso excepcional que merece ser investigado/tratado antes de virar
   variável alvo (não descartar sem checar a linha original).

5. **Sub-registro geral.** O próprio RENAEST reconhece que o número de
   sinistros é relativo apenas aos que foram oficialmente registrados, e que
   a disponibilidade de dados varia por município e estado. Bairros/municípios
   com menor presença policial tendem a ter menos sinistros *registrados*,
   não necessariamente menos sinistros *reais*.

6. **Mudança de nomenclatura ("sinistro" x "acidente").** O DETRAN-AL adota
   oficialmente o termo "sinistro" em vez de "acidente" desde 2023 (NBR 10.697
   e Código de Trânsito Brasileiro), justamente para não tratar mortes e
   lesões no trânsito como eventos aleatórios e inevitáveis — vale citar essa
   mudança conceitual na fundamentação do relatório.

7. **Uso indevido do modelo.** Deixar explícito no relatório que o modelo
   serve para **apoio a políticas públicas de segurança viária** (ex:
   priorização de fiscalização e investimento em infraestrutura), e não para
   fins discriminatórios como negar seguro ou serviços a moradores de bairros
   com score de risco alto — esse é o tipo de "discriminação indireta" que o
   critério de avaliação do curso pede para discutir.

## Como usar isso no SHAP/LIME

Depois de rodar o SHAP no modelo final, verifiquem se a variável geográfica
(bairro/município) aparece com peso desproporcional — isso seria um indício
direto do viés de cobertura descrito acima, e é exatamente o tipo de achado
que a dimensão "Ética e interpretabilidade" da rubrica está pedindo.
