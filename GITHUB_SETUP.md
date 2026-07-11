# Como criar o repositório no GitHub (passo a passo)

## 1. Criar o repositório vazio no GitHub
1. Acesse https://github.com e faça login.
2. Clique em **New repository**.
3. Nome sugerido: `projeto-transito-al` (ou outro nome descritivo).
4. Marque como **Public** (o professor precisa acompanhar).
5. NÃO marque para criar README/gitignore automaticamente (já temos os nossos).
6. Clique em **Create repository**.

## 2. Subir o projeto local para o GitHub
Dentro da pasta do projeto, no terminal do VSCode:

```bash
cd projeto-transito-al
git init
git add .
git commit -m "Estrutura inicial do projeto"
git branch -M main
git remote add origin https://github.com/SEU-USUARIO/projeto-transito-al.git
git push -u origin main
```

## 3. Adicionar a colega como colaboradora
1. No GitHub, vá em **Settings > Collaborators** do repositório.
2. Clique em **Add people** e adicione o usuário GitHub dela.
3. Ela deve clonar o repositório na máquina dela:
   ```bash
   git clone https://github.com/SEU-USUARIO/projeto-transito-al.git
   ```

## 4. Fluxo de trabalho recomendado (simples, para dois)
Como são só duas pessoas, não precisa de branches complicadas — dá pra trabalhar
direto na `main`, mas sempre puxando antes de começar a mexer:

```bash
git pull                     # antes de começar a trabalhar
# ... fazem as alterações ...
git add .
git commit -m "descrição curta do que foi feito"
git push
```

Se as duas forem mexer ao mesmo tempo em arquivos diferentes, não tem problema.
Se forem mexer no mesmo arquivo, combinem antes por mensagem pra evitar conflito.

## 5. Deixar o professor acompanhar
- Como o repositório é público, basta compartilhar o link do GitHub com ele.
- Façam commits ao longo dos dias (não só um commit gigante no final) — isso
  mostra o processo de desenvolvimento, o que costuma ser valorizado na correção.
- No README, mantenham a seção de instruções de uso sempre atualizada.
