# Publicar a app de avaliação (vários treinadores e dispositivos)

Arquitetura: **Streamlit Community Cloud** (a app) + **Postgres** (os dados, partilhados e persistentes) +
**contas individuais** dos treinadores (ou, em alternativa simples, uma palavra-passe única do clube). O Supabase tem plano gratuito e serve de exemplo; qualquer Postgres funciona (Neon, etc.).

> A app guarda dados pessoais de menores. Veja a secção «Privacidade» no fim.

## 1. Criar a base de dados (Supabase)
1. Crie conta em supabase.com → **New project**. Escolha uma região **na UE** e guarde a palavra-passe da base de dados.
2. Abra **Connect** → separador **Session pooler** → copie o URL (`postgresql://postgres.xxxx:[YOUR-PASSWORD]@…pooler.supabase.com:5432/postgres`)
   e substitua `[YOUR-PASSWORD]`. Use o *pooler* (o endereço «direct» só tem IPv6 e a cloud do Streamlit não o alcança).
3. (Recomendado) No **SQL Editor**, ative a segurança por linha, para que nada fique exposto pela API pública do Supabase.
   A app liga-se com o utilizador `postgres`, que não é afetado:
   ```sql
   ALTER TABLE coaches, players, teams, age_groups, evaluations, evaluation_items,
               defensive_movement_tests, reference_norms, test_protocols ENABLE ROW LEVEL SECURITY;
   ```
   (execute depois de a app ter criado as tabelas, no primeiro arranque do passo 3).

## 2. Código no GitHub
O Streamlit Cloud lê o repositório no GitHub. Faça merge da PR (ou escolha a branch no passo 3).

## 3. Criar a app no Streamlit Community Cloud
1. share.streamlit.io → entre com GitHub → **Create app**.
2. Repositório, branch e **Main file path: `basketball_eval/home.py`**
   (o ficheiro `basketball_eval/requirements.txt` é usado, só com o necessário).
3. **Advanced settings → Secrets**, cole (ver `.streamlit/secrets.toml.example`):
   ```toml
   DATABASE_URL = "postgresql://postgres.xxxx:SUA_PASSWORD@…pooler.supabase.com:5432/postgres"
   ADMIN_USERNAME = "admin"
   ADMIN_PASSWORD = "palavra-passe-temporaria-longa"
   ```
4. **Deploy**. No primeiro acesso as tabelas são criadas e é criado o administrador `admin`.

A app **recusa arrancar** com `DATABASE_URL` sem `ADMIN_PASSWORD` (contas) nem `APP_PASSWORD`.

## 3b. Contas dos treinadores
1. Abra o link e entre como `admin` com `ADMIN_PASSWORD`. A app obriga a definir já a sua própria palavra-passe;
   depois pode **apagar `ADMIN_PASSWORD` dos segredos**.
2. Menu **Conta → Utilizadores → Nova conta**: utilizador (ex.: `rui.costa`), nome, perfil e uma palavra-passe
   temporária. Entregue-a ao treinador: no primeiro acesso é obrigado a mudá-la.
3. As avaliações ficam associadas ao treinador com sessão iniciada, e o registo de ações (criar/editar/apagar
   avaliações, gerir contas) fica em «Registo de ações».
4. Treinador esqueceu-se da palavra-passe → administrador usa «Repor palavra-passe». Saiu do clube → «Desativar conta»
   (perde o acesso de imediato, mesmo com sessão aberta; o histórico mantém-se).
5. Segurança: palavras-passe guardadas só com hash (scrypt), mínimo de 10 caracteres, bloqueio de 5 minutos após
   5 tentativas falhadas, e nunca é possível desativar o último administrador.

Alternativa simples: em vez de contas, defina só `APP_PASSWORD` (palavra-passe única do clube).

## 4. Levar os dados que já tem no computador (opcional)
```bash
pip install psycopg2-binary
python scripts/migrate_basketball_sqlite_to_postgres.py "postgresql://…pooler…/postgres" data/basketball_eval.db
```
Copia jogadores, treinadores e avaliações mantendo as ligações. Recusa-se a copiar para tabelas que já tenham dados.

## Uso local (sem cloud)
Sem `DATABASE_URL` a app usa o ficheiro SQLite `data/basketball_eval.db`, como antes:
`streamlit run basketball_eval/home.py`. Para testar localmente contra Postgres, crie `.streamlit/secrets.toml`
a partir do exemplo.

## Testes contra Postgres
```bash
TEST_DATABASE_URL=postgresql://user@localhost:5432/teste python -m pytest tests/test_qualitative_shooting.py
```
Use uma base **descartável**: cada teste apaga o esquema `public`.

## Limitações a conhecer
- Todos os treinadores veem todas as equipas (não há permissões por equipa). Só o administrador gere contas.
- Recarregar a página no navegador termina a sessão (comportamento do Streamlit): é preciso entrar de novo.
- Não há recuperação de palavra-passe por email; é o administrador que a repõe.
- Apps gratuitas do Streamlit Cloud adormecem após inatividade (o primeiro acesso demora); projetos inativos do plano
  gratuito do Supabase podem ser pausados. Confirme os limites atuais nos respetivos planos.
- Cópias de segurança: confirme o que o seu plano inclui; caso contrário exporte periodicamente (`pg_dump` ou o export do Supabase).
- A edição simultânea da mesma avaliação por dois treinadores não é protegida (ganha a última gravação).

## Privacidade (dados de menores)
Trata-se de dados pessoais de crianças: use região UE, palavra-passe forte (mude-a quando um treinador sair),
não partilhe o link publicamente, recolha apenas o necessário (nome ou alcunha chega) e verifique com o clube o
consentimento dos encarregados de educação e as regras de proteção de dados aplicáveis.
