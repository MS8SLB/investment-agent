# Plataforma de Avaliação do Minibasquete

Aplicação para treinadores, clubes e escolas avaliarem **individual e coletivamente** jogadores **Sub-8, Sub-10 e Sub-12**,
acompanharem a evolução ao longo da época e comunicarem com os encarregados de educação.

> *Sem evolução individual, não há sucesso coletivo.*  
> Ciclo: **observar → avaliar → identificar necessidades → definir objetivos → intervir no treino → reavaliar → verificar a evolução.**

Independente do agente de investimento do resto do repositório (pacote `minibasket/`, base de dados `data/minibasket.db`).

## Executar

```bash
pip install -r requirements.txt
streamlit run minibasket/app.py
```

Na primeira execução aparece a **configuração inicial**: crie a conta de administrador. Depois:

1. **Administrador** → «Equipas»: criar clube e equipas; «Utilizadores»: criar treinadores (associar clube e equipas) e
   encarregados de educação (associar os educandos).
2. **Treinador** → «Jogadores» (criar/abrir ficha) → «Avaliar» → «Evolução do Jogador / da Equipa» → «Relatórios» (PDF).
3. **Encarregado de educação** → «O meu educando»: relatório e evolução só do seu educando.

Dados fictícios para experimentar: «Utilizadores → Dados de teste → Carregar» ou `python -m minibasket.seed load`
(`remove` / `status`). Ficam sempre marcados «dados de teste» e separados dos reais; remover só apaga o que está marcado.
As contas de teste não têm palavra-passe (o administrador define uma em «Utilizadores»).

## Escala e competências

Nove competências, avaliadas de **1 a 5** (escala **pedagógica**, guardada na base de dados e substituível; não são normas
científicas nem percentis): Lançamento, Drible / Domínio da Bola, Passe, Receção da Bola, Trabalho de Pés, Finalizações,
Defesa Individual, Tática Individual, Contra-Ataque. 1 Inicial · 2 Em desenvolvimento · 3 Adequado · 4 Bom · 5 Muito bom.

A **média global nunca é introduzida nem guardada**: calcula-se sempre a partir das competências (avaliação incompleta →
média das avaliadas, assinalada como incompleta).

## Perfis e permissões

| Perfil | Pode |
|---|---|
| **Administrador** | Tudo, incluindo contas, acessos e dados de teste |
| **Treinador** | Ver/avaliar os jogadores atuais das equipas que acompanha (e acessos individuais); criar equipas no seu clube e jogadores nas suas equipas; relatórios e estatísticas dessas equipas; exportar PDF |
| **Encarregado de educação** | Só o(s) seu(s) educando(s): relatório para pais e evolução. Nunca vê colegas, equipas, estatísticas coletivas nem notas internas |

As permissões aplicam-se na camada de dados (`access.py`), não só na interface. Palavras-passe com PBKDF2-SHA256 e sal;
bloqueio de 15 min após 5 tentativas falhadas. Sem recuperação por e-mail (o administrador redefine) e sem
tempo-limite automático de sessão nesta versão.

## Regras de dados

- Histórico **nunca se apaga nem se substitui**: corrigir uma avaliação cria uma versão nova (`supersedes_id`) e a original fica.
- Cada avaliação guarda o **escalão à data**; mudar de escalão (nova pertença à equipa) mantém todo o histórico.
- Sem normas, percentis, rankings de crianças nem comparações negativas. A comparação jogador vs. equipa só aparece com ≥ 3
  jogadores avaliados e serve para identificar necessidades.
- Linguagem pedagógica e positiva (testada contra termos como «fraco» ou «mau jogador»).

## Estrutura

```
minibasket/
  db.py            esquema SQLite + migrações      competencies.py / scale.py   competências e escala
  service.py       clubes, equipas, jogadores      evaluations.py               avaliações (imutáveis, versões)
  calc.py          média, comparação               evolution.py / teamstats.py  evolução individual e coletiva
  reports.py       relatórios (treinador, equipa, pais)    pdf.py   exportação PDF (reportlab, gráficos vetoriais)
  charts.py        radar, linhas, barras (Plotly)  auth.py / access.py          contas e permissões
  seed.py          dados de teste                  views/ + app.py              interface Streamlit
tests/test_minibasket_phase*.py  (≈ 240 testes)    python -m pytest tests/test_minibasket_*.py
```

Modelo: `Club → Teams → Players → Evaluations → resultados por competência`, mais `users` (admin / treinador /
encarregado), `team_coaches`, `guardians_players`, `player_access`.

## Fora desta versão

Exportação Excel/CSV (prevista), recuperação de palavra-passe por e-mail, tempo-limite de sessão, registo de acessos.
