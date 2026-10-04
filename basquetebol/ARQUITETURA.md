# Arquitetura — Plataforma de Avaliação Quantitativa Técnica · Basquetebol de Formação

Autor e responsável metodológico: Mário Silva · Documento de referência: `ENB_Avaliacao_Quantitativa.docx`

## Análise do documento ENB
| Teste | Unidade | Melhor | Tentativas (protocolo) |
|---|---|---|---|
| I Movimentos Defensivos (Johnson e Nelson, 1986) | s | menor | 3, recuperação 5 min |
| II Drible (Matulaitis et al., 2019) | s | menor | 3, recuperação 5 min |
| III Pontos Marcados (Matulaitis et al., 2019) | pontos/cestos | maior | **não especificado — a confirmar** |
| IV Illinois com Drible (Getchell et al., 1998) | s | menor | 1; repete (máx. 3) se perder a bola |
| V One Minute Shooting Test | pontos | maior | 2 (trocando o lado) |
| VI Velocidade e Coordenação (Defensive Movement Test) | s | menor | 2 |
| VII Lançamentos Livres (Stonkus, 2002) | cestos em 30 | maior | 1 |

Pontos assinalados na aplicação:
* O documento refere **Sub-10, Sub-12 e Sub-14**. **Sub-8 não tem protocolo** → marcado «sem protocolo ENB», nenhum teste pré-selecionado.
* **Teste III**: nº de tentativas e forma de pontuar não estão no documento (assumido 1 tentativa, resultado = cestos; marcado «a confirmar»).
* O documento traz percentis do **T-Test (12–20 anos)** dentro do teste VI; não correspondem a esse protocolo → **não foram carregados** como norma.
* Não existem normas nem percentis pré-carregados. Sem tabela ativa → valores absolutos.

## Camadas
```
Browser (HTML + CSS + Chart.js local)  <- páginas Jinja2 (servidor)
FastAPI  app/routes_*.py     páginas e formulários
app/services.py, scoring.py  regras: melhor resultado, sentido do teste, estatísticas, referências
app/reports.py               PDF (ReportLab)
SQLAlchemy  app/models.py    SQLite local  ->  PostgreSQL (variável BASQ_DATABASE_URL)
```

## Base de dados
`categories` (escalões) · `teams` · `athletes` · `tests` (protocolo, unidade, sentido, nº tentativas) ·
`category_tests` (testes por escalão) · `evaluations` (atleta + data + escalão à data + avaliador + observações) ·
`test_results` (avaliação × teste, melhor resultado, observações) · `attempts` (cada tentativa) ·
`reference_tables` / `reference_rows` (normas validadas, introduzidas pelo utilizador; vazio por defeito).

## Regras metodológicas implementadas
* Tempos: menor = melhor; pontos/cestos: maior = melhor (campo `direction` por teste; usado no melhor resultado, variações e cores).
* Nunca há média global entre testes: tudo é por teste/unidade.
* Comparação entre escalões: opcional, apenas descritiva, com aviso.
* Novas baterias: «Testes → Novo teste» (sem alterar código).

## Publicação online / vários treinadores (ainda NÃO implementado)
Trocar a base de dados com `BASQ_DATABASE_URL` (PostgreSQL) é possível. Falta: autenticação, tabela de utilizadores
(`evaluator` passará a FK), permissões por clube, migrações (Alembic), HTTPS.

## Plano por etapas
1. ✔ Atletas, equipas, testes, registo de avaliações (individual e em lote).
2. ✔ Painel, gráficos, análise coletiva, estrutura de tabelas de referência.
3. ✔ Relatórios PDF.
4. Próximo: validação metodológica (teste III, Sub-8, normas), importação/exportação CSV, cópias de segurança.
5. Autenticação e multi-treinador; publicação.
