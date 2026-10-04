# Plataforma de Avaliação Quantitativa Técnica — Basquetebol de Formação

Autor e responsável metodológico: **Mário Silva**
Documento de referência: `ENB_Avaliacao_Quantitativa.docx` (Escola Nacional de Basquetebol)

## 1. O que o documento ENB contém (e o que não contém)

| # | Teste | Tentativas (documento) | Resultado | Sentido |
|---|-------|------------------------|-----------|---------|
| I | Movimentos Defensivos (Johnson e Nelson, 1986) | 3, recuperação 5 min, melhor tempo | tempo (s) | menor = melhor |
| II | Drible (Matulaitis et al., 2019) | 3, recuperação 5 min, melhor resultado | tempo (s) — *implícito* | menor = melhor |
| III | Pontos Marcados (Matulaitis et al., 2019) | **não indicado** | **não indicado** | maior = melhor (instrução do autor) |
| IV | Illinois com Drible (Getchell et al., 1998) | 1; repete se perder a bola, máx. 3 | tempo (s) | menor = melhor |
| V | One Minute Shooting Test | 2 (trocando o lado) | pontos (1/2/3) | maior = melhor |
| VI | Velocidade e Coordenação (Defensive Movement Test) | 2, melhor tempo | tempo (s) | menor = melhor |
| VII | Lançamentos Livres (Stonkus, 2002) | 1 (30 lançamentos) | n.º de convertidos | maior = melhor |

**Lacunas assinaladas na aplicação (nada foi inventado):**
- Teste III: o documento não define a unidade de resultado nem o n.º de tentativas. A aplicação usa
  "cestos convertidos" e 1 tentativa como *configuração provisória*, marcada "A confirmar".
- Teste II: o documento não escreve a unidade, mas o teste é cronometrado → segundos (marcado "A confirmar").
- O documento intitula-se Sub10/Sub12/Sub14; **não há protocolo nem testes definidos para Sub-8**.
- O quadro de percentis do T-Test (idades 12–20) **não foi carregado** como norma: é por idade (não por
  escalão) e não é claro que se aplique a nenhum dos 7 testes. Pode ser importado mais tarde se o
  responsável metodológico o validar.
- Não existem normas/percentis validados para nenhum dos testes ⇒ a app mostra **resultados absolutos**.

## 2. Tecnologia

- **Python 3.11 + FastAPI** (servidor), **Jinja2** (páginas), **SQLAlchemy 2** (acesso a dados).
- **SQLite** localmente (um ficheiro, persistente). Troca para PostgreSQL = mudar a variável
  `DATABASE_URL` (sem alterar código).
- **Chart.js** (incluído localmente, funciona sem internet) para gráficos; **ReportLab + Matplotlib** para PDF.
- Razões: instalação simples para treinadores, uma única linguagem, fácil de publicar online
  (uvicorn/gunicorn + Postgres) e de acrescentar autenticação.

## 3. Organização do código

```
basquetebol/
  run.py                    arranque local
  data/catalogo_testes.json catálogo de testes (fonte: documento ENB) — acrescentar aqui novas baterias
  app/
    main.py  config.py  database.py  models.py  seed.py
    services/   avaliacao.py (regras de melhor resultado)  referencias.py  estatistica.py  pdf.py
    routers/    painel  atletas  equipas  testes  avaliacoes  analise  relatorios  configuracao
    templates/  static/
  tests/                    testes automáticos
```

Futuro multi-treinador: acrescentar tabela `treinador`, coluna `treinador_id` em equipas/avaliações e
autenticação (dependência FastAPI). Os routers já estão separados para o permitir.

## 4. Base de dados

`escalao` · `equipa` · `atleta` · `teste` · `escalao_teste` (testes usados por escalão) ·
`avaliacao` (atleta + data + observações) · `resultado_teste` (um teste dentro de uma avaliação, com o melhor
resultado) · `tentativa` (cada tentativa + observação) · `referencia_tabela` / `referencia_faixa`
(tabelas de referência validadas, introduzidas por CSV; vazias por defeito).

O sentido do resultado (`direcao` = `menor` ou `maior`) e as regras de validação ficam no teste, e
**toda a lógica (melhor resultado, evolução, cores dos gráficos) lê esse campo**.

## 5. Páginas

Painel · Atletas (lista, filtros, ficha com histórico e gráficos) · Equipas · Testes (catálogo e protocolos) ·
Nova avaliação (individual) · Avaliação coletiva (equipa × teste) · Avaliações (histórico) · Análise
(gráficos) · Relatórios (PDF) · Configuração (testes por escalão, referências).

## 6. Plano por etapas

1. **Atletas, equipas, escalões** + catálogo de testes ✔ (v1)
2. **Registo de avaliações** com tentativas e melhor resultado ✔ (v1)
3. **Testes por escalão** + referências (CSV) ✔ (v1)
4. **Painel e gráficos** ✔ (v1)
5. **Relatórios PDF** individual e coletivo ✔ (v1)
6. Próximo: importar/exportar Excel, autenticação e vários treinadores, publicação online, normas validadas.

## 7. Regras metodológicas implementadas

- Tempos: menor é melhor; pontos/cestos: maior é melhor (gráficos e variações respeitam o sentido).
- Nunca se calcula média entre testes/unidades diferentes (tudo é por teste).
- Sem referência ⇒ resultado absoluto, com aviso. Nunca se mostram percentis inventados.
- Comparação entre escalões: só por teste, separada por sexo, com n mínimo e aviso metodológico.
