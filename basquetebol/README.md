# Plataforma de Avaliação Quantitativa Técnica — Basquetebol de Formação

Autor e responsável metodológico: **Mário Silva** · Protocolos: documento ENB «Avaliação Quantitativa».
Arquitetura e decisões: [`docs/ARQUITETURA.md`](docs/ARQUITETURA.md).

## Como executar (em linguagem simples)

1. Instale o **Python 3.11 ou superior** (python.org).
2. Abra a pasta `basquetebol` num terminal e escreva (uma só vez):
   ```
   pip install -r requirements.txt
   ```
3. Para abrir a aplicação (sempre que quiser usar):
   ```
   python run.py
   ```
   O browser abre em **http://127.0.0.1:8000**. Para fechar, carregue em `Ctrl+C` no terminal.
4. Os dados ficam guardados no ficheiro `data/avaliacao.db`. **Para fazer uma cópia de segurança, copie esse ficheiro.**
5. No telemóvel (mesma rede Wi-Fi): execute `uvicorn app.main:app --host 0.0.0.0 --port 8000` e abra `http://IP-DO-COMPUTADOR:8000`.

## Como utilizar

1. **Equipas** → crie as equipas (clube + escalão). *(opcional)*
2. **Atletas** → «+ Novo atleta»: nome, data de nascimento, sexo, escalão, equipa, clube. Pesquise/filtre por escalão, equipa, sexo ou nome. A **ficha** mostra o histórico, a evolução e permite gerar o PDF.
3. **Nova avaliação** → escolha o atleta, a data e preencha as tentativas dos testes realizados. O **melhor resultado** é calculado automaticamente (tempos: menor; pontos/cestos: maior). Pode acrescentar observações por tentativa e por teste.
4. **Avaliação coletiva** → escolha um teste, uma equipa (ou escalão) e a data; preencha todos os atletas numa grelha.
5. **Painel** → resumo; **Análise e gráficos** → média coletiva, evolução individual, distribuição, comparação entre datas e (quando adequado) entre escalões.
6. **Relatórios PDF** → individual ou coletivo, com período e testes à escolha, e espaço para observações do treinador.
7. **Testes e protocolos** → descrição, material, tentativas e unidade de cada teste. **Configuração** → testes usados por escalão e importação de tabelas de referência.

## Pontos metodológicos (importante)

- **Não há normas nem percentis inventados.** Sem tabela de referência aparece «Sem referências» e o resultado absoluto. Tabelas validadas podem ser importadas em *Configuração → Tabelas de referência* (CSV `rotulo;minimo;maximo`).
- **Sub-8:** o documento ENB não define testes para este escalão — nenhum vem selecionado.
- **Teste de Pontos Marcados:** o documento não indica o indicador de resultado nem o n.º de tentativas (configurado provisoriamente como cestos convertidos, 1 tentativa — «a confirmar»). **Teste de Drible:** unidade assumida em segundos.
- O quadro de percentis do T-Test do documento (por idade, 12–20 anos) **não foi carregado** como norma.
- Nunca se calcula uma média global entre testes de unidades diferentes.

## Acrescentar novas baterias

Pela interface (*Testes → Novo teste*) ou acrescentando um bloco em `data/catalogo_testes.json` (é carregado no arranque, sem sobrescrever o que já existe).

## Testes automáticos

```
python -m pytest tests
```

## Publicação online / vários treinadores (futuro)

Defina `DATABASE_URL` (ex.: PostgreSQL), execute com `uvicorn`/`gunicorn` atrás de HTTPS e acrescente autenticação e a tabela de treinadores (ver arquitetura).
