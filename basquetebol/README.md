# Plataforma de Avaliação Quantitativa Técnica — Basquetebol de Formação

Autor e responsável metodológico: **Mário Silva**. Ver `ARQUITETURA.md`.

## Como executar
1. Instale o **Python 3.10 ou superior** (python.org).
2. Num terminal, na pasta `basquetebol`, instale uma vez: `pip install -r requirements.txt`
3. Arranque: `python iniciar.py` — o browser abre em http://127.0.0.1:8000
4. Para parar: `Ctrl+C` no terminal.

Os dados ficam em `data/basquetebol.db` (copie este ficheiro para fazer cópia de segurança).
No telemóvel (mesma rede Wi-Fi): `uvicorn app.main:app --host 0.0.0.0` e abra `http://IP-do-computador:8000`.

## Como usar
1. **Equipas**: crie as equipas. 2. **Atletas**: registe cada atleta.
3. **Nova avaliação** (um atleta) ou **Registo por equipa** (vários atletas no mesmo teste): preencha as tentativas;
   o melhor resultado é calculado (menor tempo / maior pontuação).
4. Veja a evolução na **ficha do atleta**, no **Painel** e em **Análise**.
5. **Relatórios**: PDF individual ou coletivo (período e testes à escolha).
6. **Testes**: protocolos, testes por escalão, novos testes e tabelas de referência validadas (quando existirem).

Testes automáticos: `python -m pytest -q`
