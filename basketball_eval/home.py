"""Ponto de entrada da aplicação de avaliação de Minibasquete (navegação).

Executar: streamlit run basketball_eval/home.py

Cada fundamento técnico futuro (Drible, Passe…) acrescenta-se como nova página no grupo
«Avaliação · Técnica Individual». O módulo quantitativo existente (app.py) mantém-se intacto.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__))

pages = {
    "Avaliação · Técnica Individual": [
        st.Page(os.path.join(HERE, "ui", "lancamento.py"), title="Lançamento", icon="🏀",
                url_path="lancamento", default=True),
    ],
    "Avaliação quantitativa": [
        st.Page(os.path.join(HERE, "app.py"), title="Movimentos Defensivos", icon="🛡️",
                url_path="movimentos-defensivos"),
    ],
}

st.navigation(pages).run()
