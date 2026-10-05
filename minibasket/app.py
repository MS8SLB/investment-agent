"""Plataforma de Avaliação do Minibasquete — esqueleto da aplicação.

Executar: streamlit run minibasket/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from minibasket import competencies as comp
from minibasket import db
from minibasket.views import evaluate as evaluate_view
from minibasket.views import player_evolution as evolution_view
from minibasket.views import players as players_view
from minibasket.views import teams as teams_view

st.set_page_config(page_title="Avaliação do Minibasquete", page_icon="🏀", layout="wide")
db.init_db()
scale_cfg = db.active_scale()

SECTIONS = [
    ("Dashboard", "Visão geral das equipas, avaliações e evolução."),
    ("Equipas", "Criar e gerir equipas Sub-8, Sub-10 e Sub-12."),
    ("Jogadores", "Fichas individuais, pesquisa e filtros."),
    ("Avaliar", "Ficha de avaliação das nove competências."),
    ("Evolução do Jogador", "Roda das Competências e evolução ao longo da época."),
    ("Evolução da Equipa", "Médias, medianas e evolução coletiva."),
    ("Relatórios", "Relatórios para treinadores e pais; exportação PDF."),
]

st.sidebar.title("🏀 Minibasquete")
section = st.sidebar.radio("Navegação", [s for s, _ in SECTIONS])
st.sidebar.caption("Sem evolução individual, não há sucesso coletivo.")

st.title(section)
st.caption(dict(SECTIONS)[section])

if section == "Dashboard":
    st.subheader("Ciclo pedagógico")
    st.markdown(
        "**Observar** → **Avaliar** → **Identificar necessidades** → **Definir objetivos** → "
        "**Intervir no treino** → **Reavaliar** → **Verificar a evolução**")
    st.subheader("Roda das Competências")
    st.write(" · ".join(c.name for c in comp.COMPETENCIES))
    st.subheader(f"Escala: {scale_cfg['name']}")
    st.write(" · ".join(f"**{v}** — {t}" for v, t in scale_cfg["levels"]))
    st.caption("Escala pedagógica de avaliação; não corresponde a normas científicas nem a percentis.")
elif section == "Equipas":
    teams_view.render()
elif section == "Avaliar":
    evaluate_view.render()
elif section == "Evolução do Jogador":
    evolution_view.render()
elif section == "Jogadores":
    players_view.render()
else:
    st.info("Secção prevista para uma fase seguinte.")
