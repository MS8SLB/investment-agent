"""Plataforma de Avaliação do Minibasquete.

Executar: streamlit run minibasket/app.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st

from minibasket import db
from minibasket.views import admin as admin_view
from minibasket.views import auth_view
from minibasket.views import dashboard as dashboard_view
from minibasket.views import evaluate as evaluate_view
from minibasket.views import guardian as guardian_view
from minibasket.views import player_evolution as evolution_view
from minibasket.views import players as players_view
from minibasket.views import reports as reports_view
from minibasket.views import team_evolution as team_evolution_view
from minibasket.views import teams as teams_view

st.set_page_config(page_title="Avaliação do Minibasquete", page_icon="🏀", layout="wide")
db.init_db()

user = auth_view.require_login()

STAFF_SECTIONS = [
    ("Dashboard", "Visão geral das equipas, avaliações e evolução."),
    ("Equipas", "Criar e gerir equipas Sub-8, Sub-10 e Sub-12."),
    ("Jogadores", "Fichas individuais, pesquisa e filtros."),
    ("Avaliar", "Ficha de avaliação das nove competências."),
    ("Evolução do Jogador", "Roda das Competências e evolução ao longo da época."),
    ("Evolução da Equipa", "Médias, medianas e evolução coletiva."),
    ("Relatórios", "Relatórios para treinadores e pais; exportação PDF."),
]
if user["role"] == "guardian":
    SECTIONS = [("O meu educando", "Relatório e evolução do seu educando.")]
elif user["role"] == "admin":
    SECTIONS = STAFF_SECTIONS + [("Utilizadores", "Contas, perfis e acessos.")]
else:
    SECTIONS = STAFF_SECTIONS

st.sidebar.title("🏀 Minibasquete")
section = st.sidebar.radio("Navegação", [s for s, _ in SECTIONS])
st.sidebar.caption("Sem evolução individual, não há sucesso coletivo.")
auth_view.sidebar_account(user)

st.title(section)
st.caption(dict(SECTIONS)[section])

if section == "Dashboard":
    dashboard_view.render()
elif section == "Equipas":
    teams_view.render()
elif section == "Avaliar":
    evaluate_view.render()
elif section == "Evolução do Jogador":
    evolution_view.render()
elif section == "Evolução da Equipa":
    team_evolution_view.render()
elif section == "Relatórios":
    reports_view.render()
elif section == "Jogadores":
    players_view.render()
elif section == "Utilizadores":
    admin_view.render()
elif section == "O meu educando":
    guardian_view.render()
