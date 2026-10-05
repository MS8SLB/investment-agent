"""Copia os dados locais (SQLite) da app de basquetebol para o Postgres partilhado.

Uso:
    python scripts/migrate_basketball_sqlite_to_postgres.py "postgresql://..." [caminho/para/basketball_eval.db]

Mantém os ids (as avaliações continuam ligadas aos jogadores). Recusa copiar para tabelas que já
tenham dados (não mistura nem sobrescreve). O SQLite local não é alterado.
"""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from basketball_eval import db, qual_db

# Ordem respeita as chaves estrangeiras. As duas últimas «de configuração» já vêm semeadas.
TABLES = ["coaches", "players", "teams", "defensive_movement_tests", "reference_norms",
          "evaluations", "evaluation_items"]
SEEDED = ["age_groups", "test_protocols"]          # copiadas com ON CONFLICT DO NOTHING
SERIAL = {"coaches", "players", "teams", "defensive_movement_tests", "reference_norms",
          "evaluations", "evaluation_items"}


def main(pg_url: str, sqlite_path: str) -> None:
    if not os.path.exists(sqlite_path):
        sys.exit(f"Ficheiro SQLite não encontrado: {sqlite_path}")
    src = sqlite3.connect(sqlite_path)
    src.row_factory = sqlite3.Row
    have = {r[0] for r in src.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    qual_db.init(pg_url)                            # cria o esquema no Postgres
    with db.connect(pg_url) as dst:
        for t in TABLES + SEEDED:
            if t not in have:
                continue
            rows = src.execute(f"SELECT * FROM {t}").fetchall()
            if not rows:
                continue
            if t in TABLES and dst.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]:
                sys.exit(f"A tabela «{t}» do destino já tem dados — nada foi copiado a partir dela. Abortado.")
            cols = rows[0].keys()
            sql = (f"INSERT INTO {t}({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))}) "
                   "ON CONFLICT DO NOTHING")
            for r in rows:
                dst.execute(sql, tuple(r))
            print(f"{t}: {len(rows)} linha(s)")
        for t in SERIAL:                            # os próximos ids continuam depois dos copiados
            dst.execute(f"SELECT setval(pg_get_serial_sequence('{t}', 'id'), COALESCE((SELECT MAX(id) FROM {t}), 1), "
                        f"(SELECT COUNT(*) FROM {t}) > 0)")
    print("Concluído.")


if __name__ == "__main__":
    if len(sys.argv) < 2 or not sys.argv[1].startswith(("postgres://", "postgresql://")):
        sys.exit(__doc__)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else db.DB_PATH)
