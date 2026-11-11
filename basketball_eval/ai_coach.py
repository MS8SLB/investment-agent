"""Proposta de objetivos individuais com IA (API da Anthropic).

PRIVACIDADE: o pedido enviado ao serviço externo contém APENAS o escalão, as
classificações numéricas e a variação percentual do teste. Nunca envia nome,
equipa, data de nascimento, fotografia nem notas de texto livre (podem conter
dados pessoais). Os jogadores são frequentemente menores.

A resposta é um apoio à decisão do treinador: só são aceites chaves de objetivos
existentes em objectives.LIBRARY; o resto é descartado. Requer ANTHROPIC_API_KEY
(ou um perfil `ant auth login`).
"""

from __future__ import annotations

import json
from typing import Optional

from . import objectives as ob
from . import qualitative as ql

MODEL = "claude-opus-5-5"
MAX_TOKENS = 4000

SYSTEM = (
    "És um assistente de apoio a treinadores de formação de basquetebol (escalões Sub-8 a Sub-14). "
    "Recebes dados anonimizados de um jogador e propões objetivos individuais concretos, observáveis "
    "e adequados à idade, escolhidos APENAS da lista fornecida. Escreve em português de Portugal. "
    "Baseia-te só nos dados recebidos: se faltarem dados, di-lo em vez de inventar. "
    "É uma sugestão para o treinador decidir, não um diagnóstico."
)

SCHEMA = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "objectives": {"type": "array", "items": {
            "type": "object",
            "properties": {"key": {"type": "string"}, "reason": {"type": "string"},
                           "drills": {"type": "array", "items": {"type": "string"}}},
            "required": ["key", "reason", "drills"], "additionalProperties": False}},
        "caveats": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["summary", "objectives", "caveats"], "additionalProperties": False,
}


def build_payload(category: str, qualitative: Optional[dict], movement_change_pct: Optional[float],
                  objective_scores: Optional[dict], limit: int = 4) -> dict:
    """Dados anonimizados enviados à IA (sem identificação nem texto livre)."""
    return {
        "escalao": category,
        "avaliacao_qualitativa_defesa_1a4": {
            ql.CRITERIA[k]["name"]: v for k, v in (qualitative or {}).items() if k in ql.CRITERIA},
        "teste_movimentos_defensivos_variacao_pct_positivo_e_melhoria": movement_change_pct,
        "ultima_avaliacao_objetivos_1a5": {
            k: v for k, v in (objective_scores or {}).items() if k in ob.LIBRARY},
        "objetivos_disponiveis": {k: v["text"] for k, v in ob.LIBRARY.items()},
        "n_objetivos_a_propor": limit,
    }


def _clean(raw: dict, limit: int) -> dict:
    seen, objs = set(), []
    for o in raw.get("objectives", []):
        k = o.get("key")
        if k in ob.LIBRARY and k not in seen:
            seen.add(k)
            objs.append({"key": k, "text": ob.LIBRARY[k]["text"], "side": ob.LIBRARY[k]["side"],
                         "reason": str(o.get("reason", "")), "drills": [str(d) for d in o.get("drills", [])][:4]})
    return {"summary": str(raw.get("summary", "")), "objectives": objs[:limit],
            "caveats": [str(c) for c in raw.get("caveats", [])]}


def propose(payload: dict, client=None) -> dict:
    """Pede à API uma proposta. Levanta RuntimeError se recusada/truncada/inválida."""
    if client is None:
        import anthropic
        client = anthropic.Anthropic()
    resp = client.beta.messages.create(
        model=MODEL, max_tokens=MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",
        thinking={"type": "adaptive"}, output_config={"effort": "medium", "format": {
            "type": "json_schema", "schema": SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": "Dados do jogador (JSON):\n" + json.dumps(payload, ensure_ascii=False)}],
    )
    if resp.stop_reason == "refusal":
        raise RuntimeError("A IA recusou o pedido; use as sugestões por regras.")
    if resp.stop_reason == "max_tokens":
        raise RuntimeError("Resposta da IA truncada; tente novamente.")
    text = next((b.text for b in resp.content if b.type == "text"), "")
    try:
        return _clean(json.loads(text), payload["n_objetivos_a_propor"])
    except (json.JSONDecodeError, AttributeError) as e:
        raise RuntimeError(f"Resposta da IA inválida: {e}") from e
