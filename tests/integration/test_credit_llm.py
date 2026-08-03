"""Fluxo de limite de crédito com extrator LLM (``deps.nlu``).

Frases em linguagem natural (extenso, MEI, negativado, …) forçam o caminho
``LlmExtractor`` — a heurística sozinha não as resolve. Sem rede: fake chat.
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.deps import build_deps
from banco_agil.graph.nodes.interview import _heuristic_field, _parse_emprego
from banco_agil.graph.workflow import build_graph, invoke_turn, last_ai_text
from banco_agil.infrastructure.session_checkpointer import build_checkpointer
from banco_agil.llm.extract import LlmExtractor
from banco_agil.llm.intent import IntentResult
from banco_agil.utils.conversation import extract_money

IntentFallback = Callable[[str, str], IntentResult]

_ANA_CPF = "52998224725"
_ANA_DOB = "15/05/1990"


class _FakeChat:
    """Reply fixo, mapeado pelo HumanMessage, fila sequencial, ou raise."""

    def __init__(
        self,
        reply: str | list[Any] | None = None,
        *,
        mapping: dict[str, str] | None = None,
        queue: list[str] | None = None,
        raises: bool = False,
    ) -> None:
        self._reply = reply
        self._mapping = {k.lower(): v for k, v in (mapping or {}).items()}
        self._queue = list(queue or [])
        self._raises = raises
        self.calls = 0

    def invoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
        self.calls += 1
        if self._raises:
            raise RuntimeError("network down")
        if self._queue:
            return AIMessage(content=self._queue.pop(0))
        if self._mapping:
            human = ""
            last = messages[-1] if messages else None
            content = getattr(last, "content", "")
            if isinstance(content, str):
                human = content.lower()
            for needle, value in self._mapping.items():
                if needle in human:
                    return AIMessage(content=value)
            return AIMessage(content="null")
        return AIMessage(content=self._reply if self._reply is not None else "null")


def _credit_intent(_text: str, _context: str = "") -> IntentResult:
    return IntentResult("credit")


def _build_env(
    tmp_path: Path,
    *,
    nlu: object | None = None,
    llm_fallback: IntentFallback | None = None,
) -> tuple[object, Any]:
    root = Path(__file__).resolve().parents[2]
    data_dir = tmp_path / "data"
    models_dir = tmp_path / "models"
    data_dir.mkdir()
    models_dir.mkdir()
    shutil.copy(root / "data" / "clientes.csv", data_dir / "clientes.csv")
    shutil.copy(root / "data" / "score_limite.csv", data_dir / "score_limite.csv")
    for name in ("intent_router.joblib", "safety_clf.joblib"):
        src = root / "models" / name
        if src.exists():
            shutil.copy(src, models_dir / name)

    deps = build_deps(data_dir=data_dir, models_dir=models_dir)
    deps.nlu = nlu  # type: ignore[assignment]
    deps.settings.fx_mock = True
    deps.fx._mock = True  # noqa: SLF001
    graph = build_graph(
        deps,
        checkpointer=build_checkpointer(memory=True),
        llm_fallback=llm_fallback or _credit_intent,
    )
    return graph, deps


def _auth(graph: object, sid: str) -> None:
    invoke_turn(graph, session_id=sid, message="oi")
    invoke_turn(graph, session_id=sid, message=_ANA_CPF)
    invoke_turn(graph, session_id=sid, message=_ANA_DOB)


def _nlu(
    reply: str | None = None,
    *,
    mapping: dict[str, str] | None = None,
    queue: list[str] | None = None,
    raises: bool = False,
) -> LlmExtractor:
    return LlmExtractor(
        cast(
            BaseChatModel,
            _FakeChat(reply, mapping=mapping, queue=queue, raises=raises),
        )
    )


def _start_increase(graph: object, sid: str) -> None:
    """Consulta limite e confirma desejo de aumento (aguarda valor)."""
    invoke_turn(graph, session_id=sid, message="quero consultar meu limite")
    invoke_turn(graph, session_id=sid, message="sim")


# Valores por extenso → reply LLM (Ana: score 450, limite 3000, teto 5000).
_LLM_LIMIT_CASES: list[tuple[str, str, float, str]] = [
    # phrase, llm_reply, expected_limit, expected_status
    ("quatro mil", "4000", 4000.0, "aprovado"),
    ("cinco mil", "5000", 5000.0, "aprovado"),
    ("seis mil", "6000", 6000.0, "rejeitado"),
    ("sete mil", "7000", 7000.0, "rejeitado"),
    ("dez mil", "10000", 10000.0, "rejeitado"),
    ("quinze mil", "15000", 15000.0, "rejeitado"),
    ("vinte e cinco mil", "25000", 25000.0, "rejeitado"),
    ("trinta mil", "30000", 30000.0, "rejeitado"),
    ("dois mil", "2000", 2000.0, "rejeitado"),  # <= atual
    ("três mil", "3000", 3000.0, "rejeitado"),  # == atual
]


@pytest.mark.parametrize(("phrase", "reply", "expected", "status"), _LLM_LIMIT_CASES)
def test_limit_phrase_is_llm_only(
    phrase: str,
    reply: str,
    expected: float,
    status: str,
) -> None:
    assert extract_money(phrase) is None, phrase
    assert float(reply) == expected
    assert status in {"aprovado", "rejeitado"}


@pytest.mark.parametrize(("phrase", "reply", "expected", "status"), _LLM_LIMIT_CASES)
def test_credit_graph_limit_via_llm(
    tmp_path: Path,
    phrase: str,
    reply: str,
    expected: float,
    status: str,
) -> None:
    """Consulta → sim → valor por extenso (LLM) → aprovação/rejeição.

    Mapping por frase evita que o 'sim' da confirmação seja lido como valor.
    """
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping={phrase: reply}))
    sid = f"credit-llm-{int(expected)}-{abs(hash(phrase)) % 10_000}"
    _auth(graph, sid)
    _start_increase(graph, sid)
    state = invoke_turn(graph, session_id=sid, message=phrase)
    assert state["active_agent"] == "credit"
    assert state["pending_new_limit"] == expected
    assert state["last_request_status"] == status
    if status == "aprovado":
        assert "aprovad" in last_ai_text(state).lower()
    else:
        assert "rejeitad" in last_ai_text(state).lower()
        if expected > 3000:
            assert state["offered_interview"] is True


def test_credit_one_shot_increase_via_llm(tmp_path: Path) -> None:
    """Pedido de aumento + valor por extenso no mesmo turno (LLM)."""
    mapping = {"sete mil": "7000"}
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping=mapping))
    sid = "credit-oneshot"
    _auth(graph, sid)
    state = invoke_turn(
        graph,
        session_id=sid,
        message="quero aumentar meu limite para sete mil",
    )
    assert state["pending_new_limit"] == 7000.0
    assert state["last_request_status"] == "rejeitado"
    assert state["offered_interview"] is True


def test_credit_approve_four_thousand_via_llm(tmp_path: Path) -> None:
    """Ana (teto 5000): 'quatro mil' via LLM aprova sem entrevista."""
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping={"quatro mil": "4000"}))
    sid = "credit-approve-4k"
    _auth(graph, sid)
    _start_increase(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="quatro mil")
    assert state["last_request_status"] == "aprovado"
    assert state.get("offered_interview") in (None, False)
    assert "aprovad" in last_ai_text(state).lower()


def test_credit_reject_increase_confirm(tmp_path: Path) -> None:
    """Consulta → 'não' → limite permanece, sem pedir valor."""
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping={}))
    sid = "credit-decline-inc"
    _auth(graph, sid)
    invoke_turn(graph, session_id=sid, message="limite de crédito")
    state = invoke_turn(graph, session_id=sid, message="não")
    assert state["awaiting_increase_confirm"] is False
    assert state["awaiting_limit_value"] is False
    assert "permanece" in last_ai_text(state).lower() or "3000" in last_ai_text(state).replace(
        ".", ""
    )


def test_credit_reject_interview_offer(tmp_path: Path) -> None:
    """Após rejeição, recusar entrevista mantém pedido rejeitado."""
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping={"sete mil": "7000"}))
    sid = "credit-decline-iv"
    _auth(graph, sid)
    _start_increase(graph, sid)
    rej = invoke_turn(graph, session_id=sid, message="sete mil")
    assert rej["offered_interview"] is True
    state = invoke_turn(graph, session_id=sid, message="não")
    assert state["offered_interview"] is False
    assert "rejeitad" in last_ai_text(state).lower()


def test_credit_llm_null_falls_back_to_heuristic(tmp_path: Path) -> None:
    """LLM null + número heurístico no texto → usa heurística."""
    graph, _ = _build_env(tmp_path, nlu=_nlu("null"))
    sid = "credit-fallback"
    _auth(graph, sid)
    _start_increase(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="quero 4000 reais")
    assert state["pending_new_limit"] == 4000.0
    assert state["last_request_status"] == "aprovado"


def test_credit_llm_null_and_opaque_reasks(tmp_path: Path) -> None:
    """LLM null + frase opaca → re-pergunta o valor."""
    graph, _ = _build_env(tmp_path, nlu=_nlu("null"))
    sid = "credit-reask"
    _auth(graph, sid)
    _start_increase(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="sei lá quanto")
    assert state["awaiting_limit_value"] is True
    assert state["clarify_attempts"] == 1
    assert "desculpe" in last_ai_text(state).lower()


def test_credit_llm_down_reasks(tmp_path: Path) -> None:
    """LLM indisponível no valor → degrada e re-pergunta (não inventa)."""
    graph, _ = _build_env(tmp_path, nlu=_nlu(raises=True))
    sid = "credit-llm-down"
    _auth(graph, sid)
    _start_increase(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="sete mil")
    assert state["awaiting_limit_value"] is True
    assert "desculpe" in last_ai_text(state).lower() or "identificar" in last_ai_text(state).lower()


def test_credit_unauthenticated_gate(tmp_path: Path) -> None:
    """Intent credit sem auth → pede autenticação."""
    graph, _ = _build_env(tmp_path, nlu=_nlu("4000"))
    sid = "credit-unauth"
    state = invoke_turn(graph, session_id=sid, message="quero meu limite")
    # Pode cair em triage pedindo CPF, ou credit pedindo auth — ambos ok.
    reply = last_ai_text(state).lower()
    assert state.get("authenticated") is not True
    assert "cpf" in reply or "autentic" in reply


def test_interview_full_path_via_llm(tmp_path: Path) -> None:
    """Rejeição → entrevista NL via LlmExtractor → reanálise."""
    mapping = {
        "sete mil": "7000",
        "quinze mil": "15000",
        "sou mei": "autônomo",
        "cinco mil": "5000",
        "moro sozinho": "0",
        "estou negativado": "sim",
    }
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping=mapping))
    sid = "credit-iv-full"
    _auth(graph, sid)
    invoke_turn(graph, session_id=sid, message="quero aumentar meu limite")
    rej = invoke_turn(graph, session_id=sid, message="sete mil")
    assert rej["last_request_status"] == "rejeitado"

    assert "renda" in last_ai_text(invoke_turn(graph, session_id=sid, message="sim")).lower()
    invoke_turn(graph, session_id=sid, message="quinze mil")
    invoke_turn(graph, session_id=sid, message="sou MEI")
    invoke_turn(graph, session_id=sid, message="cinco mil")
    invoke_turn(graph, session_id=sid, message="moro sozinho")
    final = invoke_turn(graph, session_id=sid, message="estou negativado")

    inputs = final["last_score_calculation"]["inputs"]
    assert inputs["renda_mensal"] == 15000.0
    assert inputs["tipo_emprego"] == "autônomo"
    assert inputs["despesas_fixas"] == 5000.0
    assert inputs["num_dependentes"] == 0
    assert inputs["tem_dividas"] == "sim"
    assert final["last_score_calculation"]["score_after"] is not None


def test_interview_llm_null_falls_back_to_heuristic_emprego(tmp_path: Path) -> None:
    """LLM responde null no emprego; 'CLT' é resolvido pela heurística."""
    mapping = {
        "sete mil": "7000",
        "quinze mil": "15000",
        "clt": "null",  # força fallback heurístico em "CLT"
        "mil e quinhentos": "1500",
        "moro sozinho": "0",
        "estou limpo no spc": "não",
    }
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping=mapping))
    sid = "credit-iv-fallback"
    _auth(graph, sid)
    invoke_turn(graph, session_id=sid, message="aumentar limite para sete mil")
    invoke_turn(graph, session_id=sid, message="sim")
    invoke_turn(graph, session_id=sid, message="quinze mil")
    invoke_turn(graph, session_id=sid, message="CLT")
    invoke_turn(graph, session_id=sid, message="mil e quinhentos")
    invoke_turn(graph, session_id=sid, message="moro sozinho")
    final = invoke_turn(graph, session_id=sid, message="estou limpo no SPC")
    assert final["last_score_calculation"]["inputs"]["tipo_emprego"] == "formal"
    assert final["last_score_calculation"]["inputs"]["tem_dividas"] == "não"


def test_interview_field_reask_when_llm_and_heuristic_fail(tmp_path: Path) -> None:
    """Campo opaco + LLM null → re-pergunta na entrevista."""
    mapping = {"sete mil": "7000"}
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping=mapping))
    sid = "credit-iv-reask"
    _auth(graph, sid)
    invoke_turn(graph, session_id=sid, message="aumentar limite para sete mil")
    invoke_turn(graph, session_id=sid, message="sim")
    r1 = invoke_turn(graph, session_id=sid, message="sei lá")
    assert r1["clarify_attempts"] == 1
    assert "desculpe" in last_ai_text(r1).lower()


@pytest.mark.parametrize(
    ("phrase", "check"),
    [
        ("sou MEI", lambda: _parse_emprego("sou mei") is None),
        ("trabalho PJ", lambda: _parse_emprego("trabalho pj") is None),
        ("estou sem emprego", lambda: _parse_emprego("estou sem emprego") is None),
        ("servidor público", lambda: _parse_emprego("servidor público") is None),
        (
            "moro sozinho",
            lambda: _heuristic_field("num_dependentes", "moro sozinho") is None,
        ),
        (
            "dois filhos",
            lambda: _heuristic_field("num_dependentes", "dois filhos") is None,
        ),
        (
            "estou negativado",
            lambda: _heuristic_field("tem_dividas", "estou negativado") is None,
        ),
        (
            "estou limpo no SPC",
            lambda: _heuristic_field("tem_dividas", "estou limpo no spc") is None,
        ),
        ("quinze mil", lambda: extract_money("quinze mil") is None),
        ("mil e quinhentos", lambda: extract_money("mil e quinhentos") is None),
    ],
)
def test_interview_llm_only_phrases(phrase: str, check: Callable[[], bool]) -> None:
    """Frases usadas na entrevista NL não podem ser 'roubadas' pela heurística."""
    assert check(), phrase
