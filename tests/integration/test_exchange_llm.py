"""Fluxo de câmbio com extrator LLM (``deps.nlu``) — heurística propositalmente burla.

Mensagens escolhidas para ``extract_currency_code`` devolver ``None``, forçando
o caminho ``deps.nlu.currency``. Sem rede: chat model fake + FX mock (ou
MockTransport para par inexistente).
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage

from banco_agil.deps import build_deps
from banco_agil.graph.workflow import build_graph, invoke_turn, last_ai_text
from banco_agil.infrastructure.fx_client import FxClient
from banco_agil.infrastructure.session_checkpointer import build_checkpointer
from banco_agil.llm.extract import LlmExtractor
from banco_agil.llm.intent import IntentResult
from banco_agil.utils.conversation import extract_currency_code

IntentFallback = Callable[[str, str], IntentResult]

_ANA_CPF = "52998224725"
_ANA_DOB = "15/05/1990"


class _FakeChat:
    """Devolve resposta fixa ou mapeada pelo texto do HumanMessage."""

    def __init__(
        self,
        reply: str | list[Any] | None = None,
        *,
        mapping: dict[str, str] | None = None,
        raises: bool = False,
    ) -> None:
        self._reply = reply
        self._mapping = {k.lower(): v for k, v in (mapping or {}).items()}
        self._raises = raises
        self.calls = 0

    def invoke(self, messages: Any, **_kwargs: Any) -> AIMessage:
        self.calls += 1
        if self._raises:
            raise RuntimeError("network down")
        if self._mapping:
            human = ""
            for message in messages:
                content = getattr(message, "content", "")
                # System prompt de currency começa com "Identifique"; human é a frase.
                if (
                    isinstance(content, str)
                    and content
                    and not content.startswith("Identifique")
                    and "Identifique a moeda" not in content
                ):
                    human = content.lower()
            for needle, code in self._mapping.items():
                if needle in human:
                    return AIMessage(content=code)
            return AIMessage(content="null")
        return AIMessage(content=self._reply if self._reply is not None else "null")


def _exchange_intent(_text: str, _context: str = "") -> IntentResult:
    return IntentResult("exchange")


def _build_env(
    tmp_path: Path,
    *,
    nlu: object | None = None,
    llm_fallback: IntentFallback | None = None,
    fx: FxClient | None = None,
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
    if fx is not None:
        deps.fx = fx
        deps.settings.fx_mock = False
    else:
        deps.settings.fx_mock = True
        deps.fx._mock = True  # noqa: SLF001
    graph = build_graph(
        deps,
        checkpointer=build_checkpointer(memory=True),
        llm_fallback=llm_fallback or _exchange_intent,
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
    raises: bool = False,
) -> LlmExtractor:
    return LlmExtractor(cast(BaseChatModel, _FakeChat(reply, mapping=mapping, raises=raises)))


# Frases que a heurística NÃO resolve (forçam LLM no nó de câmbio).
_LLM_PHRASES: list[tuple[str, str]] = [
    ("quero a cotação do greenback", "USD"),
    ("quanto tá o cable hoje", "GBP"),
    ("me passa o loonie", "CAD"),
    ("moeda usada em Tóquio", "JPY"),
    ("dinheiro de Bangkok", "THB"),
    ("moeda que circula em Seul", "KRW"),
    ("a moeda da Islândia", "ISK"),
    ("forint húngaro", "HUF"),
    ("leu romeno", "RON"),
    ("coroa tcheca", "CZK"),
    ("ringgit malaio", "MYR"),
    ("dong vietnamita", "VND"),
    ("riyal saudita", "SAR"),
    ("dinar do Kuwait", "KWD"),
    ("naira da Nigéria", "NGN"),
    ("a paridade do kiwi", "NZD"),
    ("eurinho pra viagem", "EUR"),
    ("câmbio de Praga", "CZK"),
    ("quanto está a moeda da Hungria", "HUF"),
    ("preciso do câmbio pra viajar a Budapeste", "HUF"),
    ("cotação pra Dubai", "AED"),
    ("moeda do Marrocos", "MAD"),
    ("quetzal da Guatemala", "GTQ"),
    ("córdoba nicaraguense", "NIO"),
    ("balboa do Panamá", "PAB"),
    ("lempira hondurenha", "HNL"),
    ("gourde haitiano", "HTG"),
    ("pula de Botswana", "BWP"),
    ("tugrik mongol", "MNT"),
    ("tenge cazaque", "KZT"),
    ("dram armênio", "AMD"),
    ("lari georgiano", "GEL"),
    ("kuna croata", "HRK"),
    ("pataca de Macau", "MOP"),
    ("taka bengali", "BDT"),
    ("kip laosiano", "LAK"),
    ("riel cambojano", "KHR"),
    ("kyat birmanês", "MMK"),
    ("dinar sérvio", "RSD"),
    ("bucks americanos", "USD"),
    ("sterling", "GBP"),
    ("greenbacks", "USD"),
]


@pytest.mark.parametrize(("phrase", "code"), _LLM_PHRASES)
def test_phrase_is_llm_only(phrase: str, code: str) -> None:
    """Sanidade: a suíte de integração não pode ser 'roubada' pela heurística."""
    assert extract_currency_code(phrase) is None, phrase
    assert len(code) == 3


@pytest.mark.parametrize(("phrase", "code"), _LLM_PHRASES)
def test_exchange_graph_quotes_via_llm(tmp_path: Path, phrase: str, code: str) -> None:
    """Grafo: intent exchange + NLU LLM → cotação mock da moeda interpretada."""
    graph, _ = _build_env(tmp_path, nlu=_nlu(code))
    sid = f"fx-llm-{code}-{abs(hash(phrase)) % 10_000}"
    _auth(graph, sid)
    state = invoke_turn(graph, session_id=sid, message=phrase)
    assert state["active_agent"] == "exchange"
    assert state["last_tool_calls"][0]["args"]["currency"] == code
    assert code in last_ai_text(state)


def test_exchange_llm_null_asks_currency(tmp_path: Path) -> None:
    """LLM responde null → pergunta a moeda (não assume USD)."""
    graph, _ = _build_env(tmp_path, nlu=_nlu("null"))
    sid = "fx-llm-null"
    _auth(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="quero ver o câmbio")
    assert state["active_agent"] == "exchange"
    reply = last_ai_text(state).lower()
    assert "moeda" in reply
    # Não cotou: tool calls residuais são só de auth, não de câmbio.
    tool_names = [c.get("name") for c in (state.get("last_tool_calls") or [])]
    assert "get_exchange_rate" not in tool_names
    assert "/brl" not in reply


def test_exchange_llm_down_asks_currency(tmp_path: Path) -> None:
    """LLM indisponível (raise) degrada e pede clarificação."""
    graph, _ = _build_env(tmp_path, nlu=_nlu(raises=True))
    sid = "fx-llm-down"
    _auth(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="greenback por favor")
    assert state["active_agent"] == "exchange"
    assert "moeda" in last_ai_text(state).lower()


def test_exchange_llm_then_second_turn_quotes(tmp_path: Path) -> None:
    """Turno 1 sem moeda → pergunta; turno 2 com gíria LLM → cotação."""
    mapping = {"greenback": "USD"}
    graph, _ = _build_env(tmp_path, nlu=_nlu(mapping=mapping))
    sid = "fx-llm-multiturn"
    _auth(graph, sid)
    ask = invoke_turn(graph, session_id=sid, message="quero ver o câmbio")
    assert "moeda" in last_ai_text(ask).lower()
    quoted = invoke_turn(graph, session_id=sid, message="o greenback")
    assert quoted["last_tool_calls"][0]["args"]["currency"] == "USD"
    assert "USD" in last_ai_text(quoted)


def test_heuristic_wins_before_llm(tmp_path: Path) -> None:
    """Se a heurística já resolve, a LLM de moeda não é consultada."""
    fake = _FakeChat("EUR")  # responderia EUR se chamada
    graph, deps = _build_env(tmp_path, nlu=LlmExtractor(cast(BaseChatModel, fake)))
    sid = "fx-heuristic-first"
    _auth(graph, sid)
    state = invoke_turn(graph, session_id=sid, message="cotação do dólar")
    assert state["last_tool_calls"][0]["args"]["currency"] == "USD"
    assert fake.calls == 0
    assert deps.nlu is not None


def test_exchange_llm_pair_not_found_message(tmp_path: Path) -> None:
    """LLM devolve ISO inexistente na fonte → mensagem de cotação indisponível."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            404,
            json={
                "status": 404,
                "code": "CoinNotExists",
                "message": "moeda nao encontrada XYZ-BRL",
            },
        )

    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    fx = FxClient(
        api_url_template="https://example.com/json/last/{pair}",
        client=http_client,
        mock=False,
    )
    graph, _ = _build_env(tmp_path, nlu=_nlu("XYZ"), fx=fx)
    sid = "fx-llm-missing"
    _auth(graph, sid)
    # Frase sem heurística; LLM devolve XYZ.
    state = invoke_turn(graph, session_id=sid, message="quero a cotação do greenback")
    reply = last_ai_text(state).lower()
    assert state["active_agent"] == "exchange"
    assert "não encontrei cotação" in reply or "nao encontrei cotacao" in reply or "xyz" in reply
    assert state.get("error")
    http_client.close()


def test_exchange_end_from_skill(tmp_path: Path) -> None:
    """Pedido de encerramento dentro do skill de câmbio."""
    graph, _ = _build_env(tmp_path, nlu=_nlu("USD"))
    sid = "fx-llm-end"
    _auth(graph, sid)
    invoke_turn(graph, session_id=sid, message="greenback")
    ended = invoke_turn(graph, session_id=sid, message="encerrar")
    reply = last_ai_text(ended).lower()
    assert ended.get("should_end") is True or "até logo" in reply or "encerrado" in reply
