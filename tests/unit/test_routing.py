"""Testes das funções puras de roteamento (edges)."""

from langgraph.graph import END

from banco_agil.graph.edges import (
    route_after_credit,
    route_after_exchange,
    route_after_guard,
    route_after_interview,
    route_after_router,
    route_after_triage,
)
from banco_agil.graph.state import SessionState, initial_state


def _state(**overrides: object) -> SessionState:
    base = initial_state("test")
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def test_guard_blocked_goes_to_safe_reply() -> None:
    assert route_after_guard(_state(input_blocked=True)) == "safe_reply"


def test_guard_unauthenticated_goes_to_triage() -> None:
    assert route_after_guard(_state(authenticated=False, input_blocked=False)) == "triage"


def test_guard_authenticated_goes_to_router() -> None:
    assert route_after_guard(_state(authenticated=True, input_blocked=False)) == "router"


def test_triage_three_failures_goes_to_end() -> None:
    assert route_after_triage(_state(authenticated=False, auth_attempts=3)) == "end"


def test_triage_waiting_for_input_ends_turn() -> None:
    assert route_after_triage(_state(authenticated=False, auth_attempts=1)) is END


def test_triage_authenticated_goes_to_router() -> None:
    assert route_after_triage(_state(authenticated=True)) == "router"


def test_router_intents() -> None:
    assert route_after_router(_state(intent="credit")) == "credit"
    assert route_after_router(_state(intent="exchange")) == "exchange"
    assert route_after_router(_state(intent="interview")) == "interview"
    assert route_after_router(_state(intent="end")) == "end"
    assert route_after_router(_state(intent="unknown")) is END


def test_credit_interview_accepted() -> None:
    assert route_after_credit(_state(interview_accepted=True)) == "interview"


def test_credit_otherwise_ends_turn() -> None:
    assert (
        route_after_credit(_state(last_request_status="rejeitado", offered_interview=True)) is END
    )


def test_interview_complete_goes_to_credit() -> None:
    assert route_after_interview(_state(interview_complete=True)) == "credit"


def test_interview_incomplete_ends_turn() -> None:
    assert route_after_interview(_state(interview_complete=False)) is END


def test_exchange_ends_turn() -> None:
    assert route_after_exchange(_state()) is END


def test_should_end_priority() -> None:
    assert route_after_credit(_state(should_end=True)) == "end"
    assert route_after_router(_state(should_end=True, intent="credit")) == "end"
