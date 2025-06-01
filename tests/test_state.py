import pytest
from assistant.state import SummaryState, SummaryStateInput, SummaryStateOutput

def test_summary_state_initialization():
    # Test default initialization
    state = SummaryState()
    assert state.research_topic is None
    assert state.search_query is None
    assert state.web_research_results == []
    assert state.sources_gathered == []
    assert state.research_loop_count == 0
    assert state.running_summary is None
    assert state.local_context == ""

    # Test initialization with values
    state = SummaryState(
        research_topic="Test Topic",
        search_query="Test Query",
        web_research_results=["result1"],
        sources_gathered=["source1"],
        research_loop_count=1,
        running_summary="Test Summary",
        local_context="Test Context"
    )
    assert state.research_topic == "Test Topic"
    assert state.search_query == "Test Query"
    assert state.web_research_results == ["result1"]
    assert state.sources_gathered == ["source1"]
    assert state.research_loop_count == 1
    assert state.running_summary == "Test Summary"
    assert state.local_context == "Test Context"

def test_summary_state_input():
    # Test default initialization
    state_input = SummaryStateInput()
    assert state_input.research_topic is None

    # Test initialization with value
    state_input = SummaryStateInput(research_topic="Test Topic")
    assert state_input.research_topic == "Test Topic"

def test_summary_state_output():
    # Test default initialization
    state_output = SummaryStateOutput()
    assert state_output.running_summary is None

    # Test initialization with value
    state_output = SummaryStateOutput(running_summary="Test Summary")
    assert state_output.running_summary == "Test Summary" 