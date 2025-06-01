import pytest
import os
from dotenv import load_dotenv

@pytest.fixture(autouse=True)
def load_env():
    """Automatically load environment variables from .env file for all tests."""
    load_dotenv()

@pytest.fixture
def mock_env_vars(monkeypatch):
    """Mock environment variables for testing."""
    monkeypatch.setenv("TAVILY_API_KEY", "test_tavily_key")
    monkeypatch.setenv("PERPLEXITY_API_KEY", "test_perplexity_key")
    monkeypatch.setenv("OPENAI_API_KEY", "test_openai_key")
    return {
        "TAVILY_API_KEY": "test_tavily_key",
        "PERPLEXITY_API_KEY": "test_perplexity_key",
        "OPENAI_API_KEY": "test_openai_key"
    } 