import pytest
from assistant.utils import (
    deduplicate_and_format_sources,
    format_sources,
    duckduckgo_search,
    tavily_search,
    perplexity_search,
)
import os

def test_deduplicate_and_format_sources():
    # Test with duplicate URLs
    search_response = {
        "results": [
            {"url": "http://example.com/1", "title": "Example 1", "snippet": "Test 1", "content": "Content 1"},
            {"url": "http://example.com/1", "title": "Example 1", "snippet": "Test 1", "content": "Content 1"},
            {"url": "http://example.com/2", "title": "Example 2", "snippet": "Test 2", "content": "Content 2"},
        ]
    }
    result = deduplicate_and_format_sources(search_response, max_tokens_per_source=100)
    assert "Example 1" in result
    assert "Example 2" in result
    assert result.count("http://example.com/1") == 1
    assert result.count("http://example.com/2") == 1

def test_format_sources():
    # Test source formatting
    search_results = {
        "results": [
            {"url": "http://example.com", "title": "Example Title", "snippet": "Example snippet"}
        ]
    }
    result = format_sources(search_results)
    assert "Example Title" in result
    assert "http://example.com" in result

def test_duckduckgo_search():
    # Test with a simple query
    result = duckduckgo_search("test query", max_results=1)
    assert isinstance(result, dict)
    assert "results" in result
    assert isinstance(result["results"], list)
    if result["results"]:  # If we got results
        assert isinstance(result["results"][0], dict)
        assert "url" in result["results"][0]
        assert "title" in result["results"][0]

@pytest.mark.asyncio
async def test_tavily_search():
    # Skip if API key is not set
    if not os.getenv("TAVILY_API_KEY"):
        pytest.skip("TAVILY_API_KEY not set")
    
    # Test with a simple query
    result = tavily_search("test query", max_results=1)
    assert isinstance(result, dict)
    assert "results" in result
    assert isinstance(result["results"], list)
    if result["results"]:  # If we got results
        assert isinstance(result["results"][0], dict)
        assert "url" in result["results"][0]
        assert "title" in result["results"][0]
        assert "content" in result["results"][0]

@pytest.mark.asyncio
async def test_perplexity_search():
    # Skip if API key is not set
    if not os.getenv("PERPLEXITY_API_KEY"):
        pytest.skip("PERPLEXITY_API_KEY not set")
    
    # Test with a simple query
    result = await perplexity_search("test query", max_results=1)
    assert isinstance(result, list)
    if result:  # If we got results
        assert isinstance(result[0], dict)
        assert "url" in result[0]
        assert "title" in result[0]
        assert "snippet" in result[0] 