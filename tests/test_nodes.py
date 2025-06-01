import pytest
from unittest.mock import Mock, patch
from assistant.nodes import local_rag_node
from assistant.state import SummaryState

def test_local_rag_node_no_query(caplog):
    # Test when no query is provided
    with caplog.at_level('INFO'):
        state = SummaryState()
        result = local_rag_node(state)
        assert result == {"local_context": ""}
        assert "[local_rag_node] No search query found in state, returning empty context" in caplog.text

@patch('assistant.nodes.retriever')
def test_local_rag_node_with_query(mock_retriever, caplog):
    # Mock the retriever's response
    mock_docs = [
        Mock(page_content="Test content 1"),
        Mock(page_content="Test content 2")
    ]
    mock_retriever.invoke.return_value = mock_docs

    # Test with a query
    with caplog.at_level('INFO'):
        state = SummaryState(search_query="test query")
        result = local_rag_node(state)
        
        # Verify the retriever was called with the correct query
        mock_retriever.invoke.assert_called_once_with("test query")
        
        # Verify the result
        assert result == {"local_context": "Test content 1\n\nTest content 2"}
        
        # Verify logging
        assert "[local_rag_node] Searching local RAG with query: test query" in caplog.text
        assert "[local_rag_node] Found 2 relevant documents" in caplog.text
        assert "[local_rag_node] Combined context length: 30 characters" in caplog.text

@patch('assistant.nodes.retriever')
def test_local_rag_node_with_subquery(mock_retriever, caplog):
    # Mock the retriever's response
    mock_docs = [Mock(page_content="Test content")]
    mock_retriever.invoke.return_value = mock_docs

    # Test with a subquery
    with caplog.at_level('INFO'):
        state = SummaryState(search_query="test subquery")
        result = local_rag_node(state)
        
        # Verify the retriever was called with the correct query
        mock_retriever.invoke.assert_called_once_with("test subquery")
        
        # Verify the result
        assert result == {"local_context": "Test content"}
        
        # Verify logging
        assert "[local_rag_node] Searching local RAG with query: test subquery" in caplog.text
        assert "[local_rag_node] Found 1 relevant documents" in caplog.text
        assert "[local_rag_node] Combined context length: 12 characters" in caplog.text

@patch('assistant.nodes.retriever')
def test_local_rag_node_error_handling(mock_retriever, caplog):
    # Mock the retriever to raise an exception
    mock_retriever.invoke.side_effect = Exception("Test error")

    # Test error handling
    with caplog.at_level('ERROR'):
        state = SummaryState(search_query="test query")
        result = local_rag_node(state)
        
        # Verify the result is empty on error
        assert result == {"local_context": ""}
        
        # Verify error logging
        assert "[local_rag_node error] Test error" in caplog.text 