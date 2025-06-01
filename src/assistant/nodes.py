from assistant.utils import get_local_rag_retriever
from assistant.state import SummaryState

retriever = get_local_rag_retriever()

def local_rag_node(state: SummaryState) -> dict:
    """
    Node that performs local RAG search using the state's search_query.
    
    Args:
        state: SummaryState object containing the search query
        
    Returns:
        dict: Dictionary containing the local context
    """
    if not state.search_query:
        print("[local_rag_node] No search query found in state, returning empty context")
        return {"local_context": ""}

    try:
        print(f"[local_rag_node] Searching local RAG with query: {state.search_query}")
        docs = retriever.invoke(state.search_query)
        combined = "\n\n".join([doc.page_content for doc in docs])
        print(f"[local_rag_node] Found {len(docs)} relevant documents")
        print(f"[local_rag_node] Combined context length: {len(combined)} characters")
        return {"local_context": combined}
    except Exception as e:
        print(f"[local_rag_node error] {e}")
        return {"local_context": ""}
