from assistant.utils import get_local_rag_retriever

retriever = get_local_rag_retriever()

def local_rag_node(state):
    query = state.get("query") or state.get("subquery")
    if not query:
        print("[local_rag_node] No query found in state, returning empty context")
        return {"local_context": ""}

    try:
        print(f"[local_rag_node] Searching local RAG with query: {query}")
        docs = retriever.get_relevant_documents(query)
        combined = "\n\n".join([doc.page_content for doc in docs])
        print(f"[local_rag_node] Found {len(docs)} relevant documents")
        print(f"[local_rag_node] Combined context length: {len(combined)} characters")
        return {"local_context": combined}
    except Exception as e:
        print(f"[local_rag_node error] {e}")
        return {"local_context": ""}
