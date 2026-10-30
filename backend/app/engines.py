"""Registry shared by serving and evaluation."""
ENGINE_NAMES = ["graphrag", "vector_rag", "hybrid_rag"]


def answer_question(engine, question, mode="auto", classification=None):
    """classification is an optional (category, entities) already computed by the router; GraphRAG
    reuses it instead of classifying the question again."""
    if engine == "graphrag":
        from . import query_engine
        response = query_engine.answer_question(question, mode=mode, classification=classification)
    elif engine == "vector_rag":
        from . import vector_baseline
        response = vector_baseline.answer_question(question)
    elif engine == "hybrid_rag":
        from . import hybrid_search
        response = hybrid_search.answer_question(question)
    else:
        raise ValueError(f"Unknown engine: {engine}")
    response.engine_used = engine
    return response
