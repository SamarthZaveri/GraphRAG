"""Registry shared by serving and evaluation."""
from . import query_engine, vector_baseline, hybrid_search

ENGINE_NAMES = ["graphrag", "vector_rag", "hybrid_rag"]


def answer_question(engine, question, mode="auto"):
    if engine == "graphrag":
        response = query_engine.answer_question(question, mode=mode)
    elif engine == "vector_rag":
        response = vector_baseline.answer_question(question)
    elif engine == "hybrid_rag":
        response = hybrid_search.answer_question(question)
    else:
        raise ValueError(f"Unknown engine: {engine}")
    response.engine_used = engine
    return response
