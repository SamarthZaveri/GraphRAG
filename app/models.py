from __future__ import annotations
from typing import List, Optional, Literal
from pydantic import BaseModel


class Triple(BaseModel):
    subject: str
    predicate: str
    object: str
    doc_id: str
    chunk_id: str
    evidence: str  # short quote/paraphrase snippet from the chunk supporting this triple


class Entity(BaseModel):
    name: str
    type: str  # Party, Property, Money, Date, Obligation, Agreement, Location, Role, Other
    description: Optional[str] = None
    source_docs: List[str] = []


class ExtractionResult(BaseModel):
    doc_id: str
    chunk_id: str
    entities: List[Entity]
    triples: List[Triple]


class CommunitySummary(BaseModel):
    community_id: int
    level: int
    members: List[str]
    title: str
    summary: str


class IngestResponse(BaseModel):
    doc_ids: List[str]
    num_chunks: int
    num_entities: int
    num_triples: int
    num_communities: int


class QueryRequest(BaseModel):
    question: str
    mode: Literal["auto", "local", "global"] = "auto"


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    snippet: str


class QueryResponse(BaseModel):
    question: str
    mode_used: Literal["local", "global"]
    answer: str
    citations: List[Citation]
    graph_path: Optional[List[str]] = None  # entity names touched, for local mode


class CompareResponse(BaseModel):
    question: str
    graphrag: QueryResponse
    vector_rag: QueryResponse


class BenchmarkQuestion(BaseModel):
    id: str
    question: str
    category: Literal["local", "global", "multi_hop", "conflict"]
    reference_answer: str


class BenchmarkResult(BaseModel):
    question_id: str
    question: str
    category: str
    graphrag_answer: str
    vector_rag_answer: str
    graphrag_score: float
    vector_rag_score: float
    judge_rationale: str


class BenchmarkSummary(BaseModel):
    results: List[BenchmarkResult]
    graphrag_avg: float
    vector_rag_avg: float