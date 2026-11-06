"""Typed read layer over bronze / silver / gold (spec 005, AC-005.12).

``TcsData`` is the single entry point other code should use instead of raw SQL / ES calls::

    from tcs_bot.medallion.read import DocumentFilters, Page, get_tcs_data

    data = get_tcs_data()
    page = data.list_documents(DocumentFilters(category=["legal_litigation"], min_risk_score=40))
    doc = data.get_document(page.items[0].doc_id)       # silver + bronze lineage + gold node
    similar = data.similar_documents(doc.doc_id, k=5)   # cosine kNN on the stored gold vector
"""

from tcs_bot.medallion.read.bronze_reader import BronzeReader, MysqlBronzeReader
from tcs_bot.medallion.read.data import TcsData, get_tcs_data, set_tcs_data
from tcs_bot.medallion.read.models import (
    BronzeLineage,
    CategoryBreakdown,
    ClusterDetail,
    ClusterKind,
    CountBucket,
    DocumentDetail,
    DocumentFilters,
    DocumentPage,
    DocumentSummary,
    GoldNode,
    Page,
    RiskBucket,
    RiskOverview,
    SilverView,
    SimilarDocuments,
)

__all__ = [
    "BronzeLineage",
    "BronzeReader",
    "CategoryBreakdown",
    "ClusterDetail",
    "ClusterKind",
    "CountBucket",
    "DocumentDetail",
    "DocumentFilters",
    "DocumentPage",
    "DocumentSummary",
    "GoldNode",
    "MysqlBronzeReader",
    "Page",
    "RiskBucket",
    "RiskOverview",
    "SilverView",
    "SimilarDocuments",
    "TcsData",
    "get_tcs_data",
    "set_tcs_data",
]
