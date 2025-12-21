import pytest
from me_ecu_agent.data_schema import DocMeta
from me_ecu_agent.query.util import (
    model_is_plus,
    docs_covering,
    base_docs_of_same_series,
    select_candidate_docs,
)


# -----------------------------
# Test fixtures
# -----------------------------

@pytest.fixture
def doc_metas():
    """
    Canonical DocMeta fixtures used across all tests.
    This mirrors the real metadata structure in the system.
    """
    return [
        DocMeta(
            doc_uid="doc-700-base",
            source_filename="ECU-700_Series_Manual.md",
            product_line="ECU",
            series="ECU-700",
            model_type="Base",
            covered_models=["ECU-750"],
            model_inherits_from=None,
            status="legacy",
        ),
        DocMeta(
            doc_uid="doc-800-base",
            source_filename="ECU-800_Series_Base.md",
            product_line="ECU",
            series="ECU-800",
            model_type="Base",
            covered_models=["ECU-850"],
            model_inherits_from=None,
            status="online",
        ),
        DocMeta(
            doc_uid="doc-800-plus",
            source_filename="ECU-800_Series_Plus.md",
            product_line="ECU",
            series="ECU-800",
            model_type="Plus",
            covered_models=["ECU-850b"],
            model_inherits_from="ECU-800-Base",
            status="online",
        ),
    ]


# -----------------------------
# model_is_plus
# -----------------------------

def test_model_is_plus_true(doc_metas):
    assert model_is_plus("ECU-850b", doc_metas) is True


def test_model_is_plus_false(doc_metas):
    assert model_is_plus("ECU-850", doc_metas) is False
    assert model_is_plus("ECU-750", doc_metas) is False


def test_model_is_plus_unknown_model(doc_metas):
    assert model_is_plus("ECU-999", doc_metas) is False


# -----------------------------
# docs_covering
# -----------------------------

def test_docs_covering_base_model(doc_metas):
    docs = docs_covering("ECU-850", doc_metas)
    assert len(docs) == 1
    assert docs[0].doc_uid == "doc-800-base"


def test_docs_covering_plus_model(doc_metas):
    docs = docs_covering("ECU-850b", doc_metas)
    assert len(docs) == 1
    assert docs[0].doc_uid == "doc-800-plus"


def test_docs_covering_unknown_model(doc_metas):
    docs = docs_covering("ECU-999", doc_metas)
    assert docs == []


# -----------------------------
# base_docs_of_same_series
# -----------------------------

def test_base_docs_of_same_series_for_plus(doc_metas):
    docs = base_docs_of_same_series("ECU-850b", doc_metas)
    assert len(docs) == 1
    assert docs[0].doc_uid == "doc-800-base"


def test_base_docs_of_same_series_for_base(doc_metas):
    docs = base_docs_of_same_series("ECU-850", doc_metas)
    assert docs == []


def test_base_docs_of_same_series_unknown_model(doc_metas):
    docs = base_docs_of_same_series("ECU-999", doc_metas)
    assert docs == []


# -----------------------------
# select_candidate_docs
# -----------------------------

def test_select_single_base_model(doc_metas):
    docs = select_candidate_docs(["ECU-850"], doc_metas)
    doc_ids = {d.doc_uid for d in docs}

    assert doc_ids == {"doc-800-base"}


def test_select_single_plus_model(doc_metas):
    docs = select_candidate_docs(["ECU-850b"], doc_metas)
    doc_ids = {d.doc_uid for d in docs}

    # Plus model should include both Plus and Base documents
    assert doc_ids == {"doc-800-plus", "doc-800-base"}


def test_select_base_and_plus_comparison(doc_metas):
    docs = select_candidate_docs(["ECU-850", "ECU-850b"], doc_metas)
    doc_ids = {d.doc_uid for d in docs}

    # Base appears only once after deduplication
    assert doc_ids == {"doc-800-base", "doc-800-plus"}


def test_select_cross_series_models(doc_metas):
    docs = select_candidate_docs(["ECU-750", "ECU-850"], doc_metas)
    doc_ids = {d.doc_uid for d in docs}

    assert doc_ids == {"doc-700-base", "doc-800-base"}


def test_select_unknown_model(doc_metas):
    docs = select_candidate_docs(["ECU-999"], doc_metas)
    assert docs == []
