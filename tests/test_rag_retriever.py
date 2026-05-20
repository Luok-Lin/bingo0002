import unittest
from unittest.mock import patch

from rag.retriever import SimpleRAG, resolve_chroma_persist_dir


class _Doc:
    def __init__(self, page_content):
        self.page_content = page_content


class _FakeVectorStore:
    def __init__(self):
        self.filters = []

    def similarity_search(self, query, k, filter=None):
        self.filters.append(filter)
        if filter and "is_fallback" in str(filter):
            return [_Doc("fallback doc")]
        return []


class SimpleRAGTests(unittest.TestCase):
    def test_normalize_marks_fallback_docs(self):
        content, metadata = SimpleRAG._normalize_item(
            {
                "page_content": "no external data",
                "metadata": {"ticker": "600519", "source": "fallback"},
            }
        )
        self.assertEqual(content, "no external data")
        self.assertTrue(metadata["is_fallback"])
        self.assertEqual(metadata["ticker"], "600519")

    def test_build_filter_uses_and_for_multiple_conditions(self):
        filt = SimpleRAG._build_filter([{"ticker": {"$eq": "600519"}}, {"date_int": {"$lte": 20260520}}])
        self.assertEqual(set(filt.keys()), {"$and"})

    def test_retrieve_falls_back_to_fallback_docs_after_strict_date_miss(self):
        rag = SimpleRAG.__new__(SimpleRAG)
        rag.vectorstore = _FakeVectorStore()
        docs = rag.retrieve("query", target_date="2026-05-20", ticker="600519", top_k=2)
        self.assertEqual(docs, ["fallback doc"])
        self.assertEqual(len(rag.vectorstore.filters), 2)

    def test_missing_vector_dependencies_use_memory_retrieval(self):
        with patch("rag.retriever.get_embedding_function", side_effect=RuntimeError("missing embeddings")):
            rag = SimpleRAG(
                data_sources=[
                    {
                        "page_content": "600519 一季度业绩稳健，机构评级增持",
                        "metadata": {"ticker": "600519", "source": "report", "date_int": 20260518},
                    }
                ]
            )

        self.assertIsNone(rag.vectorstore)
        self.assertIn("missing embeddings", rag.degraded_reason)
        docs = rag.retrieve("600519 一季度", target_date="2026-05-20", ticker="600519", top_k=1)
        self.assertEqual(docs, ["600519 一季度业绩稳健，机构评级增持"])

    def test_chroma_persist_dir_uses_env_relative_to_project_base(self):
        with patch.dict("os.environ", {"CHROMA_PERSIST_DIR": "custom_chroma"}):
            path = resolve_chroma_persist_dir("/tmp/project")
        self.assertEqual(path, "/tmp/project/custom_chroma")


if __name__ == "__main__":
    unittest.main()
