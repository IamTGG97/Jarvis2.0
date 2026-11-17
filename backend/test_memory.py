import os
import unittest
from unittest.mock import patch
from uuid import UUID

from embeddings import EmbeddingError
from memory import RETRIEVAL_WARNING, SAVE_WARNING, remember_turn, retrieve_memory

SESSION_ID = UUID("fdde5172-0d09-4eb6-8502-ad4a65448b19")


class MemoryTests(unittest.TestCase):
    def setUp(self) -> None:
        patcher = patch.dict(os.environ, {"MEMORY_EMBEDDINGS": "on", "OLLAMA_EMBED_MODEL": ""})
        patcher.start()
        self.addCleanup(patcher.stop)

    def test_hybrid_retrieval_passes_query_vector(self) -> None:
        with (
            patch("memory.embed_text", return_value=[0.1, 0.2]) as embed,
            patch("memory.database.find_conversation_context", return_value=[]) as find,
        ):
            result = retrieve_memory(SESSION_ID, "Which squad do I root for?")

        embed.assert_called_once_with("Which squad do I root for?", "query")
        find.assert_called_once_with(
            SESSION_ID,
            "Which squad do I root for?",
            query_embedding=[0.1, 0.2],
            embedding_model="nomic-embed-text",
        )
        self.assertEqual(result.mode, "hybrid")
        self.assertIsNone(result.warning)

    def test_embedding_failure_falls_back_to_keyword_with_warning(self) -> None:
        with (
            patch("memory.embed_text", side_effect=EmbeddingError("offline")),
            patch("memory.database.find_conversation_context", return_value=[]) as find,
            self.assertLogs("memory", level="WARNING"),
        ):
            result = retrieve_memory(SESSION_ID, "hello")

        find.assert_called_once_with(
            SESSION_ID, "hello", query_embedding=None, embedding_model=None
        )
        self.assertEqual(result.mode, "keyword")
        self.assertEqual(result.warning, RETRIEVAL_WARNING)

    def test_disabled_embeddings_use_keyword_without_warning(self) -> None:
        with (
            patch.dict(os.environ, {"MEMORY_EMBEDDINGS": "off"}),
            patch("memory.embed_text") as embed,
            patch("memory.database.find_conversation_context", return_value=[]),
        ):
            result = retrieve_memory(SESSION_ID, "hello")

        embed.assert_not_called()
        self.assertEqual(result.mode, "keyword")
        self.assertIsNone(result.warning)

    def test_database_errors_are_not_swallowed(self) -> None:
        with (
            patch("memory.embed_text", return_value=[0.1]),
            patch("memory.database.find_conversation_context", side_effect=RuntimeError("db down")),
            self.assertRaises(RuntimeError),
        ):
            retrieve_memory(SESSION_ID, "hello")

    def test_remember_turn_saves_document_vector(self) -> None:
        with (
            patch("memory.embed_text", return_value=[0.5]) as embed,
            patch("memory.database.save_conversation_turn") as save,
        ):
            warning = remember_turn(SESSION_ID, "Q", "A")

        embed.assert_called_once_with("User: Q\nJarvis: A", "document")
        save.assert_called_once_with(
            SESSION_ID, "Q", "A", embedding=[0.5], embedding_model="nomic-embed-text"
        )
        self.assertIsNone(warning)

    def test_remember_turn_saves_without_vector_on_embedding_failure(self) -> None:
        with (
            patch("memory.embed_text", side_effect=EmbeddingError("offline")),
            patch("memory.database.save_conversation_turn") as save,
            self.assertLogs("memory", level="WARNING"),
        ):
            warning = remember_turn(SESSION_ID, "Q", "A")

        save.assert_called_once_with(SESSION_ID, "Q", "A", embedding=None, embedding_model=None)
        self.assertEqual(warning, SAVE_WARNING)


if __name__ == "__main__":
    unittest.main()
