import os
import unittest
from unittest.mock import patch

import httpx

from embeddings import (
    EmbeddingError,
    cosine_similarity,
    embed_texts,
    embeddings_enabled,
    turn_document_text,
)


def _response(request: httpx.Request, payload: object, status: int = 200) -> httpx.Response:
    return httpx.Response(status, json=payload, request=request)


class EmbeddingTests(unittest.TestCase):
    def test_uses_task_prefixes_and_embed_endpoint(self) -> None:
        captured: dict[str, object] = {}

        def fake_post(url: str, json: dict, timeout: float) -> httpx.Response:
            captured["url"] = url
            captured["body"] = json
            request = httpx.Request("POST", url)
            return _response(request, {"embeddings": [[0.1, 0.2], [0.3, 0.4]]})

        with (
            patch.dict(os.environ, {"OLLAMA_BASE_URL": "http://localhost:11434/"}),
            patch("embeddings.httpx.post", side_effect=fake_post),
        ):
            vectors = embed_texts(["a", "b"], "query")

        self.assertEqual(vectors, [[0.1, 0.2], [0.3, 0.4]])
        self.assertEqual(captured["url"], "http://localhost:11434/api/embed")
        body = captured["body"]
        self.assertEqual(body["input"], ["search_query: a", "search_query: b"])
        self.assertEqual(body["model"], "nomic-embed-text")

    def test_document_prefix(self) -> None:
        def fake_post(url: str, json: dict, timeout: float) -> httpx.Response:
            self.assertEqual(json["input"], ["search_document: User: q\nJarvis: a"])
            return _response(httpx.Request("POST", url), {"embeddings": [[1.0]]})

        with patch("embeddings.httpx.post", side_effect=fake_post):
            embed_texts([turn_document_text("q", "a")], "document")

    def test_connection_error_raises_embedding_error(self) -> None:
        with (
            patch("embeddings.httpx.post", side_effect=httpx.ConnectError("offline")),
            self.assertRaises(EmbeddingError),
        ):
            embed_texts(["a"], "query")

    def test_http_error_raises_embedding_error(self) -> None:
        def fake_post(url: str, json: dict, timeout: float) -> httpx.Response:
            return _response(httpx.Request("POST", url), {"error": "model not found"}, 404)

        with (
            patch("embeddings.httpx.post", side_effect=fake_post),
            self.assertRaisesRegex(EmbeddingError, "failed"),
        ):
            embed_texts(["a"], "query")

    def test_malformed_payload_raises_embedding_error(self) -> None:
        def fake_post(url: str, json: dict, timeout: float) -> httpx.Response:
            return _response(httpx.Request("POST", url), {"embeddings": []})

        with (
            patch("embeddings.httpx.post", side_effect=fake_post),
            self.assertRaisesRegex(EmbeddingError, "unexpected"),
        ):
            embed_texts(["a"], "query")

    def test_empty_input_makes_no_request(self) -> None:
        with patch("embeddings.httpx.post") as post:
            self.assertEqual(embed_texts([], "query"), [])
        post.assert_not_called()

    def test_cosine_similarity(self) -> None:
        self.assertAlmostEqual(cosine_similarity([1, 0], [1, 0]), 1.0)
        self.assertAlmostEqual(cosine_similarity([1, 0], [0, 1]), 0.0)
        self.assertEqual(cosine_similarity([1, 0], [1, 0, 0]), 0.0)
        self.assertEqual(cosine_similarity([0, 0], [1, 0]), 0.0)

    def test_embeddings_can_be_disabled(self) -> None:
        with patch.dict(os.environ, {"MEMORY_EMBEDDINGS": "off"}):
            self.assertFalse(embeddings_enabled())
        with patch.dict(os.environ, {"MEMORY_EMBEDDINGS": "on"}):
            self.assertTrue(embeddings_enabled())


if __name__ == "__main__":
    unittest.main()
