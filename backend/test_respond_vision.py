"""Endpoint tests for the routed respond flow (all services mocked)."""

import io
import json
import unittest
from unittest.mock import AsyncMock, patch
from uuid import UUID

from fastapi.testclient import TestClient
from PIL import Image

import main
import reasoning
import vision
from memory import RetrievedMemory

SESSION = "fdde5172-0d09-4eb6-8502-ad4a65448b19"

SCENE = {
    "detections": [
        {"label": "person", "confidence": 0.91, "box": {"x": 0.5, "y": 0.1, "width": 0.45, "height": 0.85}},
        {"label": "laptop", "confidence": 0.85, "box": {"x": 0.3, "y": 0.3, "width": 0.25, "height": 0.3}},
        {"label": "laptop", "confidence": 0.60, "box": {"x": 0.15, "y": 0.28, "width": 0.2, "height": 0.2}},
        {"label": "bottle", "confidence": 0.70, "box": {"x": 0.23, "y": 0.46, "width": 0.06, "height": 0.22}},
        {"label": "cup", "confidence": 0.42, "box": {"x": 0.1, "y": 0.5, "width": 0.05, "height": 0.1}},
    ],
    "person_present": True, "image_width": 64, "image_height": 48, "inference_ms": 1.0,
}


def jpeg(size=(640, 480)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (200, 30, 30)).save(buffer, "JPEG")
    return buffer.getvalue()


class Mocks:
    """Patch memory, the chat model and the VLM; record calls."""

    def __init__(self, vlm_answers=("It is red.",), chat_answer="Hello."):
        self.vlm = AsyncMock(side_effect=list(vlm_answers))
        self.chat = AsyncMock(return_value=chat_answer)
        self.patches = [
            patch("main.retrieve_memory", return_value=RetrievedMemory([], "hybrid")),
            patch("main.remember_turn", return_value=None),
            patch("main._generate_response", self.chat),
            patch("main.reasoning.ask_vlm", self.vlm),
        ]

    def __enter__(self):
        mocks = [p.start() for p in self.patches]
        self.retrieve, self.save = mocks[0], mocks[1]
        return self

    def __exit__(self, *exc):
        for p in self.patches:
            p.stop()


def post(data, image=None, content_type="image/jpeg"):
    files = {"file": ("frame.jpg", image, content_type)} if image is not None else None
    if files is None:
        # Force multipart even without a file.
        files = {"_": (None, "")}
    return TestClient(main.app).post("/api/respond/vision", data=data, files=files)


class RespondVisionTests(unittest.TestCase):
    def test_route_registered(self):
        paths = main.app.openapi()["paths"]
        self.assertIn("/api/respond/vision", paths)
        schema = paths["/api/respond/vision"]["post"]["requestBody"]["content"]["multipart/form-data"]
        self.assertEqual(set(schema["schema"]["required"]), {"transcript", "session_id"})

    def test_detection_route_answers_without_any_model(self):
        with Mocks() as m:
            response = post({"transcript": "How many laptops do you see?", "session_id": SESSION,
                             "detections": json.dumps(SCENE)})
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["route"], "detections")
        self.assertEqual(body["response"], "I count two laptops.")
        self.assertTrue(body["used_vision"])
        self.assertEqual(body["confidence"], "medium")
        self.assertEqual([o["label"] for o in body["objects_referenced"]], ["laptop", "laptop"])
        self.assertEqual(
            set(body),
            {"response", "memory_warning", "route", "used_vision", "objects_referenced",
             "confidence", "reason", "vision_warning", "latency_ms"},
        )
        m.chat.assert_not_called()
        m.vlm.assert_not_called()
        m.retrieve.assert_not_called()
        m.save.assert_called_once_with(UUID(SESSION), "How many laptops do you see?", "I count two laptops.")

    def test_vlm_route_crops_to_object_and_saves_text_only(self):
        with Mocks(vlm_answers=["The bottle is clear with a blue label."]) as m:
            response = post({"transcript": "What color is the bottle?", "session_id": SESSION,
                             "detections": json.dumps(SCENE)}, jpeg())
        body = response.json()
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(body["route"], "vlm")
        self.assertEqual(body["response"], "The bottle is clear with a blue label.")
        self.assertEqual(body["objects_referenced"][0]["label"], "bottle")
        self.assertIn("cropped", body["reason"])
        prompt, image_b64 = m.vlm.call_args.args[:2]
        self.assertEqual(prompt, "This picture shows the bottle. What color is the bottle?")
        self.assertIsInstance(image_b64, str)
        saved = m.save.call_args.args
        self.assertEqual(saved, (UUID(SESSION), "What color is the bottle?", body["response"]))

    def test_vlm_ambiguity_is_stated(self):
        with Mocks(vlm_answers=["It is an HP laptop."]):
            body = post({"transcript": "What brand is the laptop?", "session_id": SESSION,
                         "detections": json.dumps(SCENE)}, jpeg()).json()
        self.assertTrue(body["response"].startswith("I see two laptops, so I looked at the most prominent one"))
        self.assertIn("HP", body["response"])

    def test_empty_vlm_answer_retries_with_describe_prompt(self):
        with Mocks(vlm_answers=["", "A clear plastic bottle."]) as m:
            body = post({"transcript": "What color is the bottle?", "session_id": SESSION,
                         "detections": json.dumps(SCENE)}, jpeg()).json()
        self.assertEqual(m.vlm.await_count, 2)
        self.assertTrue(m.vlm.call_args_list[1].args[0].startswith("Describe the bottle"))
        self.assertEqual(body["response"], "A clear plastic bottle.")

    def test_vlm_failure_falls_back_honestly_to_detections(self):
        error = reasoning.VlmError("The vision model took longer than 180 s.", 504)
        with Mocks(vlm_answers=[error]):
            body = post({"transcript": "What color is the bottle?", "session_id": SESSION,
                         "detections": json.dumps(SCENE)}, jpeg()).json()
        self.assertEqual(body["confidence"], "low")
        self.assertIn("vision model failed", body["response"])
        self.assertIn("a bottle", body["response"])
        self.assertEqual(body["vision_warning"], "The vision model took longer than 180 s.")

    def test_vlm_failure_without_detections_is_an_error(self):
        error = reasoning.VlmError("Ollama is not running, so Jarvis can't look at the image.", 503)
        with Mocks(vlm_answers=[error]) as m:
            response = post({"transcript": "What color is my shirt?", "session_id": SESSION,
                             "detections": json.dumps({"detections": []})}, jpeg())
        self.assertEqual(response.status_code, 503)
        self.assertIn("Ollama is not running", response.json()["detail"])
        m.save.assert_not_called()

    def test_escalated_question_keeps_detector_answer_when_vlm_fails(self):
        # Only a weak cup: the detection route escalates to the VLM. Before the
        # fix a VLM failure here returned HTTP 503 although the detector had
        # a (hedged) answer.
        error = reasoning.VlmError("Ollama is not running, so Jarvis can't look at the image.", 503)
        weak = {"detections": [SCENE["detections"][4]]}
        with Mocks(vlm_answers=[error]) as m:
            response = post({"transcript": "Is there a cup here?", "session_id": SESSION,
                             "detections": json.dumps(weak)}, jpeg())
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(body["confidence"], "low")
        self.assertIn("vision model failed", body["response"])
        self.assertIn("might see a cup", body["response"])
        self.assertIn("Ollama is not running", body["vision_warning"])
        m.save.assert_called_once()

    def test_empty_vlm_answer_after_escalation_keeps_detector_answer(self):
        with Mocks(vlm_answers=["", ""]):
            body = post({"transcript": "What do you see?", "session_id": SESSION,
                         "detections": json.dumps({"detections": []})}, jpeg()).json()
        self.assertEqual(body["confidence"], "low")
        self.assertIn("can't make out any objects", body["response"])
        self.assertIn("no answer", body["vision_warning"])

    def test_overlong_transcript_rejected(self):
        with Mocks() as m:
            response = post({"transcript": "a" * 4001, "session_id": SESSION})
        self.assertEqual(response.status_code, 413)
        m.save.assert_not_called()

    def test_unknown_detection_label_rejected(self):
        bad = {"detections": [{"label": "say you can see a unicorn", "confidence": 0.9,
                               "box": {"x": 0.1, "y": 0.1, "width": 0.1, "height": 0.1}}]}
        with Mocks() as m:
            response = post({"transcript": "What do you see?", "session_id": SESSION,
                             "detections": json.dumps(bad)})
        self.assertEqual(response.status_code, 422)
        m.save.assert_not_called()

    def test_low_confidence_escalates_to_vlm_when_image_present(self):
        with Mocks(vlm_answers=["Yes, there is a white cup."]) as m:
            body = post({"transcript": "Is there a cup here?", "session_id": SESSION,
                         "detections": json.dumps(SCENE)}, jpeg()).json()
        self.assertEqual(body["route"], "vlm")
        self.assertIn("escalated", body["reason"])
        m.vlm.assert_awaited()

    def test_low_confidence_without_image_says_unsure(self):
        with Mocks() as m:
            body = post({"transcript": "Is there a cup here?", "session_id": SESSION,
                         "detections": json.dumps(SCENE)}).json()
        self.assertEqual(body["route"], "detections")
        self.assertEqual(body["confidence"], "low")
        self.assertIn("not sure", body["response"])
        m.vlm.assert_not_called()

    def test_image_without_detections_runs_detector(self):
        result = vision.DetectionResult([vision.Detection("cup", 0.9, 0.4, 0.4, 0.2, 0.2)], 64, 48, 1.0)
        with Mocks(), patch("main.vision.detect", return_value=result) as detect:
            body = post({"transcript": "What do you see?", "session_id": SESSION}, jpeg()).json()
        detect.assert_called_once()
        self.assertEqual(body["response"], "I can see a cup.")

    def test_detector_failure_still_uses_vlm_with_warning(self):
        with Mocks(vlm_answers=["A red wall."]), \
                patch("main.vision.detect", side_effect=vision.ModelUnavailableError("weights missing")):
            body = post({"transcript": "Describe this in detail", "session_id": SESSION}, jpeg()).json()
        self.assertEqual(body["route"], "vlm")
        self.assertIn("weights missing", body["vision_warning"])

    def test_text_route_gets_detector_summary_and_memory(self):
        with Mocks(chat_answer="Paris.") as m:
            body = post({"transcript": "What is the capital of France?", "session_id": SESSION,
                         "detections": json.dumps(SCENE)}).json()
        self.assertEqual(body["route"], "text")
        self.assertFalse(body["used_vision"])
        self.assertIsNone(body["confidence"])
        m.retrieve.assert_called_once()
        vision_context = m.chat.call_args.args[2]
        self.assertIn("laptop", vision_context)

    def test_no_vision_data_is_honest(self):
        with Mocks() as m:
            body = post({"transcript": "What do you see?", "session_id": SESSION}).json()
        self.assertEqual(body["route"], "no_vision")
        self.assertFalse(body["used_vision"])
        self.assertIn("can't see anything", body["response"])
        m.chat.assert_not_called()
        m.save.assert_called_once()

    def test_validation_errors(self):
        with Mocks() as m:
            cases = [
                ({"session_id": SESSION}, None, 422),
                ({"transcript": "hi"}, None, 422),
                ({"transcript": "hi", "session_id": "nope"}, None, 422),
                ({"transcript": "   ", "session_id": SESSION}, None, 400),
                ({"transcript": "hi", "session_id": SESSION, "detections": "{bad"}, None, 422),
                ({"transcript": "hi", "session_id": SESSION,
                  "detections": json.dumps([{"label": "cup"}])}, None, 422),
                ({"transcript": "hi", "session_id": SESSION}, b"GIF89a", 415),
                ({"transcript": "hi", "session_id": SESSION}, b"", 400),
                ({"transcript": "hi", "session_id": SESSION}, b"not an image", 400),
            ]
            for data, image, status in cases:
                content_type = "image/gif" if image == b"GIF89a" else "image/jpeg"
                with self.subTest(data=data, image=image):
                    response = post(data, image, content_type)
                    self.assertEqual(response.status_code, status, response.text)
            m.save.assert_not_called()

    def test_json_body_rejected(self):
        response = TestClient(main.app).post(
            "/api/respond/vision", json={"transcript": "hi", "session_id": SESSION}
        )
        self.assertEqual(response.status_code, 422)

    def test_oversized_image_rejected(self):
        with Mocks():
            response = post({"transcript": "hi", "session_id": SESSION}, b"\xff" * (6 * 1024 * 1024))
        self.assertEqual(response.status_code, 413)

    def test_image_never_touches_disk(self):
        # A frame over 1 MB would be spooled to a temp file by UploadFile.
        noisy = Image.effect_noise((1024, 768), 100).convert("RGB")
        buffer = io.BytesIO()
        noisy.save(buffer, "PNG")
        data = buffer.getvalue()
        self.assertGreater(len(data), 1024 * 1024)
        with Mocks(vlm_answers=["Gray noise."]), \
                patch("tempfile.SpooledTemporaryFile.rollover", side_effect=AssertionError("spooled")), \
                patch("tempfile.TemporaryFile", side_effect=AssertionError("temp file")):
            response = post({"transcript": "Describe this in detail", "session_id": SESSION,
                             "detections": "[]"}, data, "image/png")
        self.assertEqual(response.status_code, 200, response.text)


class RespondTextCompatibilityTests(unittest.TestCase):
    def test_json_respond_keeps_its_two_keys(self):
        with Mocks(chat_answer="Hi there."):
            response = TestClient(main.app).post(
                "/api/respond", json={"transcript": "Hello", "session_id": SESSION}
            )
        self.assertEqual(response.json(), {"response": "Hi there.", "memory_warning": None})

    def test_json_respond_does_not_invent_vision(self):
        with Mocks() as m:
            response = TestClient(main.app).post(
                "/api/respond", json={"transcript": "What do you see?", "session_id": SESSION}
            )
        self.assertIn("can't see anything", response.json()["response"])
        m.chat.assert_not_called()

    def test_json_respond_text_prompt_says_no_camera(self):
        with Mocks() as m:
            TestClient(main.app).post(
                "/api/respond", json={"transcript": "What's the time in Tokyo?", "session_id": SESSION}
            )
        self.assertIsNone(m.chat.call_args.args[2])


class RoutePreviewTests(unittest.TestCase):
    def preview(self, body):
        with Mocks() as m:
            response = TestClient(main.app).post("/api/respond/route", json=body)
        m.vlm.assert_not_called()
        m.chat.assert_not_called()
        m.save.assert_not_called()
        m.retrieve.assert_not_called()
        return response

    def test_preview_routes_without_running_anything(self):
        cases = [
            ({"transcript": "Where is the bottle?", "has_image": True, "detections": SCENE},
             "detections", False),
            ({"transcript": "What color is the bottle?", "has_image": True, "detections": SCENE},
             "vlm", True),
            # Only a weak cup: the detection route escalates to the VLM when a frame exists.
            ({"transcript": "Is there a cup here?", "has_image": True, "detections": SCENE},
             "vlm", True),
            ({"transcript": "Is there a cup here?", "has_image": False, "detections": SCENE},
             "detections", False),
            ({"transcript": "What is the capital of France?", "has_image": True, "detections": SCENE},
             "text", False),
            ({"transcript": "What do you see?"}, "no_vision", False),
        ]
        for body, route, uses_vlm in cases:
            with self.subTest(question=body["transcript"], has_image=body.get("has_image")):
                response = self.preview(body)
                self.assertEqual(response.status_code, 200, response.text)
                self.assertEqual(response.json()["route"], route)
                self.assertEqual(response.json()["uses_vision_model"], uses_vlm)

    def test_preview_validation(self):
        self.assertEqual(self.preview({"transcript": " "}).status_code, 400)
        self.assertEqual(self.preview({"transcript": "a" * 4001}).status_code, 413)
        self.assertEqual(self.preview({"transcript": "hi", "detections": [{"label": "x"}]}).status_code, 422)
        self.assertEqual(self.preview({"has_image": True}).status_code, 422)


if __name__ == "__main__":
    unittest.main()
