import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image

import vision
import vision_api

SAMPLE_IMAGE = Path(__file__).resolve().parent / "test_data" / "office_public_domain.jpg"


def _image_bytes(fmt="PNG", size=(32, 24)):
    buffer = io.BytesIO()
    Image.new("RGB", size, (120, 120, 120)).save(buffer, format=fmt)
    return buffer.getvalue()


def _client() -> TestClient:
    # Only the vision router: no database lifespan is involved.
    app = FastAPI()
    app.include_router(vision_api.router)
    return TestClient(app)


FAKE_RESULT = vision.DetectionResult(
    detections=[
        vision.Detection("person", 0.91234, 0.5, 0.1, 0.4, 0.8),
        vision.Detection("laptop", 0.8, 0.2, 0.3, 0.2, 0.2),
    ],
    image_width=32,
    image_height=24,
    inference_ms=12.345,
)


class DetectRouteTests(unittest.TestCase):
    def test_route_is_registered_on_main_app(self):
        from main import app

        paths = app.openapi()["paths"]
        self.assertIn("/api/detect", paths)
        self.assertIn("post", paths["/api/detect"])

    def test_returns_contract_shape_with_mocked_detector(self):
        with mock.patch.object(vision_api.vision, "detect_bytes", return_value=FAKE_RESULT) as detect:
            response = _client().post(
                "/api/detect", files={"file": ("frame.png", _image_bytes(), "image/png")}
            )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        self.assertEqual(
            set(body), {"detections", "person_present", "image_width", "image_height", "inference_ms"}
        )
        self.assertTrue(body["person_present"])
        self.assertEqual(body["image_width"], 32)
        self.assertEqual(body["inference_ms"], 12.3)
        first = body["detections"][0]
        self.assertEqual(first["label"], "person")
        self.assertEqual(first["confidence"], 0.9123)
        self.assertEqual(first["box"], {"x": 0.5, "y": 0.1, "width": 0.4, "height": 0.8})
        # Presence only: no extra per-person fields leak into the payload.
        self.assertEqual(set(first), {"label", "confidence", "box"})
        detect.assert_called_once()

    def test_person_present_false_when_no_person(self):
        result = vision.DetectionResult([], 32, 24, 1.0)
        with mock.patch.object(vision_api.vision, "detect_bytes", return_value=result):
            response = _client().post(
                "/api/detect", files={"file": ("frame.jpg", _image_bytes("JPEG"), "image/jpeg")}
            )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["detections"], [])
        self.assertFalse(response.json()["person_present"])

    def test_missing_file_field_is_422(self):
        response = _client().post("/api/detect", files={"image": ("a.png", _image_bytes(), "image/png")})
        self.assertEqual(response.status_code, 422)

    def test_empty_file_is_400(self):
        response = _client().post("/api/detect", files={"file": ("a.png", b"", "image/png")})
        self.assertEqual(response.status_code, 400)

    def test_oversized_file_is_413(self):
        data = b"\x89PNG" + b"0" * (vision_api.MAX_IMAGE_BYTES)
        with mock.patch.object(vision_api.vision, "detect_bytes") as detect:
            response = _client().post("/api/detect", files={"file": ("a.png", data, "image/png")})
        self.assertEqual(response.status_code, 413)
        detect.assert_not_called()

    def test_wrong_content_type_is_415(self):
        response = _client().post("/api/detect", files={"file": ("a.gif", b"GIF89a", "image/gif")})
        self.assertEqual(response.status_code, 415)

    def test_gif_bytes_with_png_content_type_is_415(self):
        response = _client().post(
            "/api/detect", files={"file": ("a.png", _image_bytes("GIF"), "image/png")}
        )
        self.assertEqual(response.status_code, 415)

    def test_garbage_bytes_are_400(self):
        response = _client().post("/api/detect", files={"file": ("a.jpg", b"not really", "image/jpeg")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("JPEG or PNG", response.json()["detail"])

    def test_missing_weights_is_503_with_instructions(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VISION_MODELS_DIR": tmp}):
            vision.unload_model()
            response = _client().post(
                "/api/detect", files={"file": ("a.png", _image_bytes(), "image/png")}
            )
        vision.unload_model()
        self.assertEqual(response.status_code, 503)
        self.assertIn("download_vision_model.py", response.json()["detail"])

    def test_bad_setting_is_503(self):
        with mock.patch.dict(os.environ, {"VISION_CONFIDENCE": "7"}):
            response = _client().post(
                "/api/detect", files={"file": ("a.png", _image_bytes(), "image/png")}
            )
        self.assertEqual(response.status_code, 503)
        self.assertIn("misconfigured", response.json()["detail"])

    def test_inference_failure_is_502(self):
        with mock.patch.object(
            vision_api.vision, "detect_bytes", side_effect=vision.DetectionError("boom")
        ):
            response = _client().post(
                "/api/detect", files={"file": ("a.png", _image_bytes(), "image/png")}
            )
        self.assertEqual(response.status_code, 502)

    def test_image_is_never_written_to_disk(self):
        # Real decode/letterbox/postprocess path with a fake ONNX session; any
        # file open during the request fails the test.
        import numpy as np

        session = mock.Mock()
        node = mock.Mock()
        node.name = "images"
        session.get_inputs.return_value = [node]
        session.run.return_value = [np.zeros((1, 3549, 85), np.float32)]
        with mock.patch.object(vision, "_load_session", return_value=session), \
                mock.patch("builtins.open", side_effect=AssertionError("open() called")), \
                mock.patch("os.open", side_effect=AssertionError("os.open() called")):
            response = _client().post(
                "/api/detect", files={"file": ("a.png", _image_bytes(), "image/png")}
            )
        self.assertEqual(response.status_code, 200)

    def test_large_frame_is_never_spooled_to_disk(self):
        # Regression: FastAPI's UploadFile spooled uploads over 1 MB into a
        # temp file on disk. A ~1.9 MB noisy PNG must stay in memory.
        import numpy as np

        buffer = io.BytesIO()
        noise = np.random.default_rng(0).integers(0, 255, (800, 800, 3), dtype=np.uint8)
        Image.fromarray(noise).save(buffer, format="PNG")
        data = buffer.getvalue()
        self.assertGreater(len(data), 1024 * 1024)
        received = []
        with mock.patch.object(
            vision_api.vision, "detect_bytes", side_effect=lambda b: received.append(b) or FAKE_RESULT
        ), mock.patch("tempfile.TemporaryFile", side_effect=AssertionError("temp file created")), \
                mock.patch("builtins.open", side_effect=AssertionError("open() called")), \
                mock.patch("os.open", side_effect=AssertionError("os.open() called")):
            response = _client().post("/api/detect", files={"file": ("a.png", data, "image/png")})
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(received, [data])

    def test_oversized_chunked_body_is_413_without_content_length(self):
        boundary = "testboundary"
        crlf = bytes([13, 10])
        head = b"".join([
            f"--{boundary}".encode(), crlf,
            b'Content-Disposition: form-data; name="file"; filename="a.png"', crlf,
            b"Content-Type: image/png", crlf, crlf,
        ])

        def body():
            yield head
            for _ in range(6):
                yield b"0" * (1024 * 1024)
            yield crlf + f"--{boundary}--".encode() + crlf

        with mock.patch.object(vision_api.vision, "detect_bytes") as detect:
            response = _client().post(
                "/api/detect",
                content=body(),
                headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
            )
        self.assertEqual(response.status_code, 413)
        detect.assert_not_called()

    def test_non_multipart_body_is_422(self):
        response = _client().post("/api/detect", json={"file": "x"})
        self.assertEqual(response.status_code, 422)
        self.assertEqual(response.json()["detail"][0]["loc"], ["body", "file"])

    def test_two_files_are_rejected(self):
        response = _client().post(
            "/api/detect",
            files=[("file", ("a.png", _image_bytes(), "image/png")), ("file", ("b.png", _image_bytes(), "image/png"))],
        )
        self.assertEqual(response.status_code, 400)

    def test_decompression_bomb_is_400_not_500(self):
        buffer = io.BytesIO()
        Image.new("1", (20000, 10000)).save(buffer, format="PNG")
        response = _client().post("/api/detect", files={"file": ("b.png", buffer.getvalue(), "image/png")})
        self.assertEqual(response.status_code, 400)
        self.assertIn("pixels", response.json()["detail"])

    def test_corrupt_weights_are_503_json(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VISION_MODELS_DIR": tmp}):
            Path(tmp, "yolox_tiny.onnx").write_bytes(b"garbage")
            vision.unload_model()
            response = _client().post(
                "/api/detect", files={"file": ("a.png", _image_bytes(), "image/png")}
            )
        vision.unload_model()
        self.assertEqual(response.status_code, 503)
        self.assertIn("download_vision_model.py", response.json()["detail"])

    def test_openapi_documents_the_file_field(self):
        from main import app

        schema = app.openapi()["paths"]["/api/detect"]["post"]["requestBody"]["content"]
        self.assertIn("file", schema["multipart/form-data"]["schema"]["properties"])


@unittest.skipUnless(
    vision.model_path(vision.model_name()).is_file(),
    "Real-model test skipped: weights missing. Run "
    "backend\\download_vision_model.py to download them.",
)
class RealModelDetectTests(unittest.TestCase):
    """Runs the actual YOLOX ONNX model on a public-domain photo.

    backend/test_data/office_public_domain.jpg: FEMA photo by Patsy Lynch,
    public domain (US government work); three people, laptops, bottles.
    """

    def tearDown(self):
        vision.unload_model()

    def test_finds_people_and_laptop_in_sample_photo(self):
        response = _client().post(
            "/api/detect",
            files={"file": ("office.jpg", SAMPLE_IMAGE.read_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 200, response.text)
        body = response.json()
        labels = [d["label"] for d in body["detections"]]
        self.assertTrue(body["person_present"])
        self.assertGreaterEqual(labels.count("person"), 2)
        self.assertIn("laptop", labels)
        self.assertEqual((body["image_width"], body["image_height"]), (960, 742))
        for d in body["detections"]:
            box = d["box"]
            self.assertGreaterEqual(box["x"], 0)
            self.assertLessEqual(box["x"] + box["width"], 1.0001)
            self.assertLessEqual(box["y"] + box["height"], 1.0001)
        self.assertLess(body["inference_ms"], 1000)

    def test_blank_image_has_no_person(self):
        response = _client().post(
            "/api/detect", files={"file": ("blank.png", _image_bytes(size=(640, 480)), "image/png")}
        )
        self.assertEqual(response.status_code, 200)
        self.assertFalse(response.json()["person_present"])


if __name__ == "__main__":
    unittest.main()
