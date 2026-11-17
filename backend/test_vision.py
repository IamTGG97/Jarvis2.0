import hashlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

import download_vision_model
import vision


def _png_bytes(width=8, height=6, color=(10, 20, 30), fmt="PNG"):
    buffer = io.BytesIO()
    Image.new("RGB", (width, height), color).save(buffer, format=fmt)
    return buffer.getvalue()


def _yolox_output(size=416):
    rows = sum((size // s) ** 2 for s in vision.STRIDES)
    return np.zeros((1, rows, 85), dtype=np.float32)


def _row_index(size, stride, gx, gy):
    offset = 0
    for s in vision.STRIDES:
        cells = size // s
        if s == stride:
            return offset + gy * cells + gx
        offset += cells * cells
    raise ValueError(stride)


class LetterboxTests(unittest.TestCase):
    def test_wide_image_is_scaled_and_padded_at_bottom(self):
        image = Image.new("RGB", (832, 416), (255, 0, 0))  # pure red
        tensor, ratio = vision.letterbox(image, 416)
        self.assertEqual(tensor.shape, (1, 3, 416, 416))
        self.assertEqual(tensor.dtype, np.float32)
        self.assertAlmostEqual(ratio, 0.5)
        # BGR order, raw 0-255: red lands in channel 2.
        self.assertEqual(tensor[0, 2, 0, 0], 255.0)
        self.assertEqual(tensor[0, 0, 0, 0], 0.0)
        # Rows below the 208-px tall image are padding (114).
        self.assertTrue(np.all(tensor[0, :, 208:, :] == 114.0))
        self.assertTrue(np.all(tensor[0, 2, :208, :] == 255.0))

    def test_tall_image_pads_right_side(self):
        image = Image.new("RGB", (100, 400), (0, 0, 0))
        tensor, ratio = vision.letterbox(image, 416)
        self.assertAlmostEqual(ratio, 1.04)
        self.assertTrue(np.all(tensor[0, :, :, 104:] == 114.0))
        self.assertTrue(np.all(tensor[0, :, :, :104] == 0.0))


class DecodeTests(unittest.TestCase):
    def test_decodes_one_cell_into_original_pixels(self):
        output = _yolox_output()
        row = _row_index(416, 16, gx=10, gy=5)
        output[0, row, 0:4] = [0.5, 0.5, np.log(4.0), np.log(2.0)]
        output[0, row, 4] = 0.9  # objectness
        output[0, row, 5 + 63] = 0.8  # laptop
        boxes, scores, classes = vision.decode_yolox(output, 416, ratio=0.5, conf_threshold=0.3)
        self.assertEqual(len(boxes), 1)
        # center (10.5*16, 5.5*16) = (168, 88); size (64, 32); /0.5 back to original.
        np.testing.assert_allclose(boxes[0], [272, 144, 400, 208], atol=1e-3)
        self.assertAlmostEqual(float(scores[0]), 0.72, places=5)
        self.assertEqual(vision.COCO_LABELS[int(classes[0])], "laptop")

    def test_threshold_uses_objectness_times_class_score(self):
        output = _yolox_output()
        output[0, 0, 4] = 0.5
        output[0, 0, 5] = 0.6  # 0.30 < 0.35
        boxes, _, _ = vision.decode_yolox(output, 416, 1.0, 0.35)
        self.assertEqual(len(boxes), 0)

    def test_wrong_output_shape_is_an_error(self):
        with self.assertRaises(vision.DetectionError):
            vision.decode_yolox(np.zeros((1, 10, 85), np.float32), 416, 1.0, 0.35)


class NmsTests(unittest.TestCase):
    def test_suppresses_overlaps_and_keeps_separate_boxes(self):
        boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
        scores = np.array([0.6, 0.9, 0.5], np.float32)
        self.assertEqual(vision.nms(boxes, scores, 0.45), [1, 2])

    def test_low_overlap_is_kept(self):
        boxes = np.array([[0, 0, 10, 10], [8, 0, 18, 10]], np.float32)
        self.assertEqual(sorted(vision.nms(boxes, np.array([0.9, 0.8], np.float32))), [0, 1])

    def test_postprocess_is_class_wise_normalized_and_clipped(self):
        boxes = np.array([[-10, 0, 50, 100], [0, 0, 50, 100], [700, 300, 900, 600]], np.float32)
        scores = np.array([0.9, 0.8, 0.7], np.float32)
        classes = np.array([0, 63, 0])
        dets = vision.postprocess(boxes, scores, classes, 800, 400)
        self.assertEqual([d.label for d in dets], ["person", "laptop", "person"])
        first = dets[0]
        self.assertEqual((first.x, first.y), (0.0, 0.0))
        self.assertAlmostEqual(first.width, 50 / 800)
        self.assertAlmostEqual(first.height, 0.25)
        last = dets[2]
        self.assertAlmostEqual(last.x + last.width, 1.0)
        self.assertAlmostEqual(last.y + last.height, 1.0)


class DecodeImageTests(unittest.TestCase):
    def test_png_and_jpeg_accepted(self):
        self.assertEqual(vision.decode_image(_png_bytes()).size, (8, 6))
        self.assertEqual(vision.decode_image(_png_bytes(fmt="JPEG")).mode, "RGB")

    def test_rejects_empty_garbage_and_other_formats(self):
        for data in (b"", b"not an image", _png_bytes(fmt="GIF"), _png_bytes(fmt="BMP")):
            with self.assertRaises(vision.ImageValidationError):
                vision.decode_image(data)

    def test_rejects_truncated_image(self):
        with self.assertRaises(vision.ImageValidationError):
            vision.decode_image(_png_bytes(64, 64, fmt="JPEG")[:200])

    def test_rejects_too_many_pixels(self):
        with mock.patch.object(vision, "MAX_IMAGE_PIXELS", 40):
            with self.assertRaises(vision.ImageValidationError):
                vision.decode_image(_png_bytes(8, 6))


class FakeSession:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.inputs = []

    def get_inputs(self):
        node = mock.Mock()
        node.name = "images"
        return [node]

    def run(self, _, feeds):
        self.inputs.append(feeds)
        if self.error:
            raise self.error
        return [self.output]


class DetectTests(unittest.TestCase):
    def tearDown(self):
        vision.unload_model()

    def test_detect_with_mocked_session(self):
        output = _yolox_output()
        row = _row_index(416, 32, gx=6, gy=6)
        output[0, row, 0:4] = [0.5, 0.5, np.log(2.0), np.log(4.0)]
        output[0, row, 4] = 1.0
        output[0, row, 5] = 0.9  # person
        session = FakeSession(output)
        with mock.patch.object(vision, "_load_session", return_value=session):
            result = vision.detect(Image.new("RGB", (416, 416)))
        self.assertTrue(result.person_present)
        payload = result.as_dict()
        self.assertEqual(payload["image_width"], 416)
        self.assertEqual(payload["detections"][0]["label"], "person")
        box = payload["detections"][0]["box"]
        self.assertAlmostEqual(box["x"], (208 - 32) / 416, places=3)
        self.assertAlmostEqual(box["height"], 128 / 416, places=3)
        self.assertEqual(session.inputs[0]["images"].shape, (1, 3, 416, 416))

    def test_keep_loaded_off_unloads_after_each_request(self):
        session = FakeSession(_yolox_output())
        loads = []

        def fake_load(name):
            loads.append(name)
            vision._session = session
            vision._session_name = name
            return session

        with mock.patch.object(vision, "_load_session", side_effect=fake_load),                 mock.patch.dict(os.environ, {"VISION_KEEP_LOADED": "off"}):
            vision.detect(Image.new("RGB", (32, 32)))
            self.assertIsNone(vision._session)
        with mock.patch.dict(os.environ, {"VISION_KEEP_LOADED": ""}):
            self.assertTrue(vision.keep_loaded())

    def test_no_person_when_only_objects(self):
        result = vision.DetectionResult(
            [vision.Detection("cup", 0.9, 0.1, 0.1, 0.2, 0.2)], 10, 10, 5.0
        )
        self.assertFalse(result.person_present)

    def test_inference_failure_raises_detection_error(self):
        with mock.patch.object(vision, "_load_session", return_value=FakeSession(error=RuntimeError("boom"))):
            with self.assertRaises(vision.DetectionError):
                vision.detect(Image.new("RGB", (32, 32)))

    def test_missing_weights_raise_model_unavailable_with_instructions(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(os.environ, {"VISION_MODELS_DIR": tmp}):
            vision.unload_model()
            with self.assertRaises(vision.ModelUnavailableError) as ctx:
                vision.detect(Image.new("RGB", (32, 32)))
        self.assertIn("download_vision_model.py", str(ctx.exception))

    def test_unknown_model_name(self):
        with mock.patch.dict(os.environ, {"VISION_MODEL": "yolov8n"}):
            with self.assertRaises(vision.ModelUnavailableError):
                vision.detect(Image.new("RGB", (32, 32)))

    def test_confidence_setting_validated(self):
        with mock.patch.dict(os.environ, {"VISION_CONFIDENCE": "1.5"}):
            with self.assertRaises(ValueError):
                vision.confidence_threshold()
        with mock.patch.dict(os.environ, {"VISION_CONFIDENCE": ""}):
            self.assertEqual(vision.confidence_threshold(), 0.35)


class _FakeStream:
    def __init__(self, payload):
        self.payload = payload

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        return None

    def iter_bytes(self, _):
        yield self.payload


class DownloadTests(unittest.TestCase):
    def _spec(self, payload):
        return {
            "filename": "fake.onnx",
            "url": "https://example.invalid/fake.onnx",
            "sha256": hashlib.sha256(payload).hexdigest(),
            "input_size": 416,
        }

    def test_verified_download_is_moved_into_place(self):
        payload = b"weights"
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"VISION_MODELS_DIR": tmp}), \
                mock.patch.dict(vision.MODELS, {"fake": self._spec(payload)}), \
                mock.patch.object(download_vision_model.httpx, "stream", return_value=_FakeStream(payload)) as stream:
            path = download_vision_model.download("fake")
            self.assertEqual(path.read_bytes(), payload)
            download_vision_model.download("fake")  # already verified: no second request
            self.assertEqual(stream.call_count, 1)

    def test_checksum_mismatch_discards_file(self):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.dict(os.environ, {"VISION_MODELS_DIR": tmp}), \
                mock.patch.dict(vision.MODELS, {"fake": self._spec(b"expected")}), \
                mock.patch.object(download_vision_model.httpx, "stream", return_value=_FakeStream(b"tampered")):
            with self.assertRaises(RuntimeError):
                download_vision_model.download("fake")
            self.assertEqual(list(Path(tmp).iterdir()), [])


if __name__ == "__main__":
    unittest.main()
