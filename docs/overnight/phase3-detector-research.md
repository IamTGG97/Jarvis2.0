# Phase 3 detector research and selection

Date: 2026-11-17. Machine: AMD Athlon Gold 3150U (2 cores), 5.9 GB RAM, no GPU,
Windows 11, Python 3.14.0. Author: `vision-detector-backend` overnight agent.

## Decision

**YOLOX-Tiny (ONNX, Apache-2.0), run with `onnxruntime` 1.31.0 on CPU.**
YOLOX-Nano is supported by the same code as a low-resource fallback
(`VISION_MODEL=yolox_nano`).

Why, in one paragraph: of the candidates whose artifact I could download, verify
and decode correctly, YOLOX-Tiny has the best published COCO accuracy (32.8 mAP),
gave the most confident and complete detections on both test photos, and runs in
about 0.1-0.2 s per frame on this 2-core laptop while adding only about 65 MB of
RAM. It is hosted on GitHub releases (reliable), and its decoding is simple and
documented in the official YOLOX ONNX demo. `onnxruntime` is already installed (it
is a dependency of faster-whisper), so no new runtime was needed.

## Runtime check (Python 3.14 / win_amd64)

- `onnxruntime` **1.31.0** is already in `.venv` (pulled in by `faster-whisper`, MIT
  license). `import onnxruntime` works on Python 3.14.0 and reports
  `CPUExecutionProvider`. So the OpenCV `cv2.dnn` fallback is not needed.
- Image decoding: **Pillow 13.0.0** (license MIT-CMU, a permissive license) ships a
  `cp314-win_amd64` wheel. It is the only new dependency (`pillow>=11,<14`).
- NumPy 2.5.4 is already installed.

## Candidates

All sizes are file sizes of the exact artifact. SHA-256 was computed locally on the
downloaded file.

| Model | License | Artifact URL | SHA-256 | Size | COCO mAP (published) |
|---|---|---|---|---|---|
| **YOLOX-Tiny** | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_tiny.onnx | `427cc366d34e27ff7a03e2899b5e3671425c262ea2291f88bb942bc1cc70b0f7` | 20.2 MB | 32.8 @416 |
| YOLOX-Nano | Apache-2.0 | https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/yolox_nano.onnx | `c789161ed43c8269fcd4e67c67eeeb4e80c622da2eb296a20bc6007bd18a0b7d` | 3.7 MB | 25.8 @416 |
| NanoDet-Plus-m 320 | Apache-2.0 | https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_320.onnx | `4f12723cce3d48e47ca92cb925ba74d97a965c069208edca660bbb9f7ce2c610` | 4.8 MB | 27.0 @320 |
| NanoDet-Plus-m 416 | Apache-2.0 | https://github.com/RangiLyu/nanodet/releases/download/v1.0.0-alpha-1/nanodet-plus-m_416.onnx | `59d2f166088889c902f523bf08079391993491324f0d84847e3c4016a8f7cc3d` | 4.8 MB | 30.4 @416 |
| PP-PicoDet-S 416 (LCNet) | Apache-2.0 | https://paddledet.bj.bcebos.com/deploy/third_engine/picodet_s_416_coco_lcnet.onnx | `d080f8d9931bf9164934a7863e989605c7b3aa7751e35be406e20aa6b68256a7` | 4.8 MB | 32.5 @416 |
| D-FINE-N (HF ONNX export) | Apache-2.0 | https://huggingface.co/onnx-community/dfine_n_coco-ONNX/resolve/380d2839c327efaf65dd0fe0c2c10ab7fadd5473/onnx/model.onnx | `0f684f409618ee8a822410e754a29caa817d1aa16283ce89cad936d0a48e2f35` | 15.3 MB | 42.8 @640 |
| SSD-MobileNet v1 (ONNX model zoo, opset 12) | Apache-2.0 (TF Object Detection API weights, ONNX zoo) | https://github.com/onnx/models/raw/main/validated/vision/object_detection_segmentation/ssd-mobilenetv1/model/ssd_mobilenet_v1_12.onnx | `b8fba5e404077d4048d27fcd1667e85e27e192eb9bf51e696c46a3acd7d21058` | 29.5 MB | ~23 |

Excluded on license: Ultralytics YOLOv5/v8/v11 (AGPL-3.0), YOLOv6/v7/v9 (GPL-3.0).

### Input, normalization, output layout and decoding (verified by running them)

- **YOLOX (Tiny and Nano)**: input `images` `[1,3,416,416]` float32, **BGR, raw 0-255,
  no mean/std** (YOLOX >= 0.1.1rc0 removed the legacy normalization). Letterbox:
  resize keeping aspect ratio so the long side is 416, paste at the **top-left** of a
  416x416 canvas filled with 114. Output `output` `[1,3549,85]`: 3549 = 52²+26²+13²
  grid cells for strides 8/16/32, in that order (row-major). Per row:
  `[tx, ty, tw, th, objectness, 80 class scores]`; objectness and class scores are
  already sigmoid-activated. Decode: `cx = (tx + grid_x) * stride`,
  `cy = (ty + grid_y) * stride`, `w = exp(tw) * stride`, `h = exp(th) * stride`;
  score = objectness × class score; divide by the letterbox ratio to get original
  pixels; then class-wise NMS (IoU 0.45). Labels: the 80 COCO classes in the
  standard order (`person`, `bicycle`, `car`, ... `toothbrush`).
- **NanoDet-Plus-m**: input `data` `[1,3,S,S]`, BGR, mean `[103.53,116.28,123.675]`,
  std `[57.375,57.12,58.395]`. Output `[1,N,112]` = 80 sigmoid class scores + 4×8
  distribution (GFL/DFL) logits; strides 8/16/32/64, anchor points at `x*stride`
  (no 0.5 offset). Decode: softmax over 8 bins, expected value × stride gives
  left/top/right/bottom distances. 80 COCO labels.
- **PP-PicoDet-S (LCNet)**: input `image` `[1,3,416,416]`, RGB, `/255` then ImageNet
  mean/std. Eight outputs: four class maps `[1,K,80]` (sigmoid scores) and four
  `[1,K,32]` DFL maps for strides 8/16/32/64; anchor points at `(x+0.5)*stride`.
  Same DFL decode as NanoDet. 80 COCO labels.
- **D-FINE-N (HF export)**: input `pixel_values` `[B,3,H,W]` (dynamic), per its
  `preprocessor_config.json`: plain resize to 640x640, `/255`, no normalization.
  Outputs `logits [B,300,80]` (sigmoid) and `pred_boxes [B,300,4]` normalized
  `cx,cy,w,h`. DETR-style, no NMS needed.
- **SSD-MobileNet v1**: input `inputs` `[1,H,W,3]` uint8 RGB, any size; NMS is inside
  the graph. Outputs `detection_boxes` (normalized `y1,x1,y2,x2`),
  `detection_classes` (1-based COCO-91 ids, which need a different label map),
  `detection_scores`, `num_detections`.

## Measurements on this machine

Method: scratch script, one fresh Python process per model, `onnxruntime`
`intra_op_num_threads=2`, 1 cold run + median of 8 warm runs, process working set
read from Windows `K32GetProcessMemoryInfo`. The Python + NumPy + Pillow +
onnxruntime baseline is about 50 MB working set before any model is loaded. Free
RAM at the time: about 450 MB of 5.9 GB (VS Code and other apps open; Ollama had
no model loaded). Score threshold 0.35.

Test photos (both public domain, US government works, from Wikimedia Commons):
- `street`: "A Home Built for the Mondragon Family (20120716-RD-LSC-1519)", USDA photo
  by Lance Cheung, 960x588. A man and a girl on a pink bicycle, a woman on a bicycle
  far left, a parked car.
- `office`: "FEMA - 44994 - FEMA Public Assistance specialist in Illinois", FEMA
  photo, 960x742. Three people, three laptops, water bottles, a table, a chair.

| Model | Warm latency (street / office) | Cold first run | Working set after inference | What it found (office) |
|---|---|---|---|---|
| **YOLOX-Tiny** | 175 / 116 ms | 101-189 ms | 115 MB (+65) | 3 persons (0.92, 0.82, 0.77), 4 laptops, bottle 0.90, cup, dining table |
| YOLOX-Nano | 70 / 32 ms | 42-56 ms | 88 MB (+38) | 3 persons, 4 laptops, 2 bottles, table; lower confidence |
| NanoDet-Plus-m 416 | 64 / 54 ms | 61-68 ms | 97 MB (+47) | 3 persons (lower conf.), 3 laptops, bottle, cup, book |
| NanoDet-Plus-m 320 | 39 ms (street) | 42 ms | 88 MB | street: only 1 person, 1 bicycle |
| PP-PicoDet-S 416 | 86 / 84 ms | 77-85 ms | 108 MB (+58) | 3 persons, 2 laptops, 2 bottles, 2 chairs, book, cup, table |
| D-FINE-N 640 | 227 / 214 ms | 231-240 ms | 148 MB (+98) | **no persons**; only "keyboard 0.40", "chair 0.36" |
| SSD-MobileNet v1 | 62 ms (street) | 78 ms; **33 s session load** | 175 MB (+125) | not run (different label map) |

Session load time (reading the file and building the graph) was 70-230 ms for all
except D-FINE (~750 ms) and SSD-MobileNet (33 s).

## Why not the others

- **D-FINE-N** has by far the best published accuracy, but this ONNX export did not
  find any person in the office photo with any preprocessing variant I tried (plain
  /255, BGR, ImageNet normalization). It may be an export problem or something I
  misread, but I could not verify it tonight, so I did not adopt it. Worth a retry
  later with an official D-FINE export.
- **PP-PicoDet-S** was a close second: similar accuracy, faster (85 ms) and found more
  small objects. I chose YOLOX-Tiny because its artifact is on GitHub (the PicoDet
  file is on a Baidu CDN that may be slow or blocked), its decoding is simpler and
  documented officially, and its confidences were higher. If latency ever matters
  more, PicoDet is the first thing to try.
- **YOLOX-Nano** is 3-4x faster and 27 MB lighter but noticeably less accurate (25.8
  mAP). It is kept as a supported option because it uses the same decoder.
- **NanoDet-Plus** was fast but less confident, and the 320 version missed people.
- **SSD-MobileNet v1** is old (about 23 mAP), the largest file, and took 33 s to load.

## Accuracy caveat

Two photos are a sanity check, not an accuracy benchmark. The mAP column is from each
project's published COCO val2017 results, not measured here.
