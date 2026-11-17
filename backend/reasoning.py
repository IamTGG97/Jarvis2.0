"""Question routing and vision reasoning for Jarvis (Phase 4).

Every question goes to exactly one route:

- ``text``: no vision needed. The normal memory + chat-model path. The prompt
  tells the model it has no camera image for this turn, so it can't invent
  what it "sees".
- ``detections``: presence, counting, "where is", and "what do you see"
  questions. Answered in milliseconds from the object detector's structured
  results, with no language model at all.
- ``vlm``: color, text/brand, comparison, "what is this", and detailed
  description questions. The camera image (or a crop of the object the
  question is about) plus the detector context goes to a local vision-language
  model through Ollama.
- ``no_vision``: the question needs vision but no usable image or detections
  came with it. Jarvis says so honestly instead of guessing.

The router is a transparent keyword/rule router (see ``route_question``), so
every decision can be explained and unit-tested. Pure functions live here; the
HTTP layer is ``reasoning_api.py``.
"""

from __future__ import annotations

import base64
import io
import logging
import os
import re
from dataclasses import dataclass, field

import httpx
from PIL import Image

from vision import COCO_LABELS

logger = logging.getLogger(__name__)

# Detections at or above this score are treated as reliable. The detector's
# own floor (VISION_CONFIDENCE, default 0.35) lets weaker ones through; on the
# test photos those were often wrong (hand sanitizer read as "cup" at 0.59,
# keyboard at 0.36), so they are only ever reported as "might be".
CONFIDENT = 0.5
HIGH = 0.75

KNOWN_LABELS = frozenset(COCO_LABELS)

DEFAULT_VLM_MODEL = "moondream"

# COCO label -> natural word used in answers and prompts.
NATURAL_NAMES = {
    "cell phone": "phone",
    "tv": "TV or monitor",
    "dining table": "table",
    "potted plant": "plant",
    "sports ball": "ball",
    "hair drier": "hair dryer",
    "remote": "remote control",
    "mouse": "computer mouse",
    "wine glass": "wine glass",
    "couch": "couch",
    "motorcycle": "motorbike",
}

PLURALS = {
    "person": "people",
    "knife": "knives",
    "couch": "couches",
    "bus": "buses",
    "glass": "glasses",
    "sandwich": "sandwiches",
    "toothbrush": "toothbrushes",
    "TV or monitor": "TVs or monitors",
    "skis": "skis",
    "scissors": "scissors",
    "computer mouse": "computer mice",
}

# Words a user might say -> COCO label. Plural "s" is stripped before lookup.
WORD_TO_LABEL = {
    "person": "person", "people": "person", "persons": "person", "anyone": "person",
    "anybody": "person", "someone": "person", "somebody": "person", "man": "person",
    "men": "person", "woman": "person", "women": "person", "guy": "person",
    "kid": "person", "kids": "person", "child": "person", "children": "person",
    "human": "person", "face": "person",
    "phone": "cell phone", "cellphone": "cell phone", "smartphone": "cell phone",
    "iphone": "cell phone", "mobile": "cell phone",
    "tv": "tv", "television": "tv", "monitor": "tv", "screen": "tv",
    "table": "dining table", "desk": "dining table",
    "sofa": "couch", "couch": "couch",
    "plant": "potted plant", "houseplant": "potted plant",
    "cup": "cup", "mug": "cup",
    "bottle": "bottle",
    "laptop": "laptop", "computer": "laptop", "notebook": "laptop",
    "keyboard": "keyboard", "mouse": "mouse", "mice": "mouse",
    "remote": "remote", "controller": "remote",
    "book": "book",
    "chair": "chair",
    "dog": "dog", "puppy": "dog", "cat": "cat", "kitten": "cat", "bird": "bird",
    "car": "car", "bike": "bicycle", "bicycle": "bicycle", "motorbike": "motorcycle",
    "backpack": "backpack", "bag": "handbag", "handbag": "handbag", "purse": "handbag",
    "umbrella": "umbrella", "clock": "clock", "scissor": "scissors", "scissors": "scissors",
    "toothbrush": "toothbrush", "banana": "banana", "apple": "apple", "orange": "orange",
    "pizza": "pizza", "donut": "donut", "doughnut": "donut", "cake": "cake",
    "sandwich": "sandwich", "bowl": "bowl", "fork": "fork", "knife": "knife",
    "knive": "knife", "spoon": "spoon", "vase": "vase", "tie": "tie",
    "suitcase": "suitcase", "ball": "sports ball", "bed": "bed", "sink": "sink",
    "microwave": "microwave", "oven": "oven", "toaster": "toaster",
    "fridge": "refrigerator", "refrigerator": "refrigerator", "teddy": "teddy bear",
    "glass": "wine glass", "carrot": "carrot", "broccoli": "broccoli",
}

# Nouns that are not objects ("how many things do you see").
GENERIC_NOUNS = {
    "thing", "things", "object", "objects", "item", "items", "stuff", "something",
    "anything", "it", "this", "that", "there", "here", "you", "i", "a", "an", "the",
    "any", "my", "your", "of", "in", "on",
}

CLOTHING = r"shirt|t-shirt|hoodie|jacket|sweater|hat|cap|glasses|dress|jumper|top"

# Questions that are only answerable by looking at the image in detail.
VLM_PATTERNS = [
    r"\bcolou?rs?\b",
    # Color words ("how many red cups"). "orange" is left out: it's a COCO fruit.
    r"\b(red|blue|green|yellow|purple|pink|black|white|gr[ae]y|brown|silver|gold)\b",
    r"\bbrand\b", r"\blogo\b", r"\bwhat make\b", r"\bwhich make\b", r"\bmanufacturer\b",
    r"\bwhat model\b", r"\bwhich model\b",
    r"\bwhat does (it|this|that|the [a-z ]+?) say\b", r"\bwhat do(es)? [a-z' ]{0,60}?\bsays?\b",
    r"\bread\b", r"\btext\b", r"\bwritten\b", r"\bwriting\b", r"\bwords?\b", r"\blabel\b",
    r"\bspell", r"\btitle\b",
    r"\bwhich (one )?(is|looks|seems) (the )?(bigger|smaller|larger|taller|shorter|closer|newer|older|"
    r"brighter|darker|biggest|smallest|largest|tallest|longer|wider)\b",
    r"\b(bigger|smaller|larger|taller|shorter|closer|darker|brighter) than\b",
    r"\bwhich [a-z]+ (is|looks) (the )?(bigger|smaller|larger|taller|shorter|closer|newer|older|"
    r"brighter|darker|biggest|smallest|largest|tallest|longer|wider|heavier|nearest|closest)\b",
    r"\bcompare\b", r"\bcomparison\b", r"\bdifference\b", r"\bdifferent\b", r"\bsame\b",
    r"\bdescribe\b", r"\bin detail\b", r"\bdetails?\b", r"\blook like\b", r"\blooks? like\b",
    r"\bwhat kind of\b", r"\bwhat type of\b", r"\bwhat sort of\b",
    r"\bwhat('s| is) (this|that|it)\b", r"\bwhat are (these|those)\b", r"\bwhats (this|that)\b",
    r"\bwhat am i holding\b", r"\bin my hand\b", r"\bholding\b",
    r"\bwearing\b", r"\bmade of\b", r"\bmaterial\b", r"\bshape\b", r"\bpattern\b",
    r"\b(how|do|does) (do |does )?(i|my [a-z]+) look\b",
    r"\bhow full\b",
    r"\b(is|are) (it|this|that|these|those|the [a-z]+|my [a-z]+) "
    r"(on|off|open|closed|empty|full|clean|dirty|broken|plugged in)\b",
]

# Presence / counting / listing / locating: the detector can answer these.
DETECTION_PATTERNS = {
    "count": [r"\bhow many\b", r"\bcount\b", r"\bnumber of\b"],
    "list": [
        r"\bwhat (do|can) you see\b", r"\bwhat are you seeing\b", r"\bwhat do you notice\b",
        r"\bwhat('?s| is) (in front of|around|near) (you|me)\b",
        r"\bwhat('s| is) (here|in (the|this) (room|picture|image|frame|view))\b",
        r"\blook around\b", r"\bwhat objects\b", r"\bwhat things\b", r"\bwhat can you spot\b",
        r"\bwhat('s| is| are) (on|behind|next to|near|around|beside) (me|my|the|this)\b",
        r"\bwhat('s| is) (happening|going on) (here|in (the|this) (picture|image|frame|room|photo))\b",
        r"\blook at (this|that|me)\b",
    ],
    "locate": [r"\bwhere('s| is| are)\b", r"^(show me|point (at|to)|find) (the|my|a|an)\b"],
    "presence": [
        r"\bis there\b", r"\bare there\b", r"\bdo you see\b", r"\bcan you see\b",
        r"\bdo you notice\b", r"\bcan you spot\b", r"\bis (anyone|anybody|someone|somebody)\b",
        r"\b(anyone|anybody|someone|somebody) (here|there|in)\b",
        r"\bam i (in|on) (the )?(frame|view|camera|picture|shot)\b", r"\bsee me\b",
        r"\bvisible\b",
        r"\bwho('s| is| are) (this|that|there|here|in|behind|next to|with me|in front of)\b",
        r"\b(is|are) (my|the|a|an|any) [a-z]+( [a-z]+)? (in|on) (the |this )?"
        r"(picture|frame|view|camera|shot|image|photo)\b",
    ],
}

# A counting/presence/attribute question only counts as visual when it is
# anchored to the scene. "How many days are in a year?" must stay ``text``.
VISION_ANCHORS = (
    r"\b(see|seeing|look|looking|view|camera|frame|picture|image|photo|visible|spot)\b"
    r"|\bin front of\b|\bhere\b|\bin (the|this|my) (room|office|kitchen|desk|hand|hands)\b"
    r"|\bon (the|my|this) (desk|table|wall|shelf|screen)\b|\bholding\b|\bwearing\b"
    r"|\b(this|that|these|those)\b|\baround (me|you)\b|\bbehind me\b|\bnext to me\b"
    r"|\bmy (" + CLOTHING + r")\b"
)

# Telling or asking about remembered facts ("remember my favorite color is
# green", "what did I say my car looked like") is memory, not vision.
# Not visual even though they say "see"/"look".
NOT_VISUAL = (
    r"\b(can|do) you see (why|how|if|whether|what i mean|my point)\b"
    r"|\bsee you\b|\blook (it )?up\b|\blook into\b|\blook forward\b|\bwho am i\b"
    r"|\bwhere('s| is) my [a-z]+ (parked|kept|stored|from)\b"
)

# With a frame attached, a definite reference plus one of these means the
# scene ("What color is the mug?"). The broader VLM_PATTERNS are too loose
# here: they sent "Tell me about the Red Sea", "Is the white house in
# Washington?" and "What's the difference between a list and a tuple?" to the
# VLM (minutes on this laptop) just because a frame came along.
SCENE_DETAIL_PATTERNS = [
    r"\bwhat colou?rs? (is|are|was)\b", r"\bwhich colou?r\b",
    r"\bcolou?rs? of (the|my|this|that|these|those)\b",
    r"\bwhat does (it|this|that|the [a-z ]+?) say\b", r"\bwhat('s| is) written\b",
    r"\bread (the|this|that|it|these|those)\b",
    r"\b(what|which) (brand|make|model)\b", r"\blogo\b",
    r"\bwhat am i holding\b", r"\bin my hands?\b", r"\bwearing\b",
    r"\bwhat('s| is) (this|that)\b", r"\bwhats (this|that)\b", r"\bwhat are (these|those)\b",
    r"\bwhich [a-z]+ (is|looks) (the )?(bigger|smaller|larger|taller|shorter|closer|biggest|smallest|"
    r"largest|tallest|nearest|closest)\b",
    r"\b(is|are) (it|this|that|these|those|the [a-z]+|my [a-z]+) "
    r"(on|off|open|closed|empty|full|clean|dirty|broken|plugged in)\b",
    r"\b(how|do|does) (do |does )?(i|my [a-z]+) look\b", r"\bhow full\b",
    r"\bdescribe (the|this|my|what)\b", r"\bwhat kind of [a-z]+ is (this|that|it|my)\b",
]

# A count question about detected objects but with no scene words. Only short
# forms count: "How many people live in Paris?" must stay text.
SHORT_COUNT = (
    r"^(count (the |my |all the )?[a-z]+( (you see|here|in view|in the frame))?"
    r"|how many [a-z]+( [a-z]+)?( are there| can you see| do you see| are here| here| in view"
    r"| visible| right now| are visible)?)$"
)

# First-person statements ("My car is red", "I like the color blue") are
# facts for memory, not questions about the frame.
STATEMENT = r"^(i|i'm|im|i've|my|we|our|we're)\b"

# Only the start of a very long transcript is used for routing: the patterns
# are cheap on normal speech, but a 100k-character transcript took 21 s.
MAX_ROUTING_CHARS = 1000

MEMORY_TALK = (
    r"^(remember|note that|why|how come|explain)\b|\bfavou?rite\b|\bremember\b"
    r"|\bdid i (say|tell|mention)\b|\bi told you\b|\bdid you say\b"
)


# ------------------------------------------------------------ data types ---

@dataclass
class Detection:
    label: str
    confidence: float
    x: float
    y: float
    width: float
    height: float

    @property
    def area(self) -> float:
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        return self.x + self.width / 2, self.y + self.height / 2

    def prominence(self) -> float:
        """Bigger and nearer the middle of the frame = more likely meant."""
        cx, cy = self.center
        off_center = ((cx - 0.5) ** 2 + (cy - 0.5) ** 2) ** 0.5  # 0 .. ~0.71
        centrality = 1.0 - off_center / 0.71
        return self.area * (0.5 + 0.5 * centrality) * (0.5 + 0.5 * self.confidence)

    def as_reference(self) -> dict:
        return {
            "label": natural_name(self.label),
            "detector_label": self.label,
            "confidence": round(self.confidence, 2),
            "box": {
                "x": round(self.x, 4), "y": round(self.y, 4),
                "width": round(self.width, 4), "height": round(self.height, 4),
            },
            "position": screen_position(self),
        }


@dataclass
class RouteDecision:
    route: str  # "text" | "detections" | "vlm" | "no_vision"
    intent: str  # "chat" | "count" | "list" | "locate" | "presence" | "attribute"
    reason: str
    target_labels: list[str] = field(default_factory=list)
    unknown_target: str | None = None  # an object the detector has no class for
    text: str = ""  # the normalized question


@dataclass
class Answer:
    response: str
    route: str
    used_vision: bool
    objects_referenced: list[dict]
    confidence: str | None  # "high" | "medium" | "low" | None (text route)
    reason: str
    vision_warning: str | None = None


class VlmError(RuntimeError):
    """The vision-language model failed; ``status_code`` maps to HTTP."""

    def __init__(self, message: str, status_code: int = 502) -> None:
        super().__init__(message)
        self.status_code = status_code


# ---------------------------------------------------------------- helpers ---

def natural_name(label: str) -> str:
    return NATURAL_NAMES.get(label, label)


def plural(name: str, count: int) -> str:
    if count == 1:
        return name
    if name in PLURALS:
        return PLURALS[name]
    if name.endswith(("s", "x", "ch", "sh")):
        return name + "es"
    return name + "s"


NUMBER_WORDS = ["no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]


def count_phrase(name: str, count: int) -> str:
    number = NUMBER_WORDS[count] if count < len(NUMBER_WORDS) else str(count)
    if count == 1:
        article = "an" if name[0].lower() in "aeiou" else "a"
        return f"{article} {name}"
    return f"{number} {plural(name, count)}"


def join_words(parts: list[str]) -> str:
    if len(parts) <= 1:
        return "".join(parts)
    return ", ".join(parts[:-1]) + " and " + parts[-1]


def screen_position(detection: Detection) -> str:
    """Where the object appears on the user's (mirrored) preview.

    Boxes are relative to the unmirrored camera frame, but the app shows the
    video mirrored like a mirror, so left and right are flipped here to match
    what the user sees on screen.
    """
    cx, cy = detection.center
    cx = 1.0 - cx
    horizontal = "left" if cx < 0.36 else "right" if cx > 0.64 else "middle"
    vertical = "top" if cy < 0.33 else "bottom" if cy > 0.67 else ""
    if horizontal == "middle" and not vertical:
        return "in the middle of the screen"
    if horizontal == "middle":
        return f"at the {vertical} of the screen"
    if vertical:
        return f"at the {vertical} {horizontal} of the screen"
    return f"on the {horizontal} side of the screen"


def parse_detections(raw: object) -> list[Detection]:
    """Validate the ``/api/detect`` shape (either the full result object or
    just its ``detections`` list). Raises ``ValueError`` with a clear message."""
    if isinstance(raw, dict):
        raw = raw.get("detections")
    if not isinstance(raw, list):
        raise ValueError("detections must be a list (or the /api/detect result object).")
    if len(raw) > 100:
        raise ValueError("At most 100 detections are accepted.")
    parsed = []
    for item in raw:
        try:
            box = item["box"]
            detection = Detection(
                label=str(item["label"]).strip().lower(),
                confidence=float(item["confidence"]),
                x=float(box["x"]), y=float(box["y"]),
                width=float(box["width"]), height=float(box["height"]),
            )
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(
                "Each detection needs label, confidence and box {x, y, width, height}."
            ) from error
        if detection.label not in KNOWN_LABELS:
            raise ValueError(
                f"Unknown detection label {detection.label[:40]!r}; labels must be detector (COCO) classes."
            )
        values = (detection.confidence, detection.x, detection.y, detection.width, detection.height)
        if not all(0.0 <= v <= 1.0 for v in values):
            raise ValueError("Detection confidence and box values must be between 0 and 1.")
        parsed.append(detection)
    return parsed


def _normalize(question: str) -> str:
    text = question[:MAX_ROUTING_CHARS].lower().replace("’", "'")
    text = re.sub(r"[^a-z0-9' -]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z]+(?:-[a-z]+)?", text)


def find_target_labels(text: str) -> list[str]:
    """COCO labels the question mentions, in order of mention."""
    labels: list[str] = []
    if re.search(r"\bwine glass(es)?\b", text):
        labels.append("wine glass")
    if re.search(r"\b(stop sign|traffic light|teddy bear|hot dog|fire hydrant|tennis racket)\b", text):
        labels.append(re.search(
            r"\b(stop sign|traffic light|teddy bear|hot dog|fire hydrant|tennis racket)\b", text
        ).group(1))
    for word in _words(text):
        if word == "glasses":
            continue  # eyeglasses, not a COCO class
        label = WORD_TO_LABEL.get(word)
        if label is None and word.endswith("es"):
            label = WORD_TO_LABEL.get(word[:-2])
        if label is None and word.endswith("s"):
            label = WORD_TO_LABEL.get(word[:-1])
        if label and label not in labels:
            labels.append(label)
    return labels


def _unknown_target(text: str) -> str | None:
    """The object noun after "how many"/"is there a"/... when the detector has
    no class for it (e.g. "pens"), else None."""
    match = re.search(
        r"\b(?:how many|is there(?: an?\b)?|are there(?: any\b)?|do you see(?: (?:an?|any|my|the)\b)?"
        r"|can you see(?: (?:an?|any|my|the)\b)?|where(?:'s| is| are)(?: (?:my|the|a)\b)?"
        r"|count(?: the\b)?|number of)"
        r"\s*((?:[a-z]+ ){0,2}[a-z]+)",
        text,
    )
    if not match:
        return None
    words = [w for w in match.group(1).split() if w not in GENERIC_NOUNS]
    stop = {"are", "is", "do", "can", "you", "there", "in", "on", "see", "here", "right", "now"}
    noun_words = []
    for word in words:
        if word in stop:
            break
        noun_words.append(word)
    if not noun_words:
        return None
    if find_target_labels(" ".join(noun_words)):
        return None
    if noun_words[0] in {"people", "persons", "me", "anyone", "someone"}:
        return None
    return " ".join(noun_words)


def is_vision_question(
    question: str,
    detections: list[Detection] | None = None,
    has_image: bool = False,
) -> bool:
    text = _normalize(question)
    if re.search(MEMORY_TALK, text) or re.search(NOT_VISUAL, text):
        return False
    if re.search(STATEMENT, text) and "?" not in question[:MAX_ROUTING_CHARS]:
        return False
    if any(re.search(p, text) for p in DETECTION_PATTERNS["list"]):
        return True
    anchored = re.search(VISION_ANCHORS, text) is not None
    detail_kind = any(re.search(p, text) for p in VLM_PATTERNS)
    visual_kind = detail_kind or any(
        re.search(p, text) for key in ("count", "presence", "locate") for p in DETECTION_PATTERNS[key]
    )
    if visual_kind and anchored:
        return True
    has_vision = has_image or detections is not None
    targets = find_target_labels(text)
    count_kind = any(re.search(p, text) for p in DETECTION_PATTERNS["count"])
    # "What color is the cup?" / "Which laptop is bigger?" with that object in
    # view. Counting needs the short form ("count the cups").
    if visual_kind and detections and {d.label for d in detections}.intersection(targets):
        if not count_kind or re.search(SHORT_COUNT, text):
            return True
    # "Where's my phone?" with a frame: the detector can say it isn't in view.
    if has_vision and targets and re.search(r"^(where('s| is| are)|show me|point (at|to)|find)\b", text):
        return True
    # With a frame attached, a definite reference ("the sign", "my mug") to a
    # scene-detail question means the scene; "What color is a banana?" stays text.
    if (has_vision and re.search(r"\b(the|my|your|our|this|that)\b", text)
            and any(re.search(p, text) for p in SCENE_DETAIL_PATTERNS)):
        return True
    return False


def _subject_text(text: str) -> str:
    """The question without place phrases, so "is there a phone on my desk"
    is about the phone, not the desk."""
    stripped = re.sub(
        r"\b(on|in|at|under|near|by|behind|beside|inside|next to|in front of)"
        r"( the| my| your| this| that| a)? [a-z]+",
        " ",
        text,
    )
    return stripped if find_target_labels(stripped) or _unknown_target(stripped) else text


# ----------------------------------------------------------------- router ---

def route_question(
    question: str,
    has_image: bool,
    detections: list[Detection] | None,
) -> RouteDecision:
    """Pick the route. ``detections`` is None when the client sent none (and
    no image to run the detector on); an empty list means "detector ran and
    found nothing"."""
    decision = _route(question, has_image, detections)
    decision.text = _normalize(question)
    return decision


def _route(
    question: str,
    has_image: bool,
    detections: list[Detection] | None,
) -> RouteDecision:
    text = _normalize(question)
    if not is_vision_question(question, detections, has_image):
        return RouteDecision("text", "chat", "no visual cue in the question")

    subject = _subject_text(text)
    targets = find_target_labels(subject)
    unknown = _unknown_target(subject)
    if re.search(r"\bam i (in|on|visible)\b|\bsee me\b|\bnotice me\b|\blook at me\b"
                 r"|\bwho('s| is| are)\b", text):
        targets, unknown = ["person"], None
    wants_vlm = any(re.search(p, text) for p in VLM_PATTERNS)
    intent = "attribute"
    if not wants_vlm:
        for name in ("count", "list", "locate", "presence"):
            if any(re.search(p, text) for p in DETECTION_PATTERNS[name]):
                intent = name
                break

    if wants_vlm:
        if has_image:
            return RouteDecision("vlm", "attribute", "needs color/text/comparison/detail", targets)
        return RouteDecision(
            "no_vision", "attribute", "needs the image, but none was sent", targets
        )

    if unknown and not targets:
        # The detector only knows 80 everyday classes; a VLM may know a pen.
        if has_image:
            return RouteDecision(
                "vlm", intent, f"'{unknown}' is not something the detector knows", [], unknown
            )
        return RouteDecision(
            "no_vision", intent, f"'{unknown}' is not a detector class and no image was sent",
            [], unknown,
        )

    if detections is None:
        if has_image:  # API runs the detector first, so this is rare
            return RouteDecision("vlm", intent, "image but no detections", targets)
        return RouteDecision("no_vision", intent, "no image or detections were sent", targets)
    return RouteDecision("detections", intent, f"{intent} question", targets)


# ------------------------------------------------- detection-route answers ---

def select_objects(detections: list[Detection], targets: list[str]) -> list[Detection]:
    """Objects the question refers to: label matches first, most prominent
    first. With no target label, all objects ranked by prominence."""
    pool = [d for d in detections if not targets or d.label in targets]
    return sorted(pool, key=lambda d: d.prominence(), reverse=True)


def _level(confidences: list[float]) -> str:
    if not confidences:
        return "low"
    lowest = min(confidences)
    return "high" if lowest >= HIGH else "medium" if lowest >= CONFIDENT else "low"


@dataclass
class DetectionOutcome:
    answer: Answer | None
    escalate: bool = False  # ask the VLM instead
    reason: str = ""


def answer_from_detections(decision: RouteDecision, detections: list[Detection]) -> DetectionOutcome:
    """Answer presence/count/list/locate questions without any language model.

    Returns ``escalate=True`` when the detections are too uncertain to answer
    and the image should go to the VLM instead (the caller decides whether an
    image exists; if not it uses ``answer`` which then states the uncertainty).
    """
    sure = [d for d in detections if d.confidence >= CONFIDENT]
    unsure = [d for d in detections if d.confidence < CONFIDENT]

    if decision.intent == "list" or (decision.intent in {"count", "presence"} and not decision.target_labels):
        if decision.intent == "count" and not decision.target_labels:
            ranked = select_objects(sure, [])
            text = (
                f"I can make out {len(ranked)} object{'s' if len(ranked) != 1 else ''} clearly: "
                + _group_phrase(ranked) + "." if ranked else ""
            )
        else:
            ranked = select_objects(sure, [])
            text = f"I can see {_group_phrase(ranked)}." if ranked else ""
        maybe = [d for d in select_objects(unsure, []) if d.label not in {r.label for r in ranked}]
        if not ranked:
            uncertain = (
                f" I might see {_group_phrase(maybe)}, but I'm not sure." if maybe else ""
            )
            return DetectionOutcome(
                Answer(
                    "I can't make out any objects clearly right now." + uncertain,
                    "detections", True, [d.as_reference() for d in maybe], "low",
                    "nothing detected with confidence",
                ),
                escalate=True, reason="no confident detections",
            )
        if maybe:
            text += f" I might also see {_group_phrase(maybe)}, but I'm not sure about that."
        if any(d.label == "person" for d in ranked):
            text += " (I only notice that people are present; I don't recognize anyone.)"
        return DetectionOutcome(Answer(
            text, "detections", True, [d.as_reference() for d in ranked + maybe],
            _level([d.confidence for d in ranked]), "listed confident detections",
        ))

    label = decision.target_labels[0]
    name = natural_name(label)
    hits = select_objects([d for d in sure if d.label == label], [])
    maybes = select_objects([d for d in unsure if d.label == label], [])
    refs = [d.as_reference() for d in hits + maybes]
    other = (
        f" I'm only answering about the {name}; you also asked about "
        + join_words([natural_name(t) for t in decision.target_labels[1:]]) + "."
        if len(decision.target_labels) > 1 and decision.intent != "count" else ""
    )

    if not hits and maybes:
        text = (
            f"I'm not sure. I might see {count_phrase(name, 1)} {maybes[0].as_reference()['position']},"
            " but my detector isn't confident about it."
        )
        return DetectionOutcome(
            Answer(text + other, "detections", True, refs, "low", "only low-confidence matches"),
            escalate=True, reason="only low-confidence matches",
        )

    if decision.intent == "count":
        parts = []
        for target in decision.target_labels:
            target_hits = [d for d in sure if d.label == target]
            target_maybes = [d for d in unsure if d.label == target]
            target_name = natural_name(target)
            if target_hits:
                phrase = count_phrase(target_name, len(target_hits))
            else:
                phrase = f"no {plural(target_name, 2)}"
            if target_maybes:
                phrase += f" (and possibly {len(target_maybes)} more I'm unsure about)"
            parts.append(phrase)
        text = f"I count {join_words(parts)}."
        if label == "person":
            text += " That's presence only; I don't identify people."
        confidences = [d.confidence for d in sure if d.label in decision.target_labels]
        level = _level(confidences) if confidences else "medium"
        if any(d.label in decision.target_labels for d in unsure):
            level = "low" if level == "low" else "medium"
        all_refs = [
            d.as_reference() for d in select_objects(detections, decision.target_labels)
        ]
        return DetectionOutcome(Answer(text, "detections", True, all_refs, level, "counted detections"))

    if not hits:
        text = (
            f"I don't see {'anyone' if label == 'person' else count_phrase(name, 1)} right now. "
            "Small or partly hidden things can be missed, so try moving it closer to the camera."
        )
        return DetectionOutcome(Answer(text + other, "detections", True, [], "medium", "no matching detections"))

    best = hits[0]
    position = best.as_reference()["position"]
    if label == "person" and re.search(r"^who\b|\bwho('s| is| are)\b", decision.text):
        seen = f"one person {position}" if len(hits) == 1 else count_phrase("person", len(hits))
        text = f"I don't identify people; I can only tell that someone is there. I can see {seen}."
    elif label == "person":
        yes = "Yes, I" if decision.intent == "presence" else "I"
        if len(hits) == 1:
            text = f"{yes} can see one person {position}."
        else:
            text = f"{yes} can see {count_phrase('person', len(hits))}."
        text += " I can only tell that someone is there, not who it is."
    elif decision.intent == "locate":
        if len(hits) == 1:
            text = f"Your {name} is {position}." if "my" in _words(decision.text) else (
                f"The {name} is {position}."
            )
        else:
            text = f"I see {count_phrase(name, len(hits))}. The most prominent one is {position}."
    else:
        if len(hits) == 1:
            text = f"Yes, I can see {count_phrase(name, 1)} {position}."
        else:
            text = f"Yes, I can see {count_phrase(name, len(hits))}; the most prominent one is {position}."
    return DetectionOutcome(Answer(
        text + other, "detections", True, refs, _level([d.confidence for d in hits]),
        f"{decision.intent} from detections",
    ))


def _group_phrase(detections: list[Detection]) -> str:
    counts: dict[str, int] = {}
    for detection in detections:
        name = natural_name(detection.label)
        counts[name] = counts.get(name, 0) + 1
    parts = [count_phrase(name, n) for name, n in counts.items()]
    if len(parts) > 6:
        parts = parts[:6] + ["a few other things"]
    return join_words(parts)


def vision_summary(detections: list[Detection] | None) -> str | None:
    """Plain-text detector summary for the chat model's ``<vision>`` block.
    None when there are no detections at all (the prompt then says Jarvis
    cannot see)."""
    if detections is None:
        return None
    sure = select_objects([d for d in detections if d.confidence >= CONFIDENT], [])
    unsure = select_objects([d for d in detections if d.confidence < CONFIDENT], [])
    if not sure and not unsure:
        return (
            "A camera frame was captured, but the object detector found nothing in it. "
            "You know nothing else about the picture."
        )
    lines = ["The object detector found these things in the camera frame:"]
    for detection in sure[:12]:
        lines.append(f"- {natural_name(detection.label)}, {screen_position(detection)}")
    for detection in unsure[:6]:
        lines.append(f"- maybe {count_phrase(natural_name(detection.label), 1)} (not sure)")
    lines.append(
        "That list is all you know about the picture: you can't see colors, text, "
        "brands or details, and you never recognize who a person is."
    )
    return "\n".join(lines)


def vlm_failure_answer(
    decision: RouteDecision, detections: list[Detection], error: str
) -> Answer | None:
    """When the VLM fails but the detector saw something, say plainly that the
    close look failed and give only what the detector knows. Returns None when
    there is nothing to fall back on (the API then returns the error)."""
    sure = select_objects([d for d in detections if d.confidence >= CONFIDENT], [])
    if not sure:
        return None
    relevant = [d for d in sure if d.label in decision.target_labels] or sure
    text = (
        "I couldn't take a close look just now because the vision model failed, so I can't "
        f"answer that properly. My object detector only tells me I can see {_group_phrase(relevant)}."
    )
    return Answer(
        text, "vlm", True, [d.as_reference() for d in relevant], "low",
        "vision model failed; detector-only fallback", vision_warning=error,
    )


def honest_no_vision(decision: RouteDecision) -> Answer:
    if decision.unknown_target:
        text = (
            "I can't check that right now: no camera image came with your question, "
            f"and my object detector doesn't know what a {decision.unknown_target} is."
        )
    elif decision.intent == "attribute":
        text = (
            "I can't tell right now: no camera image came with your question, so I can't "
            "see colors, text or details. Try again with the camera on."
        )
    else:
        text = (
            "I can't see anything right now: no camera image or detections came with your "
            "question. Make sure the camera is on and try again."
        )
    return Answer(text, "no_vision", False, [], None, decision.reason)


# ------------------------------------------------------------------- VLM ---

def vlm_model() -> str:
    return os.getenv("VLM_MODEL", DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL


def vlm_settings() -> dict:
    def number(name: str, default: float) -> float:
        raw = os.getenv(name, "").strip()
        try:
            return float(raw) if raw else default
        except ValueError as error:
            raise VlmError(f"{name} must be a number.", 503) from error

    return {
        "keep_alive": os.getenv("VLM_KEEP_ALIVE", "0").strip() or "0",
        "timeout": number("VLM_TIMEOUT_SECONDS", 180.0),
        "max_edge": int(number("VLM_MAX_IMAGE_EDGE", 768)),
        "num_predict": int(number("VLM_NUM_PREDICT", 80)),
    }


def choose_vlm_focus(
    decision: RouteDecision, detections: list[Detection]
) -> tuple[list[Detection], str | None]:
    """Which detected object(s) the VLM question is about, plus a note to the
    user when the choice was ambiguous (several of the same object)."""
    if not decision.target_labels or not detections:
        return [], None
    sure = [d for d in detections if d.confidence >= CONFIDENT]
    matches = select_objects([d for d in sure if d.label in decision.target_labels], [])
    if not matches:
        return [], None
    comparison = len(decision.target_labels) > 1 or re.search(
        r"\b(than|which|compare|difference|same)\b", decision.text
    )
    if comparison:
        picked: list[Detection] = []
        for label in decision.target_labels:
            best = next((d for d in matches if d.label == label), None)
            if best:
                picked.append(best)
        if len(picked) == 1:  # "which cup is bigger": several of one label
            picked = matches[:2]
        return picked, None
    first = matches[0]
    same = [d for d in matches if d.label == first.label]
    note = None
    if len(same) > 1:
        name = natural_name(first.label)
        note = (
            f"I see {count_phrase(name, len(same))}, so I looked at the most prominent one, "
            f"{first.as_reference()['position']}."
        )
    return [first], note


def crop_for(image: Image.Image, focus: list[Detection]) -> Image.Image:
    """Crop to the object(s) with a margin so the small VLM sees detail.
    Large objects (over 60% of the frame) keep the whole image."""
    if not focus:
        return image
    left = min(d.x for d in focus)
    top = min(d.y for d in focus)
    right = max(d.x + d.width for d in focus)
    bottom = max(d.y + d.height for d in focus)
    if (right - left) * (bottom - top) > 0.6:
        return image
    pad_x = max(0.08, (right - left) * 0.25)
    pad_y = max(0.08, (bottom - top) * 0.25)
    box = (
        int(max(0.0, left - pad_x) * image.width),
        int(max(0.0, top - pad_y) * image.height),
        int(min(1.0, right + pad_x) * image.width),
        int(min(1.0, bottom + pad_y) * image.height),
    )
    if box[2] - box[0] < 16 or box[3] - box[1] < 16:
        return image
    return image.crop(box)


def encode_for_vlm(image: Image.Image, max_edge: int) -> str:
    image = image.convert("RGB")
    if max(image.size) > max_edge:
        scale = max_edge / max(image.size)
        image = image.resize(
            (max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.LANCZOS
        )
    buffer = io.BytesIO()
    image.save(buffer, "JPEG", quality=88)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _focus_phrase(focus: list[Detection]) -> str:
    """"the bottle", "the bottle and the cup", or "two laptops" (not "the
    laptop and laptop") when the focus holds several of one kind."""
    names = [natural_name(d.label) for d in focus]
    if len(set(names)) == len(names):
        return join_words([f"the {name}" for name in names])
    return _group_phrase(focus)


def build_vlm_prompt(question: str, focus: list[Detection], cropped: bool) -> str:
    """Short, plain prompt. Small VLMs follow long instructions poorly, so the
    detector context is one sentence and only about the object in question."""
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    if focus and cropped:
        return f"This picture shows {_focus_phrase(focus)}. {question}"
    return question


def retry_prompt(question: str, focus: list[Detection]) -> str:
    """Second try after an empty answer. moondream reliably answers
    "describe" requests even when it stops at once on a terse question."""
    subject = _focus_phrase(focus) if focus else "this image"
    question = question.strip()
    if not question.endswith("?"):
        question += "?"
    return f"Describe {subject} in one sentence, then answer: {question}"


async def ask_vlm(prompt: str, image_b64: str, settings: dict | None = None) -> str:
    """Ask the local VLM through Ollama. Raises ``VlmError`` with an honest,
    user-facing message on any failure (never a made-up answer)."""
    settings = settings or vlm_settings()
    model = vlm_model()
    base_url = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434").rstrip("/")
    payload = {
        "model": model,
        "stream": False,
        "keep_alive": settings["keep_alive"],
        "prompt": prompt,
        "images": [image_b64],
        "options": {"temperature": 0, "num_predict": settings["num_predict"], "num_ctx": 2048},
    }
    try:
        async with httpx.AsyncClient(timeout=settings["timeout"]) as client:
            result = await client.post(f"{base_url}/api/generate", json=payload)
            if result.status_code == 404:
                raise VlmError(
                    f"The vision model '{model}' is not installed. Run `ollama pull {model}`.", 503
                )
            result.raise_for_status()
            answer = (result.json().get("response") or "").strip()
    except VlmError:
        raise
    except httpx.ConnectError as error:
        raise VlmError(
            "Ollama is not running, so Jarvis can't look at the image. Start Ollama and try again.",
            503,
        ) from error
    except httpx.TimeoutException as error:
        raise VlmError(
            f"The vision model took longer than {settings['timeout']:.0f} s. The computer may be "
            "low on memory; close other apps and try again.",
            504,
        ) from error
    except httpx.HTTPStatusError as error:
        raise VlmError(
            f"The vision model failed (Ollama HTTP {error.response.status_code}). It may have run "
            "out of memory.",
            502,
        ) from error
    except ValueError as error:
        raise VlmError("The vision model returned an invalid response.", 502) from error
    return answer


def tidy_vlm_answer(answer: str, max_sentences: int = 3) -> str:
    """Trim a rambling VLM answer to a few sentences for speech."""
    answer = re.sub(r"\s+", " ", answer).strip()
    sentences = re.split(r"(?<=[.!?])\s+", answer)
    trimmed = " ".join(sentences[:max_sentences]).strip()
    if trimmed and trimmed[-1] not in ".!?":
        # Cut off by the token limit: drop the unfinished sentence if possible.
        if len(sentences) > 1:
            trimmed = " ".join(sentences[: min(max_sentences, len(sentences) - 1)])
        else:
            trimmed += "..."
    return trimmed
