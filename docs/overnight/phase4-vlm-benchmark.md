# Phase 4 VLM benchmark (moondream vs granite3.2-vision)

Date: 2026-11-17 (overnight, unattended). Machine: AMD Athlon Gold 3150U
(2 cores), 5.9 GB RAM, no GPU, Windows 11, Ollama 0.40.13.

Script: `backend/bench_vlm.py` (reproducible; run it yourself with Ollama up).

## Short answer

- **granite3.2-vision does not fit on this machine.** Every request timed out
  (900 s twice; see the retry below). Ollama loaded 2.0 GB of weights, but with
  only ~100-250 MB of RAM free Windows kept paging the model out.
- **moondream loads and runs, but answers almost nothing correctly here.**
  3/12 at best. Most targeted questions ("What color is the circle?") came back
  **empty** (the model emits one token and stops), and the Ollama runner
  **crashed** on 2 of 12 requests in each run (`tokenize: EOF`). Latency was
  1.5-11 minutes per question while the machine was swapping.
- **Default: `VLM_MODEL=moondream`** (Apache-2.0), because it is the only
  candidate that runs at all. It is **not good enough to rely on**, so the
  backend treats it as a best-effort extra: the router answers presence,
  counting and "what do you see" questions from the object detector without
  any VLM, and when the VLM fails or returns nothing Jarvis says so plainly.
- Configure with **`VLM_MODEL`** (not `VISION_MODEL`: that name was already
  taken by the object detector, `yolox_tiny`). See "Recommendation".

## Licenses (checked with `ollama show <model> --license`)

| Model | Ollama tag | Size on disk | License | OK under our rules? |
|---|---|---|---|---|
| moondream (moondream2, phi-1.5 base + SigLIP) | `moondream:latest` (55fc3abd3867) | 1.7 GB | Apache-2.0 | Yes |
| IBM granite3.2-vision 2B | `granite3.2-vision:latest` (3be41a661804) | 2.4 GB | Apache-2.0 | Yes |

No 3B+ model was tried: free RAM was ~100-250 MB before loading anything (the
owner's VS Code windows, Chrome and Claude use the rest; Windows "Memory
Compression" held 2-3 GB). Even the 2B model did not fit, so a 3B model would
not either.

## Method

**Images** (no camera frames; nothing is written to disk, all images are made in memory):

| Image | Source / license |
|---|---|
| `office_public_domain.jpg` (960x742) | FEMA photo, public domain (`backend/test_data/README.md`) |
| bottle crop of the office photo (240x370) | derived from the above |
| blue circle, "ACME COFFEE" sign, red octagon "STOP" sign, 3 red squares, 5 blue circles, big green vs small yellow rectangle (640x480) | drawn by the script with Pillow (no third-party content) |

**Questions and expected answers** (written in `CASES` before any model ran;
pass = an expected keyword appears as a whole word and no "reject" keyword does):

| id | category | question | expected |
|---|---|---|---|
| color-circle | color | What color is the circle? | blue |
| text-sign | text/brand | What does the sign say? | ACME |
| text-stop | text/brand | What word is written on the sign? | STOP |
| count-3 | counting | How many red squares are there? | 3 |
| count-5 | counting | How many blue circles are in the image? | 5 |
| compare-size | comparison | Which rectangle is bigger, the green one or the yellow one? | green |
| office-what | what is this | What is happening in this picture? | office/meeting/table/laptops/working |
| office-count | counting | How many people are in this picture? | 3 |
| office-shirt | color | What color is the shirt of the man with glasses on the right? | blue/navy |
| office-folder | color | What color is the folder in the bottom left corner? | red |
| office-brand | text/brand | What brand is the black laptop on the left? | HP |
| bottle-what | what is this | What is this object? | bottle/water |

**Timing and memory.** Before each model, everything is unloaded. The first
question is the **cold** call (includes loading the model); the rest are
**warm** (model kept with `keep_alive: 5m`, then unloaded at the end). Ollama's
own `load_duration` / `prompt_eval_duration` / `eval_duration` are recorded. A
thread samples the working set of all `ollama*` processes and the system's
available RAM every 0.5 s. Temperature 0, `num_predict` 80.

Runs: (1) `/api/chat`, `num_ctx` 4096, plain question; (2) `/api/generate`
(the call the backend uses), `num_ctx` 2048, "describe" style prompt
(`Describe this image in one sentence, then answer: <question>`), because in
manual probes moondream reliably answered "Describe this image." while
returning nothing for short questions.

## Results

### moondream

| Run | Correct | Cold | Warm median | Warm max | Runner crashes | Peak Ollama working set | Min free RAM |
|---|---|---|---|---|---|---|---|
| plain question (`/api/chat`) | **2/12** | 129 s | 115 s | 546 s | 2 | 638 MB | 3 MB |
| describe style (`/api/generate`) | **3/12** | 145 s | 116 s | 680 s | 2 | 694 MB | 9 MB |

Raw answers (plain / describe):

| id | plain question | describe style |
|---|---|---|
| color-circle | *(empty)* | "urn" |
| text-sign | `?".` | runner crash |
| text-stop | *(empty)* | "!!!STOP!!!" (pass) |
| count-3 | runner crash | runner crash |
| count-5 | *(empty)* | *(empty)* |
| compare-size | *(empty)* | *(empty)* |
| office-what | "a group of people sitting around a table, working on their laptops. There are four people..." (pass) | "urns of water and a laptop are on a table, with a man in a blue shirt working on the laptop." (pass) |
| office-count | *(empty)* | *(empty)* |
| office-shirt | *(empty)* | *(empty)* |
| office-folder | runner crash | "urn" |
| office-brand | "iphone" | "iphone" |
| bottle-what | "a clear plastic water bottle with a white cap." (pass) | "urn of water" (pass) |

Manual probes (same model, not scored): "Describe this image." on the office
photo gave a good paragraph three times (blue shirt, glasses, water bottle,
whiteboard), in 18-19 s when the model was already in RAM and 99 s when it had
to page back in. "What color is the circle?", "Describe the color of the
circle." and "What is the color of the shape?" all returned an empty string
(one generated token).

**Where the time goes.** moondream turns every image into ~740 tokens. When
the model was actually in RAM, encoding them took **3-8 s** and a short answer
1-4 s (e.g. office-brand: 9 s total). When Windows had paged it out, the same
step took **75-110 s**, and reloads after a runner crash took up to 560 s. The
machine was swapping the whole time (free RAM fell to 3-9 MB).

**Empty answers are not just memory pressure**: they also happened on warm,
fast calls (office-shirt: 8.4 s, empty). It looks like moondream on this
Ollama build stops right away on most short questions. I couldn't read the
Ollama server log (it was empty), so the crash cause isn't confirmed. Out of
memory is likely, since commit space was fine (5.6 GB free) but physical RAM
was not.

### granite3.2-vision

| Attempt | Result |
|---|---|
| Run 1, color-circle (cold) | timed out after 900 s |
| Run 1, text-sign | timed out after 900 s (stopped the run here) |
| Retry (`/api/generate`, `num_ctx` 2048), office-shirt | HTTP 400: "request (**5167 tokens**) exceeds the available context size (2048 tokens)" |
| Retry, bottle crop | runner crashed (`tokenize: EOF`) |

`ollama ps` during run 1 showed `granite3.2-vision 2.0 GB 100% CPU, context
4096`, but the runner process wasn't among the top memory users, so its pages
were mostly swapped out. With ~200 MB free, a 2 GB model can't run at a usable
speed on this laptop.

**granite retry:** granite splits a 960x742 photo into several tiles and
needs **5,167 tokens** for it, about 7x moondream's ~740. That means a context
of at least ~6k tokens (more KV-cache RAM) and roughly 7x the image-encoding
work on 2 cores. The small bottle crop fit in 2,048 tokens, but the runner
crashed while loading. Granite answered **0 of 4** attempts.

## Recommendation

1. **Default `VLM_MODEL=moondream`, `VLM_KEEP_ALIVE=0`.** It's the only candidate
   that runs at all, and it unloads after every answer so it never sits next to
   qwen in RAM. Expect the VLM route to be **slow (minutes) and unreliable** on
   this laptop until there's more free RAM. Closing VS Code/Chrome before a demo
   should bring a call down to the ~10-20 s seen when the model stayed in RAM.
2. Don't make anything depend on the VLM. The router answers presence,
   counting, location and "what do you see" questions from YOLOX in
   milliseconds, and falls back to "the vision model failed; my detector sees
   ..." when the VLM breaks.
3. Worth trying later (not done tonight; each needs RAM headroom or new
   downloads): a newer moondream2 build (the Ollama `moondream` tag is the 2024
   model), `granite3.2-vision` on a machine with 8+ GB free, and SmolVLM-256M or
   500M (Apache-2.0, small enough for this laptop, but not in the Ollama
   library). Re-run `backend/bench_vlm.py --models <tag>` to compare.
