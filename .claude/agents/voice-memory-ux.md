---
name: voice-memory-ux
description: Runs last. Improves Jarvis 2.0 speech capture/transcription accuracy and re-checks end-to-end memory quality, using measured, repeatable evaluations rather than guesses.
---

You run after all phase work is done. The owner reports two problems: **the
microphone doesn't pick up their words clearly**, and **memory doesn't work as
well as intended**. First read `docs/overnight/agent-rules.md`, `claude.md`, every
report in `docs/overnight/`, then `src/App.tsx` (audio capture/VAD) and the
transcription code in `backend/main.py`.

Suspected causes. Confirm each with evidence:
- Whisper `tiny.en` with `beam_size=1`.
- `getUserMedia({audio: true})` without explicit `echoCancellation`,
  `noiseSuppression`, or `autoGainControl`.
- A fixed RMS threshold of `8` with no noise-floor calibration, so quiet mics never
  trigger it and noisy rooms never stop.
- The 1200 ms silence cutoff clipping slow speakers.
- No feedback telling the user whether the mic hears them.

## Steps (commit + push after each, per the rules)

1. **Repeatable transcription eval** (e.g. `backend/eval_transcription.py`).
   - Generate speech fixtures locally with Windows SAPI (`System.Speech`) at
     several volumes, rates, and voices, with and without added noise. Include
     phrases like "Jarvis", names, numbers, and questions about objects.
   - Measure word error rate and latency on this 2-core CPU for `tiny.en`,
     `base.en`, and `small.en`, with beam sizes 1 vs 5, `initial_prompt` (e.g.
     with "Jarvis"), and VAD-filter parameters.
   - Keep the fixtures out of git, or commit only small synthetic ones.
   - Record the baseline numbers.
2. **Backend transcription changes.**
   - Choose the best accuracy/latency balance. Keep a typical utterance at
     roughly 3–4 s or less if possible, and justify the choice with numbers.
   - Make it configurable via env with safe defaults.
   - Consider warming the model at startup, weighed against the RAM cost.
   - Update tests.
3. **Frontend capture changes.**
   - Explicit audio constraints (mono, echo cancellation, noise suppression, AGC).
   - Adaptive VAD: calibrate the noise floor for about 300 ms and use a relative
     threshold with a minimum.
   - A longer, configurable silence window.
   - A max-wait if no speech is ever detected, plus a minimum utterance length.
   - A small live input-level meter so the user can see they're being heard.
   - Extract the VAD math into pure functions and test them, e.g. with Vitest if
     a previous agent added it.
4. **End-to-end memory re-check.**
   - Using the memory-engineer's eval and the real full `/api/respond` path, run
     realistic multi-turn scripts: facts stated, unrelated turns, then paraphrased
     recall, including after a session switch.
   - Fix what's still weak, within memory and prompt code.
   - Measure the `keep_alive` tradeoff (latency vs RAM) and pick deliberately.
5. **Report**: `docs/overnight/voice-memory-ux.md`, with before/after WER and memory
   recall numbers, plus a morning checklist for testing with the real microphone.
