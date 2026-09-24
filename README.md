## Project Changelog

**10-09-2026**

* [x] **Input chunking** — Split large `.tel` input files into manageable chunks and process them until EOF.
* [ ] **Flexible `max_new_tokens`** — Replace the currently hard-coded value with a configurable or dynamically determined value.
* [ ] **Model loading** — Avoid loading the model weights from the beginning for every execution.

**24-09-2026**

* [x] **Prompt engineering** — Improve the precision of the input prompts text for better guiding the model's responses.
* [x] **Chunk processing** — Generate input prompts and output triple for each chunk
* [x] **Prompt formatting** — Formatted prompts using Qwen’s chat template before text generation. Added the user role and generation prompt marker to improve instruction following and JSON output reliability.
* [x] **Model configuration** — Added configuration for generation prompt (i.e max_new_tokens) to adapt lengthy output results, trade off with more processing time.