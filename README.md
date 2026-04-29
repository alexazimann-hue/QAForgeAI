
># 🧪 QAForge — AI Test Case Generator
**Version 0.5** 
> **QAForge** is a web application that automatically generates professional, structured QA test cases from a user story or feature description, using AI.
---

## ✨ Features

- **5 LLM providers**: Gemini, OpenAI, Groq, Mistral, OpenRouter
- **Multi-file upload**: PDF (text + images), DOCX (paragraphs + tables + images), TXT, MD, direct images (max 5 files)
- **Smart extraction**: PDF via PyMuPDF (positional order), DOCX with Markdown tables, `[IMAGE_N — file]` markers
- **Image resizing**: images auto-resized to max 1024px before API call (reduces cost and latency)
- **3 structured phases** with locked navigation and progress tracking
- **Export**: Markdown · JSON · CSV
- **Temperature slider**: 0.0 → 1.0 (default 0.2)
- **Auto-retry**: automatic retry with backoff on 429 / rate-limit errors
- **Multilingual**: output language matches the language of the User Story
- **Tooltips on every phase button** — technique glossary and phase guidance built in

---

## 🗺️ The 3 Phases

### Phase 1 — Analysis & Clarification
The AI analyzes the User Story and:
1. Identifies **applicable ISO 29119-4 test design techniques** before generating anything
2. Generates **typed clarifying questions** (boolean / multiple_choice / text) grouped by category (Functional, Validation, Error Handling, Edge Cases, System Dependencies)
3. Displays an analysis panel: business rules, actors, identified screens, ISO techniques with rationale

### Phase 2 — Test Checklist
The AI generates a test checklist per ISO 29119-4 technique:
- Each scenario is prefixed by its technique: `BVA —`, `DT —`, `ST —`, `EP —`, `FC —`, `EG —`, `ET —`
- Per-scenario validation ✅ / rejection ❌ / priority adjustment
- Batch generation (6 scenarios per batch) to avoid timeouts

### Phase 3 — Full Test Cases & Export
The AI writes execution-ready test cases with:
- **Technique** field (ISO 29119-4 traceability)
- Real test data in steps
- Expected result in natural language
- MD / JSON / CSV export

> ⚠️ ISO 29119-4 coverage techniques favour exhaustiveness — a human review pass to remove duplicates is normal and expected.

---

## 🚀 Installation

### Requirements

```bash
pip install -r requirements.txt
```

Or manually:

```bash
pip install streamlit google-genai openai pymupdf python-docx Pillow pypdf
```

### Run the app

```bash
streamlit run app.py
```

---

## ⚙️ Configuration

In the **sidebar**:

| Parameter | Description |
|---|---|
| **LLM Provider** | Gemini · OpenAI · Groq · Mistral · OpenRouter |
| **API Key** | API key for the selected provider |
| **Model** | Exact model ID (e.g. `gemini-2.0-flash`, `gpt-4o-mini`) |
| **🌡️ Temperature** | `0.0` = reproducible (ISO 29119-4) · `0.2` = balanced default · `>0.5` = creative but less stable JSON |

### Recommended models

| Provider | Recommended model | Notes |
|---|---|---|
| Gemini | `gemini-2.0-flash` | Best quality/speed ratio, native vision |
| OpenAI | `gpt-4o-mini` | Cost-efficient, reliable structured JSON |
| Groq | `llama-3.3-70b-versatile` | Very fast, free tier available |
| Mistral | `mistral-small-latest` | Good European alternative |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | Free |

---

## 🔬 ISO/IEC/IEEE 29119-4 Techniques

QAForge applies **ISO/IEC/IEEE 29119-4 test design techniques** systematically across all 3 phases.

### Technique identification (Phase 1)

Before asking any clarifying question, the AI identifies which techniques apply to the requirements. This ensures test coverage is grounded in the actual feature characteristics — not generated arbitrarily.

### Supported techniques

| Prefix | Technique | When applied |
|---|---|---|
| `BVA` | Boundary Value Analysis | Numeric fields, ranges, limits |
| `DT` | Decision Table Testing | Multi-condition logic (IF x AND y THEN z) |
| `ST` | State Transition Testing | Lifecycles, statuses (draft/active/archived) |
| `EP` | Equivalence Partitioning | Valid/invalid input classes |
| `FC` | Function Combinations | Interactions between features/modules |
| `EG` | Error Guessing | Likely failure points from experience |
| `ET` | Exploratory Testing | Unexpected user paths, free-form exploration |
| _(none)_ | Happy Path / Alternate Flow | Nominal user journeys |

### Coverage vs. precision

These techniques are designed to maximise **test coverage (recall)**. This means the generated test suite will be exhaustive but may include overlapping scenarios. A human review pass to consolidate duplicates before execution is recommended.

---

## 📁 Supported file formats

| Format | Text extraction | Image extraction | Notes |
|---|---|---|---|
| PDF | ✅ PyMuPDF | ✅ (positional order) | Fallback to pypdf if PyMuPDF unavailable · Scanned pages rasterized |
| DOCX | ✅ paragraphs + tables | ✅ inline images | Tables converted to Markdown |
| TXT / MD | ✅ | ❌ | Plain text |
| PNG / JPG / WEBP | — | ✅ direct | Visual analysis (Gemini / OpenAI only) |

> **Limits**: 5 files max · Max 80,000 chars per file · Images smaller than 50×50px filtered out · Images resized to max 1024px before API call

---

## 📤 Exports

| Format | Content |
|---|---|
| **Markdown** | Formatted test cases with tables, numbered steps, expected results |
| **JSON** | Structured array: `id`, `title`, `technique`, `type`, `priority`, `automation`, `preconditions`, `steps`, `expected_result`, `failure_signature` |
| **CSV** | Excel / Google Sheets / Jira compatible |

---

## 🏗️ Architecture

```
app.py
├── Image utils           resize_image (max 1024px before API)
├── LLM Adapters          _gemini_client / _openai_client (module-level cached)
│                         call_gemini / call_openai / call_llm (unified router)
│                         call_llm_structured (native Gemini JSON + fallback)
│                         _retry (auto-retry on 429 / rate-limit errors)
├── File Parsers          pdf_smart_extract / docx_smart_extract / extract_text_plain
├── Generation helpers    generate_until_complete (anti-truncation [[GENERATION_COMPLETE]])
│                         generate_test_cases_in_batches (batches of 6)
├── HELP_TEXTS            Centralised tooltip strings — edit without touching UI logic
├── Prompts               PROMPT_P1_QUESTIONS · PROMPT_P1_CHAT
│                         PROMPT_P2 · PROMPT_P3_MARKDOWN · PROMPT_P3_JSON
└── UI                    render_tab_bar / Phase 1 / Phase 2 / Phase 3
```

### Anti-truncation mechanism

For long generations, QAForge uses a `[[GENERATION_COMPLETE]]` signal:
- The AI emits this token when all test cases have been generated
- If the token is absent, a "Continue..." iteration is triggered automatically (max 2 iterations)
- An **"Auto-complete"** button is available if generation remains incomplete

### Centralised help texts

All user-visible tooltip strings are stored in the `HELP_TEXTS` dictionary, placed just before `# ── FILE PARSING`. To update any tooltip, edit only this block — no UI logic is touched.

---

## ⚠️ Known limitations

- **Groq, Mistral, OpenRouter** do not support image analysis — `[IMAGE_N]` markers remain in text but are not visually processed
- **Mistral Small** context window (~32k tokens) limits effective document size to ~25,000 chars
- `localStorage` is disabled in Streamlit Cloud iframes — session is lost on page reload

---
