
># 🧪 QAForge — AI Test Case Generator

> **QAForge** is a web application that automatically generates professional, structured QA test cases from a user story or feature description, using AI (Gemini or OpenAI).
---

## ✨ Features

- **5 LLM providers**: Gemini, OpenAI, Groq, Mistral, OpenRouter
- **Multi-file upload**: PDF (text + images), DOCX (paragraphs + tables + images), TXT, MD, direct images (max 5 files)
- **Smart extraction**: PDF via PyMuPDF (positional order), DOCX with Markdown tables, `[IMAGE_N — file]` markers
- **3 structured phases** with locked navigation and progress tracking
- **Export**: Markdown · JSON · CSV
- **Temperature slider**: 0.0 → 1.0 (default 0.2)

---

## 🗺️ The 3 Phases

### Phase 1 — Analysis & Questions
The AI analyzes the User Story and:
1. Identifies **applicable ISO 29119-4 techniques** (GHL Step 1): BVA, Equivalence Partitioning, Decision Table, State Transition, Error Guessing, Exploratory Testing, Function Combinations
2. Generates **typed clarifying questions** (boolean / multiple_choice / text) grouped by category (Functional, Validation, Error Handling, Edge Cases, System Dependencies)
3. Displays an analysis panel: business rules, actors, identified screens, ISO techniques

### Phase 2 — Test Plan
The AI generates a test plan per ISO technique (GHL-F method):
- Each scenario is prefixed by its technique: `BVA —`, `DT —`, `ST —`, `EP —`, `FC —`, `EG —`
- Per-scenario validation / rejection / priority adjustment
- Batch generation (6 scenarios per batch) to avoid timeouts

### Phase 3 — Full Test Cases
The AI writes test cases with:
- **Technique** field (ISO 29119-4 traceability)
- Real test data in steps
- Expected result in natural language (optimized for cosine semantic similarity ≥ 0.7)
- MD / JSON / CSV export

---

## 🚀 Installation

### Requirements

```bash
pip install streamlit google-generativeai openai pymupdf python-docx Pillow pypdf
```

### Run the app

```bash
streamlit run app_v04.py
```

---

## ⚙️ Configuration

In the **sidebar**:

| Parameter | Description |
|---|---|
| **LLM Provider** | Gemini · OpenAI · Groq · Mistral · OpenRouter |
| **API Key** | API key for the selected provider |
| **Model** | Exact model ID (e.g. `gemini-2.0-flash`, `gpt-4o-mini`) |
| **🌡️ Temperature** | `0.0` = reproducible (GHL study) · `0.2` = balanced default · `>0.5` = creative but less stable JSON |

### Recommended models

| Provider | Recommended model | Notes |
|---|---|---|
| Gemini | `gemini-2.0-flash` | Best quality/speed ratio, native vision |
| OpenAI | `gpt-4o-mini` | Cost-efficient, reliable structured JSON |
| Groq | `llama-3.3-70b-versatile` | Very fast, free tier available |
| Mistral | `mistral-small-latest` | Good European alternative |
| OpenRouter | `meta-llama/llama-3.3-70b-instruct:free` | Free |

---

## 🔬 GHL Methodology (ISO/IEC/IEEE 29119-4)

QAForge implements the **GHL (Generating High-Level test cases)** method described in:

> Masuda et al., *"Generating High-Level Test Cases from Requirements using LLM: An Industry Study"*, arXiv:2510.03641, 2025.

### Why GHL?

| Method | Recall (Bluetooth) | Recall (Mozilla) |
|---|---|---|
| Zero-shot (baseline) | 0.65 | 0.02 |
| GHL (Step 1 + Step 2) | 0.69 | 0.20 |
| **GHL-F (+ Function Combinations)** | **0.84** | **0.37** |

### The 2 key steps

**Step 1** — The AI identifies applicable ISO 29119-4 test design techniques from the requirements *before* generating anything.

**Step 2** — For each identified technique, the AI generates test cases specific to that technique, ensuring exhaustive coverage.

### Supported techniques

| Prefix | Technique | When to use |
|---|---|---|
| `BVA` | Boundary Value Analysis | Numeric fields, ranges, limits |
| `DT` | Decision Table Testing | Multi-condition logic (IF x AND y THEN z) |
| `ST` | State Transition Testing | Lifecycles, statuses (draft/active/archived) |
| `EP` | Equivalence Partitioning | Valid/invalid input classes |
| `FC` | Function Combinations (GHL-F) | Interactions between features/modules |
| `EG` | Error Guessing | Likely failure points |
| _(none)_ | Happy Path / Alternate Flow | Nominal user journeys |

---

## 📁 Supported file formats

| Format | Text extraction | Image extraction | Notes |
|---|---|---|---|
| PDF | ✅ PyMuPDF | ✅ (positional order) | Fallback to pypdf if PyMuPDF unavailable · Scanned pages rasterized |
| DOCX | ✅ paragraphs + tables | ✅ inline images | Tables converted to Markdown |
| TXT / MD | ✅ | ❌ | Plain text |
| PNG / JPG / WEBP | — | ✅ direct | Visual analysis (Gemini / OpenAI only) |

> **Limits**: 5 files max · Images smaller than 50×50px filtered out (decorative icons ignored)

---

## 📤 Exports

| Format | Content |
|---|---|
| **Markdown** | Formatted test cases with tables, numbered steps, expected results |
| **JSON** | Structured array: `id`, `title`, `type`, `technique`, `priority`, `automation`, `preconditions`, `steps`, `expected_result`, `failure_signature` |
| **CSV** | Excel / Google Sheets compatible |

---

## 🏗️ Architecture

```
app_v04.py
├── LLM Adapters          call_gemini / call_openai / call_llm (unified router)
│                         call_llm_structured (native Gemini JSON + fallback)
├── File Parsers          pdf_smart_extract / docx_smart_extract / extract_text_plain
├── Generation helpers    generate_until_complete (anti-truncation [[GENERATION_COMPLETE]])
│                         generate_test_cases_in_batches (batches of 6)
├── Prompts               PROMPT_P1_QUESTIONS · PROMPT_P1_CHAT
│                         PROMPT_P2 · PROMPT_P3_MARKDOWN · PROMPT_P3_JSON
└── UI                    render_tab_bar / Phase 1 / Phase 2 / Phase 3
```

### Anti-truncation mechanism

For long generations, QAForge uses a `[[GENERATION_COMPLETE]]` signal:
- The AI emits this token when all test cases have been generated
- If the token is absent, a "Continue..." iteration is triggered automatically (max 2 iterations)
- An **"Auto-complete"** button is available if generation remains incomplete

---

## 📝 Changelog

### v0.4 (2026-03-31)
- **GHL prompts**: `PROMPT_P1_QUESTIONS` identifies ISO 29119-4 techniques before asking questions (GHL Step 1)
- **GHL-F prompts**: `PROMPT_P2` generates scenarios per technique with prefixes (`BVA —`, `DT —`, `ST —`, `EP —`, `FC —`, `EG —`)
- **Technique field** in `PROMPT_P3_MARKDOWN` for ISO traceability and semantic similarity validation
- **`iso_ctx`**: techniques extracted in Phase 1 are automatically injected into Phase 2 context
- **🌡️ Temperature slider** in sidebar (0.0 → 1.0, default 0.2) propagated to all providers
- **ISO panel**: "🔬 ISO 29119-4 techniques identified" expander in Phase 1

### v0.3
- Multi-provider support: Groq, Mistral, OpenRouter added
- Improved PDF extraction with positional order (PyMuPDF)
- `generate_until_complete` mechanism + `[[GENERATION_COMPLETE]]` signal
- Batch generation of 6 scenarios per call

### v0.2
- PDF / DOCX support with image extraction
- JSON + CSV export
- Phase navigation with progressive locking

---

## ⚠️ Known limitations

- **Groq, Mistral, OpenRouter** providers do not support image analysis — `[IMAGE_N]` markers remain in the text but are not visually analyzed
- **Temperature** max is `1.0` (Groq and Mistral do not support values above 1.0)
- Files larger than 15,000 characters after extraction are automatically truncated
- `localStorage` is disabled in Streamlit Cloud iframes — session is lost on page reload

---

