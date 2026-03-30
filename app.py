import streamlit as st
import time
from PIL import Image
import io
import docx
import json
import csv

# ── Smart extraction — requires: pip install pymupdf python-docx Pillow
# PyMuPDF is imported lazily so the app still starts even without it

#Hide Streamlit logo - (Streamlit-specific)
import streamlit.components.v1 as components

components.html("""<script>
function h(){try{var d=window.parent.parent.document;
['[class*="profileContainer"]','[class*="viewerBadge"]'].forEach(s=>
d.querySelectorAll(s).forEach(e=>e.style.setProperty('display','none','important')));
}catch(e){}}
h();[500,1500,3000].forEach(t=>setTimeout(h,t));
try{new MutationObserver(h).observe(window.parent.parent.document.body,{childList:true,subtree:true});}catch(e){}
</script>""", height=0)


# ── LLM ADAPTERS ──────────────────────────────────────────────────────────────

def call_gemini(history, system_prompt, user_message, images=None, max_tokens=3000):
    from google import genai
    from google.genai import types

    @st.cache_resource
    def get_gemini_client(key):
        return genai.Client(api_key=key)

    client = get_gemini_client(st.session_state.api_key)
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    parts = [types.Part(text=user_message)]
    for img in (images or []):
        buf = io.BytesIO(); img.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    contents.append(types.Content(role="user", parts=parts))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.2,
    )
    result = client.models.generate_content(
        model=st.session_state.model_choice.strip(), contents=contents, config=config
    )
    if not result or not result.text or not result.text.strip():
        raise Exception("Empty response from Gemini.")
    return result.text

def call_openai(history, system_prompt, user_message, images=None, max_tokens=3000, base_url=None):
    from openai import OpenAI

    @st.cache_resource
    def get_openai_client(key, url, provider):
        if provider == "OpenRouter":
            return OpenAI(api_key=key, base_url=url,
                default_headers={
                    "HTTP-Referer": "https://testcasegenerator-draft.streamlit.app",
                    "X-Title": "QAForge"
                })
        return OpenAI(api_key=key, base_url=url) if url else OpenAI(api_key=key)

    client = get_openai_client(st.session_state.api_key, base_url, st.session_state.get("provider", "OpenAI"))
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    # Build user content (text + images)
    if images:
        content = [{"type": "text", "text": user_message}]
        for img in images:
            buf = io.BytesIO(); img.save(buf, format="PNG")
            import base64
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_message})

    result = client.chat.completions.create(
        model=st.session_state.model_choice.strip(),
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
    )
    text = result.choices[0].message.content
    if not text or not text.strip():
        raise Exception("Empty response from OpenAI.")
    return text

def call_llm(history, system_prompt, user_message, images=None, max_tokens=3000):
    """Unified entry point — routes to the right provider."""
    provider = st.session_state.provider
    if provider == "Gemini":
        return call_gemini(history, system_prompt, user_message, images, max_tokens)
    elif provider in ("Groq", "Mistral", "OpenRouter"):
        # These providers don't support image input via API
        if images:
            st.warning(
                f"⚠️ **{provider}** does not support image input via API. "
                "Images from documents will be described by their markers in the text only. "
                "Switch to **Gemini** or **OpenAI** for full visual analysis.",
                icon="🖼️"
            )
        base_url = PROVIDER_DEFAULTS[provider]["base_url"]
        return call_openai(history, system_prompt, user_message, None, max_tokens, base_url)
    else:  # OpenAI
        return call_openai(history, system_prompt, user_message, images, max_tokens)

def call_llm_structured(system_prompt, user_message, max_tokens=8000):
    """Structured JSON output — uses native mode per provider, with fallback."""
    provider = st.session_state.provider

    if provider == "Gemini":
        try:
            from google import genai
            from google.genai import types
            import typing_extensions as typing

            class TestStep(typing.TypedDict):
                step_number: int
                action: str

            class TestCase(typing.TypedDict):
                id: str
                title: str
                type: str
                priority: str
                automation: str
                preconditions: list[str]
                steps: list[TestStep]
                expected_result: str
                failure_signature: str

            class TestCaseList(typing.TypedDict):
                test_cases: list[TestCase]

            @st.cache_resource
            def get_gemini_client(key):
                return genai.Client(api_key=key)

            client = get_gemini_client(st.session_state.api_key)
            contents = [types.Content(role="user", parts=[types.Part(text=user_message)])]
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=max_tokens,
                temperature=0.2,
                response_mime_type="application/json",
                response_schema=TestCaseList,
            )
            result = client.models.generate_content(
                model=st.session_state.model_choice.strip(), contents=contents, config=config
            )
            return json.loads(result.text).get("test_cases", [])
        except Exception:
            pass  # fall through to manual parsing

    elif provider == "OpenAI":
        try:
            from openai import OpenAI

            @st.cache_resource
            def get_openai_client(key):
                return OpenAI(api_key=key)

            client = get_openai_client(st.session_state.api_key)
            result = client.chat.completions.create(
                model=st.session_state.model_choice.strip(),
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                max_tokens=max_tokens,
                temperature=0.2,
                response_format={"type": "json_object"},
            )
            parsed = json.loads(result.choices[0].message.content)
            if isinstance(parsed, list):
                return parsed
            return parsed.get("test_cases", [])
        except Exception:
            pass  # fall through to manual parsing

    # Universal fallback: ask for raw JSON, parse manually
    fallback_msg = (
        user_message +
        """\n\nOutput ONLY a valid JSON array, no markdown, no explanation. Each item:
{"id":"TC-1","title":"...","type":"...","priority":"...","automation":"...",
"preconditions":["..."],"steps":[{"step_number":1,"action":"..."}],
"expected_result":"...","failure_signature":"..."}"""
    )
    raw = call_llm([], system_prompt, fallback_msg, max_tokens=max_tokens)
    raw = raw.strip()
    if raw.startswith("```"):
        raw = raw.split("\n", 1)[-1].rsplit("```", 1)[0]
    parsed = json.loads(raw)
    if isinstance(parsed, list):
        return parsed
    return parsed.get("test_cases", [])


def generate_test_cases_in_batches(system_prompt, plan_ctx, scenario_titles, batch_size=6):
    """Split scenario list into batches, generate TCs per batch, concatenate results."""
    batches = [scenario_titles[i:i+batch_size] for i in range(0, len(scenario_titles), batch_size)]
    all_markdown = []
    total = len(batches)

    progress = st.progress(0, text=f"Generating test cases… batch 1/{total}")

    for idx, batch in enumerate(batches):
        batch_list = "\n".join(f"- {t}" for t in batch)
        batch_prompt = (
            f"{plan_ctx}\n\n"
            f"Generate DETAILED test cases ONLY for these {len(batch)} scenarios (batch {idx+1}/{total}):\n"
            f"{batch_list}\n\n"
            f"Number them starting from TC-{idx * batch_size + 1}."
        )
        # Markdown
        md, _ = generate_until_complete(system_prompt, [], batch_prompt, max_iterations=2, max_tokens=6000)
        all_markdown.append(md)

        progress.progress((idx + 1) / total,
                          text=f"Generating test cases… batch {idx+2}/{total}" if idx+1 < total else "✅ Done!")

    progress.empty()
    return "\n\n---\n\n".join(all_markdown), []


def extract_scenario_titles(plan_text):
    """Extract TC titles from Phase 2 plan (lines starting with '- TC:')."""
    import re
    titles = re.findall(r"-\s*TC:\s*(.+)", plan_text)
    if not titles:
        # Fallback: any bullet line
        titles = re.findall(r"^\s*[-•]\s*(.+)", plan_text, re.MULTILINE)
    return [t.strip() for t in titles if t.strip()]


COMPLETION_SIGNAL = "[[GENERATION_COMPLETE]]"

def generate_until_complete(system_prompt, history, initial_prompt, max_iterations=2, max_tokens=8000):
    """
    Call LLM in a loop until it emits COMPLETION_SIGNAL or max_iterations is reached.
    Returns the full concatenated markdown (clean, without the signal token).
    """
    messages = list(history)
    full_parts = []

    for i in range(max_iterations):
        if i == 0:
            user_msg = initial_prompt + (
                "\n\n---\nIMPORTANT: When you have finished generating ALL test cases, "
                f"end your response with the exact token: {COMPLETION_SIGNAL}"
            )
        else:
            user_msg = (
                "Continue EXACTLY where you stopped. Generate the remaining test cases. "
                f"When ALL test cases are done, end with: {COMPLETION_SIGNAL}"
            )

        response = call_llm(messages, system_prompt, user_msg, max_tokens=max_tokens)
        full_parts.append(response.replace(COMPLETION_SIGNAL, "").rstrip())
        messages.append({"role": "user", "content": user_msg})
        messages.append({"role": "assistant", "content": response})

        if COMPLETION_SIGNAL in response:
            break
        if i < max_iterations - 1:
            time.sleep(2)  # Avoid RPM rate limit between iterations

    return "\n\n".join(p for p in full_parts if p.strip()), messages

# ── PROVIDER DEFAULTS ─────────────────────────────────────────────────────────
PROVIDER_DEFAULTS = {
    "Gemini": {
        "placeholder": "gemini-2.0-flash",
        "examples": "`gemini-2.0-flash` · `gemini-2.5-flash-lite-preview-06-17` · `gemini-2.5-pro`",
        "docs": "https://ai.google.dev/gemini-api/docs/models",
        "base_url": None,
    },
    "OpenAI": {
        "placeholder": "gpt-4o-mini",
        "examples": "`gpt-4o-mini` · `gpt-4o` · `gpt-4-turbo` · `gpt-3.5-turbo`",
        "docs": "https://platform.openai.com/docs/models",
        "base_url": None,
    },
    "Groq": {
        "placeholder": "llama-3.3-70b-versatile",
        "examples": "`llama-3.3-70b-versatile` · `llama-3.1-8b-instant` · `mixtral-8x7b-32768`",
        "docs": "https://console.groq.com/keys",
        "base_url": "https://api.groq.com/openai/v1",
    },
    "Mistral": {
        "placeholder": "mistral-small-latest",
        "examples": "`mistral-small-latest` · `mistral-medium-latest` · `mistral-large-latest`",
        "docs": "https://console.mistral.ai/api-keys",
        "base_url": "https://api.mistral.ai/v1",
    },
    "OpenRouter": {
        "placeholder": "meta-llama/llama-3.3-70b-instruct:free",
        "examples": "`meta-llama/llama-3.3-70b-instruct:free` · `nvidia/nemotron-3-super-120b-a12b:free` · `deepseek/deepseek-r1:free` · `google/gemma-3-27b-it:free`",
        "docs": "https://openrouter.ai/keys",
        "base_url": "https://openrouter.ai/api/v1",
    },
}

# ── PAGE CONFIG ───────────────────────────────────────────────────────────────
st.set_page_config(page_title="QAForge – AI Test Case Generator", page_icon="🧪", layout="wide")
st.markdown("""
<style>
.badge{display:inline-block;padding:6px 16px;border-radius:20px;font-weight:700;font-size:13px;margin-bottom:16px;}
.b1{background:#1a3a5c;color:#60aaff;border:1px solid #2255aa;}
.b2{background:#1a3a25;color:#60cc88;border:1px solid #226644;}
.b3{background:#3a1a2a;color:#cc6699;border:1px solid #882255;}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("🧪 QAForge — AI Test Case Generator V.0.2")

    provider = st.radio("LLM Provider", list(PROVIDER_DEFAULTS.keys()), horizontal=True)
    cfg = PROVIDER_DEFAULTS[provider]

    api_key = st.text_input(
        f"{provider} API Key", type="password",
        help=f"Get your key at: {cfg['docs']}"
    )
    model_choice = st.text_input(
        "Model", value=cfg["placeholder"],
        help=f"Exact model ID — {cfg['docs']}"
    )
    st.caption(cfg["examples"])

    # Store in session state so adapters can access them
    st.session_state.provider = provider
    st.session_state.api_key = api_key
    st.session_state.model_choice = model_choice

    st.divider()
    st.markdown("""
### 🗺️ How it works
1. **Phase 1** — Submit your User Story → AI asks questions → answer → validate
2. **Phase 2** — AI generates test plan → refine → validate
3. **Phase 3** — AI writes full test cases → export (MD / JSON / CSV)
""")
    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "active_phase": 1, "phase_reached": 1,
    "p1_msgs": [], "p2_msgs": [], "p3_msgs": [],
    "p1_validated": False, "p2_validated": False,
    "us_submitted": False, "p1_context": "", "p2_draft": "",
    "structured_test_cases": None,
    "p1_questions": [], "p1_answers": {}, "p1_summary": "", "p1_user_story": "", "p1_raw_prompt": "", "p1_extra_ctx": "", "p1_chat_msgs": [],
    "p2_scenarios": [], "p2_summary": "", "p2_review": {},
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PROMPTS ───────────────────────────────────────────────────────────────────
PROMPT_P1_QUESTIONS = """You are a Senior QA Analyst and Requirements Engineer with 10+ years of experience.

## YOUR ROLE
Analyze the provided User Story and generate clarifying questions to resolve ambiguities before test planning.

## QUESTION STRATEGY
- Ask ONLY questions whose answer would meaningfully change the test strategy
- Simple, unambiguous user stories → fewer questions (3–5)
- Complex user stories (multi-step flows, payments, permissions, integrations) → more questions (up to 15)
- Every question must target a REAL ambiguity — never ask what is already stated
- 1 question = 1 specific piece of missing information
- Never combine two questions into one

## QUESTION TYPES
Use the most appropriate type for each question:
- "boolean" → yes/no questions (e.g. "Is this field mandatory?")
- "multiple_choice" → when there are 2–5 known possible answers
- "text" → when the answer is a free value (limit, rule, description)

## CATEGORIES
Classify each question into one of:
- Functional
- Validation
- Error Handling
- Edge Cases
- System / Dependencies

## VISUAL ANALYSIS
When [IMAGE_N — filename] markers appear in the document context:
- Identify the type of visual: wireframe, UI screenshot, form mockup, flow diagram, table, error state
- Extract ALL visible form fields and their apparent constraints (required, format, length)
- Note navigation elements, buttons, links and the flows they imply
- Identify visible validation rules, error messages, or status indicators
- Treat every visual as a functional specification — it defines behaviour, not just appearance
- If a visual contradicts or extends the written text, flag it in your questions
- Reference visuals explicitly in your questions (e.g. "In the login screen shown in IMAGE_1...")

## OUTPUT FORMAT (STRICT JSON — no markdown, no explanation)
{
  "summary": "2-3 sentence summary of your current understanding of the feature",
  "key_business_rules": ["rule extracted from text or visuals"],
  "actors": ["User", "Admin"],
  "screens_identified": ["Login screen (IMAGE_1)", "Dashboard (IMAGE_2)"],
  "questions": [
    {
      "id": 1,
      "category": "Functional",
      "type": "boolean",
      "question": "Is the user required to be logged in to access this feature?"
    },
    {
      "id": 2,
      "category": "Validation",
      "type": "multiple_choice",
      "question": "Which email formats are accepted?",
      "options": ["All valid email formats", "Professional emails only", "Specific domain only"]
    },
    {
      "id": 3,
      "category": "Edge Cases",
      "type": "text",
      "question": "What is the maximum character length allowed for this field?"
    }
  ]
}

HARD CONSTRAINTS:
- Output ONLY valid JSON. No markdown fences, no preamble.
- Do NOT generate test cases, scenarios, or test plan content.
- Do NOT invent business rules not present in the User Story or attached visuals.
- If no visuals are present, leave screens_identified as an empty array.
"""

PROMPT_P1_CHAT = """You are a Senior QA Analyst reviewing answers to your clarifying questions.
Acknowledge the answers, identify any remaining ambiguities, and ask follow-up questions if needed.
If all critical questions are answered, confirm readiness to proceed to test planning.
Keep responses concise and professional.
"""

PROMPT_P2 = """You are a Lead QA Engineer specializing in test design and coverage strategy.

## YOUR ROLE
Generate a comprehensive TEST PLAN as scenario TITLES ONLY with metadata.
FORBIDDEN: steps, preconditions, or expected results.

## COVERAGE — apply ALL applicable techniques:
- Happy Path, Alternate Flows
- Equivalence Partitioning, Boundary Value Analysis (BVA)
- Error Guessing, State Transitions, Negative Testing
- Security / Non-Functional if applicable

## OUTPUT FORMAT (STRICT JSON — no markdown, no explanation)
{
  "summary": "2-3 sentence feature summary",
  "scenarios": [
    {
      "id": 1,
      "title": "Successful login with valid credentials",
      "category": "Happy Path",
      "priority": "Very High"
    },
    {
      "id": 2,
      "title": "Login with invalid password",
      "category": "Negative",
      "priority": "High"
    }
  ]
}

## CATEGORIES (use exactly these values):
Happy Path | Alternate Flow | BVA | Equivalence | Negative | Edge Case | Security | Non-Functional

## PRIORITIES (use exactly these values):
Very High | High | Medium | Low

## HARD CONSTRAINTS
- Output ONLY valid JSON. No markdown fences, no preamble.
- Minimum 12 scenarios.
- Assign realistic priorities based on business impact.
"""

PROMPT_P3_MARKDOWN = """You are a Senior QA Test Architect writing execution-ready test cases.
Generate detailed human-readable test cases in Markdown format.

### TEST CASE [N]: [Scenario Title]
| Field | Detail |
|-------|--------|
| **ID** | TC-[N] |
| **Type** | [Happy Path / Alternate / BVA / Equivalence / Negative / Edge Case / Security] |
| **Priority** | [Very High / High / Medium / Low] |
| **Automation** | [✅ Good candidate / 🖐️ Manual only] — [reason] |

**📌 Preconditions:** - [state, role, data]
**🔢 Test Steps:** 1. [action + exact data] 2. ...
**✅ Expected Result:** [exact observable outcome]
**🔴 Failure Signature:** [what tester sees on failure]

HARD CONSTRAINTS: Real test data in steps. If unclear: ⚠️ *Assumption: [...] — confirm with PO.*"""

PROMPT_P3_JSON = """You are a Senior QA Test Architect.
Your task is to convert the provided Markdown test cases into a structured JSON array.
DO NOT invent or add new test cases. Extract EXACTLY what is in the Markdown.

Each object must have:
- id (string, e.g. "TC-1")
- title (string)
- type (string: Happy Path / Alternate / BVA / Equivalence / Negative / Edge Case / Security)
- priority (string: Very High / High / Medium / Low)
- automation (string: "Good candidate" or "Manual only")
- preconditions (array of strings)
- steps (array of {"step_number": int, "action": string})
- expected_result (string)
- failure_signature (string)

Output ONLY a valid JSON array. No markdown, no explanation, no preamble."""

# ── FILE PARSING ──────────────────────────────────────────────────────────────
ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp", "pdf", "txt", "md", "docx"]
MAX_FILES = 5
MAX_CHARS = 15000

# Minimum image dimensions — filters out decorative icons, bullets, artefacts
IMG_MIN_WIDTH  = 50
IMG_MIN_HEIGHT = 50


def pdf_smart_extract(file_bytes: bytes, fname: str):
    """
    Extract text + embedded images from a PDF, preserving their positional order.

    Returns:
        text  (str)        — full text with [IMAGE_N — fname page X] markers intercalated
        images (list[PIL]) — PIL images in marker order
    """
    try:
        import fitz  # PyMuPDF
    except ImportError:
        # Graceful degradation: fall back to pypdf text-only
        import pypdf
        reader = pypdf.PdfReader(io.BytesIO(file_bytes))
        text = "\n".join(p.extract_text() or "" for p in reader.pages)
        if not text.strip():
            return f"[⚠️ {fname}: image-based PDF — install pymupdf for full extraction]", []
        return text, []

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    full_text_parts = []
    extracted_images = []
    img_counter = 0
    seen_xrefs = set()  # deduplicate images shared across pages

    for page_num, page in enumerate(doc):
        blocks = page.get_text("dict")["blocks"]
        page_header = f"--- {fname} | Page {page_num + 1} ---"

        # Build ordered element list: (y_position, kind, content)
        elements = []

        for block in blocks:
            btype = block.get("type", -1)
            y0 = block["bbox"][1]

            if btype == 0:  # text block
                spans_text = " ".join(
                    span["text"]
                    for line in block.get("lines", [])
                    for span in line.get("spans", [])
                ).strip()
                if spans_text:
                    elements.append((y0, "text", spans_text))

            elif btype == 1:  # image block
                xref = block.get("xref")
                if xref and xref not in seen_xrefs:
                    elements.append((y0, "image", xref))

        # Sort by vertical position so order matches reading order
        elements.sort(key=lambda e: e[0])

        has_text = any(e[1] == "text" for e in elements)
        page_parts = [page_header]

        for _, kind, content in elements:
            if kind == "text":
                page_parts.append(content)
            elif kind == "image":
                try:
                    base_img = doc.extract_image(content)
                    img = Image.open(io.BytesIO(base_img["image"])).convert("RGB")
                    w, h = img.size

                    # Skip tiny decorative images (icons, rules, bullets…)
                    if w < IMG_MIN_WIDTH or h < IMG_MIN_HEIGHT:
                        continue

                    seen_xrefs.add(content)
                    img_counter += 1
                    extracted_images.append(img)
                    page_parts.append(
                        f"[IMAGE_{img_counter} — {fname} page {page_num + 1}]"
                    )
                except Exception:
                    pass  # corrupt image — skip silently

        # Fallback for scanned pages (no extractable text): rasterise full page
        if not has_text:
            try:
                pix = page.get_pixmap(dpi=150)
                img = Image.open(io.BytesIO(pix.tobytes("png"))).convert("RGB")
                img_counter += 1
                extracted_images.append(img)
                page_parts.append(
                    f"[IMAGE_{img_counter} — {fname} page {page_num + 1} — scanned page]"
                )
            except Exception:
                pass

        full_text_parts.append("\n".join(page_parts))

    doc.close()
    return "\n\n".join(full_text_parts), extracted_images


def docx_smart_extract(file_bytes: bytes, fname: str):
    """
    Extract text + embedded images from a DOCX, preserving document order.

    Returns:
        text  (str)        — paragraphs + [IMAGE_N — fname] markers + Markdown tables
        images (list[PIL]) — PIL images in marker order
    """
    doc = docx.Document(io.BytesIO(file_bytes))
    text_parts = []
    extracted_images = []
    img_counter = 0

    # Correct XML namespaces for image lookup in DOCX
    NS = {
        "a":   "http://schemas.openxmlformats.org/drawingml/2006/main",
        "pic": "http://schemas.openxmlformats.org/drawingml/2006/picture",
        "r":   "http://schemas.openxmlformats.org/officeDocument/2006/relationships",
    }

    # We need to walk the document body in XML order so tables and paragraphs
    # are interleaved correctly (doc.paragraphs skips tables entirely).
    from docx.oxml.ns import qn

    for child in doc.element.body:
        tag = child.tag

        # ── Paragraph ────────────────────────────────────────────────────────
        if tag == qn("w:p"):
            from docx.text.paragraph import Paragraph as DocxParagraph
            para = DocxParagraph(child, doc)

            # Extract text
            para_text = para.text.strip()
            if para_text:
                text_parts.append(para_text)

            # Extract images inside this paragraph's runs
            for run in para.runs:
                blips = run._r.findall(".//pic:blipFill/a:blip", NS)
                for blip in blips:
                    embed_id = blip.get(
                        "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                    )
                    if embed_id and embed_id in doc.part.rels:
                        rel = doc.part.rels[embed_id]
                        if "image" in rel.reltype:
                            try:
                                img = Image.open(
                                    io.BytesIO(rel.target_part.blob)
                                ).convert("RGB")
                                w, h = img.size
                                if w < IMG_MIN_WIDTH or h < IMG_MIN_HEIGHT:
                                    continue
                                img_counter += 1
                                extracted_images.append(img)
                                text_parts.append(
                                    f"[IMAGE_{img_counter} — {fname}]"
                                )
                            except Exception:
                                pass

        # ── Table ─────────────────────────────────────────────────────────────
        elif tag == qn("w:tbl"):
            from docx.table import Table as DocxTable
            table = DocxTable(child, doc)
            rows = []
            for i, row in enumerate(table.rows):
                cells = [cell.text.strip().replace("\n", " ") for cell in row.cells]
                rows.append("| " + " | ".join(cells) + " |")
                if i == 0:
                    rows.append("|" + "|".join(["---"] * len(cells)) + "|")
            if rows:
                text_parts.append("\n[TABLE]\n" + "\n".join(rows))

    return "\n".join(text_parts), extracted_images


def extract_text_plain(f):
    """Fallback plain-text extraction for .txt and .md files."""
    return f.read().decode("utf-8", errors="ignore")


def is_image(f):
    return f.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))


def file_icon(f):
    n = f.name.lower()
    if n.endswith(".pdf"):  return "📕"
    if n.endswith(".docx"): return "📘"
    if n.endswith((".txt", ".md")): return "📄"
    return "🖼️" if is_image(f) else "📎"


# ── ERROR HANDLER ─────────────────────────────────────────────────────────────
def handle_error(e):
    err = str(e)
    if "429" in err or "RESOURCE_EXHAUSTED" in err or "rate_limit" in err.lower():
        st.error("⚠️ Quota/rate limit reached. Wait a moment or switch model.")
    elif "404" in err or "NOT_FOUND" in err or "model_not_found" in err.lower():
        st.error(f"⚠️ Model not found: **{st.session_state.model_choice}**. Check the docs: {PROVIDER_DEFAULTS[st.session_state.provider]['docs']}")
    elif "401" in err or "invalid_api_key" in err.lower() or "API_KEY" in err:
        st.error("⚠️ Invalid API key. Check your key in the sidebar.")
    else:
        st.error(f"LLM Error: {err}")

def render_chat(msgs):
    for m in msgs:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])

# ── CSV BUILDER ───────────────────────────────────────────────────────────────
def build_csv(data):
    if not data: return ""
    out = io.StringIO()
    fields = ["id","title","type","priority","automation","preconditions","steps","expected_result","failure_signature"]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        r = dict(row)
        pre = r.get("preconditions", [])
        r["preconditions"] = " | ".join(pre) if isinstance(pre, list) else str(pre)
        steps = r.get("steps", [])
        if steps and isinstance(steps, list):
            r["steps"] = " | ".join(
                f"{s.get('step_number','')}.{s.get('action','')}" if isinstance(s, dict) else str(s)
                for s in steps
            )
        else:
            r["steps"] = str(steps)
        writer.writerow(r)
    return out.getvalue()

# ── TAB BAR ───────────────────────────────────────────────────────────────────
def render_tab_bar():
    pr, ap = st.session_state.phase_reached, st.session_state.active_phase
    for i, (n, label) in enumerate({1:"Analysis", 2:"Test Plan", 3:"Test Cases"}.items()):
        with st.columns(3)[i]:
            if n > pr:
                st.button(f"🔒 Phase {n} — {label}", key=f"tab_{n}", disabled=True, use_container_width=True)
            else:
                prefix = "▶" if n == ap else "✅"
                if st.button(f"{prefix} Phase {n} — {label}", key=f"tab_{n}",
                              use_container_width=True, type="primary" if n == ap else "secondary"):
                    st.session_state.active_phase = n
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
st.title("🧪 QAForge — AI Test Case Generator")

if not api_key:
    st.warning(f"⚠️ Enter your {provider} API key in the sidebar.")
    st.stop()

render_tab_bar()
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.active_phase == 1:
    st.markdown('<div class="badge b1">🔍 Phase 1 — Senior QA Analyst: Requirements Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        us_input = st.text_area("User Story + Acceptance Criteria", height=180, max_chars=5000,
            placeholder="As a [user], I want to [action] so that [benefit].\n\nAcceptance Criteria:\n- ...")
        if us_input: st.caption(f"{len(us_input)}/5000 characters")

        uploaded_files = st.file_uploader(f"📎 Attach files (max {MAX_FILES})",
            type=ALLOWED_TYPES, accept_multiple_files=True,
            help="PNG, JPG, WEBP · PDF · DOCX · TXT / MD")
        if uploaded_files:
            if len(uploaded_files) > MAX_FILES:
                st.warning(f"⚠️ Max {MAX_FILES} files. First {MAX_FILES} used.")
                uploaded_files = uploaded_files[:MAX_FILES]
            fcols = st.columns(len(uploaded_files))
            for idx, f in enumerate(uploaded_files):
                with fcols[idx]:
                    if is_image(f): st.image(f, caption=f.name, use_column_width=True)
                    else: st.markdown(f"{file_icon(f)} **{f.name}**"); st.caption(f"{round(f.size/1024,1)} KB")

        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not us_input or len(us_input.strip()) < 20:
                st.warning("Please provide a more detailed User Story (min. 20 characters).")
            else:
                # ── Smart document extraction ─────────────────────────────────
                # images: all PIL images to send to the LLM (direct uploads + extracted)
                # doc_texts: list of text blocks with positional markers
                images, doc_texts = [], []
                doc_image_count = 0  # images extracted from documents
                direct_image_count = 0  # images uploaded directly

                for f in (uploaded_files or []):
                    f.seek(0)
                    fname = f.name

                    if is_image(f):
                        # Direct image upload — send as-is
                        images.append(Image.open(f).convert("RGB"))
                        direct_image_count += 1

                    elif fname.lower().endswith(".pdf"):
                        file_bytes = f.read()
                        with st.spinner(f"🔍 Extracting {fname}…"):
                            text, doc_imgs = pdf_smart_extract(file_bytes, fname)
                        if text:
                            if len(text) > MAX_CHARS:
                                text = text[:MAX_CHARS] + f"\n[...truncated at {MAX_CHARS} chars]"
                                st.info(f"ℹ️ {fname} truncated to {MAX_CHARS} chars.")
                            doc_texts.append(text)
                        images.extend(doc_imgs)
                        doc_image_count += len(doc_imgs)

                    elif fname.lower().endswith(".docx"):
                        file_bytes = f.read()
                        with st.spinner(f"🔍 Extracting {fname}…"):
                            text, doc_imgs = docx_smart_extract(file_bytes, fname)
                        if text:
                            if len(text) > MAX_CHARS:
                                text = text[:MAX_CHARS] + f"\n[...truncated at {MAX_CHARS} chars]"
                                st.info(f"ℹ️ {fname} truncated to {MAX_CHARS} chars.")
                            doc_texts.append(text)
                        images.extend(doc_imgs)
                        doc_image_count += len(doc_imgs)

                    elif fname.lower().endswith((".txt", ".md")):
                        f.seek(0)
                        text = extract_text_plain(f)
                        if text:
                            if len(text) > MAX_CHARS:
                                text = text[:MAX_CHARS] + f"\n[...truncated at {MAX_CHARS} chars]"
                                st.info(f"ℹ️ {fname} truncated to {MAX_CHARS} chars.")
                            doc_texts.append(f"--- {fname} ---\n{text}")

                # ── Build prompt ──────────────────────────────────────────────
                prompt = f"Please analyze the following User Story:\n\n{us_input}"

                if doc_texts:
                    prompt += "\n\n=== ATTACHED DOCUMENTS ===\n" + "\n\n".join(doc_texts)

                # Summarise what images are included (context for providers without vision)
                if images:
                    img_summary_parts = []
                    if direct_image_count:
                        img_summary_parts.append(f"{direct_image_count} directly uploaded image(s)")
                    if doc_image_count:
                        img_summary_parts.append(
                            f"{doc_image_count} image(s) extracted from documents "
                            "(referenced by [IMAGE_N] markers in the text above)"
                        )
                    prompt += f"\n\n[Visuals attached: {' + '.join(img_summary_parts)}]"

                with st.spinner(f"Analyzing with {provider} / `{model_choice}`…"):
                    try:
                        raw = call_llm([], PROMPT_P1_QUESTIONS, prompt, images or None, max_tokens=3000)
                        # Parse JSON — strip markdown fences if present
                        clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                        parsed = json.loads(clean)
                        st.session_state.p1_questions = parsed.get("questions", [])
                        st.session_state.p1_summary = parsed.get("summary", "")
                        # Store enriched fields from new schema
                        st.session_state.p1_business_rules = parsed.get("key_business_rules", [])
                        st.session_state.p1_actors = parsed.get("actors", [])
                        st.session_state.p1_screens = parsed.get("screens_identified", [])
                        st.session_state.p1_answers = {}
                        st.session_state.p1_raw_prompt = prompt
                        st.session_state.p1_user_story = us_input
                        st.session_state.us_submitted = True
                        st.rerun()
                    except Exception as e: handle_error(e)

    elif st.session_state.p1_validated or (st.session_state.us_submitted and not st.session_state.p1_validated):
        # ── Unified editable view (works both before and after validation) ───
        # ── Display summary ───────────────────────────────────────────────────
        st.info(f"📋 **Current Understanding:** {st.session_state.p1_summary}")

        # ── Display enriched analysis (new schema fields) ─────────────────────
        with st.expander("🔎 Extracted analysis details", expanded=False):
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                rules = st.session_state.get("p1_business_rules", [])
                if rules:
                    st.markdown("**⚖️ Business Rules**")
                    for r in rules: st.markdown(f"- {r}")
            with col_b:
                actors = st.session_state.get("p1_actors", [])
                if actors:
                    st.markdown("**👤 Actors**")
                    for a in actors: st.markdown(f"- {a}")
            with col_c:
                screens = st.session_state.get("p1_screens", [])
                if screens:
                    st.markdown("**🖥️ Screens identified**")
                    for s in screens: st.markdown(f"- {s}")

        st.markdown("### 🔍 Clarifying Questions")
        st.caption("Answer the questions below — click or type as appropriate.")

        questions = st.session_state.p1_questions
        answers = st.session_state.p1_answers

        # Group by category
        from collections import defaultdict
        by_cat = defaultdict(list)
        for q in questions:
            by_cat[q.get("category", "General")].append(q)

        cat_icons = {
            "Functional": "⚙️", "Validation": "✅", "Error Handling": "❌",
            "Edge Cases": "⚠️", "System / Dependencies": "🔗", "General": "💬"
        }

        for cat, qs in by_cat.items():
            icon = cat_icons.get(cat, "💬")
            st.markdown(f"#### {icon} {cat}")
            for q in qs:
                qid = q["id"]
                qtype = q.get("type", "text")
                label = f"**{q['question']}**"

                if qtype == "boolean":
                    col1, col2, col3 = st.columns([3, 1, 1])
                    with col1: st.markdown(label)
                    with col2:
                        if st.button("✅ Yes", key=f"yes_{qid}", use_container_width=True,
                                     type="primary" if answers.get(qid) == "Yes" else "secondary"):
                            st.session_state.p1_answers[qid] = "Yes"; st.rerun()
                    with col3:
                        if st.button("❌ No", key=f"no_{qid}", use_container_width=True,
                                     type="primary" if answers.get(qid) == "No" else "secondary"):
                            st.session_state.p1_answers[qid] = "No"; st.rerun()
                    if qid in answers:
                        st.caption(f"→ Your answer: **{answers[qid]}**")

                elif qtype == "multiple_choice":
                    opts = q.get("options", [])
                    current = answers.get(qid, None)
                    chosen = st.radio(label, opts, index=opts.index(current) if current in opts else None,
                                      key=f"mc_{qid}", horizontal=True)
                    if chosen:
                        st.session_state.p1_answers[qid] = chosen

                else:  # text
                    current_val = answers.get(qid, "")
                    val = st.text_input(label, value=current_val, key=f"txt_{qid}",
                                        placeholder="Your answer…")
                    if val:
                        st.session_state.p1_answers[qid] = val

        st.divider()

        # ── Optional free-text context ────────────────────────────────────────
        extra = st.text_area("💬 Additional context (optional)",
                             placeholder="Any extra details, constraints or remarks…",
                             height=80, key="p1_extra")

        # ── Progress indicator ────────────────────────────────────────────────
        answered = sum(1 for q in questions if q["id"] in st.session_state.p1_answers)
        total_q = len(questions)
        st.progress(answered / total_q if total_q else 1,
                    text=f"{answered}/{total_q} questions answered")

        # ── Chat libre avec l'agent (avant soumission) ───────────────────────
        st.divider()
        st.markdown("#### 💬 Discuss with the agent")
        st.caption("Ask for clarification, correct a misunderstanding, or request new questions.")
        if "p1_chat_msgs" not in st.session_state:
            st.session_state.p1_chat_msgs = []
        for m in st.session_state.p1_chat_msgs:
            with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
                st.markdown(m["content"])
        p1_reply = st.chat_input("Message the agent…", key="p1_agent_chat")
        if p1_reply:
            st.session_state.p1_chat_msgs.append({"role": "user", "content": p1_reply})
            with st.spinner("Thinking…"):
                try:
                    cur_answers = "\n".join(
                        f"- {q['question']} → {st.session_state.p1_answers.get(q['id'], 'not answered yet')}"
                        for q in st.session_state.p1_questions
                    )
                    ctx_msg = (
                        f"Current understanding: {st.session_state.p1_summary}\n\n"
                        f"Questions and answers so far:\n{cur_answers}\n\n"
                        f"User says: {p1_reply}"
                    )
                    response = call_llm(st.session_state.p1_chat_msgs[:-1], PROMPT_P1_CHAT, ctx_msg, max_tokens=2000)
                    st.session_state.p1_chat_msgs.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e: handle_error(e)

        st.divider()

        if st.session_state.p1_validated:
            st.warning("⚠️ Phase 1 already validated. Re-submitting will regenerate Phase 2 and reset Phase 3.")

        btn_label = "🔄 Re-submit → Regenerate Phase 2" if st.session_state.p1_validated else "✅ Submit Answers → Phase 2"
        if st.button(btn_label, type="primary", use_container_width=True, key="p1_val"):
            st.session_state.p1_extra_ctx = extra
            answers_text = "\n".join(
                f"- [{q.get('category','')}] {q['question']}\n  → {st.session_state.p1_answers.get(q['id'], 'Not answered')}"
                for q in questions
            )
            if extra:
                answers_text += f"\n\nAdditional context:\n{extra}"

            # Include enriched schema fields in Phase 2 context
            rules_ctx = ""
            if st.session_state.get("p1_business_rules"):
                rules_ctx = "\nKey Business Rules:\n" + "\n".join(
                    f"- {r}" for r in st.session_state.p1_business_rules
                )
            screens_ctx = ""
            if st.session_state.get("p1_screens"):
                screens_ctx = "\nScreens identified:\n" + "\n".join(
                    f"- {s}" for s in st.session_state.p1_screens
                )

            ctx = (
                f"User Story:\n{st.session_state.p1_user_story}\n\n"
                f"Requirements Analysis Summary:\n{st.session_state.p1_summary}"
                f"{rules_ctx}{screens_ctx}\n\n"
                f"Clarification Q&A:\n{answers_text}\n\n"
                f"Generate the test plan (titles only)."
            )
            with st.spinner("📋 Generating test plan…"):
                try:
                    raw_p2 = call_llm([], PROMPT_P2, ctx, max_tokens=3000)
                    clean_p2 = raw_p2.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                    parsed_p2 = json.loads(clean_p2)
                    st.session_state.p2_scenarios = parsed_p2.get("scenarios", [])
                    st.session_state.p2_summary = parsed_p2.get("summary", "")
                    st.session_state.p2_draft = raw_p2
                    st.session_state.p2_msgs = [{"role":"user","content":ctx},{"role":"assistant","content":raw_p2}]
                    st.session_state.p2_review = {}
                    st.session_state.p2_validated = False
                    st.session_state.p3_msgs = []
                    st.session_state.p3_full_md = ""
                    st.session_state.structured_test_cases = None
                    st.session_state.p1_context = ctx
                    st.session_state.p1_validated = True
                    st.session_state.phase_reached = max(st.session_state.phase_reached, 2)
                    st.session_state.active_phase = 2
                    st.rerun()
                except Exception as e: handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_phase == 2:
    st.markdown('<div class="badge b2">📋 Phase 2 — Lead QA Engineer: Test Plan</div>', unsafe_allow_html=True)

    scenarios = st.session_state.get("p2_scenarios", [])

    if scenarios:
        # ── Init review state ─────────────────────────────────────────────────
        if "p2_review" not in st.session_state or len(st.session_state.p2_review) != len(scenarios):
            st.session_state.p2_review = {
                s["id"]: {"selected": True, "priority": s.get("priority", "P2")}
                for s in scenarios
            }

        review = st.session_state.p2_review
        PRIORITY_COLORS = {"Very High": "🔴", "High": "🟠", "Medium": "🟡", "Low": "🟢"}
        CAT_ICONS = {
            "Happy Path": "✅", "Alternate Flow": "🔄", "BVA": "🔢",
            "Equivalence": "🔀", "Negative": "❌", "Edge Case": "⚠️",
            "Security": "🔒", "Non-Functional": "⚙️"
        }

        st.markdown(f"📋 **{st.session_state.get('p2_summary', 'Test Plan')}**")
        st.divider()

        for s in scenarios:
            sid = s["id"]
            rv = review[sid]
            is_sel = rv["selected"]
            cur_prio = rv["priority"]
            cat = s.get("category", "")
            cat_icon = CAT_ICONS.get(cat, "📌")

            c1, c2, c3, c4, c5, c6, c7 = st.columns([0.5, 0.5, 3.5, 1.2, 1.2, 1.2, 1.2])
            with c1:
                if st.button("✅", key=f"sel_{sid}", help="Include in Phase 3",
                             type="primary" if is_sel else "secondary"):
                    st.session_state.p2_review[sid]["selected"] = True; st.rerun()
            with c2:
                if st.button("❌", key=f"del_{sid}", help="Exclude from Phase 3",
                             type="primary" if not is_sel else "secondary"):
                    st.session_state.p2_review[sid]["selected"] = False; st.rerun()
            with c3:
                label = f"{cat_icon} {s['title']}"
                st.markdown(f"~~{label}~~" if not is_sel else label)
            with c4:
                if st.button("🔴 Very High", key=f"pvh_{sid}",
                             type="primary" if cur_prio=="Very High" else "secondary"):
                    st.session_state.p2_review[sid]["priority"] = "Very High"; st.rerun()
            with c5:
                if st.button("🟠 High", key=f"phi_{sid}",
                             type="primary" if cur_prio=="High" else "secondary"):
                    st.session_state.p2_review[sid]["priority"] = "High"; st.rerun()
            with c6:
                if st.button("🟡 Medium", key=f"pmd_{sid}",
                             type="primary" if cur_prio=="Medium" else "secondary"):
                    st.session_state.p2_review[sid]["priority"] = "Medium"; st.rerun()
            with c7:
                if st.button("🟢 Low", key=f"plw_{sid}",
                             type="primary" if cur_prio=="Low" else "secondary"):
                    st.session_state.p2_review[sid]["priority"] = "Low"; st.rerun()

    else:
        render_chat(st.session_state.p2_msgs)

    # ── Chat modifications ────────────────────────────────────────────────────
    st.markdown("#### 💬 Request global modifications")
    reply2 = st.chat_input("Add scenarios, change coverage, request modifications…", key="p2_chat")
    if reply2:
        with st.spinner("Updating plan…"):
            try:
                raw = call_llm(st.session_state.p2_msgs, PROMPT_P2, reply2, max_tokens=3000)
                clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
                parsed = json.loads(clean)
                st.session_state.p2_scenarios = parsed.get("scenarios", [])
                st.session_state.p2_summary = parsed.get("summary", "")
                st.session_state.p2_draft = raw
                st.session_state.p2_msgs.append({"role":"user","content":reply2})
                st.session_state.p2_msgs.append({"role":"assistant","content":raw})
                st.session_state.p2_review = {}
                st.rerun()
            except Exception as e: handle_error(e)

    if st.button("✅ Validate Plan → Phase 3", type="primary", use_container_width=True, key="p2_val"):
        review = st.session_state.get("p2_review", {})
        all_scenarios = st.session_state.get("p2_scenarios", [])
        selected_scenarios = [
            s for s in all_scenarios
            if review.get(s["id"], {}).get("selected", True)
        ]
        if not selected_scenarios:
            st.warning("⚠️ No scenarios selected. Please select at least one scenario.")
            st.stop()
        plan_lines = "\n".join(
            f"- TC: {s['title']} [{review.get(s['id'], {}).get('priority', s.get('priority','P2'))}]"
            for s in selected_scenarios
        )
        plan_ctx = (
            f"Validated test plan ({len(selected_scenarios)} scenarios):\n\n"
            f"{plan_lines}\n\n"
            f"Feature summary: {st.session_state.get('p2_summary', '')}\n\n"
            f"Context:\n{st.session_state.p1_context}"
        )
        scenario_titles = [s["title"] for s in selected_scenarios]
        n_scenarios = len(scenario_titles)

        if n_scenarios == 0:
            st.warning("⚠️ Could not extract scenario titles from the plan. Generating in single call.")
            scenario_titles = None

        if scenario_titles and n_scenarios > 6:
            st.info(f"📦 {n_scenarios} scenarios detected — generating in batches of 6 to avoid truncation.")
            try:
                md, structured = generate_test_cases_in_batches(
                    PROMPT_P3_MARKDOWN, plan_ctx, scenario_titles, batch_size=6
                )
                st.session_state.p3_msgs = [{"role":"user","content":plan_ctx},{"role":"assistant","content":md}]
                st.session_state.p3_full_md = md
                st.session_state.structured_test_cases = None
            except Exception as e:
                handle_error(e); st.stop()
        else:
            with st.spinner("📝 Generating test cases (auto-completing…)"):
                try:
                    md, final_msgs = generate_until_complete(
                        PROMPT_P3_MARKDOWN, [],
                        plan_ctx + "\n\nGenerate COMPLETE test cases for every scenario."
                    )
                    st.session_state.p3_msgs = final_msgs + [{"role":"assistant","content":md}] if not final_msgs else final_msgs
                    st.session_state.p3_full_md = md
                except Exception as e: handle_error(e); st.stop()

        st.session_state.structured_test_cases = None
        st.session_state.p3_bg_json_pending = False
        st.session_state.p3_bg_json_ctx = plan_ctx
        st.session_state.p2_validated = True
        st.session_state.phase_reached = max(st.session_state.phase_reached, 3)
        st.session_state.active_phase = 3
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_phase == 3:
    st.markdown('<div class="badge b3">📝 Phase 3 — Test Architect: Detailed Test Cases</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p3_msgs)
    if st.session_state.p3_msgs:
        all_md = "\n\n".join([m["content"] for m in st.session_state.p3_msgs if m["role"]=="assistant"])
        tc_data = st.session_state.structured_test_cases
        st.divider()
        st.markdown("### 📥 Export")
        c1,c2,c3,c4 = st.columns(4)
        with c1:
            st.download_button("📝 Markdown", data=all_md, file_name="test_cases.md", mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Text", data=all_md, file_name="test_cases.txt", mime="text/plain", use_container_width=True)
        with c3:
            if tc_data:
                st.download_button("🗂️ JSON", data=json.dumps(tc_data, indent=2, ensure_ascii=False),
                                   file_name="test_cases.json", mime="application/json", use_container_width=True)
            else:
                st.button("🗂️ JSON", disabled=True, use_container_width=True, help="Structured export unavailable")
        with c4:
            csv_d = build_csv(tc_data) if tc_data else ""
            if csv_d:
                st.download_button("📊 CSV", data=csv_d, file_name="test_cases.csv", mime="text/csv", use_container_width=True)
            else:
                st.button("📊 CSV", disabled=True, use_container_width=True, help="Structured export unavailable")
        if tc_data:
            with st.expander(f"👁️ Preview JSON ({len(tc_data)} test cases)", expanded=False):
                st.json(tc_data)

    # ── On-demand JSON/CSV generation ────────────────────────────────────────
    if st.session_state.get("p3_full_md") and st.session_state.structured_test_cases is None:
        st.info("💡 JSON & CSV exports are ready to generate on demand.")
        if st.button("⚙️ Generate JSON & CSV exports", use_container_width=True, key="p3_gen_exports"):
            with st.spinner("Structuring test cases from Markdown…"):
                try:
                    markdown_content = st.session_state.p3_full_md
                    tc = call_llm_structured(
                        PROMPT_P3_JSON,
                        f"Convert the following Markdown test cases into a JSON array:\n\n{markdown_content}",
                        max_tokens=8000
                    )
                    st.session_state.structured_test_cases = tc
                    st.success("✅ JSON & CSV ready!")
                    st.rerun()
                except Exception as e:
                    st.warning(f"⚠️ Export generation failed: {e}")

    st.divider()

    # ── Auto-repair ───────────────────────────────────────────────────────────
    if st.session_state.p3_msgs:
        with st.expander("⚠️ Generation incomplete? Click to auto-complete", expanded=False):
            st.caption("This will automatically continue until all test cases are generated.")
            if st.button("🔄 Auto-complete remaining test cases", use_container_width=True, key="p3_autocomplete"):
                progress = st.progress(0, text="Auto-completing… iteration 1")
                try:
                    existing_md = st.session_state.get("p3_full_md", "")
                    extra_md, new_msgs = generate_until_complete(
                        PROMPT_P3_MARKDOWN,
                        st.session_state.p3_msgs,
                        "Continue EXACTLY where you stopped. Generate ALL remaining test cases.",
                        max_iterations=2, max_tokens=8000
                    )
                    st.session_state.p3_full_md = (existing_md + "\n\n" + extra_md).strip()
                    st.session_state.p3_msgs = new_msgs
                    progress.progress(1.0, text="✅ Complete!")
                    st.rerun()
                except Exception as e:
                    handle_error(e)

    reply3 = st.chat_input("Request adjustments or additional test cases…", key="p3_chat")
    if reply3:
        st.session_state.p3_msgs.append({"role":"user","content":reply3})
        with st.spinner("Updating…"):
            try:
                response = call_llm(st.session_state.p3_msgs[:-1], PROMPT_P3_MARKDOWN, reply3, max_tokens=8000)
                st.session_state.p3_msgs.append({"role":"assistant","content":response})
                st.session_state.p3_full_md = response
                st.session_state.structured_test_cases = None
                st.rerun()
            except Exception as e: handle_error(e)
