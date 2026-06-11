import streamlit as st
import time
import re
import io
import json
import csv
import base64
from collections import defaultdict
from PIL import Image
import docx

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


# ── IMAGE UTILS ───────────────────────────────────────────────────────────────

def resize_image(img: Image.Image, max_dim: int = 1024) -> Image.Image:
    """Downscale image so its longest side ≤ max_dim. Avoids costly API payloads."""
    w, h = img.size
    if max(w, h) <= max_dim:
        return img
    scale = max_dim / max(w, h)
    return img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

# ── LLM ADAPTERS ──────────────────────────────────────────────────────────────

def _is_rate_limit(e: Exception) -> bool:
    msg = str(e)
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "rate_limit" in msg.lower()

def _retry(fn, *args, max_retries: int = 3, **kwargs):
    """Call fn(*args, **kwargs) up to max_retries times on rate-limit errors."""
    for attempt in range(max_retries):
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            if _is_rate_limit(e) and attempt < max_retries - 1:
                wait = 5 * (attempt + 1)  # 5s, 10s
                time.sleep(wait)
            else:
                raise

@st.cache_resource
def _gemini_client(key: str):
    from google import genai
    return genai.Client(api_key=key)

@st.cache_resource
def _openai_client(key: str, base_url, provider: str):
    from openai import OpenAI
    if provider == "OpenRouter":
        return OpenAI(api_key=key, base_url=base_url,
            default_headers={
                "HTTP-Referer": "https://testcasegenerator-draft.streamlit.app",
                "X-Title": "QAForge"
            })
    return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)


def call_gemini(history, system_prompt, user_message, images=None, max_tokens=3000):
    from google.genai import types

    client = _gemini_client(st.session_state.api_key)
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    parts = [types.Part(text=user_message)]
    for img in (images or []):
        buf = io.BytesIO(); resize_image(img).save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    contents.append(types.Content(role="user", parts=parts))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=st.session_state.get("temperature", 0.2),
    )
    result = _retry(
        client.models.generate_content,
        model=st.session_state.model_choice.strip(), contents=contents, config=config
    )
    if not result or not result.text or not result.text.strip():
        raise Exception("Empty response from Gemini.")
    return result.text

def call_openai(history, system_prompt, user_message, images=None, max_tokens=3000, base_url=None):
    client = _openai_client(st.session_state.api_key, base_url, st.session_state.get("provider", "OpenAI"))
    messages = [{"role": "system", "content": system_prompt}]
    for m in history:
        messages.append({"role": m["role"], "content": m["content"]})

    # Build user content (text + images)
    if images:
        content = [{"type": "text", "text": user_message}]
        for img in images:
            buf = io.BytesIO(); resize_image(img).save(buf, format="PNG")
            b64 = base64.b64encode(buf.getvalue()).decode()
            content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
        messages.append({"role": "user", "content": content})
    else:
        messages.append({"role": "user", "content": user_message})

    result = _retry(
        client.chat.completions.create,
        model=st.session_state.model_choice.strip(),
        messages=messages,
        max_tokens=max_tokens,
        temperature=st.session_state.get("temperature", 0.2),
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

def extract_json(raw: str):
    """Robustly extract the first JSON object/array from an LLM response.
    Tolerates markdown fences, preambles and trailing text.
    (Replaces the fragile lstrip("```json") character-strip hack.)"""
    txt = raw.strip()
    txt = re.sub(r"^```[a-zA-Z]*\s*", "", txt)
    txt = re.sub(r"\s*```\s*$", "", txt)
    starts = [i for i in (txt.find("{"), txt.find("[")) if i != -1]
    if not starts:
        raise ValueError(f"No JSON found in LLM response: {txt[:200]}")
    obj, _ = json.JSONDecoder().raw_decode(txt[min(starts):])
    return obj


def call_llm_json(system_prompt, user_message, max_tokens=8000):
    """Call the LLM and return parsed JSON.
    Uses the provider's NATIVE JSON mode when available (Gemini, OpenAI);
    instruction-based JSON for the others (Groq / Mistral / OpenRouter).
    API errors (auth, rate-limit, model not found) are RAISED — never
    silently swallowed into a second doomed call."""
    provider = st.session_state.provider

    if provider == "Gemini":
        from google.genai import types
        client = _gemini_client(st.session_state.api_key)
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
            max_output_tokens=max_tokens,
            temperature=st.session_state.get("temperature", 0.2),
            response_mime_type="application/json",
        )
        result = _retry(
            client.models.generate_content,
            model=st.session_state.model_choice.strip(),
            contents=[types.Content(role="user", parts=[types.Part(text=user_message)])],
            config=config,
        )
        if not result or not result.text:
            raise Exception("Empty response from Gemini.")
        return extract_json(result.text)

    if provider == "OpenAI":
        client = _openai_client(st.session_state.api_key, None, "OpenAI")
        result = _retry(
            client.chat.completions.create,
            model=st.session_state.model_choice.strip(),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            max_tokens=max_tokens,
            temperature=st.session_state.get("temperature", 0.2),
            response_format={"type": "json_object"},
        )
        text = result.choices[0].message.content
        if not text:
            raise Exception("Empty response from OpenAI.")
        return extract_json(text)

    # Groq / Mistral / OpenRouter — instruction-based JSON
    raw = call_llm(
        [], system_prompt,
        user_message + "\n\nOutput ONLY valid JSON. No markdown fences, no explanation.",
        max_tokens=max_tokens,
    )
    return extract_json(raw)


def _generate_tc_batch(plan_ctx, batch):
    """Generate structured test cases for one batch of scenarios.
    `batch` = list of {"id": "TC-3", "title": ..., "priority": ...}.
    Verifies completeness against requested ids; retries missing ones ONCE."""
    want = {s["id"]: s for s in batch}
    batch_list = "\n".join(
        f'- {s["id"]} | {s["title"]} | priority: {s["priority"]} | covers: {", ".join(s.get("covers") or []) or "—"}'
        for s in batch
    )
    msg = (
        f"{st.session_state.get('lang_directive', '')}"
        f"{plan_ctx}\n\n"
        f"Write ONE detailed test case for EACH of these {len(batch)} scenarios, "
        f"keeping the exact `id`, `title`, `priority` and `covers` given:\n{batch_list}"
    )
    parsed = call_llm_json(PROMPT_P3_GEN, msg, max_tokens=8000)
    tcs = parsed.get("test_cases", []) if isinstance(parsed, dict) else parsed
    got = {tc.get("id"): tc for tc in tcs if isinstance(tc, dict) and tc.get("id") in want}

    missing = [want[i] for i in want if i not in got]
    if missing:
        retry_list = "\n".join(
            f'- {s["id"]} | {s["title"]} | priority: {s["priority"]} | covers: {", ".join(s.get("covers") or []) or "—"}'
            for s in missing
        )
        parsed2 = call_llm_json(
            PROMPT_P3_GEN,
            f"{plan_ctx}\n\nWrite ONE detailed test case for EACH of these scenarios, "
            f"keeping the exact `id`, `title` and `priority` given:\n{retry_list}",
            max_tokens=8000,
        )
        tcs2 = parsed2.get("test_cases", []) if isinstance(parsed2, dict) else parsed2
        for tc in tcs2:
            if isinstance(tc, dict) and tc.get("id") in want and tc.get("id") not in got:
                got[tc["id"]] = tc

    return [got[i] for i in want if i in got]  # preserve requested order


def generate_test_cases_in_batches(plan_ctx, scenarios, batch_size=6):
    """Generate ALL test cases as structured JSON, batch by batch.
    Returns (test_cases, missing_scenarios). Completeness is verified per
    batch by id — no completion-token heuristic, no duplicated content."""
    batches = [scenarios[i:i + batch_size] for i in range(0, len(scenarios), batch_size)]
    all_tcs, total = [], len(batches)
    progress = st.progress(0, text=f"Generating test cases… batch 1/{total}")

    for idx, batch in enumerate(batches):
        all_tcs.extend(_generate_tc_batch(plan_ctx, batch))
        progress.progress(
            (idx + 1) / total,
            text=f"Generating test cases… batch {idx + 2}/{total}" if idx + 1 < total else "✅ Done!",
        )
        if idx + 1 < total:
            time.sleep(1)  # be gentle with RPM limits between batches

    progress.empty()
    generated_ids = {tc.get("id") for tc in all_tcs}
    missing = [s for s in scenarios if s["id"] not in generated_ids]
    return all_tcs, missing


def tc_to_markdown(tcs):
    """Deterministic Markdown rendering of structured test cases.
    The JSON is the single source of truth — MD/CSV are DERIVED from it,
    so exports can never diverge from what is displayed."""
    parts = []
    for tc in tcs:
        pre = tc.get("preconditions", [])
        pre_md = "<br>".join(f"- {p}" for p in pre) if isinstance(pre, list) else str(pre)
        steps = tc.get("steps", [])
        def _step_md(i, s):
            if not isinstance(s, dict):
                return f"{i + 1}. {s}"
            line = f'{s.get("step_number", i + 1)}. {s.get("action", "")}'
            if s.get("expected"):
                line += f'\n   → *Expected:* {s["expected"]}'
            return line
        steps_md = "\n".join(_step_md(i, s) for i, s in enumerate(steps))
        covers = tc.get("covers") or []
        covers_md = ", ".join(str(c) for c in covers) if covers else "—"
        parts.append(
            f"""---
### {tc.get('id', 'TC-?')} — {tc.get('title', '')}

| Field | Detail |
|---|---|
| **ID** | {tc.get('id', '')} |
| **Technique** | {tc.get('technique', '')} |
| **Type** | {tc.get('type', '')} |
| **Priority** | {tc.get('priority', '')} |
| **Automation** | {tc.get('automation', '')} |
| **Covers** | {covers_md} |
| **Preconditions** | {pre_md} |

**🔢 Test Steps**
{steps_md}

**✅ Expected Result**
{tc.get('expected_result', '')}

**🔴 Failure Signature**
{tc.get('failure_signature', '')}
"""
        )
    return "\n".join(parts)


# ── DIFF-BASED MODIFICATION HELPERS ──────────────────────────────────────────
# Iterating on the plan/test cases applies add/remove/modify OPERATIONS to the
# structured state instead of regenerating everything — so human review work
# (✅/❌, priorities) and untouched content are always preserved.

def apply_scenario_ops(scenarios, review, ops):
    """Apply Phase 2 diff operations, preserving review state of untouched scenarios."""
    by_id = {s["id"]: s for s in scenarios}
    for rid in ops.get("remove") or []:
        by_id.pop(rid, None)
        review.pop(rid, None)
    for mod in ops.get("modify") or []:
        sid = mod.get("id")
        if sid in by_id:
            by_id[sid].update({k: v for k, v in mod.items() if k != "id" and v is not None})
            if mod.get("priority") and sid in review:
                review[sid]["priority"] = mod["priority"]
    next_id = max(by_id.keys(), default=0) + 1
    for add in ops.get("add") or []:
        add = dict(add)
        add["id"] = next_id
        by_id[next_id] = add
        review[next_id] = {"selected": True, "priority": add.get("priority", "Medium")}
        next_id += 1
    return list(by_id.values()), review


def apply_tc_ops(tcs, ops):
    """Apply Phase 3 diff operations to the structured test cases."""
    by_id = {tc.get("id"): tc for tc in tcs}
    for rid in ops.get("remove") or []:
        by_id.pop(rid, None)
    for mod in ops.get("modify") or []:
        tid = mod.get("id")
        if tid in by_id:
            by_id[tid].update({k: v for k, v in mod.items() if k != "id" and v is not None})
    nums = [int(m.group(1)) for i in by_id if i for m in [re.match(r"TC-(\d+)$", str(i))] if m]
    nxt = max(nums, default=0) + 1
    for add in ops.get("add") or []:
        add = dict(add)
        if not add.get("id") or add.get("id") in by_id:
            add["id"] = f"TC-{nxt}"
            nxt += 1
        by_id[add["id"]] = add
    return list(by_id.values())


# ── COVERAGE / TRACEABILITY ──────────────────────────────────────────────────

LANG_NAMES = {
    "fr": "French", "en": "English", "de": "German", "es": "Spanish", "it": "Italian",
    "pt": "Portuguese", "nl": "Dutch", "pl": "Polish", "ar": "Arabic", "ja": "Japanese",
    "zh-cn": "Chinese", "zh-tw": "Chinese", "ko": "Korean", "ru": "Russian", "tr": "Turkish",
}

def output_language_directive(text: str) -> str:
    """Detect the input language deterministically and return an explicit
    output-language directive to prepend to LLM messages.
    Research note: an instruction computed in code beats one inferred by the
    model — small models often drift to English with English system prompts."""
    try:
        from langdetect import detect
        code = detect(text[:2000])
        lang = LANG_NAMES.get(code, code)
        return f"OUTPUT LANGUAGE: {lang}. Write ALL user-facing text in {lang}.\n\n"
    except Exception:
        return "OUTPUT LANGUAGE: the same language as the User Story below.\n\n"


def normalize_scenarios(raw_scenarios):
    """Coerce scenario ids to unique ints (models sometimes return strings or
    duplicate ids) — apply_scenario_ops and the review state rely on int keys."""
    out, seen = [], set()
    for i, s in enumerate(raw_scenarios or [], 1):
        if not isinstance(s, dict):
            continue
        try:
            sid = int(s.get("id"))
        except (TypeError, ValueError):
            sid = i
        while sid in seen:
            sid += 1
        seen.add(sid)
        s = dict(s); s["id"] = sid
        out.append(s)
    return out


def normalize_rules(raw_rules):
    """Normalize business rules to [{"id": "BR-1", "rule": "..."}] whatever the model returned."""
    out = []
    for i, r in enumerate(raw_rules or [], 1):
        if isinstance(r, dict):
            out.append({"id": r.get("id") or f"BR-{i}", "rule": r.get("rule") or r.get("text") or str(r)})
        else:
            out.append({"id": f"BR-{i}", "rule": str(r)})
    return out


def coverage_gaps(rules, scenarios, review):
    """Business rules not covered by any SELECTED scenario (traceability check)."""
    covered = set()
    for s in scenarios:
        if review.get(s["id"], {}).get("selected", True):
            for br in s.get("covers") or []:
                covered.add(str(br).strip().upper())
    return [r for r in rules if str(r.get("id", "")).strip().upper() not in covered]

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
    st.title("🧪 QAForge — AI Test Case Generator V.0.6")

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
    temperature = st.slider(
        "🌡️ Temperature",
        min_value=0.0, max_value=1.0,
        value=st.session_state.get("temperature", 0.2),
        step=0.05,
        help="0 = reproducible (ISO 29119-4)  ·  0.2 = balanced default  ·  >0.5 = creative but less stable JSON"
    )
    st.session_state.temperature = temperature

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
    "p1_validated": False, "p2_validated": False,
    "us_submitted": False, "p1_context": "",
    "structured_test_cases": None,          # ← single source of truth for Phase 3
    "p3_chat_log": [], "p3_missing": [], "p3_plan_ctx": "",
    "p1_questions": [], "p1_answers": {}, "p1_summary": "", "p1_user_story": "", "p1_raw_prompt": "", "p1_extra_ctx": "", "p1_iso_techniques": [], "p1_chat_msgs": [],
    "p1_business_rules": [], "p1_actors": [], "p1_screens": [],
    "temperature": 0.2, "p2_scenarios": [], "p2_summary": "", "p2_review": {}, "p2_last_reply": "", "p2_overlaps": [],
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PROMPTS ───────────────────────────────────────────────────────────────────
PROMPT_P1_QUESTIONS = """
You are a Senior QA Analyst and Requirements Engineer with 10+ years of experience
applying ISO/IEC/IEEE 29119 standards in industrial software testing projects.

## YOUR ROLE
Analyze the provided User Story and:
1. FIRST identify which test design techniques apply to these requirements.
2. THEN generate clarifying questions to resolve ambiguities before test planning.

## TECHNIQUE IDENTIFICATION (Step 1)
Before writing any question, reason about the requirements and identify applicable techniques.

Specification-based techniques (ISO/IEC/IEEE 29119-4):
- Boundary Value Analysis (BVA) → if numeric fields, ranges, limits, or thresholds exist
- Equivalence Partitioning (EP) → if inputs can be grouped into valid/invalid classes
- Decision Table Testing (DT) → if complex multi-condition logic (IF x AND y THEN z)
- State Transition Testing (ST) → if the feature has lifecycle states (draft/active/archived, open/closed)
- Combinatorial Testing / Pairwise → if 3+ independent parameters or options interact
  (pairwise keeps the number of combinations manageable)

Experience-based:
- Error Guessing (EG) → ISO 29119-4 experience-based technique; likely failure points
- Exploratory Testing (ET) → test PRACTICE (not a 29119-4 design technique); included for pragmatic coverage

Feature interactions:
- Function Combinations (FC) → interactions between multiple independent features/modules

Include Error Guessing and Exploratory Testing unless they are clearly irrelevant to the feature.

## QUESTION STRATEGY
- Ask ONLY questions whose answer would meaningfully change the test strategy
- Simple, unambiguous user stories: fewer questions (3–5)
- Complex user stories (multi-step flows, payments, permissions, integrations): more questions (up to 15)
- Every question must target a REAL ambiguity — never ask what is already stated
- 1 question = 1 specific piece of missing information
- Never combine two questions into one

## QUESTION TYPES
- "boolean" → yes/no questions (e.g. "Is this field mandatory?")
- "multiple_choice" → when there are 2–5 known possible answers
- "text" → when the answer is a free value (limit, rule, description)

## CATEGORIES
- Functional | Validation | Error Handling | Edge Cases | System / Dependencies

## VISUAL ANALYSIS
When [IMAGE_N — filename] markers appear in the document context:
- Identify the type of visual (wireframe, UI screenshot, form mockup, flow diagram, table, error state)
- Extract ALL visible form fields and their apparent constraints (required, format, length)
- Note navigation elements, buttons, links and the flows they imply
- Identify visible validation rules, error messages, or status indicators
- Treat every visual as a functional specification — it defines behaviour, not just appearance
- If a visual contradicts or extends the written text, flag it in your questions
- Reference visuals explicitly in your questions (e.g. "In the login screen shown in [IMAGE_1]...")

## OUTPUT FORMAT (STRICT JSON — no markdown, no explanation)
The "analysis" field comes FIRST: use it as your reasoning space (3-5 sentences identifying
inputs, constraints, states and condition logic) BEFORE committing to techniques and questions.
It is internal — it is never shown to the user.
{
  "analysis": "3-5 sentences of reasoning about inputs, constraints, states, condition logic",
  "summary": "2-3 sentence summary of your current understanding of the feature",
  "applicable_iso_techniques": [
    {"name": "Boundary Value Analysis", "rationale": "Password field has min/max character constraints"},
    {"name": "Decision Table Testing", "rationale": "Login logic varies by role AND account status"}
  ],
  "key_business_rules": [
    {"id": "BR-1", "rule": "Password must be 8–128 characters"},
    {"id": "BR-2", "rule": "Account locks after 5 failed login attempts"}
  ],
  "actors": ["User", "Admin"],
  "screens_identified": ["Login screen — [IMAGE_1]", "Dashboard — [IMAGE_2]"],
  "questions": [
    {
      "id": 1, "category": "Functional", "type": "boolean",
      "question": "Is the user required to be logged in to access this feature?"
    },
    {
      "id": 2, "category": "Validation", "type": "multiple_choice",
      "question": "Which email formats are accepted?",
      "options": ["All valid email formats", "Professional emails only", "Specific domain only"]
    },
    {
      "id": 3, "category": "Edge Cases", "type": "text",
      "question": "What is the maximum character length allowed for this field?"
    }
  ]
}

HARD CONSTRAINTS:
- Output ONLY valid JSON. No markdown fences, no preamble.
- Do NOT generate test cases, scenarios, or test plan content.
- Do NOT invent business rules not present in the User Story or attached visuals.
- Business rule ids MUST be sequential: BR-1, BR-2, … They are used later for coverage traceability.
- If no visuals are present, leave screens_identified as an empty array.
- Write all text fields (summary, questions, business rules, actors) in the SAME LANGUAGE as the User Story.
"""

PROMPT_P1_CHAT = """You are a Senior QA Analyst conducting a requirements clarification session.

## YOUR ROLE
Review the current state of the session (summary, business rules, questions, answers so far)
and the user's message. Your answer is APPLIED to the session state — corrections must be
reflected in the structured fields, not just acknowledged in prose.

## OUTPUT FORMAT (STRICT JSON object — no markdown, no preamble)
{
  "reply": "conversational answer to the user (under 150 words)",
  "updated_summary": null,
  "updated_business_rules": null,
  "new_questions": null
}

Rules:
- "reply" is ALWAYS required. Same language as the user's message.
- If the user CORRECTS a misunderstanding → rewrite the WHOLE summary in "updated_summary" (else null).
- If a correction changes, adds or removes business rules → return the FULL updated list in
  "updated_business_rules" as [{"id": "BR-1", "rule": "..."}], keeping existing ids stable
  and continuing the BR-x sequence for new rules (else null).
- If a critical NEW ambiguity emerges → add typed questions in "new_questions" using the same
  structure as the initial questions ({"id", "category", "type", "question", "options"?}),
  with ids continuing the existing sequence (else null).
- If the user simply answers questions or asks for clarification → "reply" only, other fields null.
- Never re-ask questions already answered. Never invent business rules the user did not state or imply.
- If all critical questions are answered, say so in "reply": the user can proceed to Phase 2.
"""

PROMPT_P2 = """
You are a Lead QA Engineer specialising in test design using ISO/IEC/IEEE 29119-4
and experience-based techniques.

## YOUR ROLE
Generate a comprehensive TEST CHECKLIST as scenario TITLES ONLY with metadata.
FORBIDDEN: steps, preconditions, or expected results in this phase.

## COVERAGE — test design techniques
The techniques identified in Phase 1 are provided in the context.
For EACH applicable technique, generate dedicated scenarios:

- **Equivalence Partitioning** → valid class, invalid class scenarios
- **Boundary Value Analysis (BVA)** → cover min-1, min, max, max+1 for every constrained field.
  GROUPING RULE: one scenario MAY cover several boundary values of the SAME field
  (e.g. "BVA — Password length at boundaries (7, 8, 128, 129 chars)") — the exact
  values are detailed in Phase 3 steps. Do NOT create 4 scenarios per field.
- **Decision Table Testing** → one scenario per significant condition combination
  (if 3+ independent conditions interact, prefer PAIRWISE coverage of combinations
  instead of exhaustive enumeration)
- **State Transition Testing** → each state, each valid/invalid transition
- **Error Guessing** → likely failure points (empty inputs, nulls, concurrent access, special chars)
- **Exploratory Testing** → at least 1 scenario covering unexpected user paths
- **Function Combinations** → interactions between identified features/modules

## TRACEABILITY (MANDATORY)
The context lists numbered business rules (BR-1, BR-2, …).
- Each scenario MUST declare which business rules it covers in its "covers" array (use [] if none).
- EVERY business rule must be covered by at least one scenario. Do not leave a rule uncovered.

## SCENARIO TITLE FORMAT
Prefix each title with its technique abbreviation:
- "BVA — Login with password at maximum length (128 chars)"
- "DT — Admin user with expired account attempts login"
- "ST — Password reset token transitions from valid to expired state"
- "EP — Registration with invalid email format (missing @ symbol)"
- "FC — Login followed immediately by password change in same session"
- "EG — Submit form with all fields empty"
- "ET — Navigate through checkout by skipping optional steps in random order"
- Happy Path and Alternate Flow titles: no prefix needed.

## OUTPUT FORMAT (STRICT JSON — no markdown, no explanation)
{
  "summary": "2-3 sentence feature summary highlighting testing strategy and techniques applied",
  "scenarios": [
    {"id": 1, "title": "Successful login with valid credentials", "category": "Happy Path", "priority": "Very High", "covers": ["BR-1"]},
    {"id": 2, "title": "BVA — Login with password length at boundaries (7, 8, 128, 129 chars)", "category": "BVA", "priority": "High", "covers": ["BR-1"]},
    {"id": 3, "title": "DT — Premium user with active subscription accesses restricted content", "category": "Decision Table", "priority": "Very High", "covers": ["BR-2", "BR-3"]}
  ],
  "coverage_check": [
    {"rule": "BR-1", "covered_by": [1, 2]},
    {"rule": "BR-2", "covered_by": [3]},
    {"rule": "BR-3", "covered_by": [3]}
  ],
  "potential_overlaps": [[2, 5]]
}

"coverage_check" comes AFTER "scenarios" on purpose: fill it by RE-READING your own
scenario list rule by rule. If a rule has an empty "covered_by", go back and add a scenario.
"potential_overlaps": pairs of scenario ids that may test the same thing — flag them
honestly so the human reviewer can consolidate (empty array if none).

## CATEGORIES (use exactly these values):
Happy Path | Alternate Flow | BVA | Equivalence | Decision Table | State Transition | Negative | Edge Case | Security | Non-Functional | Function Combination | Error Guessing

## PRIORITIES (use exactly these values):
Very High | High | Medium | Low

## HARD CONSTRAINTS — in priority order (if constraints conflict, the higher one wins)
1. TRACEABILITY: every business rule covered by at least one scenario.
2. TECHNIQUE COMPLETENESS: apply ALL relevant techniques — do NOT skip one to reduce count.
3. SCENARIO BUDGET: target 6–20 scenarios based on complexity
   (Simple 1–2 flows: 6–9 · Moderate 3–5 flows + validation: 10–15 · Complex multi-actor/payments/permissions: 15–20).
   You MAY exceed 20 only if constraints 1 and 2 genuinely require it.
- Output ONLY valid JSON. No markdown fences, no preamble.
- Do NOT invent scenarios to reach a quota — every scenario must cover a real test need.
- Assign realistic priorities based on business impact.
- Write all text fields (summary, scenario titles) in the SAME LANGUAGE as the User Story.
"""

# Few-shot example appended ONLY for smaller models (Groq / Mistral / OpenRouter).
# Evidence: few-shot stabilises weaker models but can anchor and degrade strong ones
# (zero-shot outperformed few-shot for GPT-4 on industrial user stories) — so
# Gemini / OpenAI stay zero-shot.
PROMPT_P2_FEWSHOT = """

## EXAMPLE (illustrative only — adapt to the actual feature, do NOT copy)
Input: "As a user, I want to reset my password via an emailed link valid 24h.
BR-1: link expires after 24h. BR-2: new password must be 8–128 chars."
Output:
{
  "summary": "Password reset via emailed time-limited link; coverage focuses on link lifecycle (ST), password constraints (BVA) and failure handling (EG).",
  "scenarios": [
    {"id": 1, "title": "Successful password reset via valid emailed link", "category": "Happy Path", "priority": "Very High", "covers": ["BR-1"]},
    {"id": 2, "title": "ST — Reset link transitions from valid to expired after 24h", "category": "State Transition", "priority": "High", "covers": ["BR-1"]},
    {"id": 3, "title": "BVA — New password length at boundaries (7, 8, 128, 129 chars)", "category": "BVA", "priority": "High", "covers": ["BR-2"]},
    {"id": 4, "title": "EG — Submit reset form with empty password fields", "category": "Error Guessing", "priority": "Medium", "covers": ["BR-2"]}
  ],
  "coverage_check": [
    {"rule": "BR-1", "covered_by": [1, 2]},
    {"rule": "BR-2", "covered_by": [3, 4]}
  ],
  "potential_overlaps": []
}
"""

# Self-Refine pattern (generate → critique → refine) reusing the diff machinery:
# the model reviews ITS OWN plan against the business rules and returns operations.
PROMPT_P2_REVIEW = """
You are a Lead QA Engineer performing a CRITICAL SELF-REVIEW of a test checklist.
You receive the business rules, the full context and the CURRENT scenario list (JSON).
Review it like a demanding peer reviewer:

1. COVERAGE: is any business rule weakly covered or uncovered? Any obvious risk
   (security, concurrency, data validation) with no scenario?
2. DUPLICATES: do any scenarios test essentially the same thing? Propose removing
   or merging the weakest one.
3. QUALITY: are titles concrete and testable (exact values, conditions, states)?
   Are priorities realistic given business impact?

Return ONLY the operations that IMPROVE the plan — if the plan is already solid,
say so in "reply" and return empty arrays. Do NOT inflate the plan.

## OUTPUT FORMAT (STRICT JSON object — no markdown, no preamble)
{
  "reply": "2-4 sentence review summary: what was weak and what you changed (same language as the plan)",
  "add": [{"title": "…", "category": "…", "priority": "…", "covers": ["BR-x"]}],
  "remove": [ids of duplicate/valueless scenarios],
  "modify": [{"id": n, "title": "sharper title with exact values"}]
}
"""

PROMPT_P2_MODIFY = """
You are a Lead QA Engineer maintaining a test checklist that a human is reviewing.
You receive the CURRENT scenario list (JSON) and a modification request from the user.
Return ONLY the operations needed — do NOT regenerate the whole plan. The human's
review work (selections, priorities) on untouched scenarios must survive your changes.

## OUTPUT FORMAT (STRICT JSON object — no markdown, no preamble)
{
  "reply": "1-2 sentence summary of what you changed (same language as the user)",
  "add": [{"title": "EG — …", "category": "Error Guessing", "priority": "Medium", "covers": ["BR-2"]}],
  "remove": [4, 7],
  "modify": [{"id": 2, "priority": "Very High"}]
}

Rules:
- "add": new scenarios WITHOUT id (ids are assigned by the system). Use the same title
  prefixes (BVA —, DT —, ST —, EP —, FC —, EG —, ET —), categories, priorities and
  "covers" (BR-x ids) conventions as the existing plan.
- "remove": ids of scenarios to delete.
- "modify": only the id plus the fields that change (title, category, priority, covers).
- Untouched scenarios must NOT appear anywhere in your output.
- Use empty arrays when nothing applies. If the user only asks a question, answer in
  "reply" and leave the three arrays empty.
- If the request is AMBIGUOUS or too vague to translate into precise operations
  (e.g. "improve the plan"), do NOT guess: ask a clarifying question in "reply"
  and leave the three arrays empty.
"""

PROMPT_P3_GEN = """
You are a Senior QA Test Architect writing execution-ready test cases aligned with
ISO/IEC/IEEE 29119-4 and experience-based test design techniques.

## GUIDELINES
- Each test case derives directly from the technique in its scenario title prefix
  (BVA, DT, ST, EP, FC, EG, ET — no prefix = Happy Path / Alternate Flow).
- Real, concrete test data in steps. If a value is unknown, embed:
  "⚠️ Assumption: […] — confirm with PO." inside the relevant step.
- For BVA: state the EXACT boundary value tested in the expected result.
- For Decision Table: state the EXACT combination of conditions tested.
- Use terminology strictly consistent with the requirements document
  (consistent terminology → higher recall against reference tests).
- Write ALL text content in the SAME LANGUAGE as the requirements.

## OUTPUT FORMAT (STRICT JSON object — no markdown, no preamble)
{
  "test_cases": [
    {
      "id": "TC-1",
      "title": "…",
      "technique": "BVA | Decision Table | Equivalence | State Transition | Error Guessing | Exploratory | Function Combination | Happy Path | Alternate Flow",
      "type": "Happy Path | Alternate | BVA | Equivalence | Decision Table | State Transition | Negative | Edge Case | Security | Function Combination | Error Guessing | Exploratory",
      "priority": "Very High | High | Medium | Low",
      "automation": "Good candidate" or "Manual only — (reason)",
      "covers": ["BR-1"],
      "preconditions": ["state, role, data"],
      "steps": [{"step_number": 1, "action": "action with exact data or boundary value", "expected": "observable intermediate outcome (OPTIONAL — only when the step has one)"}],
      "expected_result": "exact observable outcome in natural language",
      "failure_signature": "what the tester sees on failure"
    }
  ]
}

## HARD CONSTRAINTS
- Generate EXACTLY one test case per requested scenario — no more, no less.
- Keep the EXACT `id`, `title`, `priority` and `covers` provided for each scenario.
- Per-step "expected" is OPTIONAL: include it only when a step has an observable
  intermediate outcome (test-management tools like Squash TM / TestLink use it).
- Do NOT add commentary, summaries, or extra keys.
"""

PROMPT_P3_MODIFY = """
You are a Senior QA Test Architect maintaining a set of structured test cases.
You receive the CURRENT test cases (JSON) and a user request.
Return ONLY operations — never regenerate untouched test cases, and never put
test case content in "reply".

## OUTPUT FORMAT (STRICT JSON object — no markdown, no preamble)
{
  "reply": "1-2 sentence answer / summary of changes (same language as the user)",
  "add": [ full test case objects WITHOUT id — same schema as existing test cases ],
  "remove": ["TC-4"],
  "modify": [{"id": "TC-2", "expected_result": "…"}]
}

Rules:
- If the user only asks a QUESTION (e.g. "explain TC-3"), answer in "reply" and leave
  add/remove/modify as empty arrays — the exported test cases must NOT be polluted.
- "modify": id plus ONLY the fields that change. If "steps" or "preconditions" change,
  return the COMPLETE new array for that field.
- "add": ids are assigned by the system — do not include them.
- Use empty arrays when nothing applies.
- If the request is AMBIGUOUS or too vague to translate into precise operations,
  do NOT guess: ask a clarifying question in "reply" and leave the arrays empty.
"""

# ── HELP TEXTS ────────────────────────────────────────────────────────────────
# All user-visible tooltip/help strings centralised here.
# Edit this block to update tooltips without touching UI logic.

HELP_TEXTS = {

    "phase1": (
        "PHASE 1 — Analysis & Clarification\n"
        "─────────────────────────────────────\n"
        "What happens here:\n"
        "  • Paste your User Story (max 20,000 chars) and attach files (PDF, DOCX, images)\n"
        "  • The AI identifies applicable ISO 29119-4 test techniques (BVA, Decision Table…)\n"
        "  • The AI asks typed clarifying questions (Yes/No · Multiple choice · Free text)\n"
        "  • Answer all questions, then validate to unlock Phase 2\n"
        "\n"
        "Output: structured context (business rules, actors, ISO techniques) passed to Phase 2\n"
        "\n"
        "Tip: the more detailed your User Story + Acceptance Criteria, the better the coverage."
    ),

    "phase2": (
        "PHASE 2 — Test Checklist (Scenario Titles)\n"
        "──────────────────────────────────────\n"
        "What happens here:\n"
        "  • The AI generates scenario titles using ISO/IEC/IEEE 29119-4 & experience-based techniques\n"
        "  • Each scenario is prefixed by its technique:\n"
        "      BVA  — Boundary Value Analysis (min-1, min, max, max+1)\n"
        "      DT   — Decision Table (multi-condition logic combinations / pairwise)\n"
        "      ST   — State Transition (lifecycle states & transitions)\n"
        "      EP   — Equivalence Partitioning (valid / invalid input classes)\n"
        "      FC   — Function Combination (interactions between features)\n"
        "      EG   — Error Guessing (likely failure points from experience)\n"
        "      (none) — Happy Path / Alternate Flow\n"
        "  • Each scenario declares the business rules (BR-x) it covers — uncovered rules are flagged\n"
        "  • Accept ✅ or reject ❌ each scenario, adjust priorities\n"
        "  • Chat modifications apply as a DIFF — your ✅/❌ and priority choices are preserved\n"
        "  • Validate the plan to unlock Phase 3\n"
        "\n"
        "Output: a prioritised, traceable list of scenarios passed to Phase 3\n"
        "\n"
        "Note: coverage techniques favour exhaustiveness — some overlap is normal."
    ),

    "phase3": (
        "PHASE 3 — Full Test Cases & Export\n"
        "────────────────────────────────────\n"
        "What happens here:\n"
        "  • The AI writes execution-ready test cases for every validated scenario\n"
        "    (generated as structured data, verified for completeness scenario by scenario)\n"
        "  • Each test case contains:\n"
        "      Technique  : test design method used (BVA, DT, ST, EP, FC, EG…)\n"
        "      Type       : Happy Path / Negative / Edge Case / Security…\n"
        "      Priority   : Very High / High / Medium / Low\n"
        "      Automation : Good candidate ✅ or Manual only 🖐️\n"
        "      Steps      : numbered actions with real test data\n"
        "      Expected Result & Failure Signature\n"
        "  • Chat requests modify test cases IN PLACE (questions never pollute the export)\n"
        "  • Export instantly in Markdown, JSON, or CSV (Excel / Jira compatible)\n"
        "\n"
        "⚠️ This tool optimises for exhaustive coverage (high recall).\n"
        "   A human review pass to remove duplicates is normal and expected."
    ),

    "phase1_locked": (
        "🔒 Locked — complete Phase 1 first.\n"
        "Submit your User Story and answer all clarifying questions."
    ),

    "phase2_locked": (
        "🔒 Locked — complete Phase 2 first.\n"
        "Validate your test plan before generating full test cases."
    ),
}

# ── FILE PARSING ──────────────────────────────────────────────────────────────
ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp", "pdf", "txt", "md", "docx"]
MAX_FILES = 5
MAX_CHARS = 80000

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

# ── CSV BUILDER ───────────────────────────────────────────────────────────────
def _csv_safe(v):
    """Neutralise spreadsheet formula injection (cells starting with = + - @)."""
    s = str(v)
    return "'" + s if s[:1] in ("=", "+", "-", "@") else s

def build_csv(data):
    if not data: return ""
    out = io.StringIO()
    fields = ["id","title","technique","type","priority","automation","covers","preconditions","steps","expected_result","failure_signature"]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        r = dict(row)
        pre = r.get("preconditions", [])
        r["preconditions"] = " | ".join(pre) if isinstance(pre, list) else str(pre)
        cov = r.get("covers", [])
        r["covers"] = " | ".join(str(c) for c in cov) if isinstance(cov, list) else str(cov)
        steps = r.get("steps", [])
        if steps and isinstance(steps, list):
            def _step_csv(s):
                if not isinstance(s, dict):
                    return str(s)
                base = f"{s.get('step_number','')}.{s.get('action','')}"
                return f"{base} [expected: {s['expected']}]" if s.get("expected") else base
            r["steps"] = " | ".join(_step_csv(s) for s in steps)
        else:
            r["steps"] = str(steps)
        writer.writerow({k: _csv_safe(r.get(k, "")) for k in fields})
    return out.getvalue()

# ── TAB BAR ───────────────────────────────────────────────────────────────────
def render_tab_bar():
    pr, ap = st.session_state.phase_reached, st.session_state.active_phase
    phase_meta = {
        1: ("Analysis",   HELP_TEXTS["phase1"],      HELP_TEXTS["phase1_locked"]),
        2: ("Test Checklist",  HELP_TEXTS["phase2"],      HELP_TEXTS["phase2_locked"]),
        3: ("Test Cases", HELP_TEXTS["phase3"],      HELP_TEXTS["phase2_locked"]),
    }
    cols = st.columns(3)
    for i, (n, (label, help_active, help_locked)) in enumerate(phase_meta.items()):
        with cols[i]:
            if n > pr:
                st.button(f"🔒 Phase {n} — {label}", key=f"tab_{n}", disabled=True,
                          use_container_width=True, help=help_locked)
            else:
                prefix = "▶" if n == ap else "✅"
                if st.button(f"{prefix} Phase {n} — {label}", key=f"tab_{n}",
                              use_container_width=True,
                              type="primary" if n == ap else "secondary",
                              help=help_active):
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
        us_input = st.text_area("User Story + Acceptance Criteria", height=180, max_chars=20000,
            placeholder="As a [user], I want to [action] so that [benefit].\n\nAcceptance Criteria:\n- ...")
        if us_input: st.caption(f"{len(us_input):,}/20,000 characters")

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
                lang_directive = output_language_directive(us_input)
                st.session_state.lang_directive = lang_directive
                prompt = f"{lang_directive}Please analyze the following User Story:\n\n{us_input}"

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
                        parsed = extract_json(raw)
                        st.session_state.p1_questions = parsed.get("questions", [])
                        st.session_state.p1_summary = parsed.get("summary", "")
                        # Store enriched fields — business rules normalised to {id, rule}
                        st.session_state.p1_business_rules = normalize_rules(parsed.get("key_business_rules", []))
                        st.session_state.p1_actors = parsed.get("actors", [])
                        st.session_state.p1_screens = parsed.get("screens_identified", [])
                        st.session_state.p1_iso_techniques = parsed.get("applicable_iso_techniques", [])
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
                    for r in rules: st.markdown(f"- **{r['id']}** — {r['rule']}")
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
            # ISO techniques display
            iso_techs = st.session_state.get("p1_iso_techniques", [])
            if iso_techs:
                with st.expander("🔬 ISO 29119-4 techniques identified", expanded=False):
                    for t in iso_techs:
                        st.markdown(f"- **{t['name']}** — {t.get('rationale', '')}")

        st.markdown("### 🔍 Clarifying Questions")
        st.caption("Answer the questions below — click or type as appropriate.")

        questions = st.session_state.p1_questions
        answers = st.session_state.p1_answers

        # Group by category
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
                    cur_rules = "\n".join(
                        f"- {r['id']}: {r['rule']}" for r in st.session_state.p1_business_rules
                    )
                    ctx_msg = (
                        f"Current understanding: {st.session_state.p1_summary}\n\n"
                        f"Current business rules:\n{cur_rules}\n\n"
                        f"Questions and answers so far:\n{cur_answers}\n\n"
                        f"User says: {p1_reply}"
                    )
                    raw = call_llm(st.session_state.p1_chat_msgs[:-1], PROMPT_P1_CHAT, ctx_msg, max_tokens=2000)
                    # Apply structured updates so corrections actually reach Phase 2
                    try:
                        ops = extract_json(raw)
                        reply_text = ops.get("reply") or "(updated)"
                        if ops.get("updated_summary"):
                            st.session_state.p1_summary = ops["updated_summary"]
                            reply_text += "\n\n📋 *Summary updated.*"
                        if ops.get("updated_business_rules"):
                            st.session_state.p1_business_rules = normalize_rules(ops["updated_business_rules"])
                            reply_text += "\n\n⚖️ *Business rules updated.*"
                        if ops.get("new_questions"):
                            existing_ids = {q["id"] for q in st.session_state.p1_questions}
                            added = [q for q in ops["new_questions"] if q.get("id") not in existing_ids]
                            st.session_state.p1_questions.extend(added)
                            if added:
                                reply_text += f"\n\n🔍 *{len(added)} new question(s) added.*"
                    except (ValueError, json.JSONDecodeError):
                        reply_text = raw  # graceful fallback: plain conversational answer
                    st.session_state.p1_chat_msgs.append({"role": "assistant", "content": reply_text})
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
                rules_ctx = "\nKey Business Rules (numbered — used for coverage traceability):\n" + "\n".join(
                    f"- {r['id']}: {r['rule']}" for r in st.session_state.p1_business_rules
                )
            screens_ctx = ""
            if st.session_state.get("p1_screens"):
                screens_ctx = "\nScreens identified:\n" + "\n".join(
                    f"- {s}" for s in st.session_state.p1_screens
                )
            iso_ctx = ""
            if st.session_state.get("p1_iso_techniques"):
                iso_ctx = "\nTest design techniques to apply:\n" + "\n".join(
                    f"- {t['name']}: {t.get('rationale', '')}" for t in st.session_state.p1_iso_techniques
                )
            # Clarification chat: user corrections take precedence over the initial analysis
            chat_ctx = ""
            if st.session_state.p1_chat_msgs:
                transcript = "\n".join(
                    f"{m['role'].upper()}: {m['content']}" for m in st.session_state.p1_chat_msgs[-12:]
                )
                chat_ctx = (
                    "\nClarification discussion (USER corrections take precedence over the initial analysis):\n"
                    f"{transcript}\n"
                )

            ctx = (
                f"{st.session_state.get('lang_directive', '')}"
                f"User Story:\n{st.session_state.p1_user_story}\n\n"
                f"Requirements Analysis Summary:\n{st.session_state.p1_summary}"
                f"{rules_ctx}{screens_ctx}{iso_ctx}{chat_ctx}\n\n"
                f"Clarification Q&A:\n{answers_text}\n\n"
                f"Generate the test plan (titles only)."
            )
            # Few-shot ONLY for smaller models — strong models perform better zero-shot
            p2_prompt = PROMPT_P2 + (
                PROMPT_P2_FEWSHOT if st.session_state.provider in ("Groq", "Mistral", "OpenRouter") else ""
            )
            with st.spinner("📋 Generating test plan…"):
                try:
                    parsed_p2 = call_llm_json(p2_prompt, ctx, max_tokens=5000)
                    st.session_state.p2_scenarios = normalize_scenarios(parsed_p2.get("scenarios", []))
                    st.session_state.p2_summary = parsed_p2.get("summary", "")
                    st.session_state.p2_overlaps = parsed_p2.get("potential_overlaps", []) or []
                    st.session_state.p2_review = {}
                    st.session_state.p2_last_reply = ""
                    st.session_state.p2_validated = False
                    st.session_state.structured_test_cases = None
                    st.session_state.p3_chat_log = []
                    st.session_state.p3_missing = []
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
    st.markdown('<div class="badge b2">📋 Phase 2 — Lead QA Engineer: Test Checklist</div>', unsafe_allow_html=True)

    scenarios = st.session_state.get("p2_scenarios", [])

    if scenarios:
        # ── Sync review state (NEVER reset — preserves human ✅/❌/priority work) ──
        review = st.session_state.p2_review
        scenario_ids = {s["id"] for s in scenarios}
        for s in scenarios:
            review.setdefault(s["id"], {"selected": True, "priority": s.get("priority", "Medium")})
        for rid in list(review.keys()):
            if rid not in scenario_ids:
                del review[rid]

        CAT_ICONS = {
            "Happy Path": "✅", "Alternate Flow": "🔄", "BVA": "🔢",
            "Equivalence": "🔀", "Negative": "❌", "Edge Case": "⚠️",
            "Security": "🔒", "Non-Functional": "⚙️"
        }

        st.markdown(f"📋 **{st.session_state.get('p2_summary', 'Test Checklist')}**")

        if st.session_state.get("p2_last_reply"):
            st.info(f"🤖 {st.session_state.p2_last_reply}")

        # ── Coverage traceability panel (BR-x ↔ scenarios) ────────────────────
        rules = st.session_state.get("p1_business_rules", [])
        if rules:
            gaps = coverage_gaps(rules, scenarios, review)
            if gaps:
                st.warning(
                    "🕳️ **Coverage gaps — business rules not covered by any selected scenario:**\n"
                    + "\n".join(f"- **{g['id']}** — {g['rule']}" for g in gaps)
                    + "\n\n*Ask for scenarios in the chat below, or accept the gap knowingly.*"
                )
            else:
                st.success("✅ All identified business rules are covered by at least one selected scenario.")

        # ── Self-flagged potential duplicates (recall-first → consolidate by hand) ──
        overlaps = st.session_state.get("p2_overlaps", [])
        if overlaps:
            valid_pairs = [
                o for o in overlaps
                if isinstance(o, (list, tuple)) and len(o) == 2
                and o[0] in scenario_ids and o[1] in scenario_ids
            ]
            if valid_pairs:
                pairs = " · ".join(f"#{a} ↔ #{b}" for a, b in valid_pairs)
                st.caption(f"♻️ Potential overlaps flagged by the AI (consider consolidating): {pairs}")

        # ── AI self-review (Self-Refine pattern: critique own plan vs business rules) ──
        if st.button("🔍 AI self-review of the plan", use_container_width=True, key="p2_selfreview",
                     help="The AI critically reviews its own checklist against the business rules "
                          "(coverage, duplicates, title quality) and applies improvements as a diff — "
                          "your ✅/❌ selections are preserved."):
            with st.spinner("Self-reviewing the plan…"):
                try:
                    current_plan = json.dumps(
                        [{k: s.get(k) for k in ("id", "title", "category", "priority", "covers")} for s in scenarios],
                        ensure_ascii=False,
                    )
                    rules_txt = "\n".join(f"- {r['id']}: {r['rule']}" for r in st.session_state.get("p1_business_rules", []))
                    msg = (
                        f"{st.session_state.get('lang_directive', '')}"
                        f"BUSINESS RULES:\n{rules_txt}\n\n"
                        f"CONTEXT:\n{st.session_state.p1_context}\n\n"
                        f"CURRENT PLAN (JSON):\n{current_plan}\n\n"
                        f"Perform the critical self-review now."
                    )
                    ops = call_llm_json(PROMPT_P2_REVIEW, msg, max_tokens=4000)
                    new_scenarios, new_review = apply_scenario_ops(scenarios, st.session_state.p2_review, ops)
                    st.session_state.p2_scenarios = new_scenarios
                    st.session_state.p2_review = new_review
                    n_add = len(ops.get("add") or []); n_rem = len(ops.get("remove") or []); n_mod = len(ops.get("modify") or [])
                    st.session_state.p2_last_reply = (
                        f"🔍 Self-review: {ops.get('reply', '')} ({n_add} added · {n_mod} modified · {n_rem} removed)"
                    )
                    st.rerun()
                except Exception as e: handle_error(e)

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
                covers = s.get("covers") or []
                if covers:
                    st.caption("covers: " + ", ".join(str(c) for c in covers))
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
        st.info("No scenarios yet — validate Phase 1 to generate the test plan.")

    # ── Chat modifications — applied as a DIFF, review state preserved ────────
    st.markdown("#### 💬 Request modifications")
    st.caption("Add / remove / modify scenarios — your ✅/❌ selections and priorities are preserved.")
    reply2 = st.chat_input("Add scenarios, change coverage, request modifications…", key="p2_chat")
    if reply2:
        with st.spinner("Updating plan…"):
            try:
                current_plan = json.dumps(
                    [{k: s.get(k) for k in ("id", "title", "category", "priority", "covers")} for s in scenarios],
                    ensure_ascii=False,
                )
                rules_txt = "\n".join(f"- {r['id']}: {r['rule']}" for r in st.session_state.get("p1_business_rules", []))
                msg = (
                    f"{st.session_state.get('lang_directive', '')}"
                    f"CURRENT PLAN (JSON):\n{current_plan}\n\n"
                    f"BUSINESS RULES:\n{rules_txt}\n\n"
                    f"CONTEXT:\n{st.session_state.p1_context}\n\n"
                    f"USER REQUEST:\n{reply2}"
                )
                ops = call_llm_json(PROMPT_P2_MODIFY, msg, max_tokens=4000)
                new_scenarios, new_review = apply_scenario_ops(
                    scenarios, st.session_state.p2_review, ops
                )
                st.session_state.p2_scenarios = new_scenarios
                st.session_state.p2_review = new_review
                st.session_state.p2_last_reply = ops.get("reply", "")
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

        # Pre-assign stable TC ids — completeness is verified against them
        tc_scenarios = [
            {
                "id": f"TC-{i + 1}",
                "title": s["title"],
                "priority": review.get(s["id"], {}).get("priority", s.get("priority", "Medium")),
                "covers": s.get("covers") or [],
            }
            for i, s in enumerate(selected_scenarios)
        ]
        plan_lines = "\n".join(
            f'- {sc["id"]}: {sc["title"]} [{sc["priority"]}] covers: {", ".join(sc["covers"]) or "—"}'
            for sc in tc_scenarios
        )
        plan_ctx = (
            f"Validated test plan ({len(tc_scenarios)} scenarios):\n\n"
            f"{plan_lines}\n\n"
            f"Feature summary: {st.session_state.get('p2_summary', '')}\n\n"
            f"Context:\n{st.session_state.p1_context}"
        )

        try:
            tcs, missing = generate_test_cases_in_batches(plan_ctx, tc_scenarios, batch_size=6)
        except Exception as e:
            handle_error(e); st.stop()

        st.session_state.structured_test_cases = tcs   # ← single source of truth
        st.session_state.p3_missing = missing
        st.session_state.p3_plan_ctx = plan_ctx
        st.session_state.p3_chat_log = []
        st.session_state.p2_validated = True
        st.session_state.phase_reached = max(st.session_state.phase_reached, 3)
        st.session_state.active_phase = 3
        st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_phase == 3:
    st.markdown('<div class="badge b3">📝 Phase 3 — Test Architect: Detailed Test Cases</div>', unsafe_allow_html=True)

    tc_data = st.session_state.get("structured_test_cases") or []

    # ── Targeted repair: regenerate ONLY missing scenarios (no duplicates) ────
    missing = st.session_state.get("p3_missing", [])
    if missing:
        st.warning(
            f"⚠️ {len(missing)} scenario(s) could not be generated:\n"
            + "\n".join(f"- {m['id']} — {m['title']}" for m in missing)
        )
        if st.button("🔄 Regenerate missing test cases", use_container_width=True, key="p3_repair"):
            try:
                new_tcs, still_missing = generate_test_cases_in_batches(
                    st.session_state.p3_plan_ctx, missing, batch_size=6
                )
                st.session_state.structured_test_cases = tc_data + new_tcs
                st.session_state.p3_missing = still_missing
                st.rerun()
            except Exception as e:
                handle_error(e)

    if tc_data:
        # Display is DERIVED from the structured data — always in sync with exports
        all_md = tc_to_markdown(tc_data)
        st.markdown(f"**{len(tc_data)} test cases**")
        st.markdown(all_md)
        st.divider()
        st.markdown("### 📥 Export")
        st.caption("All formats are derived from the same structured data — always consistent, no extra LLM call.")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("📝 Markdown", data=all_md, file_name="test_cases.md",
                               mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Text", data=all_md, file_name="test_cases.txt",
                               mime="text/plain", use_container_width=True)
        with c3:
            st.download_button("🗂️ JSON", data=json.dumps(tc_data, indent=2, ensure_ascii=False),
                               file_name="test_cases.json", mime="application/json", use_container_width=True)
        with c4:
            st.download_button("📊 CSV", data=build_csv(tc_data), file_name="test_cases.csv",
                               mime="text/csv", use_container_width=True)
        with st.expander(f"👁️ Preview JSON ({len(tc_data)} test cases)", expanded=False):
            st.json(tc_data)
    else:
        st.info("No test cases yet — validate the Phase 2 plan to generate them.")

    st.divider()

    # ── Chat: modifications applied IN PLACE on the structured test cases ────
    st.markdown("#### 💬 Adjust the test cases")
    st.caption("Modify, add or remove test cases — questions are answered without polluting the export.")
    for m in st.session_state.p3_chat_log:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])

    reply3 = st.chat_input("Request adjustments, additions, removals, or ask a question…", key="p3_chat")
    if reply3:
        st.session_state.p3_chat_log.append({"role": "user", "content": reply3})
        with st.spinner("Updating…"):
            try:
                current_tcs = json.dumps(tc_data, ensure_ascii=False)
                msg = (
                    f"CURRENT TEST CASES (JSON):\n{current_tcs}\n\n"
                    f"CONTEXT:\n{st.session_state.get('p3_plan_ctx', '')}\n\n"
                    f"USER REQUEST:\n{reply3}"
                )
                ops = call_llm_json(PROMPT_P3_MODIFY, msg, max_tokens=8000)
                changed = bool(ops.get("add") or ops.get("remove") or ops.get("modify"))
                if changed:
                    st.session_state.structured_test_cases = apply_tc_ops(tc_data, ops)
                reply_text = ops.get("reply", "Done.")
                if changed:
                    n_add = len(ops.get("add") or [])
                    n_rem = len(ops.get("remove") or [])
                    n_mod = len(ops.get("modify") or [])
                    reply_text += f"\n\n*({n_add} added · {n_mod} modified · {n_rem} removed)*"
                st.session_state.p3_chat_log.append({"role": "assistant", "content": reply_text})
                st.rerun()
            except Exception as e: handle_error(e)
