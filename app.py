import streamlit as st
from PIL import Image
import io
import docx
import pypdf
import json
import csv

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

def call_openai(history, system_prompt, user_message, images=None, max_tokens=3000):
    from openai import OpenAI

    @st.cache_resource
    def get_openai_client(key):
        return OpenAI(api_key=key)

    client = get_openai_client(st.session_state.api_key)
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
    if st.session_state.provider == "Gemini":
        return call_gemini(history, system_prompt, user_message, images, max_tokens)
    else:
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

# ── PROVIDER DEFAULTS ─────────────────────────────────────────────────────────
PROVIDER_DEFAULTS = {
    "Gemini": {
        "placeholder": "gemini-2.5-flash",
        "examples": "`gemini-2.5-flash` · `gemini-2.0-flash` · `gemini-2.5-pro`",
        "docs": "https://ai.google.dev/gemini-api/docs/models",
    },
    "OpenAI": {
        "placeholder": "gpt-4o-mini",
        "examples": "`gpt-4o-mini` · `gpt-4o` · `gpt-4-turbo` · `gpt-3.5-turbo`",
        "docs": "https://platform.openai.com/docs/models",
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
    st.title("⚙️ Configuration")

    provider = st.radio("LLM Provider", ["Gemini", "OpenAI"], horizontal=True)
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PROMPTS ───────────────────────────────────────────────────────────────────
PROMPT_P1 = """You are a Senior QA Analyst and Requirements Engineer with 10+ years of experience in agile software testing.

## YOUR ROLE IN THIS PHASE
Perform a deep requirements analysis of the provided User Story.
Your ONLY output is a structured list of clarifying questions.
You are STRICTLY FORBIDDEN from generating test scenarios, test case titles, test plans, or any test-related content.

## ANALYSIS FRAMEWORK
Analyze the User Story across these 6 dimensions:
1. **Functional Scope** — Are all business rules explicitly stated?
2. **Input Validation** — Field constraints (type, length, format, mandatory/optional)?
3. **Error Handling** — Invalid input, system errors, timeouts, concurrent access?
4. **Boundary Conditions** — Min/max values, empty states, limit behaviors?
5. **System Dependencies** — External systems, APIs, permissions, states?
6. **Non-Functional Requirements** — Performance, security, accessibility?

## OUTPUT FORMAT (STRICT)
🔍 **PHASE 1 — Requirements Analysis & Clarifications**

**Current Understanding:**
[2–4 sentences summarizing the feature]

**Clarifying Questions:**
*Functional:*
1. [Question]
*Validation & Constraints:*
2. [Question]
*Error Handling:*
3. [Question]
*Edge Cases & Boundaries:*
4. [Question]
*System & Context:*
5. [Question]

## HARD CONSTRAINTS
- Do NOT suggest test cases or scenarios.
- Do NOT invent business rules not in the User Story."""

PROMPT_P2 = """You are a Lead QA Engineer specializing in test design and coverage strategy.

## YOUR ROLE
Generate a comprehensive TEST PLAN as scenario TITLES ONLY.
FORBIDDEN: steps, preconditions, or expected results.

## COVERAGE — apply ALL applicable techniques:
- Happy Path, Alternate Flows
- Equivalence Partitioning, Boundary Value Analysis (BVA)
- Error Guessing, State Transitions, Negative Testing
- Security / Non-Functional if applicable

## OUTPUT FORMAT (STRICT)
📋 **PHASE 2 — Test Plan (Draft)**
**Feature Summary:** [2–3 sentences]
**✅ Happy Path:** - TC: [Title]
**🔄 Alternate Flows:** - TC: [Title]
**🔢 Boundary Value Analysis:** - TC: [Title — specify boundary]
**🔀 Equivalence Partitioning:** - TC: [Title — specify partition]
**❌ Negative / Error Cases:** - TC: [Title]
**⚠️ Edge Cases:** - TC: [Title]
**🔒 Security / Non-Functional (if applicable):** - TC: [Title]

## HARD CONSTRAINTS
Titles only. Minimum 12 scenarios."""

PROMPT_P3_MARKDOWN = """You are a Senior QA Test Architect writing execution-ready test cases.
Generate detailed human-readable test cases in Markdown format.

### TEST CASE [N]: [Scenario Title]
| Field | Detail |
|-------|--------|
| **ID** | TC-[N] |
| **Type** | [Happy Path / Alternate / BVA / Equivalence / Negative / Edge Case / Security] |
| **Priority** | [P1-Critical / P2-High / P3-Medium / P4-Low] |
| **Automation** | [✅ Good candidate / 🖐️ Manual only] — [reason] |

**📌 Preconditions:** - [state, role, data]
**🔢 Test Steps:** 1. [action + exact data] 2. ...
**✅ Expected Result:** [exact observable outcome]
**🔴 Failure Signature:** [what tester sees on failure]

HARD CONSTRAINTS: Real test data in steps. If unclear: ⚠️ *Assumption: [...] — confirm with PO.*"""

PROMPT_P3_JSON = """You are a Senior QA Test Architect.
Generate ALL test cases from the validated test plan in structured JSON format only.
Each test case must have: id, title, type, priority, automation, preconditions (array),
steps (array of {step_number, action}), expected_result, failure_signature.
Be precise and use real test data in steps."""

# ── FILE PARSING ──────────────────────────────────────────────────────────────
ALLOWED_TYPES = ["png","jpg","jpeg","webp","pdf","txt","md","docx"]
MAX_FILES = 5
MAX_CHARS = 15000

def extract_text(f):
    name = f.name.lower()
    try:
        if name.endswith((".txt",".md")): return f.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(f.read()))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            return text if text.strip() else f"[⚠️ {f.name}: image-based PDF, text extraction failed]"
        elif name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(f.read()))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[Error: {f.name}: {e}]"
    return ""

def is_image(f): return f.name.lower().endswith((".png",".jpg",".jpeg",".webp"))
def file_icon(f):
    n = f.name.lower()
    if n.endswith(".pdf"): return "📕"
    if n.endswith(".docx"): return "📘"
    if n.endswith((".txt",".md")): return "📄"
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
        pre = r.get("preconditions",[])
        r["preconditions"] = " | ".join(pre) if isinstance(pre, list) else str(pre)
        steps = r.get("steps",[])
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
                images, doc_texts = [], []
                for f in (uploaded_files or []):
                    f.seek(0)
                    if is_image(f): images.append(Image.open(f))
                    else:
                        f.seek(0); text = extract_text(f)
                        if text:
                            if len(text) > MAX_CHARS:
                                text = text[:MAX_CHARS] + f"\n[...truncated at {MAX_CHARS} chars]"
                                st.info(f"ℹ️ {f.name} truncated to {MAX_CHARS} chars.")
                            doc_texts.append(f"--- {f.name} ---\n{text}")
                prompt = f"Please analyze the following User Story:\n\n{us_input}"
                if doc_texts: prompt += "\n\n=== ATTACHED DOCUMENTS ===\n" + "\n\n".join(doc_texts)
                if images: prompt += f"\n\n[{len(images)} wireframe(s) attached.]"
                with st.spinner(f"Analyzing with {provider} / `{model_choice}`…"):
                    try:
                        response = call_llm([], PROMPT_P1, prompt, images or None, max_tokens=2000)
                        st.session_state.p1_msgs = [{"role":"user","content":prompt},{"role":"assistant","content":response}]
                        st.session_state.p1_context = f"User Story:\n{us_input}\n\nAnalysis:\n{response}"
                        st.session_state.us_submitted = True
                        st.rerun()
                    except Exception as e: handle_error(e)
    else:
        render_chat(st.session_state.p1_msgs)
        st.divider()
        reply = st.chat_input("Answer the clarifying questions…", key="p1_chat")
        if reply:
            st.session_state.p1_msgs.append({"role":"user","content":reply})
            with st.spinner("Processing…"):
                try:
                    response = call_llm(st.session_state.p1_msgs[:-1], PROMPT_P1, reply, max_tokens=2000)
                    st.session_state.p1_msgs.append({"role":"assistant","content":response})
                    st.session_state.p1_context += f"\n\nQ: {reply}\nA: {response}"
                    st.rerun()
                except Exception as e: handle_error(e)
        if st.button("✅ Validate Analysis → Phase 2", type="primary", use_container_width=True, key="p1_val"):
            ctx = f"Validated context:\n\n{st.session_state.p1_context}\n\nGenerate the test plan (titles only)."
            with st.spinner("📋 Generating test plan…"):
                try:
                    response = call_llm([], PROMPT_P2, ctx, max_tokens=3000)
                    st.session_state.p2_msgs = [{"role":"user","content":ctx},{"role":"assistant","content":response}]
                    st.session_state.p2_draft = response
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
    render_chat(st.session_state.p2_msgs)
    st.divider()
    reply2 = st.chat_input("Request changes to the test plan…", key="p2_chat")
    if reply2:
        st.session_state.p2_msgs.append({"role":"user","content":reply2})
        with st.spinner("Updating…"):
            try:
                response = call_llm(st.session_state.p2_msgs[:-1], PROMPT_P2, reply2, max_tokens=3000)
                st.session_state.p2_msgs.append({"role":"assistant","content":response})
                st.session_state.p2_draft = response
                st.rerun()
            except Exception as e: handle_error(e)
    if st.button("✅ Validate Plan → Phase 3", type="primary", use_container_width=True, key="p2_val"):
        plan_ctx = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nContext:\n{st.session_state.p1_context}"
        with st.spinner("📝 Generating Markdown test cases…"):
            try:
                md = call_llm([], PROMPT_P3_MARKDOWN, plan_ctx + "\n\nGenerate COMPLETE test cases for every scenario.", max_tokens=8000)
                st.session_state.p3_msgs = [{"role":"user","content":plan_ctx},{"role":"assistant","content":md}]
            except Exception as e: handle_error(e); st.stop()
        with st.spinner("🗂️ Generating structured JSON…"):
            try:
                tc = call_llm_structured(PROMPT_P3_JSON, plan_ctx + "\n\nGenerate ALL test cases in structured JSON.", max_tokens=8000)
                st.session_state.structured_test_cases = tc
            except Exception as e:
                st.warning(f"⚠️ JSON export unavailable: {e}")
                st.session_state.structured_test_cases = None
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
    st.divider()
    reply3 = st.chat_input("Request adjustments or additional test cases…", key="p3_chat")
    if reply3:
        st.session_state.p3_msgs.append({"role":"user","content":reply3})
        with st.spinner("Updating…"):
            try:
                response = call_llm(st.session_state.p3_msgs[:-1], PROMPT_P3_MARKDOWN, reply3, max_tokens=8000)
                st.session_state.p3_msgs.append({"role":"assistant","content":response})
                st.rerun()
            except Exception as e: handle_error(e)
