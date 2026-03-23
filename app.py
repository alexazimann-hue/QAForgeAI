import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io
import docx
import pypdf
import json
import csv

st.set_page_config(
    page_title="QA Copilot – AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
.badge {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-weight:700; font-size:13px; margin-bottom:16px;
}
.b1{background:#1a3a5c;color:#60aaff;border:1px solid #2255aa;}
.b2{background:#1a3a25;color:#60cc88;border:1px solid #226644;}
.b3{background:#3a1a2a;color:#cc6699;border:1px solid #882255;}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Gemini API Key", type="password",
                             help="Get your key at https://aistudio.google.com/")
    model_choice = st.text_input(
        "Gemini Model",
        value="gemini-2.5-flash-lite-preview-06-17",
        help="Exact model ID — https://ai.google.dev/gemini-api/docs/models"
    )
    st.caption("`gemini-2.5-flash-lite-preview-06-17` · `gemini-2.0-flash` · `gemini-2.5-pro` · `gemma-3-27b-it`")

    st.divider()
    st.markdown("""
### 🗺️ How it works
1. **Phase 1** — Submit your User Story → AI asks questions → answer → validate
2. **Phase 2** — AI generates test plan → refine → validate
3. **Phase 3** — AI writes full test cases → export
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
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── GEMINI CLIENT (cached — fix: no re-creation on every call) ────────────────
@st.cache_resource
def get_gemini_client(key: str):
    return genai.Client(api_key=key)

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

**✅ Happy Path:**
- TC: [Title]

**🔄 Alternate Flows:**
- TC: [Title]

**🔢 Boundary Value Analysis:**
- TC: [Title — specify boundary]

**🔀 Equivalence Partitioning:**
- TC: [Title — specify partition]

**❌ Negative / Error Cases:**
- TC: [Title]

**⚠️ Edge Cases:**
- TC: [Title]

**🔒 Security / Non-Functional (if applicable):**
- TC: [Title]

## HARD CONSTRAINTS
- Titles only. Minimum 12 scenarios."""

PROMPT_P3 = """You are a Senior QA Test Architect writing execution-ready test cases.

## OUTPUT FORMAT (repeat for every test case)

---

### TEST CASE [N]: [Scenario Title]

| Field | Detail |
|-------|--------|
| **ID** | TC-[N] |
| **Type** | [Happy Path / Alternate / BVA / Equivalence / Negative / Edge Case / Security] |
| **Priority** | [P1-Critical / P2-High / P3-Medium / P4-Low] |
| **Automation** | [✅ Good candidate / 🖐️ Manual only] — [reason] |

**📌 Preconditions:**
- [System state, user role, required data]

**🔢 Test Steps:**
1. [Precise action with exact test data]
2. ...

**✅ Expected Result:**
[Exact observable outcome]

**🔴 Failure Signature:**
[What the tester sees when this test FAILS]

---

## HARD CONSTRAINTS
- Never vague expected results. Use real specific test data in steps.
- If a rule is unclear: ⚠️ *Assumption: [assumption] — confirm with PO.*
- At the very end, after ALL test cases, output a JSON block for structured export:
```json
[
  {
    "id": "TC-1",
    "title": "...",
    "type": "...",
    "priority": "...",
    "automation": "...",
    "preconditions": ["..."],
    "steps": ["..."],
    "expected_result": "...",
    "failure_signature": "..."
  }
]
```"""

# ── FILE PARSING ──────────────────────────────────────────────────────────────
ALLOWED_TYPES = ["png", "jpg", "jpeg", "webp", "pdf", "txt", "md", "docx"]
MAX_FILES = 5
MAX_CHARS_PER_FILE = 15000

def extract_text_from_file(uploaded_file):
    name = uploaded_file.name.lower()
    try:
        if name.endswith((".txt", ".md")):
            return uploaded_file.read().decode("utf-8", errors="ignore")
        elif name.endswith(".pdf"):
            reader = pypdf.PdfReader(io.BytesIO(uploaded_file.read()))
            text = "\n".join(p.extract_text() or "" for p in reader.pages)
            if not text.strip():
                return f"[⚠️ Could not extract text from {uploaded_file.name} — PDF may be image-based or scanned.]"
            return text
        elif name.endswith(".docx"):
            doc = docx.Document(io.BytesIO(uploaded_file.read()))
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except Exception as e:
        return f"[Error reading {uploaded_file.name}: {e}]"
    return ""

def is_image(f):
    return f.name.lower().endswith((".png", ".jpg", ".jpeg", ".webp"))

def file_icon(f):
    n = f.name.lower()
    if n.endswith(".pdf"):   return "📕"
    if n.endswith(".docx"):  return "📘"
    if n.endswith((".txt", ".md")): return "📄"
    if is_image(f):          return "🖼️"
    return "📎"

# ── GEMINI CALL ───────────────────────────────────────────────────────────────
def call_gemini(history, system_prompt, user_message, images=None, max_tokens=8000):
    client = get_gemini_client(api_key)
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    parts = [types.Part(text=user_message)]
    for img in (images or []):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    contents.append(types.Content(role="user", parts=parts))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=max_tokens,
        temperature=0.2,
    )
    result = client.models.generate_content(
        model=model_choice.strip(), contents=contents, config=config
    )
    # Fix: guard against empty response
    if not result or not result.text or not result.text.strip():
        raise Exception("Empty response received from the model. Try again or switch model.")
    return result.text

def handle_error(e):
    err = str(e)
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        st.error("⚠️ Quota exhausted. Change the model ID in the sidebar.")
    elif "404" in err or "NOT_FOUND" in err:
        st.error(f"⚠️ Model not found: **{model_choice}**. Check https://ai.google.dev/gemini-api/docs/models")
    else:
        st.error(f"Gemini Error: {err}")

def render_chat(msgs):
    for m in msgs:
        with st.chat_message(m["role"], avatar="🧑‍💻" if m["role"] == "user" else "🤖"):
            st.markdown(m["content"])

# ── EXPORT HELPERS ────────────────────────────────────────────────────────────
def extract_json_from_text(text):
    """Extract the JSON array appended at end of Phase 3 output."""
    try:
        start = text.rfind("```json")
        end   = text.rfind("```", start + 6)
        if start != -1 and end != -1:
            return json.loads(text[start + 7:end].strip())
    except Exception:
        pass
    return None

def build_csv(data):
    out = io.StringIO()
    if not data:
        return ""
    fields = ["id", "title", "type", "priority", "automation",
              "preconditions", "steps", "expected_result", "failure_signature"]
    writer = csv.DictWriter(out, fieldnames=fields, extrasaction="ignore")
    writer.writeheader()
    for row in data:
        row["preconditions"] = " | ".join(row.get("preconditions", []))
        row["steps"]         = " | ".join(row.get("steps", []))
        writer.writerow(row)
    return out.getvalue()

# ── TAB BAR ───────────────────────────────────────────────────────────────────
def render_tab_bar():
    pr = st.session_state.phase_reached
    ap = st.session_state.active_phase
    labels = {1: "Analysis", 2: "Test Plan", 3: "Test Cases"}
    cols = st.columns(3)
    for i, (n, label) in enumerate(labels.items()):
        with cols[i]:
            if n > pr:
                st.button(f"🔒 Phase {n} — {label}", key=f"tab_{n}", disabled=True, use_container_width=True)
            else:
                prefix = "▶" if n == ap else "✅"
                if st.button(f"{prefix} Phase {n} — {label}", key=f"tab_{n}",
                              use_container_width=True, type="primary" if n == ap else "secondary"):
                    st.session_state.active_phase = n
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
st.title("🧪 QA Copilot — AI Test Case Generator")

if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar.")
    st.stop()

render_tab_bar()
st.divider()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.active_phase == 1:
    st.markdown('<div class="badge b1">🔍 Phase 1 — Senior QA Analyst: Requirements Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        us_input = st.text_area(
            "User Story + Acceptance Criteria", height=180,
            placeholder="As a [user], I want to [action] so that [benefit].\n\nAcceptance Criteria:\n- ...",
            max_chars=5000,
        )
        st.caption(f"{len(us_input) if us_input else 0}/5000 characters")

        uploaded_files = st.file_uploader(
            f"📎 Attach documents or images (max {MAX_FILES} files)",
            type=ALLOWED_TYPES,
            accept_multiple_files=True,
            help="PNG, JPG, WEBP · PDF · Word (DOCX) · Text (TXT, MD) — Max 15 000 chars extracted per doc"
        )

        if uploaded_files:
            if len(uploaded_files) > MAX_FILES:
                st.warning(f"⚠️ Max {MAX_FILES} files. Only the first {MAX_FILES} will be used.")
                uploaded_files = uploaded_files[:MAX_FILES]
            file_cols = st.columns(len(uploaded_files))
            for idx, f in enumerate(uploaded_files):
                with file_cols[idx]:
                    if is_image(f):
                        st.image(f, caption=f.name, use_column_width=True)
                    else:
                        st.markdown(f"{file_icon(f)} **{f.name}**")
                        st.caption(f"{round(f.size/1024, 1)} KB")

        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not us_input or not us_input.strip():
                st.warning("Please enter a User Story (minimum content required).")
            elif len(us_input.strip()) < 20:
                st.warning("User Story too short — please provide more details.")
            else:
                images, doc_texts = [], []
                total_chars = 0
                for f in (uploaded_files or []):
                    f.seek(0)
                    if is_image(f):
                        images.append(Image.open(f))
                    else:
                        f.seek(0)
                        text = extract_text_from_file(f)
                        if text:
                            if len(text) > MAX_CHARS_PER_FILE:
                                text = text[:MAX_CHARS_PER_FILE] + f"\n[... truncated at {MAX_CHARS_PER_FILE} chars]"
                                st.info(f"ℹ️ {f.name} was truncated to {MAX_CHARS_PER_FILE} chars to stay within context limits.")
                            total_chars += len(text)
                            doc_texts.append(f"--- {f.name} ---\n{text}")

                prompt = f"Please analyze the following User Story:\n\n{us_input}"
                if doc_texts:
                    prompt += "\n\n=== ATTACHED DOCUMENTS ===\n" + "\n\n".join(doc_texts)
                if images:
                    prompt += f"\n\n[{len(images)} wireframe/screenshot(s) attached — analyze them alongside.]"

                with st.spinner(f"Analyzing with `{model_choice.strip()}`…"):
                    try:
                        # Phase 1: lighter token budget (questions only)
                        response = call_gemini([], PROMPT_P1, prompt, images or None, max_tokens=2000)
                        st.session_state.p1_msgs = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p1_context = f"User Story:\n{us_input}\n\nAnalysis:\n{response}"
                        st.session_state.us_submitted = True
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
    else:
        render_chat(st.session_state.p1_msgs)
        st.divider()

        reply = st.chat_input("Answer the clarifying questions…", key="p1_chat_input")
        if reply:
            st.session_state.p1_msgs.append({"role": "user", "content": reply})
            with st.spinner("Processing…"):
                try:
                    response = call_gemini(st.session_state.p1_msgs[:-1], PROMPT_P1, reply, max_tokens=2000)
                    st.session_state.p1_msgs.append({"role": "assistant", "content": response})
                    st.session_state.p1_context += f"\n\nQ: {reply}\nA: {response}"
                    st.rerun()
                except Exception as e:
                    handle_error(e)

        if st.button("✅ Validate Analysis → Phase 2", type="primary", use_container_width=True, key="p1_validate"):
            context_msg = f"Validated context:\n\n{st.session_state.p1_context}\n\nGenerate the test plan (titles only)."
            with st.spinner("📋 Generating test plan…"):
                try:
                    response = call_gemini([], PROMPT_P2, context_msg, max_tokens=3000)
                    st.session_state.p2_msgs = [
                        {"role": "user", "content": context_msg},
                        {"role": "assistant", "content": response},
                    ]
                    st.session_state.p2_draft = response
                    st.session_state.p1_validated = True
                    st.session_state.phase_reached = max(st.session_state.phase_reached, 2)
                    st.session_state.active_phase = 2
                    st.rerun()
                except Exception as e:
                    handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_phase == 2:
    st.markdown('<div class="badge b2">📋 Phase 2 — Lead QA Engineer: Test Plan</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p2_msgs)
    st.divider()

    reply2 = st.chat_input("Request changes to the test plan…", key="p2_chat_input")
    if reply2:
        st.session_state.p2_msgs.append({"role": "user", "content": reply2})
        with st.spinner("Updating…"):
            try:
                response = call_gemini(st.session_state.p2_msgs[:-1], PROMPT_P2, reply2, max_tokens=3000)
                st.session_state.p2_msgs.append({"role": "assistant", "content": response})
                st.session_state.p2_draft = response
                st.rerun()
            except Exception as e:
                handle_error(e)

    if st.button("✅ Validate Plan → Phase 3", type="primary", use_container_width=True, key="p2_validate"):
        plan_msg = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nContext:\n{st.session_state.p1_context}\n\nGenerate COMPLETE and DETAILED test cases for every scenario."
        with st.spinner("📝 Generating test cases…"):
            try:
                response = call_gemini([], PROMPT_P3, plan_msg, max_tokens=8000)
                st.session_state.p3_msgs = [
                    {"role": "user", "content": plan_msg},
                    {"role": "assistant", "content": response},
                ]
                st.session_state.p2_validated = True
                st.session_state.phase_reached = max(st.session_state.phase_reached, 3)
                st.session_state.active_phase = 3
                st.rerun()
            except Exception as e:
                handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.active_phase == 3:
    st.markdown('<div class="badge b3">📝 Phase 3 — Test Architect: Detailed Test Cases</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p3_msgs)

    if st.session_state.p3_msgs:
        all_content = "\n\n".join([m["content"] for m in st.session_state.p3_msgs if m["role"] == "assistant"])

        st.divider()
        st.markdown("### 📥 Export")
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.download_button("📝 Markdown", data=all_content,
                               file_name="test_cases.md", mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Text", data=all_content,
                               file_name="test_cases.txt", mime="text/plain", use_container_width=True)
        with c3:
            json_data = extract_json_from_text(all_content)
            if json_data:
                st.download_button("🗂️ JSON", data=json.dumps(json_data, indent=2, ensure_ascii=False),
                                   file_name="test_cases.json", mime="application/json", use_container_width=True)
            else:
                st.button("🗂️ JSON", disabled=True, use_container_width=True, help="JSON not yet available")
        with c4:
            csv_data = build_csv(json_data) if json_data else ""
            if csv_data:
                st.download_button("📊 CSV", data=csv_data,
                                   file_name="test_cases.csv", mime="text/csv", use_container_width=True)
            else:
                st.button("📊 CSV", disabled=True, use_container_width=True, help="CSV not yet available")

    st.divider()
    reply3 = st.chat_input("Request adjustments or additional test cases…", key="p3_chat_input")
    if reply3:
        st.session_state.p3_msgs.append({"role": "user", "content": reply3})
        with st.spinner("Updating…"):
            try:
                response = call_gemini(st.session_state.p3_msgs[:-1], PROMPT_P3, reply3, max_tokens=8000)
                st.session_state.p3_msgs.append({"role": "assistant", "content": response})
                st.rerun()
            except Exception as e:
                handle_error(e)
