import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io

st.set_page_config(
    page_title="QA Copilot – AI Test Case Generator",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
.phase-badge {
    display:inline-block; padding:8px 20px; border-radius:20px;
    font-weight:bold; font-size:15px; margin-bottom:16px;
}
.phase-1{background:#1a3a5c;color:#60aaff;border:1px solid #2255aa;}
.phase-2{background:#1a3a25;color:#60cc88;border:1px solid #226644;}
.phase-3{background:#3a1a2a;color:#cc6699;border:1px solid #882255;}
.stepper{display:flex;align-items:center;gap:8px;margin-bottom:20px;flex-wrap:wrap;}
.step{padding:6px 14px;border-radius:20px;font-size:13px;font-weight:600;}
.step-active{background:#1e3a5f;color:#60aaff;border:1px solid #2255aa;}
.step-done{background:#1a3a25;color:#60cc88;border:1px solid #226644;}
.step-pending{background:#222;color:#555;border:1px solid #333;}
.step-arrow{color:#444;font-size:18px;}
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input(
        "Gemini API Key", type="password",
        help="Get your key at https://aistudio.google.com/"
    )
    model_choice = st.text_input(
        "Gemini Model",
        value="gemini-2.5-flash-lite-preview-06-17",
        help="Enter the exact model ID. Full list: https://ai.google.dev/gemini-api/docs/models"
    )
    st.caption("e.g. `gemini-2.5-flash-lite-preview-06-17` · `gemini-2.0-flash` · `gemini-2.5-pro` · `gemma-3-27b-it`")
    st.divider()
    st.markdown("""
### 🗺️ How it works
1. **Phase 1** – Submit your User Story  
   AI asks clarifying questions → you answer → validate
2. **Phase 2** – Test Plan  
   AI lists test scenarios → modify if needed → validate
3. **Phase 3** – Detailed Test Cases  
   AI generates full test cases → export
""")
    st.divider()
    if st.button("🔄 New Session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "phase": 1,
    "p1_msgs": [], "p2_msgs": [], "p3_msgs": [],
    "p1_done": False, "p2_done": False,
    "us_text": "", "us_submitted": False,
    "p1_context": "", "p2_draft": "",
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── PROMPTS (Expert QA / Prompt Engineering best practices) ───────────────────

PROMPT_P1 = """You are a Senior QA Analyst and Requirements Engineer with 10+ years of experience in agile software testing.

## YOUR ROLE IN THIS PHASE
Perform a deep requirements analysis of the provided User Story.
Your ONLY output is a structured list of clarifying questions.
You are STRICTLY FORBIDDEN from generating test scenarios, test case titles, test plans, or any test-related content.

## ANALYSIS FRAMEWORK
Analyze the User Story across these 6 dimensions:

1. **Functional Scope** — Are all business rules explicitly stated? Are there implicit assumptions?
2. **Input Validation** — What are the field constraints (type, length, format, mandatory/optional)?
3. **Error Handling** — What happens on invalid input, system errors, timeouts, concurrent access?
4. **Boundary Conditions** — What are the min/max values, empty states, limit behaviors?
5. **System Dependencies** — What external systems, APIs, permissions, or states are involved?
6. **Non-Functional Requirements** — Any performance, security, accessibility, or UX considerations?

## OUTPUT FORMAT (STRICT)

🔍 **PHASE 1 — Requirements Analysis & Clarifications**

**Current Understanding:**
[2–4 sentences summarizing what the feature does, as you understand it from the User Story]

**Clarifying Questions:**

*Functional:*
1. [Specific question]
2. [Specific question]

*Validation & Constraints:*
3. [Specific question]

*Error Handling:*
4. [Specific question]

*Edge Cases & Boundaries:*
5. [Specific question]

*System & Context:*
6. [Specific question]

---
*Please answer the questions above, then click "✅ Validate & Go to Phase 2" when ready.*

## HARD CONSTRAINTS
- Do NOT suggest test cases or scenarios — not even as examples.
- Do NOT invent business rules not present in the User Story.
- If a question seems obvious, ask it anyway — ambiguity is the root of test gaps."""

PROMPT_P2 = """You are a Lead QA Engineer specializing in test design and coverage strategy.

## YOUR ROLE IN THIS PHASE
Based on the validated User Story and all clarification answers, generate a comprehensive TEST PLAN as a list of scenario TITLES ONLY.
You are STRICTLY FORBIDDEN from writing steps, preconditions, or expected results in this phase.

## COVERAGE STRATEGY
Apply the following QA techniques to maximize coverage:
- **Happy Path** — Standard successful user flows
- **Alternate Flows** — Valid but non-standard paths
- **Equivalence Partitioning** — One representative per valid/invalid class
- **Boundary Value Analysis (BVA)** — min, min+1, max-1, max, just outside boundaries
- **Error Guessing** — Known common failure points (empty fields, special chars, SQL injection hints, XSS)
- **State Transitions** — Different system states that affect behavior
- **Negative Testing** — Invalid data, missing permissions, wrong formats
- **Non-Functional** — Performance, security, accessibility if applicable

## OUTPUT FORMAT (STRICT)

📋 **PHASE 2 — Test Plan (Draft)**

**Feature Summary:**
[2–3 sentences describing the feature scope being tested]

**✅ Happy Path:**
- TC: [Concise scenario title]

**🔄 Alternate Flows:**
- TC: [Concise scenario title]

**🔢 Boundary Value Analysis:**
- TC: [Concise scenario title — specify the boundary]

**🔀 Equivalence Partitioning:**
- TC: [Concise scenario title — specify the partition]

**❌ Negative / Error Cases:**
- TC: [Concise scenario title]

**⚠️ Edge Cases:**
- TC: [Concise scenario title]

**🔒 Security / Non-Functional (if applicable):**
- TC: [Concise scenario title]

---
*Review and adjust this test plan, then click "✅ Validate Plan → Generate Test Cases" when ready.*

## HARD CONSTRAINTS
- Titles only — no steps, no preconditions, no expected results.
- Each scenario must be unique, atomic, and independently executable.
- Minimum 12 scenarios total across all categories."""

PROMPT_P3 = """You are a Senior QA Test Architect with deep expertise in writing execution-ready test cases for both manual testers and automation engineers.

## YOUR ROLE IN THIS PHASE
Transform each validated scenario title into a complete, precise, and executable test case.

## QUALITY STANDARDS FOR EACH TEST CASE
- **Preconditions**: exact system state, user role, data setup required
- **Steps**: one action per step, written at the UI/API/system level — no vague verbs like "verify", "check", "use"
- **Expected Result**: observable, verifiable, unambiguous — reference exact messages, UI states, HTTP codes, DB changes when relevant
- **Failure Signature**: what a tester would see when the test FAILS — helps distinguish real bugs from test issues
- **Data**: include specific test data values (valid AND invalid) directly in the steps
- **Automation Hint**: flag if the case is a good candidate for automation and why

## OUTPUT FORMAT (STRICT — repeat for every test case)

---

### TEST CASE [N]: [Scenario Title]

| Field | Detail |
|-------|--------|
| **ID** | TC-[N] |
| **Type** | [Happy Path / Alternate Flow / BVA / Equivalence / Negative / Edge Case / Security] |
| **Priority** | [P1-Critical / P2-High / P3-Medium / P4-Low] |
| **Automation** | [Good candidate ✅ / Manual only 🖐️] — [1-line reason] |

**📌 Preconditions:**
- [Exact system state]
- [User role / permissions]
- [Required test data already in system]

**🔢 Test Steps:**
1. [Navigate to / Open / Enter / Click — be precise]
2. [Next action with exact data: e.g. "Enter 'test@email.com' in the Email field"]
3. ...

**✅ Expected Result:**
[Exact observable outcome: UI message, redirect URL, DB state, API response code, etc.]

**🔴 Failure Signature:**
[What the tester sees when this test FAILS — e.g. "No error message displayed", "User is redirected instead of blocked"]

---

## HARD CONSTRAINTS
- Never write vague expected results like "it works" or "the system responds correctly".
- Never invent business rules not provided in the User Story or clarifications.
- If a rule is unclear, add a note: ⚠️ *Assumption: [your assumption] — to be confirmed with PO.*
- Use real, specific test data values in the steps.
- Steps must be granular enough for a junior tester to execute without guidance."""

# ── GEMINI CALL ───────────────────────────────────────────────────────────────
def call_gemini(history, system_prompt, user_message, image=None):
    client = genai.Client(api_key=api_key)
    contents = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        contents.append(types.Content(role=role, parts=[types.Part(text=m["content"])]))
    parts = [types.Part(text=user_message)]
    if image:
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        parts.append(types.Part.from_bytes(data=buf.getvalue(), mime_type="image/png"))
    contents.append(types.Content(role="user", parts=parts))
    config = types.GenerateContentConfig(
        system_instruction=system_prompt,
        max_output_tokens=8000,
        temperature=0.2,
    )
    return client.models.generate_content(
        model=model_choice.strip(),
        contents=contents,
        config=config,
    ).text

def handle_error(e):
    err = str(e)
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        st.error("⚠️ Quota exhausted for this model. Change the model ID in the sidebar.")
    elif "404" in err or "NOT_FOUND" in err:
        st.error(f"⚠️ Model not found: **{model_choice}**. Check exact IDs at https://ai.google.dev/gemini-api/docs/models")
    else:
        st.error(f"Gemini Error: {err}")

# ── STEPPER ───────────────────────────────────────────────────────────────────
def render_stepper():
    p = st.session_state.phase
    steps = [("🔍 Phase 1 — Analysis", 1), ("📋 Phase 2 — Test Plan", 2), ("📝 Phase 3 — Test Cases", 3)]
    html = '<div class="stepper">'
    for i, (lbl, n) in enumerate(steps):
        if n < p:    css, icon = "step step-done", "✅ "
        elif n == p: css, icon = "step step-active", "▶ "
        else:        css, icon = "step step-pending", "⏳ "
        html += f'<div class="{css}">{icon}{lbl}</div>'
        if i < 2: html += '<span class="step-arrow">→</span>'
    html += '</div>'
    st.markdown(html, unsafe_allow_html=True)

def render_chat(msgs):
    for m in msgs:
        av = "🧑‍💻" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=av):
            st.markdown(m["content"])

# ═════════════════════════════════════════════════════════════════════════════
render_stepper()
st.title("🧪 QA Copilot — AI Test Case Generator")

if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to get started.")
    st.stop()
if not model_choice.strip():
    st.warning("⚠️ Enter a model ID in the sidebar.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 1:
    st.markdown('<div class="phase-badge phase-1">🔍 Phase 1 — QA Analyst: Requirements Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        st.markdown("### 📝 Submit your User Story")
        us_input = st.text_area(
            "User Story + Acceptance Criteria",
            height=200,
            placeholder="As a [user], I want to [action] so that [benefit].\n\nAcceptance Criteria:\n- ..."
        )
        uploaded = st.file_uploader("📎 Wireframe / Figma screenshot (optional)", type=["png","jpg","jpeg","webp"])

        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not us_input.strip():
                st.warning("Please enter a User Story.")
            else:
                image_pil = Image.open(uploaded) if uploaded else None
                prompt = f"Please analyze the following User Story:\n\n{us_input}"
                if uploaded:
                    prompt += "\n\n[A wireframe/screenshot has been provided — analyze it alongside the User Story.]"
                with st.spinner(f"🔍 Analyzing with `{model_choice.strip()}`…"):
                    try:
                        response = call_gemini([], PROMPT_P1, prompt, image_pil)
                        st.session_state.us_text = us_input
                        st.session_state.p1_msgs = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p1_context = f"User Story:\n{us_input}\n\nInitial Analysis:\n{response}"
                        st.session_state.us_submitted = True
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
    else:
        render_chat(st.session_state.p1_msgs)
        reply = st.text_area("💬 Answer the clarifying questions above:", height=140, key="p1_reply")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📨 Send Answer", use_container_width=True):
                if reply.strip():
                    st.session_state.p1_msgs.append({"role": "user", "content": reply})
                    with st.spinner("Processing your answers…"):
                        try:
                            response = call_gemini(st.session_state.p1_msgs[:-1], PROMPT_P1, reply)
                            st.session_state.p1_msgs.append({"role": "assistant", "content": response})
                            st.session_state.p1_context += f"\n\nQ: {reply}\nA: {response}"
                            st.rerun()
                        except Exception as e:
                            handle_error(e)
        with c2:
            if st.button("✅ Validate Analysis → Phase 2", type="primary", use_container_width=True):
                context_msg = f"Validated context from Phase 1:\n\n{st.session_state.p1_context}\n\nNow generate the test plan (titles only)."
                with st.spinner("📋 Generating test plan…"):
                    try:
                        response = call_gemini([], PROMPT_P2, context_msg)
                        st.session_state.p2_msgs = [
                            {"role": "user", "content": context_msg},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p2_draft = response
                        st.session_state.p1_done = True
                        st.session_state.phase = 2
                        st.rerun()
                    except Exception as e:
                        handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    st.markdown('<div class="phase-badge phase-2">📋 Phase 2 — Lead QA: Test Plan</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p2_msgs)
    reply2 = st.text_area("💬 Request changes to the test plan:", height=100, key="p2_reply")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📨 Update Plan", use_container_width=True):
            if reply2.strip():
                st.session_state.p2_msgs.append({"role": "user", "content": reply2})
                with st.spinner("Updating test plan…"):
                    try:
                        response = call_gemini(st.session_state.p2_msgs[:-1], PROMPT_P2, reply2)
                        st.session_state.p2_msgs.append({"role": "assistant", "content": response})
                        st.session_state.p2_draft = response
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
    with c2:
        if st.button("✅ Validate Plan → Generate Test Cases (Phase 3)", type="primary", use_container_width=True):
            plan_msg = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nUser Story context:\n{st.session_state.p1_context}\n\nGenerate COMPLETE and DETAILED test cases for every scenario."
            with st.spinner("📝 Generating detailed test cases…"):
                try:
                    response = call_gemini([], PROMPT_P3, plan_msg)
                    st.session_state.p3_msgs = [
                        {"role": "user", "content": plan_msg},
                        {"role": "assistant", "content": response},
                    ]
                    st.session_state.p2_done = True
                    st.session_state.phase = 3
                    st.rerun()
                except Exception as e:
                    handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 3:
    st.markdown('<div class="phase-badge phase-3">📝 Phase 3 — Test Architect: Detailed Test Cases</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p3_msgs)

    if st.session_state.p3_msgs:
        all_content = "\n\n".join([m["content"] for m in st.session_state.p3_msgs if m["role"] == "assistant"])
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Export as Markdown (.md)", data=all_content,
                               file_name="test_cases_QA.md", mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Export as Text (.txt)", data=all_content,
                               file_name="test_cases_QA.txt", mime="text/plain", use_container_width=True)

    st.divider()
    reply3 = st.text_area("💬 Request adjustments or additional test cases:", height=100, key="p3_reply")
    if st.button("📨 Send", use_container_width=True):
        if reply3.strip():
            st.session_state.p3_msgs.append({"role": "user", "content": reply3})
            with st.spinner("Updating test cases…"):
                try:
                    response = call_gemini(st.session_state.p3_msgs[:-1], PROMPT_P3, reply3)
                    st.session_state.p3_msgs.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    handle_error(e)
