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
/* ── Custom tab buttons ── */
div[data-testid="stHorizontalBlock"] .phase-btn button {
    border-radius: 8px 8px 0 0 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    height: 52px !important;
    border-bottom: none !important;
}
.badge {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-weight:700; font-size:13px; margin-bottom:16px;
}
.b1{background:#1a3a5c;color:#60aaff;border:1px solid #2255aa;}
.b2{background:#1a3a25;color:#60cc88;border:1px solid #226644;}
.b3{background:#3a1a2a;color:#cc6699;border:1px solid #882255;}
.lock-box {
    background:#1a1a1a; border:1px solid #333; border-radius:8px;
    padding:32px; text-align:center; color:#666; margin-top:20px;
}
.tab-bar {
    display: flex;
    gap: 4px;
    border-bottom: 2px solid #1e2530;
    margin-bottom: 20px;
    padding-bottom: 0;
}
.tab-btn {
    padding: 10px 24px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 14px;
    cursor: pointer;
    border: 1px solid #1e2530;
    border-bottom: none;
    background: #161b22;
    color: #666;
    text-decoration: none;
}
.tab-btn.active {
    background: #1e3a5f;
    color: #60aaff;
    border-color: #2255aa;
}
.tab-btn.done {
    background: #1a3a25;
    color: #60cc88;
    border-color: #226644;
}
.tab-btn.locked {
    background: #111;
    color: #444;
    cursor: not-allowed;
}
.tab-content {
    padding: 8px 0;
}
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
    "active_phase": 1,
    "phase_reached": 1,
    "p1_msgs": [], "p2_msgs": [], "p3_msgs": [],
    "p1_validated": False, "p2_validated": False,
    "us_submitted": False,
    "p1_context": "", "p2_draft": "",
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
- If a rule is unclear: ⚠️ *Assumption: [assumption] — confirm with PO.*"""

# ── GEMINI ────────────────────────────────────────────────────────────────────
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
        model=model_choice.strip(), contents=contents, config=config
    ).text

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

# ═════════════════════════════════════════════════════════════════════════════
st.title("🧪 QA Copilot — AI Test Case Generator")

if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar.")
    st.stop()

# ── CUSTOM TAB BAR (boutons contrôlables via session_state) ──────────────────
pr = st.session_state.phase_reached
ap = st.session_state.active_phase

labels = {
    1: ("🔍", "Phase 1 — Analysis"),
    2: ("📋", "Phase 2 — Test Plan"),
    3: ("📝", "Phase 3 — Test Cases"),
}

cols = st.columns(3)
for i, (phase_n, (icon, name)) in enumerate(labels.items()):
    with cols[i]:
        if phase_n > pr:
            # Locked — disabled appearance
            st.button(
                f"🔒 {name}",
                key=f"tab_btn_{phase_n}",
                disabled=True,
                use_container_width=True,
            )
        else:
            prefix = "▶" if phase_n == ap else ("✅" if phase_n < pr else icon)
            clicked = st.button(
                f"{prefix} {name}",
                key=f"tab_btn_{phase_n}",
                use_container_width=True,
                type="primary" if phase_n == ap else "secondary",
            )
            if clicked:
                st.session_state.active_phase = phase_n
                st.rerun()

st.divider()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.active_phase == 1:
    st.markdown('<div class="badge b1">🔍 Phase 1 — Senior QA Analyst: Requirements Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        us_input = st.text_area(
            "User Story + Acceptance Criteria", height=200,
            placeholder="As a [user], I want to [action] so that [benefit].\n\nAcceptance Criteria:\n- ..."
        )
        uploaded = st.file_uploader("📎 Wireframe / Figma (optional)", type=["png","jpg","jpeg","webp"])
        if st.button("🚀 Start Analysis", type="primary", use_container_width=True):
            if not us_input.strip():
                st.warning("Please enter a User Story.")
            else:
                image_pil = Image.open(uploaded) if uploaded else None
                prompt = f"Please analyze the following User Story:\n\n{us_input}"
                if uploaded:
                    prompt += "\n\n[A wireframe has been provided — analyze it too.]"
                with st.spinner(f"Analyzing with `{model_choice.strip()}`…"):
                    try:
                        response = call_gemini([], PROMPT_P1, prompt, image_pil)
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
        reply = st.text_area("💬 Answer the clarifying questions:", height=120, key="p1_reply")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📨 Send Answer", use_container_width=True, key="p1_send"):
                if reply.strip():
                    st.session_state.p1_msgs.append({"role": "user", "content": reply})
                    with st.spinner("Processing…"):
                        try:
                            response = call_gemini(st.session_state.p1_msgs[:-1], PROMPT_P1, reply)
                            st.session_state.p1_msgs.append({"role": "assistant", "content": response})
                            st.session_state.p1_context += f"\n\nQ: {reply}\nA: {response}"
                            st.rerun()
                        except Exception as e:
                            handle_error(e)
        with c2:
            if st.button("✅ Validate → Phase 2", type="primary", use_container_width=True, key="p1_validate"):
                context_msg = f"Validated context:\n\n{st.session_state.p1_context}\n\nGenerate the test plan (titles only)."
                with st.spinner("📋 Generating test plan…"):
                    try:
                        response = call_gemini([], PROMPT_P2, context_msg)
                        st.session_state.p2_msgs = [
                            {"role": "user", "content": context_msg},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p2_draft = response
                        st.session_state.p1_validated = True
                        st.session_state.phase_reached = max(st.session_state.phase_reached, 2)
                        st.session_state.active_phase = 2   # ← auto-switch
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
    reply2 = st.text_area("💬 Request changes to the test plan:", height=100, key="p2_reply")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📨 Update Plan", use_container_width=True, key="p2_send"):
            if reply2.strip():
                st.session_state.p2_msgs.append({"role": "user", "content": reply2})
                with st.spinner("Updating…"):
                    try:
                        response = call_gemini(st.session_state.p2_msgs[:-1], PROMPT_P2, reply2)
                        st.session_state.p2_msgs.append({"role": "assistant", "content": response})
                        st.session_state.p2_draft = response
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
    with c2:
        if st.button("✅ Validate → Phase 3", type="primary", use_container_width=True, key="p2_validate"):
            plan_msg = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nContext:\n{st.session_state.p1_context}\n\nGenerate COMPLETE and DETAILED test cases for every scenario."
            with st.spinner("📝 Generating test cases…"):
                try:
                    response = call_gemini([], PROMPT_P3, plan_msg)
                    st.session_state.p3_msgs = [
                        {"role": "user", "content": plan_msg},
                        {"role": "assistant", "content": response},
                    ]
                    st.session_state.p2_validated = True
                    st.session_state.phase_reached = max(st.session_state.phase_reached, 3)
                    st.session_state.active_phase = 3   # ← auto-switch
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
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Export .md", data=all_content,
                               file_name="test_cases_QA.md", mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Export .txt", data=all_content,
                               file_name="test_cases_QA.txt", mime="text/plain", use_container_width=True)

    st.divider()
    reply3 = st.text_area("💬 Request adjustments or additional test cases:", height=100, key="p3_reply")
    if st.button("📨 Send", use_container_width=True, key="p3_send"):
        if reply3.strip():
            st.session_state.p3_msgs.append({"role": "user", "content": reply3})
            with st.spinner("Updating…"):
                try:
                    response = call_gemini(st.session_state.p3_msgs[:-1], PROMPT_P3, reply3)
                    st.session_state.p3_msgs.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    handle_error(e)
