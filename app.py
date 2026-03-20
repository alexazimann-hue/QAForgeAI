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
/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 8px;
    background: #0e1117;
    padding: 8px 8px 0 8px;
    border-bottom: 2px solid #1e2530;
}
.stTabs [data-baseweb="tab"] {
    height: 52px;
    padding: 0 24px;
    border-radius: 8px 8px 0 0;
    font-weight: 600;
    font-size: 14px;
    color: #888;
    background: #161b22;
    border: 1px solid #1e2530;
    border-bottom: none;
}
.stTabs [aria-selected="true"] {
    background: #1e3a5f !important;
    color: #60aaff !important;
    border-color: #2255aa !important;
}
/* ── Chat bubbles ── */
.stChatMessage { border-radius: 10px; margin-bottom: 8px; }
/* ── Phase badges ── */
.badge {
    display:inline-block; padding:6px 16px; border-radius:20px;
    font-weight:700; font-size:13px; margin-bottom:12px;
}
.b1{background:#1a3a5c;color:#60aaff;border:1px solid #2255aa;}
.b2{background:#1a3a25;color:#60cc88;border:1px solid #226644;}
.b3{background:#3a1a2a;color:#cc6699;border:1px solid #882255;}
/* ── Lock banner ── */
.lock-box {
    background:#1a1a1a; border:1px solid #333; border-radius:8px;
    padding:24px; text-align:center; color:#666; margin-top:20px;
}
/* ── Warning reset box ── */
.reset-warn {
    background:#2a1a0a; border:1px solid #7a4a00; border-radius:8px;
    padding:12px 16px; margin-top:8px; font-size:13px; color:#cc9944;
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
1. **Phase 1** — Submit US → AI asks clarifying questions → answer → validate
2. **Phase 2** — AI generates test plan → refine → validate
3. **Phase 3** — AI writes full test cases → export

**Navigation rules:**
- 🔒 A tab is locked until the previous phase is validated
- ✅ Validated phases stay editable — go back anytime
- 🔄 Re-running a phase resets subsequent phases (with confirmation)
""")
    st.divider()
    if st.button("🔄 New Session", use_container_width=True, key="new_session"):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
defaults = {
    "phase_reached": 1,          # highest phase unlocked (1, 2, or 3)
    "p1_msgs": [], "p2_msgs": [], "p3_msgs": [],
    "p1_validated": False, "p2_validated": False,
    "us_text": "", "us_submitted": False,
    "p1_context": "", "p2_draft": "",
    "confirm_reset_p1": False,
    "confirm_reset_p2": False,
    "confirm_reset_p3": False,
    "active_tab": 0,
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
1. **Functional Scope** — Are all business rules explicitly stated? Are there implicit assumptions?
2. **Input Validation** — What are the field constraints (type, length, format, mandatory/optional)?
3. **Error Handling** — What happens on invalid input, system errors, timeouts, concurrent access?
4. **Boundary Conditions** — What are the min/max values, empty states, limit behaviors?
5. **System Dependencies** — What external systems, APIs, permissions, or states are involved?
6. **Non-Functional Requirements** — Any performance, security, accessibility, or UX considerations?

## OUTPUT FORMAT (STRICT)
🔍 **PHASE 1 — Requirements Analysis & Clarifications**

**Current Understanding:**
[2–4 sentences summarizing what the feature does]

**Clarifying Questions:**

*Functional:*
1. [Specific question]

*Validation & Constraints:*
2. [Specific question]

*Error Handling:*
3. [Specific question]

*Edge Cases & Boundaries:*
4. [Specific question]

*System & Context:*
5. [Specific question]

---
*Please answer the questions above, then click "✅ Validate & Go to Phase 2" when ready.*

## HARD CONSTRAINTS
- Do NOT suggest test cases or scenarios.
- Do NOT invent business rules not present in the User Story.
- Ask every relevant question even if it seems obvious."""

PROMPT_P2 = """You are a Lead QA Engineer specializing in test design and coverage strategy.

## YOUR ROLE IN THIS PHASE
Based on the validated User Story and clarification answers, generate a comprehensive TEST PLAN as a list of scenario TITLES ONLY.
You are STRICTLY FORBIDDEN from writing steps, preconditions, or expected results.

## COVERAGE STRATEGY — apply ALL applicable techniques:
- **Happy Path** — Standard successful flows
- **Alternate Flows** — Valid but non-standard paths
- **Equivalence Partitioning** — One representative per valid/invalid class
- **Boundary Value Analysis (BVA)** — min, min+1, max-1, max, just outside boundaries
- **Error Guessing** — Empty fields, special chars, SQL injection, XSS hints
- **State Transitions** — Different system states affecting behavior
- **Negative Testing** — Invalid data, missing permissions, wrong formats
- **Security / Non-Functional** — Performance, accessibility if applicable

## OUTPUT FORMAT (STRICT)
📋 **PHASE 2 — Test Plan (Draft)**

**Feature Summary:**
[2–3 sentences]

**✅ Happy Path:**
- TC: [Title]

**🔄 Alternate Flows:**
- TC: [Title]

**🔢 Boundary Value Analysis:**
- TC: [Title — specify the boundary value]

**🔀 Equivalence Partitioning:**
- TC: [Title — specify the partition]

**❌ Negative / Error Cases:**
- TC: [Title]

**⚠️ Edge Cases:**
- TC: [Title]

**🔒 Security / Non-Functional (if applicable):**
- TC: [Title]

---
*Review and adjust, then click "✅ Validate Plan → Generate Test Cases".*

## HARD CONSTRAINTS
- Titles only — no steps, no preconditions, no expected results.
- Each scenario must be unique, atomic, and independently executable.
- Minimum 12 scenarios total."""

PROMPT_P3 = """You are a Senior QA Test Architect writing execution-ready test cases for manual testers and automation engineers.

## YOUR ROLE
Transform each validated scenario title into a complete, precise, and executable test case.

## QUALITY STANDARDS
- **Preconditions**: exact system state, user role, data setup
- **Steps**: one action per step — no vague verbs like "verify" or "check"
- **Expected Result**: observable, verifiable — reference exact messages, UI states, HTTP codes, DB changes
- **Failure Signature**: what a tester sees when the test FAILS
- **Test Data**: include specific values directly in the steps
- **Automation Hint**: flag if good candidate for automation and why

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
- [Exact system state]
- [User role / permissions]
- [Required test data]

**🔢 Test Steps:**
1. [Navigate to / Open / Enter / Click — precise action with exact data]
2. ...

**✅ Expected Result:**
[Exact observable outcome]

**🔴 Failure Signature:**
[What the tester sees when this test FAILS]

---

## HARD CONSTRAINTS
- Never write vague expected results.
- Never invent business rules not in the User Story.
- Use real specific test data values in steps.
- If a rule is unclear: ⚠️ *Assumption: [assumption] — confirm with PO.*"""

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
        av = "🧑‍💻" if m["role"] == "user" else "🤖"
        with st.chat_message(m["role"], avatar=av):
            st.markdown(m["content"])

def reset_from_phase(from_phase):
    """Reset all phases >= from_phase."""
    if from_phase <= 1:
        st.session_state.p1_msgs = []
        st.session_state.p1_validated = False
        st.session_state.us_submitted = False
        st.session_state.p1_context = ""
    if from_phase <= 2:
        st.session_state.p2_msgs = []
        st.session_state.p2_validated = False
        st.session_state.p2_draft = ""
    if from_phase <= 3:
        st.session_state.p3_msgs = []
    st.session_state.phase_reached = from_phase
    st.session_state.confirm_reset_p1 = False
    st.session_state.confirm_reset_p2 = False
    st.session_state.confirm_reset_p3 = False

# ═════════════════════════════════════════════════════════════════════════════
st.title("🧪 QA Copilot — AI Test Case Generator")

if not api_key:
    st.warning("⚠️ Enter your Gemini API key in the sidebar to get started.")
    st.stop()
if not model_choice.strip():
    st.warning("⚠️ Enter a model ID in the sidebar.")
    st.stop()

pr = st.session_state.phase_reached

# ── TAB LABELS with status icons ─────────────────────────────────────────────
def tab_label(n, name):
    if st.session_state.phase_reached > n:
        return f"✅ Phase {n} — {name}"
    elif st.session_state.phase_reached == n:
        return f"▶ Phase {n} — {name}"
    else:
        return f"🔒 Phase {n} — {name}"

tab1, tab2, tab3 = st.tabs([
    tab_label(1, "Analysis"),
    tab_label(2, "Test Plan"),
    tab_label(3, "Test Cases"),
])

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 1 — PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="badge b1">🔍 Phase 1 — Senior QA Analyst: Requirements Analysis</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        st.markdown("### 📝 Submit your User Story")
        us_input = st.text_area(
            "User Story + Acceptance Criteria", height=200, key="us_input_ta",
            placeholder="As a [user], I want to [action] so that [benefit].\n\nAcceptance Criteria:\n- ..."
        )
        uploaded = st.file_uploader("📎 Wireframe / Figma screenshot (optional)", type=["png","jpg","jpeg","webp"], key="p1_upload")

        if st.button("🚀 Start Analysis", type="primary", use_container_width=True, key="p1_start"):
            if not us_input.strip():
                st.warning("Please enter a User Story.")
            else:
                image_pil = Image.open(uploaded) if uploaded else None
                prompt = f"Please analyze the following User Story:\n\n{us_input}"
                if uploaded:
                    prompt += "\n\n[A wireframe has been provided — analyze it alongside the User Story.]"
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
        # ── Re-analyze button (with confirmation) ──
        if st.session_state.confirm_reset_p1:
            st.warning("⚠️ This will **reset Phase 1, Phase 2 and Phase 3**. All generated content will be lost.")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("✅ Yes, reset everything", type="primary", use_container_width=True, key="confirm_yes_p1"):
                    reset_from_phase(1)
                    st.rerun()
            with rc2:
                if st.button("❌ Cancel", use_container_width=True, key="confirm_no_p1"):
                    st.session_state.confirm_reset_p1 = False
                    st.rerun()
        else:
            if st.button("🔄 Re-analyze (new User Story)", use_container_width=True, key="p1_reanalyze"):
                st.session_state.confirm_reset_p1 = True
                st.rerun()

        render_chat(st.session_state.p1_msgs)

        st.divider()
        reply = st.text_area("💬 Answer the clarifying questions:", height=130, key="p1_reply_ta")
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
            if st.button("✅ Validate Analysis → Phase 2", type="primary", use_container_width=True, key="p1_validate"):
                # If phase 2 already done, warn reset
                if st.session_state.p2_validated or st.session_state.p2_msgs:
                    st.session_state.confirm_reset_p2 = True
                    st.rerun()
                else:
                    context_msg = f"Validated context from Phase 1:\n\n{st.session_state.p1_context}\n\nNow generate the test plan (titles only)."
                    with st.spinner("📋 Generating test plan…"):
                        try:
                            response = call_gemini([], PROMPT_P2, context_msg)
                            st.session_state.p2_msgs = [
                                {"role": "user", "content": context_msg},
                                {"role": "assistant", "content": response},
                            ]
                            st.session_state.p2_draft = response
                            st.session_state.p1_validated = True
                            if st.session_state.phase_reached < 2:
                                st.session_state.phase_reached = 2
                            st.rerun()
                        except Exception as e:
                            handle_error(e)

        # Confirm reset p2+p3 when re-validating p1
        if st.session_state.confirm_reset_p2 and not st.session_state.confirm_reset_p1:
            st.warning("⚠️ Re-validating Phase 1 will **reset Phase 2 and Phase 3**.")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("✅ Yes, regenerate Phase 2 & 3", type="primary", use_container_width=True, key="confirm_regen_p2"):
                    reset_from_phase(2)
                    context_msg = f"Validated context from Phase 1:\n\n{st.session_state.p1_context}\n\nNow generate the test plan (titles only)."
                    with st.spinner("📋 Generating test plan…"):
                        try:
                            response = call_gemini([], PROMPT_P2, context_msg)
                            st.session_state.p2_msgs = [
                                {"role": "user", "content": context_msg},
                                {"role": "assistant", "content": response},
                            ]
                            st.session_state.p2_draft = response
                            st.session_state.p1_validated = True
                            st.session_state.phase_reached = 2
                            st.rerun()
                        except Exception as e:
                            handle_error(e)
            with rc2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_regen_p2"):
                    st.session_state.confirm_reset_p2 = False
                    st.rerun()

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 2 — PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    if st.session_state.phase_reached < 2:
        st.markdown('<div class="lock-box"><h3>🔒 Phase 2 Locked</h3><p>Complete and validate <strong>Phase 1 — Analysis</strong> first.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge b2">📋 Phase 2 — Lead QA Engineer: Test Plan</div>', unsafe_allow_html=True)

        if st.session_state.confirm_reset_p3 and not st.session_state.confirm_reset_p2:
            st.warning("⚠️ Re-validating Phase 2 will **reset Phase 3** (test cases).")
            rc1, rc2 = st.columns(2)
            with rc1:
                if st.button("✅ Yes, regenerate Phase 3", type="primary", use_container_width=True, key="confirm_regen_p3"):
                    reset_from_phase(3)
                    plan_msg = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nUser Story context:\n{st.session_state.p1_context}\n\nGenerate COMPLETE and DETAILED test cases for every scenario."
                    with st.spinner("📝 Generating test cases…"):
                        try:
                            response = call_gemini([], PROMPT_P3, plan_msg)
                            st.session_state.p3_msgs = [
                                {"role": "user", "content": plan_msg},
                                {"role": "assistant", "content": response},
                            ]
                            st.session_state.p2_validated = True
                            st.session_state.phase_reached = 3
                            st.rerun()
                        except Exception as e:
                            handle_error(e)
            with rc2:
                if st.button("❌ Cancel", use_container_width=True, key="cancel_regen_p3"):
                    st.session_state.confirm_reset_p3 = False
                    st.rerun()

        render_chat(st.session_state.p2_msgs)
        st.divider()
        reply2 = st.text_area("💬 Request changes to the test plan:", height=100, key="p2_reply_ta")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📨 Update Plan", use_container_width=True, key="p2_send"):
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
            if st.button("✅ Validate Plan → Phase 3", type="primary", use_container_width=True, key="p2_validate"):
                if st.session_state.p3_msgs:
                    st.session_state.confirm_reset_p3 = True
                    st.rerun()
                else:
                    plan_msg = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nUser Story context:\n{st.session_state.p1_context}\n\nGenerate COMPLETE and DETAILED test cases for every scenario."
                    with st.spinner("📝 Generating detailed test cases…"):
                        try:
                            response = call_gemini([], PROMPT_P3, plan_msg)
                            st.session_state.p3_msgs = [
                                {"role": "user", "content": plan_msg},
                                {"role": "assistant", "content": response},
                            ]
                            st.session_state.p2_validated = True
                            if st.session_state.phase_reached < 3:
                                st.session_state.phase_reached = 3
                            st.rerun()
                        except Exception as e:
                            handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  TAB 3 — PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    if st.session_state.phase_reached < 3:
        st.markdown('<div class="lock-box"><h3>🔒 Phase 3 Locked</h3><p>Complete and validate <strong>Phase 2 — Test Plan</strong> first.</p></div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="badge b3">📝 Phase 3 — Test Architect: Detailed Test Cases</div>', unsafe_allow_html=True)

        render_chat(st.session_state.p3_msgs)

        if st.session_state.p3_msgs:
            all_content = "\n\n".join([m["content"] for m in st.session_state.p3_msgs if m["role"] == "assistant"])
            st.divider()
            c1, c2, c3 = st.columns(3)
            with c1:
                st.download_button("📥 Export .md", data=all_content,
                                   file_name="test_cases_QA.md", mime="text/markdown", use_container_width=True)
            with c2:
                st.download_button("📄 Export .txt", data=all_content,
                                   file_name="test_cases_QA.txt", mime="text/plain", use_container_width=True)
            with c3:
                if st.button("🔄 Regenerate from Plan", use_container_width=True, key="p3_regen_btn"):
                    st.session_state.confirm_reset_p3 = True
                    # trigger from tab3 directly
                    plan_msg = f"Validated test plan:\n\n{st.session_state.p2_draft}\n\nUser Story context:\n{st.session_state.p1_context}\n\nGenerate COMPLETE and DETAILED test cases for every scenario."
                    with st.spinner("📝 Regenerating test cases…"):
                        try:
                            response = call_gemini([], PROMPT_P3, plan_msg)
                            st.session_state.p3_msgs = [
                                {"role": "user", "content": plan_msg},
                                {"role": "assistant", "content": response},
                            ]
                            st.session_state.confirm_reset_p3 = False
                            st.rerun()
                        except Exception as e:
                            handle_error(e)

        st.divider()
        reply3 = st.text_area("💬 Request adjustments or additional test cases:", height=100, key="p3_reply_ta")
        if st.button("📨 Send", use_container_width=True, key="p3_send"):
            if reply3.strip():
                st.session_state.p3_msgs.append({"role": "user", "content": reply3})
                with st.spinner("Updating test cases…"):
                    try:
                        response = call_gemini(st.session_state.p3_msgs[:-1], PROMPT_P3, reply3)
                        st.session_state.p3_msgs.append({"role": "assistant", "content": response})
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
