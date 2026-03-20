import streamlit as st
import google.generativeai as genai
from PIL import Image

st.set_page_config(
    page_title="QA Copilot – 3 Phases",
    page_icon="🧪",
    layout="wide"
)

st.markdown("""
<style>
.phase-badge {
    display: inline-block;
    padding: 8px 20px;
    border-radius: 20px;
    font-weight: bold;
    font-size: 15px;
    margin-bottom: 16px;
}
.phase-1 { background-color: #1a3a5c; color: #60aaff; border: 1px solid #2255aa; }
.phase-2 { background-color: #1a3a25; color: #60cc88; border: 1px solid #226644; }
.phase-3 { background-color: #3a1a2a; color: #cc6699; border: 1px solid #882255; }
.stepper { display: flex; align-items: center; gap: 8px; margin-bottom: 20px; flex-wrap: wrap; }
.step { padding: 6px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; }
.step-active { background: #1e3a5f; color: #60aaff; border: 1px solid #2255aa; }
.step-done { background: #1a3a25; color: #60cc88; border: 1px solid #226644; }
.step-pending { background: #222; color: #555; border: 1px solid #333; }
.step-arrow { color: #444; font-size: 18px; }
.val-box { background:#0d1a0d; border:1px solid #2a5a2a; border-radius:8px; padding:16px; margin-top:16px; }
</style>
""", unsafe_allow_html=True)

# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")
    api_key = st.text_input("Clé API Gemini", type="password",
                             help="Obtenez votre clé sur https://aistudio.google.com/")
    model_choice = st.selectbox("Modèle Gemini", [
        "gemini-1.5-pro", "gemini-1.5-flash", "gemini-2.0-flash"
    ])
    st.divider()
    st.markdown("""
### 🗺️ Comment ça marche
1. **Phase 1** – Soumettez votre US  
   L'IA pose des questions → répondez → validez  
2. **Phase 2** – Plan de test  
   L'IA liste les scénarios → modifiez → validez  
3. **Phase 3** – Cas détaillés  
   L'IA rédige tout → exportez
""")
    st.divider()
    if st.button("🔄 Nouvelle session", use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()

# ── SESSION STATE ─────────────────────────────────────────────────────────────
def init_state():
    defaults = {
        "phase": 1,
        "p1_msgs": [],
        "p2_msgs": [],
        "p3_msgs": [],
        "p1_done": False,
        "p2_done": False,
        "us_text": "",
        "us_submitted": False,
        "p1_context": "",
        "p2_draft": "",
        "p1_reply": "",
        "p2_reply": "",
        "p3_reply": "",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ── PROMPTS ───────────────────────────────────────────────────────────────────
PROMPT_P1 = """Tu es un Analyste QA expérimenté.
MISSION : Analyser la User Story et poser des questions de clarification uniquement.
INTERDIT : générer des cas de test, des titres de scénarios, ou un plan de test.

Identifie les zones floues : règles métier manquantes, comportements d'erreur non définis,
validations de champs, dépendances systèmes, cas limites non précisés.

FORMAT DE RÉPONSE OBLIGATOIRE :

🔍 **PHASE 1 — Analyse & Clarifications**

**Compréhension actuelle :**
[2-3 phrases résumant l'US]

**Questions de clarification :**
1. [Question précise]
2. [Question précise]
...

*Répondez à ces questions pour passer à la Phase 2.*

RAPPEL : Ne jamais proposer de scénarios ni inventer une règle métier."""

PROMPT_P2 = """Tu es un Lead QA.
Tu disposes d'une User Story et de ses clarifications validées.
MISSION : Générer UNIQUEMENT les TITRES des scénarios de test. Pas de détails.
Couvre : cas nominaux, alternatifs, erreurs, edge cases/boundary.

FORMAT OBLIGATOIRE :

📋 **PHASE 2 — Plan de Test (Draft)**

**Contexte :**
[2-3 phrases]

**✅ Cas Nominaux (Happy Path) :**
- [Titre]

**🔄 Cas Alternatifs :**
- [Titre]

**❌ Cas d'Erreur :**
- [Titre]

**⚠️ Edge Cases / Boundary :**
- [Titre]

*Validez ce plan ou demandez des modifications.*

INTERDIT : rédiger pré-requis, étapes ou résultats attendus."""

PROMPT_P3 = """Tu es un Expert QA (Senior Test Architect).
Tu as une liste de scénarios VALIDÉS. Rédige les cas de test COMPLETS pour chaque scénario.

FORMAT OBLIGATOIRE pour chaque cas :

---

### CAS DE TEST [N] : [Titre]

| Champ | Détail |
|-------|--------|
| **ID** | TC-[N] |
| **Type** | [Nominal / Alternatif / Erreur / Limite] |
| **Priorité** | [Haute / Moyenne / Basse] |

**📌 Pré-requis :**
- [Pré-requis]

**🔢 Étapes de Reproduction :**
1. [Étape concrète]
2. ...

**✅ Résultat Attendu :**
[Description vérifiable]

**🔴 Résultat en cas d'échec :**
[Ce que verrait le testeur]

---

CONTRAINTES : résultats attendus vérifiables, ne jamais inventer de règle métier."""


# ── APPEL GEMINI ──────────────────────────────────────────────────────────────
def call_gemini(history, system_prompt, user_message, image=None):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_choice,
        system_instruction=system_prompt
    )
    chat_history = []
    for m in history:
        role = "user" if m["role"] == "user" else "model"
        chat_history.append({"role": role, "parts": [m["content"]]})
    chat = model.start_chat(history=chat_history)
    parts = [user_message]
    if image:
        parts.append(image)
    return chat.send_message(parts).text


# ── STEPPER ───────────────────────────────────────────────────────────────────
def render_stepper():
    p = st.session_state.phase
    html = '<div class="stepper">'
    steps = [("🔍 Phase 1 — Analyse", 1), ("📋 Phase 2 — Plan de test", 2), ("📝 Phase 3 — Cas détaillés", 3)]
    for i, (lbl, n) in enumerate(steps):
        if n < p:   css = "step step-done";    icon = "✅ "
        elif n == p: css = "step step-active"; icon = "▶ "
        else:        css = "step step-pending"; icon = "⏳ "
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
#  PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
render_stepper()
st.title("🧪 QA Copilot — Générateur de Cas de Tests")

if not api_key:
    st.warning("⚠️ Entrez votre clé API Gemini dans la barre latérale pour commencer.")
    st.stop()

if st.session_state.phase == 1:
    st.markdown('<div class="phase-badge phase-1">🔍 Phase 1 — Analyste : Clarifications</div>', unsafe_allow_html=True)

    # ── Saisie initiale US ──
    if not st.session_state.us_submitted:
        st.markdown("### 📝 Soumettez votre User Story")
        us_input = st.text_area(
            "User Story + Critères d'acceptation",
            height=180,
            placeholder="Ex : En tant qu'utilisateur, je veux me connecter avec email/mot de passe..."
        )
        uploaded = st.file_uploader("📎 Maquette / Capture Figma (optionnel)", type=["png","jpg","jpeg","webp"])

        if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
            if not us_input.strip():
                st.warning("Veuillez saisir une User Story.")
            else:
                st.session_state.us_text = us_input
                image_pil = Image.open(uploaded) if uploaded else None
                prompt = f"Voici la User Story à analyser :\n\n{us_input}"
                if uploaded:
                    prompt += "\n\n[Une image de maquette a été fournie, analyse-la également.]"
                with st.spinner("🔍 Analyse en cours…"):
                    try:
                        response = call_gemini([], PROMPT_P1, prompt, image_pil)
                        st.session_state.p1_msgs = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p1_context = f"US : {us_input}\n\nRéponse initiale :\n{response}"
                        st.session_state.us_submitted = True
                    except Exception as e:
                        st.error(f"Erreur Gemini : {e}")
                st.rerun()

    # ── Conversation Phase 1 ──
    else:
        render_chat(st.session_state.p1_msgs)

        reply = st.text_area("💬 Répondez aux questions de clarification :", key="p1_ta", height=120)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📨 Envoyer la réponse", use_container_width=True):
                if reply.strip():
                    st.session_state.p1_msgs.append({"role": "user", "content": reply})
                    with st.spinner("L'IA traite votre réponse…"):
                        try:
                            response = call_gemini(
                                st.session_state.p1_msgs[:-1], PROMPT_P1, reply
                            )
                            st.session_state.p1_msgs.append({"role": "assistant", "content": response})
                            st.session_state.p1_context += f"\n\nQ: {reply}\nR: {response}"
                        except Exception as e:
                            st.error(f"Erreur : {e}")
                    st.rerun()
        with c2:
            if st.button("✅ Valider l'analyse → Passer à la Phase 2", type="primary", use_container_width=True):
                context_msg = f"""Contexte validé par le testeur :

{st.session_state.p1_context}

Génère maintenant le plan de test (titres uniquement)."""
                with st.spinner("📋 Génération du plan de test…"):
                    try:
                        response = call_gemini([], PROMPT_P2, context_msg)
                        st.session_state.p2_msgs = [
                            {"role": "user", "content": context_msg},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p2_draft = response
                        st.session_state.p1_done = True
                        st.session_state.phase = 2
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    st.markdown('<div class="phase-badge phase-2">📋 Phase 2 — QA Lead : Plan de Test</div>', unsafe_allow_html=True)

    render_chat(st.session_state.p2_msgs)

    reply2 = st.text_area("💬 Demandez des modifications au plan :", key="p2_ta", height=100)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📨 Modifier le plan", use_container_width=True):
            if reply2.strip():
                st.session_state.p2_msgs.append({"role": "user", "content": reply2})
                with st.spinner("Mise à jour du plan…"):
                    try:
                        response = call_gemini(
                            st.session_state.p2_msgs[:-1], PROMPT_P2, reply2
                        )
                        st.session_state.p2_msgs.append({"role": "assistant", "content": response})
                        st.session_state.p2_draft = response
                    except Exception as e:
                        st.error(f"Erreur : {e}")
                st.rerun()
    with c2:
        if st.button("✅ Valider le plan → Générer les cas détaillés (Phase 3)", type="primary", use_container_width=True):
            plan_msg = f"""Plan de test VALIDÉ :

{st.session_state.p2_draft}

Contexte US :
{st.session_state.p1_context}

Génère les cas de tests COMPLETS et DÉTAILLÉS pour chaque scénario."""
            with st.spinner("📝 Génération des cas de tests détaillés…"):
                try:
                    response = call_gemini([], PROMPT_P3, plan_msg)
                    st.session_state.p3_msgs = [
                        {"role": "user", "content": plan_msg},
                        {"role": "assistant", "content": response},
                    ]
                    st.session_state.p2_done = True
                    st.session_state.phase = 3
                except Exception as e:
                    st.error(f"Erreur : {e}")
            st.rerun()


# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 3
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 3:
    st.markdown('<div class="phase-badge phase-3">📝 Phase 3 — Expert : Cas de Tests Détaillés</div>', unsafe_allow_html=True)

    render_chat(st.session_state.p3_msgs)

    if st.session_state.p3_msgs:
        all_content = "\n\n".join(
            [m["content"] for m in st.session_state.p3_msgs if m["role"] == "assistant"]
        )
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Exporter en Markdown (.md)", data=all_content,
                               file_name="cas_de_tests_QA.md", mime="text/markdown",
                               use_container_width=True)
        with c2:
            st.download_button("📄 Exporter en Texte (.txt)", data=all_content,
                               file_name="cas_de_tests_QA.txt", mime="text/plain",
                               use_container_width=True)

    st.divider()
    reply3 = st.text_area("💬 Demandez des ajustements ou des cas supplémentaires :", key="p3_ta", height=100)
    if st.button("📨 Envoyer", use_container_width=True):
        if reply3.strip():
            st.session_state.p3_msgs.append({"role": "user", "content": reply3})
            with st.spinner("Mise à jour des cas de tests…"):
                try:
                    response = call_gemini(
                        st.session_state.p3_msgs[:-1], PROMPT_P3, reply3
                    )
                    st.session_state.p3_msgs.append({"role": "assistant", "content": response})
                except Exception as e:
                    st.error(f"Erreur : {e}")
            st.rerun()
