import streamlit as st
from google import genai
from google.genai import types
from PIL import Image
import io

st.set_page_config(
    page_title="QA Copilot – 3 Phases",
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
        "Clé API Gemini", type="password",
        help="Obtenez votre clé sur https://aistudio.google.com/"
    )

    model_choice = st.text_input(
        "Modèle Gemini",
        value="gemini-2.5-flash-lite-preview-06-17",
        help="Entrez l'identifiant exact du modèle. Trouvez la liste sur https://ai.google.dev/gemini-api/docs/models"
    )
    st.caption("Exemples : `gemini-2.5-flash-lite-preview-06-17` · `gemini-2.0-flash` · `gemini-2.5-pro` · `gemma-3-27b-it`")

    st.divider()
    st.markdown("""
### 🗺️ Comment ça marche
1. **Phase 1** – Soumettez votre US → l'IA pose des questions → répondez → validez
2. **Phase 2** – L'IA liste les scénarios → modifiez → validez
3. **Phase 3** – L'IA rédige tout → exportez
""")
    st.divider()
    if st.button("🔄 Nouvelle session", use_container_width=True):
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
        max_output_tokens=4000,
        temperature=0.25,
    )
    response = client.models.generate_content(
        model=model_choice.strip(),
        contents=contents,
        config=config,
    )
    return response.text

def handle_error(e):
    err = str(e)
    if "429" in err or "RESOURCE_EXHAUSTED" in err:
        st.error("⚠️ Quota épuisé pour ce modèle. Changez l'identifiant du modèle dans la barre latérale.")
    elif "404" in err or "NOT_FOUND" in err:
        st.error(f"⚠️ Modèle introuvable : **{model_choice}**. Vérifiez l'identifiant exact sur https://ai.google.dev/gemini-api/docs/models")
    else:
        st.error(f"Erreur Gemini : {err}")

# ── STEPPER ───────────────────────────────────────────────────────────────────
def render_stepper():
    p = st.session_state.phase
    steps = [("🔍 Phase 1 — Analyse", 1), ("📋 Phase 2 — Plan de test", 2), ("📝 Phase 3 — Cas détaillés", 3)]
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
st.title("🧪 QA Copilot — Générateur de Cas de Tests")

if not api_key:
    st.warning("⚠️ Entrez votre clé API Gemini dans la barre latérale pour commencer.")
    st.stop()

if not model_choice.strip():
    st.warning("⚠️ Entrez un identifiant de modèle dans la barre latérale.")
    st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 1
# ═════════════════════════════════════════════════════════════════════════════
if st.session_state.phase == 1:
    st.markdown('<div class="phase-badge phase-1">🔍 Phase 1 — Analyste : Clarifications</div>', unsafe_allow_html=True)

    if not st.session_state.us_submitted:
        st.markdown("### 📝 Soumettez votre User Story")
        us_input = st.text_area(
            "User Story + Critères d'acceptation", height=180,
            placeholder="Ex : En tant qu'utilisateur, je veux me connecter avec email/mot de passe..."
        )
        uploaded = st.file_uploader("📎 Maquette / Capture Figma (optionnel)", type=["png","jpg","jpeg","webp"])

        if st.button("🚀 Lancer l'analyse", type="primary", use_container_width=True):
            if not us_input.strip():
                st.warning("Veuillez saisir une User Story.")
            else:
                image_pil = Image.open(uploaded) if uploaded else None
                prompt = f"Voici la User Story à analyser :\n\n{us_input}"
                if uploaded:
                    prompt += "\n\n[Une image de maquette a été fournie, analyse-la également.]"
                with st.spinner(f"🔍 Analyse en cours avec `{model_choice.strip()}`…"):
                    try:
                        response = call_gemini([], PROMPT_P1, prompt, image_pil)
                        st.session_state.us_text = us_input
                        st.session_state.p1_msgs = [
                            {"role": "user", "content": prompt},
                            {"role": "assistant", "content": response},
                        ]
                        st.session_state.p1_context = f"US : {us_input}\n\nRéponse initiale :\n{response}"
                        st.session_state.us_submitted = True
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
    else:
        render_chat(st.session_state.p1_msgs)
        reply = st.text_area("💬 Répondez aux questions de clarification :", height=120, key="p1_reply")
        c1, c2 = st.columns(2)
        with c1:
            if st.button("📨 Envoyer la réponse", use_container_width=True):
                if reply.strip():
                    st.session_state.p1_msgs.append({"role": "user", "content": reply})
                    with st.spinner("L'IA traite votre réponse…"):
                        try:
                            response = call_gemini(st.session_state.p1_msgs[:-1], PROMPT_P1, reply)
                            st.session_state.p1_msgs.append({"role": "assistant", "content": response})
                            st.session_state.p1_context += f"\n\nQ: {reply}\nR: {response}"
                            st.rerun()
                        except Exception as e:
                            handle_error(e)
        with c2:
            if st.button("✅ Valider l'analyse → Phase 2", type="primary", use_container_width=True):
                context_msg = f"Contexte validé :\n\n{st.session_state.p1_context}\n\nGénère le plan de test (titres uniquement)."
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
                        st.rerun()
                    except Exception as e:
                        handle_error(e)

# ═════════════════════════════════════════════════════════════════════════════
#  PHASE 2
# ═════════════════════════════════════════════════════════════════════════════
elif st.session_state.phase == 2:
    st.markdown('<div class="phase-badge phase-2">📋 Phase 2 — QA Lead : Plan de Test</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p2_msgs)
    reply2 = st.text_area("💬 Demandez des modifications au plan :", height=100, key="p2_reply")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("📨 Modifier le plan", use_container_width=True):
            if reply2.strip():
                st.session_state.p2_msgs.append({"role": "user", "content": reply2})
                with st.spinner("Mise à jour du plan…"):
                    try:
                        response = call_gemini(st.session_state.p2_msgs[:-1], PROMPT_P2, reply2)
                        st.session_state.p2_msgs.append({"role": "assistant", "content": response})
                        st.session_state.p2_draft = response
                        st.rerun()
                    except Exception as e:
                        handle_error(e)
    with c2:
        if st.button("✅ Valider le plan → Phase 3", type="primary", use_container_width=True):
            plan_msg = f"Plan validé :\n\n{st.session_state.p2_draft}\n\nContexte US :\n{st.session_state.p1_context}\n\nGénère les cas de tests COMPLETS et DÉTAILLÉS."
            with st.spinner("📝 Génération des cas de tests détaillés…"):
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
    st.markdown('<div class="phase-badge phase-3">📝 Phase 3 — Expert : Cas de Tests Détaillés</div>', unsafe_allow_html=True)
    render_chat(st.session_state.p3_msgs)

    if st.session_state.p3_msgs:
        all_content = "\n\n".join([m["content"] for m in st.session_state.p3_msgs if m["role"] == "assistant"])
        st.divider()
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("📥 Exporter en Markdown (.md)", data=all_content,
                               file_name="cas_de_tests_QA.md", mime="text/markdown", use_container_width=True)
        with c2:
            st.download_button("📄 Exporter en Texte (.txt)", data=all_content,
                               file_name="cas_de_tests_QA.txt", mime="text/plain", use_container_width=True)

    st.divider()
    reply3 = st.text_area("💬 Demandez des ajustements ou des cas supplémentaires :", height=100, key="p3_reply")
    if st.button("📨 Envoyer", use_container_width=True):
        if reply3.strip():
            st.session_state.p3_msgs.append({"role": "user", "content": reply3})
            with st.spinner("Mise à jour des cas de tests…"):
                try:
                    response = call_gemini(st.session_state.p3_msgs[:-1], PROMPT_P3, reply3)
                    st.session_state.p3_msgs.append({"role": "assistant", "content": response})
                    st.rerun()
                except Exception as e:
                    handle_error(e)
