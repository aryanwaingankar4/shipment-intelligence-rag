"""Streamlit front end. Thin client: all RAG logic lives behind the API (API_URL),
so the UI and API scale and deploy independently."""

import hmac
import html
import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8080").rstrip("/")
API_KEY = os.environ.get("API_KEY", "")
REQUEST_TIMEOUT_S = float(os.environ.get("REQUEST_TIMEOUT_S", "90"))

COMPANY_NAME = os.environ.get("COMPANY_NAME", "NexaChain BioMed")
TAGLINE = "Intelligent Pharmaceutical Supply Chain"
# Demo login only. Replace with SSO (OIDC) before real users — see PROJECT_STATUS.md.
UI_USERNAME = os.environ.get("UI_USERNAME", "demo")
UI_PASSWORD = os.environ.get("UI_PASSWORD", "demo123")
SHOW_DEMO_HINT = os.environ.get("SHOW_DEMO_HINT", "true").lower() == "true"

st.set_page_config(page_title=COMPANY_NAME, page_icon="\U0001F9EC",
                   layout="centered", initial_sidebar_state="collapsed")

# ---------------- DESIGN SYSTEM (CSS) — CLINICAL PRECISION ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:#F1F5F9; color:#334155; }
#MainMenu, footer, header {visibility:hidden;}
.block-container { padding-top:2rem; padding-bottom:2rem; max-width:900px; }

h1,h2,h3,h4 { color:#0A1628; font-weight:700; letter-spacing:-0.01em; }
h3 { font-size:1.4rem; }
p { font-size:0.92rem; }
.subtle { color:#94A3B8; font-size:0.82rem; line-height:1.55; }

.brand-row { display:flex; align-items:center; gap:.7rem; }
.brand-name { font-size:1.4rem; font-weight:700; color:#0A1628; }
.brand-tag  { color:#94A3B8; font-size:.82rem; margin-top:-.2rem; }

.rule { border:none; border-top:2px solid #2563EB; width:40px;
        margin:1.2rem auto 1.4rem auto; }

.card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px;
        padding:1.4rem 1.5rem; height:100%; position:relative;
        box-shadow:0 2px 12px rgba(10,22,40,0.07);
        border-top:3px solid transparent;
        transition:box-shadow .18s ease, border-top-color .18s ease; }
.card:hover { box-shadow:0 8px 28px rgba(37,99,235,0.10);
        border-top:3px solid #2563EB; }
.card h4 { margin:0 0 .5rem 0; font-size:1.02rem; color:#0A1628; }
.card p  { margin:0; color:#94A3B8; font-size:.88rem; line-height:1.55; }

.login-card { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:16px;
        padding:2.5rem; max-width:420px; margin:0 auto;
        box-shadow:0 8px 30px rgba(10,22,40,0.18); }
.login-header { text-align:center; margin-bottom:1.2rem; }
.login-header .brand-tag { color:#94A3B8; }

.stButton > button { background:#2563EB; color:#fff; border:none;
        border-radius:10px; padding:.6rem 1.5rem; font-weight:600;
        font-family:'DM Sans',sans-serif; transition:background .18s ease,
        box-shadow .18s ease; }
.stButton > button:hover { background:#1D4ED8; color:#fff;
        box-shadow:0 4px 14px rgba(37,99,235,0.25); }

.ghost .stButton > button { background:#FFFFFF; color:#334155;
        border:1px solid #E2E8F0; font-weight:500; }
.ghost .stButton > button:hover { background:#F8FAFC; color:#0A1628;
        border:1px solid #CBD5E1; box-shadow:none; }

.stTextInput > div > div > input,
.stTextArea textarea {
        background:#0A1628; color:#FFFFFF; border-radius:10px;
        border:1px solid #1E293B; }
.stTextInput > div > div > input:focus,
.stTextArea textarea:focus { outline:2px solid #2563EB; border:1px solid #2563EB; }
.stTextInput > div > div > input::placeholder { color:#94A3B8; }

.chat-row { display:flex; align-items:flex-start; gap:.6rem; margin:.55rem 0; }
.chat-row.user { justify-content:flex-end; }
.avatar { width:36px; height:36px; border-radius:50%; flex:0 0 36px;
        display:flex; align-items:center; justify-content:center;
        color:#fff; font-weight:700; font-size:.9rem; }
.avatar.nova { background:#2563EB; }
.avatar.user { background:#DC2626; }
.bubble { max-width:74%; padding:.7rem 1rem; font-size:.92rem; line-height:1.55;
        white-space:pre-wrap; }
.bubble.nova { background:#FFFFFF; color:#334155; border:1px solid #E2E8F0;
        border-radius:18px 18px 18px 4px; }
.bubble.user { background:#0A1628; color:#fff;
        border-radius:18px 18px 4px 18px; }
.sources { color:#94A3B8; font-size:.76rem; margin:.15rem 0 0 2.9rem; }

.error-card { background:#FFFFFF; border:1px solid #FCA5A5;
        border-left:4px solid #DC2626; border-radius:12px;
        padding:.9rem 1.1rem; color:#334155; font-size:.9rem; }
.error-card b { color:#DC2626; }

@media (max-width: 640px) {
    [data-testid="column"] { width:100% !important; flex:1 1 100% !important; }
    .card { margin-bottom:.9rem; }
    .login-card { padding:1.6rem; }
}
</style>
""", unsafe_allow_html=True)

DNA_SVG = ('<svg width="26" height="26" viewBox="0 0 24 24" fill="none" '
           'stroke="#2563EB" stroke-width="2" stroke-linecap="round">'
           '<path d="M5 3c0 6 14 6 14 18M19 3c0 6-14 6-14 18"/>'
           '<path d="M7 6h10M7 18h10M9 9h6M9 15h6"/></svg>')
CHAT_SVG = ('<svg width="24" height="24" viewBox="0 0 24 24" fill="none" '
            'stroke="#2563EB" stroke-width="2" stroke-linecap="round" '
            'stroke-linejoin="round"><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 '
            '2 0 0 1 2-2h14a2 2 0 0 1 2 2z"/></svg>')


class AssistantUnavailable(Exception):
    pass


def ask_api(question: str, custom_instructions: str) -> dict:
    headers = {"X-API-Key": API_KEY} if API_KEY else {}
    try:
        r = requests.post(
            f"{API_URL}/v1/query",
            json={"question": question, "custom_instructions": custom_instructions},
            headers=headers,
            timeout=REQUEST_TIMEOUT_S,
        )
    except requests.RequestException as exc:
        raise AssistantUnavailable() from exc
    if r.status_code != 200:
        raise AssistantUnavailable()
    return r.json()


# ---------------- SESSION STATE ----------------
ss = st.session_state
ss.setdefault("authenticated", False)
ss.setdefault("page", "home")
ss.setdefault("messages", [])
ss.setdefault("assistant_name", "Nova")
ss.setdefault("custom_instructions", "")
ss.setdefault("login_error", False)


def check_credentials(user: str, password: str) -> bool:
    user_ok = hmac.compare_digest(user.encode(), UI_USERNAME.encode())
    pass_ok = hmac.compare_digest(password.encode(), UI_PASSWORD.encode())
    return user_ok and pass_ok


# ===================== LOGIN PAGE =====================
def login_page():
    st.markdown(
        f"<div class='login-header'><div style='display:flex;align-items:center;"
        f"justify-content:center;gap:.6rem;'>{DNA_SVG}"
        f"<span class='brand-name'>{html.escape(COMPANY_NAME)}</span></div>"
        f"<p class='brand-tag'>{TAGLINE}</p></div>",
        unsafe_allow_html=True)
    st.write("")

    col = st.columns([1, 5, 1])[1]
    with col:
        st.markdown("#### Sign in to your workspace")
        u = st.text_input("Username", placeholder="Username", label_visibility="collapsed")
        p = st.text_input("Password", type="password", placeholder="Password",
                          label_visibility="collapsed")
        if st.button("Sign In", use_container_width=True):
            if check_credentials(u, p):
                ss.authenticated = True
                ss.login_error = False
                st.rerun()
            else:
                ss.login_error = True
        if ss.login_error:
            st.markdown("<div class='error-card'><b>Sign in failed.</b> "
                        "Invalid username or password.</div>", unsafe_allow_html=True)
        if SHOW_DEMO_HINT:
            st.markdown(f"<p class='subtle' style='margin-top:.8rem'>Demo access — "
                        f"username: <b>{html.escape(UI_USERNAME)}</b> &middot; password: "
                        f"<b>{html.escape(UI_PASSWORD)}</b></p>", unsafe_allow_html=True)


# ===================== HOME / DASHBOARD =====================
def home_page():
    top = st.columns([5, 1])
    with top[0]:
        st.markdown(f"<div class='brand-row'>{DNA_SVG}"
                    f"<span class='brand-name'>{html.escape(COMPANY_NAME)}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<p class='brand-tag'>{TAGLINE}</p>", unsafe_allow_html=True)
    with top[1]:
        st.markdown("<div class='ghost'>", unsafe_allow_html=True)
        if st.button("Sign Out"):
            ss.authenticated = False
            ss.page = "home"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### Welcome back")
    st.markdown(f"<p class='subtle'>{html.escape(COMPANY_NAME)} delivers precision cold-chain "
                "and pharmaceutical logistics with full regulatory traceability. "
                "Our AI assistant gives your team instant, document-grounded "
                "answers on products, shipments, vendors, and compliance — "
                "available 24/7.</p>", unsafe_allow_html=True)
    st.markdown("<hr class='rule'>", unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("<div class='card'><h4>Cold Chain Monitoring</h4><p>End-to-end "
                    "temperature-controlled distribution with real-time sensor "
                    "monitoring and automated alerts.</p></div>", unsafe_allow_html=True)
    with c2:
        st.markdown("<div class='card'><h4>Regulatory Compliance</h4><p>WHO-GMP "
                    "and FDA-aligned quality management at every step. Full audit "
                    "trail and document traceability.</p></div>", unsafe_allow_html=True)
    with c3:
        st.markdown("<div class='card'><h4>AI-Powered Insights</h4><p>Instant "
                    "answers grounded in your verified company knowledge base. "
                    "Document-cited responses only.</p></div>", unsafe_allow_html=True)

    st.write("")
    cta = st.columns([1, 1, 1])
    with cta[1]:
        if st.button(f"Chat with {ss.assistant_name}", use_container_width=True):
            ss.page = "chat"
            st.rerun()

    st.write("")
    with st.expander("Personalize your assistant"):
        name = st.text_input("Assistant name", value=ss.assistant_name, max_chars=30)
        instr = st.text_area("Custom instructions (optional)",
                             value=ss.custom_instructions, max_chars=500)
        if st.button("Save"):
            ss.assistant_name = name.strip() or "Nova"
            ss.custom_instructions = instr.strip()
            st.success(f"Saved. Assistant name set to {ss.assistant_name}.")
            st.rerun()


# ===================== CHAT PAGE =====================
def render_message(role, content, sources=None):
    # escape everything user- or model-generated before it goes into raw HTML
    safe = html.escape(content)
    if role == "user":
        st.markdown(f"<div class='chat-row user'><div class='bubble user'>{safe}</div>"
                    f"<div class='avatar user'>U</div></div>", unsafe_allow_html=True)
        return
    initial = html.escape(ss.assistant_name[:1].upper() or "N")
    st.markdown(f"<div class='chat-row'><div class='avatar nova'>{initial}</div>"
                f"<div class='bubble nova'>{safe}</div></div>", unsafe_allow_html=True)
    if sources:
        names = sorted({s["source"] for s in sources})
        st.markdown(f"<div class='sources'>Sources: {html.escape(', '.join(names))}</div>",
                    unsafe_allow_html=True)


def chat_page():
    bot = ss.assistant_name
    top = st.columns([5, 1])
    with top[0]:
        st.markdown(f"<div class='brand-row'>{CHAT_SVG}"
                    f"<span class='brand-name'>{html.escape(bot)}</span></div>",
                    unsafe_allow_html=True)
        st.markdown(f"<p class='subtle'>Your {html.escape(COMPANY_NAME)} supply-chain "
                    "assistant. Answers are grounded in verified company "
                    "documents.</p>", unsafe_allow_html=True)
    with top[1]:
        st.markdown("<div class='ghost'>", unsafe_allow_html=True)
        if st.button("Back"):
            ss.page = "home"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    for msg in ss.messages:
        render_message(msg["role"], msg["content"], msg.get("sources"))

    if prompt := st.chat_input(f"Ask {bot} about products, shipments, compliance..."):
        ss.messages.append({"role": "user", "content": prompt})
        render_message("user", prompt)
        with st.spinner("Thinking..."):
            try:
                data = ask_api(prompt, ss.custom_instructions)
                msg = {"role": "assistant", "content": data["answer"],
                       "sources": data.get("sources", [])}
                ss.messages.append(msg)
                render_message("assistant", msg["content"], msg["sources"])
            except AssistantUnavailable:
                st.markdown(f"<div class='error-card'><b>{html.escape(bot)} is currently "
                            "unavailable.</b> Please try again shortly.</div>",
                            unsafe_allow_html=True)

    st.write("")
    if ss.messages:
        st.markdown("<div class='ghost'>", unsafe_allow_html=True)
        if st.button("Clear conversation"):
            ss.messages = []
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)


# ===================== ROUTER =====================
if not ss.authenticated:
    login_page()
elif ss.page == "chat":
    chat_page()
else:
    home_page()
