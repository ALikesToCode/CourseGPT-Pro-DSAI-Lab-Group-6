import streamlit as st
import streamlit.components.v1 as components

# =====================================================================
#  NOTION-LIKE THEME (LIGHT + DARK)
# =====================================================================

BASE_CSS = r"""
:root {
  --accent: #5763D8;

  /* Light */
  --bg-light: #F7F7F5;
  --panel-light: #FAFAF8;
  --card-light: #FFFFFF;
  --border-light: #E5E5E3;
  --text-light: #1C1C1C;
  --text-muted-light: #6F6F6F;

  /* Dark */
  --bg-dark: #0F1012;
  --panel-dark: #1A1C1F;
  --card-dark: #1F2124;
  --border-dark: #2D2F33;
  --text-dark: #EAEAEA;
  --text-muted-dark: #A8A8A8;

  --radius: 8px;
  --shadow-light: 0px 2px 6px rgba(0, 0, 0, 0.06);
  --shadow-dark: 0px 2px 10px rgba(0, 0, 0, 0.4);

  --app-max-width: 1080px;
}

/* Topbar button sizing and utility tokens */
:root {
  --nav-btn-padding: 8px 16px;
  --nav-btn-radius: 10px;
}

/* =====================================================================
   GLOBAL LAYOUT RESET
===================================================================== */
html, body, .stApp, [data-testid="stAppViewContainer"] {
  margin: 0 !important;
  padding: 0 !important;
  background: var(--bg-light) !important;
  color: var(--text-light) !important;
  font-family: Inter, system-ui, sans-serif !important;
  letter-spacing: -0.1px;
}

.block-container {
  max-width: var(--app-max-width) !important;
  margin: 0 auto !important;
  padding-top: 10px !important;
}

/* Remove empty layout divs */
.block-container > div:empty {
  display: none !important;
}

/* =====================================================================
   TOP BAR — Notion minimal
===================================================================== */
.cg-topbar {
  padding: 16px 0;
  margin-bottom: 14px;
  border-bottom: 1px solid var(--border-light);
}

.cg-logo {
  font-size: 20px;
  font-weight: 600;
  color: var(--accent);
}

/* =====================================================================
   BUTTONS
===================================================================== */
.stButton > button, button {
  background: var(--accent) !important;
  color: white !important;
  border: none !important;
  padding: 8px 15px !important;
  border-radius: 6px !important;
  font-weight: 500 !important;
  box-shadow: var(--shadow-light) !important;
  transition: 0.15s ease;
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;
}

/* Topbar-specific button sizing for consistent nav tabs */
.cg-topbar [data-testid="stButton"] button {
  padding: var(--nav-btn-padding) !important;
  border-radius: var(--nav-btn-radius) !important;
  min-width: 92px !important;
  height: 36px !important;
  white-space: nowrap !important;
}

.stButton > button:hover {
  opacity: 0.92;
  transform: translateY(-1px);
}

/* =====================================================================
   INPUTS — Notion style
===================================================================== */
input, textarea, select,
.stTextInput input, .stTextArea textarea, .stSelectbox select {
  background: var(--panel-light) !important;
  color: var(--text-light) !important;
  border: 1px solid var(--border-light) !important;
  padding: 10px 12px !important;
  border-radius: 6px !important;
  box-shadow: none !important;
}

input:focus, textarea:focus {
  border-color: var(--accent) !important;
}

/* =====================================================================
   CARDS
===================================================================== */
.cg-card {
  background: var(--card-light);
  border-radius: var(--radius);
  padding: 20px;
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-light);
}

/* CHAT BUBBLES */
.msg-user {
  background: #E8E9EB !important;
  color: var(--text-light);
  border-radius: 8px;
  padding: 12px 15px;
  max-width: 78%;
}

.msg-ai {
  background: #E1E5FF !important;
  color: #1A1A1A;
  border-radius: 8px;
  padding: 12px 15px;
  max-width: 78%;
}

/* =====================================================================
   DOCUMENT CARDS
===================================================================== */
.doc-card {
  background: var(--panel-light);
  padding: 16px;
  border-radius: 8px;
  border: 1px solid var(--border-light);
  box-shadow: var(--shadow-light);
  transition: 0.2s ease;
}

.doc-card:hover {
  transform: translateY(-3px);
  box-shadow: 0 4px 14px rgba(0,0,0,0.08);
}

/* Chat column wrapper: limit width and center within its column */
.chat-column {
  max-width: 720px;
  margin-left: 0;
  margin-right: auto;
  padding-right: 12px;
}

/* Slight spacing between document cards and upload area */
.doc-card {
  margin-bottom: 14px;
}

/* Upload area tweaks: make uploader background softer and upload button full width */
.doc-upload [data-testid="stFileUploaderDropzone"] {
  background: #f5f6f8 !important;
  border-radius: 8px !important;
}
.doc-upload [data-testid="stButton"] button {
  width: 100% !important;
}

/* Ensure card buttons don't wrap and have consistent padding */
.doc-card [data-testid="stButton"] button {
  white-space: nowrap !important;
  padding: 6px 12px !important;
}

/* Stronger rule: force inline-flex layout so icon + text never wraps
   and buttons keep consistent sizing inside narrow columns. */
.doc-card [data-testid="stButton"] button,
.doc-card button {
  display: inline-flex !important;
  align-items: center !important;
  justify-content: center !important;
  gap: 8px !important;
  white-space: nowrap !important;
  width: auto !important;
  min-width: 64px !important;
  max-width: none !important;
  box-sizing: border-box !important;
}

/* =====================================================================
   DARK MODE — FULL OVERRIDE
===================================================================== */
html.dark-mode, 
html.dark-mode body,
html.dark-mode .stApp,
html.dark-mode [data-testid="stAppViewContainer"],
html.dark-mode .block-container {
  background: var(--bg-dark) !important;
  color: var(--text-dark) !important;
}

html.dark-mode .cg-topbar {
  border-color: var(--border-dark) !important;
}

html.dark-mode .cg-card {
  background: var(--card-dark) !important;
  border: 1px solid var(--border-dark) !important;
  box-shadow: var(--shadow-dark) !important;
}

html.dark-mode .doc-card {
  background: var(--panel-dark) !important;
  border: 1px solid var(--border-dark) !important;
}

html.dark-mode .msg-user {
  background: #2A2E33 !important;
  color: var(--text-dark) !important;
}

html.dark-mode .msg-ai {
  background: #2D3458 !important;
  color: #E9EAFF !important;
}

/* =====================================================================
   DARK MODE — STREAMLIT WIDGET PATCHES
===================================================================== */
html.dark-mode input,
html.dark-mode textarea,
html.dark-mode select,
html.dark-mode .stTextInput input,
html.dark-mode .stTextArea textarea,
html.dark-mode .stSelectbox select {
  background: var(--panel-dark) !important;
  color: var(--text-dark) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
}

html.dark-mode input::placeholder,
html.dark-mode textarea::placeholder {
  color: rgba(255,255,255,0.5) !important;
}

/* Buttons fix */
html.dark-mode button,
html.dark-mode .stButton > button {
  background: var(--accent) !important;
  box-shadow: none !important;
}

/* Expander */
html.dark-mode [data-testid="stExpander"] > details {
  background: var(--panel-dark) !important;
  border: 1px solid var(--border-dark) !important;
}

/* File uploader */
html.dark-mode [data-testid="stFileUploaderDropzone"] {
  background: var(--panel-dark) !important;
  border: 2px dashed rgba(255,255,255,0.15) !important;
}

/* Selectbox popup */
html.dark-mode [data-baseweb="popover"] div {
  background: var(--panel-dark) !important;
  border: 1px solid var(--border-dark) !important;
}

/* Slider */
html.dark-mode [data-baseweb="slider"] * {
  color: var(--text-dark) !important;
  background: var(--panel-dark) !important;
}
html.dark-mode [role="slider"] {
  background: var(--accent) !important;
}

/* Global text override */
html.dark-mode * {
  color: var(--text-dark) !important;
}

/* =========================================================== */
"""

# =====================================================================
#  JS (unchanged, your working version)
# =====================================================================

JS_TEMPLATE = r"""
<div id="coursegpt-style-injector"></div>
<script>
(function(){
  const MODE = "__MODE__";
  const CSS = `__CSSTEXT__`;

  const targetDoc = (window.parent && window.parent.document)
      ? window.parent.document
      : document;

  function ensureStyle() {
    if (!targetDoc.getElementById('coursegpt-styles')) {
      const s = targetDoc.createElement('style');
      s.id = 'coursegpt-styles';
      s.textContent = CSS;
      (targetDoc.head || targetDoc.documentElement).appendChild(s);
    }
  }

  function applyMode() {
    if (MODE === 'dark') {
      targetDoc.documentElement.classList.add('dark-mode');
      targetDoc.body?.classList.add('dark-mode');
    } else {
      targetDoc.documentElement.classList.remove('dark-mode');
      targetDoc.body?.classList.remove('dark-mode');
    }
  }

  function run(){ ensureStyle(); applyMode(); }
  run();

  new MutationObserver(run).observe(document.body, { childList:true, subtree:true });
})();
</script>
"""

def apply_styles(mode: str = "light"):
    safe_css = BASE_CSS.replace("`", "\\`")
    js = JS_TEMPLATE.replace("__MODE__", "dark" if mode == "dark" else "light") \
                    .replace("__CSSTEXT__", safe_css)
    components.html(js, height=0)
