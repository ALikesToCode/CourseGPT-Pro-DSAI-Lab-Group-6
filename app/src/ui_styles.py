import streamlit as st
import streamlit.components.v1 as components

# Minimal Notion-like styles: only layout, vars, cards, bubbles, plus sticky composer.
BASE_CSS = r"""
:root{
  --bg-light: #FAFAFB;
  --bg-dark: #0F1115;
  --card-light: #FFFFFF;
  --card-dark: #0F1318;
  --muted: #6B7280;
  --accent: #5763D8;
  --radius: 10px;
  --container-width: 1080px;
}

html, body, .stApp { background: var(--bg-light); color: #0B1220; font-family: Inter, system-ui, -apple-system, 'Segoe UI', Roboto; }
.block-container{ max-width: var(--container-width) !important; margin: 0 auto !important; padding: 18px !important; }

.cg-topbar{ display:flex; align-items:center; gap:18px; padding:8px 0; margin-bottom:16px; }
.cg-logo{ font-weight:600; color:var(--accent); }
.cg-logo-stack{ display:flex; flex-direction:column; line-height:1.2; }
.cg-logo-sub{ font-size:12px; color:var(--muted); }

/* Visual polish */
.cg-topbar{ padding:14px 6px; align-items:center; justify-content:space-between; }
.cg-topbar-inner{ display:flex; align-items:center; gap:18px; width:100%; max-width:var(--container-width); margin:0 auto; }
.cg-left{ flex:0 0 180px; }
.cg-center{ display:flex; gap:12px; justify-content:center; flex:1 1 auto; }
.cg-right{ display:flex; gap:12px; align-items:center; justify-content:flex-end; flex:0 0 220px; }

.cg-logo{ font-weight:700; color:var(--accent); font-size:20px; }
.cg-tab-indicator{ width:100%; height:3px; border-radius:999px; background:transparent; margin-top:4px; }
.cg-tab-indicator.active{ background:var(--accent); box-shadow:0 4px 10px rgba(87,99,216,0.35); }

/* Style Streamlit buttons globally to look modern */
.stButton>button, button {
  border-radius:10px !important;
  padding:8px 14px !important;
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02)) !important;
  border: 1px solid rgba(15,20,25,0.06) !important;
  box-shadow: 0 1px 2px rgba(12,18,24,0.04) inset !important;
  color: inherit !important;
}
.stButton>button:active, button:active{ transform: translateY(1px); }

/* Active-looking tab: slightly raised using box-shadow when focused/active */
.stButton>button[aria-pressed="true"], .stButton>button:focus {
  box-shadow: 0 6px 18px rgba(87,99,216,0.12) !important;
  border-color: rgba(87,99,216,0.18) !important;
}

/* Profile badge */
.cg-profile{ display:flex; align-items:center; gap:10px; justify-content:flex-end; }
.cg-avatar{ width:34px; height:34px; border-radius:50%; background:#E6EEF8; display:inline-flex; align-items:center; justify-content:center; box-shadow: 0 3px 10px rgba(12,18,24,0.06); font-size:14px; }
.cg-profile-label{ font-size:13px; color:var(--muted); }

/* Cards and inputs */
.cg-card{ padding:20px; border-radius:12px; }
.doc-panel{ margin-bottom:24px; }
.doc-card{ padding:16px; border-radius:10px; }
.chat-column .stTextArea>div>textarea{ min-height:120px; border-radius:10px !important; }
.chat-window{ gap:12px; }

/* Input focus */
input:focus, textarea:focus, .stTextInput:focus { outline: 2px solid rgba(87,99,216,0.12) !important; }

/* Stronger base contrast for form elements (light mode) */
.stApp .stTextInput>div>input, .stApp .stTextArea>div>textarea, .stApp .stNumberInput>div>input {
  background: #FFFFFF; color: #0B1220; border: 1px solid rgba(15,20,25,0.12); box-shadow: none; border-radius:8px; padding:10px; }

/* Additional selectors to ensure textarea/input inner elements are visible */
.stApp .stTextArea textarea, .stApp .stTextInput input, .stApp textarea, .stApp input {
  background: #FFFFFF !important; color: #0B1220 !important; border: 1px solid rgba(15,20,25,0.12) !important; padding:10px !important; border-radius:8px !important; box-shadow:none !important;
}

/* File uploader light card */
.stApp .stFileUploader div[role="button"], .stApp .stFileUploader .upload {
  background: #FFFFFF; border: 1px solid rgba(15,20,25,0.06); color: #0B1220; padding:12px; border-radius:8px;
}

/* Ensure placeholder readability in light mode */
.stApp ::placeholder { color: rgba(11,18,32,0.45); }

/* Slightly stronger card borders for contrast */
.cg-card, .doc-card { border: 1px solid rgba(15,20,25,0.06); }

.cg-card{ background:var(--card-light); border-radius:var(--radius); padding:16px; box-shadow: 0 2px 8px rgba(12,18,24,0.06); border:1px solid rgba(15,20,25,0.04); }

.doc-card{ border-radius:8px; padding:14px; background:var(--card-light); margin-bottom:12px; }
.doc-card:hover{ transform:translateY(-4px); box-shadow:0 6px 20px rgba(12,18,24,0.06); }

.chat-window{ display:flex; flex-direction:column; gap:10px; max-height:52vh; overflow:auto; padding-bottom:12px; }
.msg-user{ align-self:flex-end; background:linear-gradient(90deg,var(--accent),#6B64E6); color:white; padding:10px 14px; border-radius:14px; max-width:75%; }
.msg-ai{ align-self:flex-start; background:#F3F4F6; color:#111827; padding:10px 14px; border-radius:14px; max-width:75%; }
.typing{ font-size:13px; color:var(--muted); margin-top:6px; }
.tag-pill{ display:inline-flex; align-items:center; padding:2px 10px; border-radius:999px; background:rgba(87,99,216,0.08); color:var(--accent); font-size:11px; margin-right:6px; }
.doc-upload{ background:#F7F8FF; padding:16px; border-radius:12px; border:1px dashed rgba(87,99,216,0.35); }
.doc-upload .stFileUploader{ background:transparent !important; }
.doc-upload div[data-testid="stFileUploaderDropzone"]{
  border:1px dashed rgba(87,99,216,0.4) !important;
  background:rgba(87,99,216,0.08) !important;
  color:#0B1220 !important;
}
.doc-upload div[data-testid="stFileUploaderDropzone"] p{
  color:#0B1220 !important;
  font-weight:500;
}

/* Dark mode tweaks applied via injected .dark-mode class (JS handles toggling) */
/* Dark mode tweaks applied via injected .dark-mode class (JS handles toggling) */
html.dark-mode, html.dark-mode .stApp{ background:var(--bg-dark) !important; color:#E6EEF8 !important; }
html.dark-mode .cg-card, html.dark-mode .doc-card{ background:var(--card-dark) !important; border-color: rgba(255,255,255,0.04) !important; }
html.dark-mode .msg-ai{ background: #0F1724 !important; color: #DDE9F7 !important; }

/* Global element overrides to catch Streamlit's default widgets */
html.dark-mode .stApp h1, html.dark-mode .stApp h2, html.dark-mode .stApp h3, html.dark-mode .stApp h4, html.dark-mode .stApp h5, html.dark-mode .stApp h6 {
  color: #E6EEF8 !important;
}

html.dark-mode .stApp button, html.dark-mode .stApp .stButton>button, html.dark-mode button {
  background: rgba(255,255,255,0.04) !important;
  color: #E6EEF8 !important;
  border-color: rgba(255,255,255,0.06) !important;
}

html.dark-mode .stApp input, html.dark-mode .stApp textarea, html.dark-mode .stApp .stTextInput>div>input, html.dark-mode .stApp .stTextArea>div>textarea {
  background: rgba(255,255,255,0.02) !important;
  color: #E6EEF8 !important;
  border-color: rgba(255,255,255,0.06) !important;
}

/* Additional Streamlit-specific selectors to cover input widgets */
html.dark-mode .stApp .stTextInput input, html.dark-mode .stApp .stTextArea textarea,
html.dark-mode .stApp .stNumberInput input, html.dark-mode .stApp .stDateInput input,
html.dark-mode .stApp .stFileUploader, html.dark-mode .stApp .stFileUploader .css-1t5f0wy {
  background: rgba(255,255,255,0.02) !important;
  color: #E6EEF8 !important;
  border-color: rgba(255,255,255,0.06) !important;
}

/* Expander/header/content backgrounds */
html.dark-mode .stApp .stExpander, html.dark-mode .stApp .stExpanderHeader, html.dark-mode .stApp .stExpanderContent {
  background: transparent !important;
  color: #E6EEF8 !important;
}

/* Make sure textarea/input inner containers are dark */
html.dark-mode .stApp .stTextArea>div, html.dark-mode .stApp .stTextInput>div {
  background: rgba(255,255,255,0.02) !important;
}

/* Ensure placeholder and label colors are readable */
html.dark-mode .stApp label, html.dark-mode .stApp .stTextInput label {
  color: rgba(230,238,248,0.85) !important;
}

/* Broad catch-all widget container rules to ensure no bright white areas */
html.dark-mode .stApp .stTextInput, html.dark-mode .stApp .stTextArea,
html.dark-mode .stApp .stNumberInput, html.dark-mode .stApp .stDateInput,
html.dark-mode .stApp .stFileUploader, html.dark-mode .stApp .stSelectbox,
html.dark-mode .stApp .stMultiSelect, html.dark-mode .stApp .stCheckbox,
html.dark-mode .stApp .stRadio {
  background: transparent !important;
}

/* Force inner input/textarea elements dark */
html.dark-mode .stApp input[type="text"], html.dark-mode .stApp input[type="search"],
html.dark-mode .stApp input[type="url"], html.dark-mode .stApp input[type="email"],
html.dark-mode .stApp input[type="number"], html.dark-mode .stApp textarea,
html.dark-mode .stApp .stTextInput>div>input, html.dark-mode .stApp .stTextArea>div>textarea {
  background: rgba(255,255,255,0.02) !important;
  color: #E6EEF8 !important;
  border: 1px solid rgba(255,255,255,0.06) !important;
  box-shadow: none !important;
}

/* File uploader inner card and dropzone */
html.dark-mode .stApp .stFileUploader div[role="button"],
html.dark-mode .stApp .stFileUploader .css-1t5f0wy, html.dark-mode .stApp .stFileUploader .upload {
  background: rgba(255,255,255,0.03) !important;
  color: #E6EEF8 !important;
  border: 1px solid rgba(255,255,255,0.04) !important;
}

html.dark-mode .stApp .block-container, html.dark-mode .stApp .css-1d391kg, html.dark-mode .stApp .css-1v3fvcr {
  background-color: transparent !important;
}

/* Muted text and placeholders */
html.dark-mode ::placeholder { color: rgba(230,238,248,0.5) !important; }
html.dark-mode .stApp .css-1adrfps, html.dark-mode .stApp .stMarkdown, html.dark-mode .stApp p { color: #D0D9E6 !important; }

/* Ensure badges/cards contrast in dark mode */
html.dark-mode .doc-card, html.dark-mode .cg-card { box-shadow: 0 2px 10px rgba(0,0,0,0.4) !important; }

/* Keep CSS minimal — shadcn-ui will style buttons/inputs when present */

/* Fix for weird typing/interaction: ensure inputs have proper background and text color in all states */
.stTextInput input, .stTextArea textarea {
    color: #0B1220 !important;
    background-color: #FFFFFF !important;
}

/* Dark mode overrides */
html.dark-mode .stTextInput input, html.dark-mode .stTextArea textarea {
    color: #E6EEF8 !important;
    background-color: rgba(255,255,255,0.05) !important; /* Slightly lighter for better visibility */
    caret-color: #E6EEF8 !important; /* Ensure cursor is visible */
}

/* Fix for "some elements looking wrong" - ensure all text in dark mode is readable */
html.dark-mode, html.dark-mode .stApp, html.dark-mode p, html.dark-mode h1, html.dark-mode h2, html.dark-mode h3, html.dark-mode h4, html.dark-mode h5, html.dark-mode h6, html.dark-mode span, html.dark-mode div, html.dark-mode label, html.dark-mode .stMarkdown {
    color: #E6EEF8 !important;
}

/* Specific fix for chat input area in dark mode */
html.dark-mode .stTextArea>div>div>textarea {
    background-color: #1F2937 !important; /* Dark gray background */
    color: #F3F4F6 !important; /* Light gray text */
    border: 1px solid #374151 !important;
}

/* Fix for buttons in dark mode */
html.dark-mode .stButton>button {
    background-color: #374151 !important;
    color: #F3F4F6 !important;
    border: 1px solid #4B5563 !important;
}
html.dark-mode .stButton>button:hover {
    background-color: #4B5563 !important;
}
"""


# Small JS injector template (keeps existing behavior: inject and toggle dark-mode)
JS_TEMPLATE = r"""
<div id="coursegpt-style-injector"></div>
<script>
(function(){
  const MODE = "__MODE__";
  const CSS = `__CSSTEXT__`;

  const targetDoc = (window.parent && window.parent.document) ? window.parent.document : document;

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
