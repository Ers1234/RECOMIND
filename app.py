import io, re, json, html, base64
from urllib.parse import quote
import requests
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from concurrent.futures import ThreadPoolExecutor
from sklearn.metrics.pairwise import cosine_similarity

try:
    from PIL import Image
except ImportError:
    Image = None

from recommender import movies, recommend_movies
try:
    from recommender import tfidf, tfidf_matrix
except ImportError:
    tfidf, tfidf_matrix = None, None

st.set_page_config(page_title="RECOMIND | Movie Discovery", page_icon="🎬", layout="wide",
                   initial_sidebar_state="collapsed")
UA = {"User-Agent": "RECOMIND student project"}
PICK_COLORS = ["#ff7a59", "#ffd166", "#4fd1c5"]

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
:root{color-scheme:light;--bg:#fbf5ee;--panel:rgba(255,255,255,.78);--panel-solid:#ffffff;--text:#2b1c26;--muted:#6e5a66;--line:rgba(43,28,38,.13);--accent:#e8553a;--accent2:#f59e3b;--accent-soft:rgba(232,85,58,.12);--grad:linear-gradient(135deg,#ff5e7e 0%,#ff8a4c 55%,#ffb347 100%);--gradtxt:linear-gradient(135deg,#e11d5a,#f0592b 60%,#e8890c)}
@property --p{syntax:'<number>';inherits:false;initial-value:0}
html,body,.stApp,.stApp *{font-family:'Inter',system-ui,sans-serif}
.stApp{background:radial-gradient(900px 520px at 95% -10%,rgba(255,122,89,.20),transparent 60%),
radial-gradient(800px 520px at -10% 20%,rgba(255,94,126,.12),transparent 60%),var(--bg);color:var(--text)}
.stApp p,.stApp label,.stApp li{color:var(--text)}
.block-container{max-width:1200px;padding:1.4rem 2rem 3rem}
header[data-testid="stHeader"]{background:transparent}
#MainMenu,footer{visibility:hidden}
[data-testid="stCaptionContainer"],.stCaption,.stCaption p{color:var(--muted)!important}
.nav{display:flex;align-items:center;justify-content:space-between}
.wm{font-weight:800;font-size:25px;letter-spacing:-.04em;line-height:1}
.wm b,.hero h1 span{background:var(--gradtxt);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.tag{color:var(--muted);font-size:11px;font-weight:600;letter-spacing:.14em;text-transform:uppercase;margin-top:4px}
.status{color:var(--muted);font-size:13px;border:1px solid var(--line);padding:7px 14px;border-radius:999px;background:var(--panel)}
.status b{color:#1a9d5a}
.hero{position:relative;overflow:hidden;margin:22px 0 36px;padding:78px 56px;min-height:360px;border-radius:28px;background:#f1e4d8;border:1px solid var(--line);box-shadow:0 18px 50px rgba(120,70,40,.12)}
.wall{position:absolute;inset:-25% -10%;transform:rotate(-8deg);display:flex;flex-direction:column;gap:14px;opacity:.6}
.wrow{display:flex;gap:14px;width:max-content;animation:mq 80s linear infinite}
.wrow:nth-child(2){animation-duration:100s;animation-direction:reverse}
@keyframes mq{to{transform:translateX(-50%)}}
.pp{width:120px;aspect-ratio:2/3;flex:none;border-radius:12px;background-size:cover;background-position:center;box-shadow:0 12px 30px rgba(0,0,0,.5)}
.shade{position:absolute;inset:0;pointer-events:none;background:linear-gradient(90deg,rgba(251,245,238,.97) 0%,rgba(251,245,238,.86) 42%,rgba(251,245,238,.15) 100%)}
.hero-in{position:relative;z-index:2}
.pill{display:inline-flex;align-items:center;gap:9px;font-size:13px;color:var(--text);border:1px solid var(--line);padding:7px 14px;border-radius:999px;background:rgba(255,255,255,.75);margin-bottom:22px}
.dot{width:8px;height:8px;border-radius:50%;background:#6ee7a0;animation:pulse 2s ease-in-out infinite}
@keyframes pulse{50%{opacity:.3;transform:scale(.8)}}
.hero h1{font-weight:800;font-size:clamp(38px,5.6vw,68px);line-height:1.05;letter-spacing:-.035em;margin:0 0 18px}
.hero p{color:#4a3843;font-size:17px;line-height:1.65;max-width:540px;margin:0}
.section-heading{font-size:26px;font-weight:700;letter-spacing:-.02em;margin:6px 0 4px}
.section-subtitle{color:var(--muted);font-size:15px;line-height:1.6;margin-bottom:16px;max-width:720px}
div[data-testid="stImage"] img{border-radius:14px;transition:transform .3s,box-shadow .3s}
div[data-testid="stImage"] img:hover{transform:translateY(-6px) scale(1.03);box-shadow:0 20px 40px rgba(0,0,0,.6)}
.cover{aspect-ratio:2/3;border-radius:14px;box-sizing:border-box;position:relative;overflow:hidden;color:#fff;
background:linear-gradient(160deg,hsl(var(--h) 68% 46%),hsl(calc(var(--h) + 40) 62% 17%));border:1px solid rgba(255,255,255,.14);
display:flex;flex-direction:column;justify-content:space-between;padding:18px 20px}
.cover .big{position:absolute;right:-10px;bottom:24px;font-size:96px;opacity:.22;transform:rotate(-12deg)}
.cover .g{position:relative;font-size:11px;font-weight:700;letter-spacing:.14em;text-transform:uppercase;opacity:.85}
.cover .t{position:relative;font-weight:800;font-size:19px;line-height:1.15;overflow-wrap:anywhere}
.cover .y{position:relative;font-size:12px;opacity:.75;margin-top:5px}
.tcap{font-size:13px;font-weight:600;margin:8px 0 6px;line-height:1.3}
div[data-baseweb="select"]>div{background:var(--panel-solid);border:1.5px solid var(--line);border-radius:14px;min-height:54px}
div[data-baseweb="select"]>div:focus-within{border-color:var(--accent);box-shadow:0 0 0 3px var(--accent-soft)}
div[data-baseweb="select"] input{color:var(--text)!important;-webkit-text-fill-color:var(--text)!important}
span[data-baseweb="tag"]{background:var(--grad)!important;border-radius:8px}
span[data-baseweb="tag"],span[data-baseweb="tag"] *{color:#2a1208!important;font-weight:700}
.stButton>button{width:100%;border-radius:12px;border:1px solid var(--line);background:var(--panel);color:var(--text);font-weight:600;min-height:40px}
.stButton>button:hover{border-color:var(--accent);background:var(--accent-soft);color:var(--text)}
.stButton>button p{color:inherit!important}
.stButton>button[kind="primary"],.stButton>button[data-testid="stBaseButton-primary"]{min-height:58px;font-size:17px;font-weight:700;border:none;background:var(--grad);color:#2a1208}
.stButton>button[kind="primary"] p,.stButton>button[data-testid="stBaseButton-primary"] p{color:#2a1208!important}
div[data-testid="stVerticalBlockBorderWrapper"]{background:var(--panel);border:1px solid var(--line);border-radius:22px}
.movie-title{font-size:24px;font-weight:700;letter-spacing:-.02em;margin:2px 0 6px;overflow-wrap:anywhere;line-height:1.2}
.movie-meta{color:var(--muted);font-size:13.5px;line-height:1.55}
.rank-tag{display:inline-block;background:var(--grad);color:#2a1208;font-size:12px;font-weight:800;padding:4px 11px;border-radius:999px;margin-bottom:8px}
.overview{color:#4a3843;font-size:14px;line-height:1.65;margin-top:10px}
.why{color:var(--muted);font-size:12.5px;font-weight:600;margin:14px 0 7px}
.chips{display:flex;flex-wrap:wrap;gap:7px}
.chip{background:rgba(43,28,38,.07);border-radius:999px;padding:5px 12px;font-size:12.5px}
.chip.g{background:var(--accent-soft);color:#b03a1c}
.ring{--p:0;width:88px;height:88px;border-radius:50%;margin:0 auto;position:relative;background:conic-gradient(var(--accent2) calc(var(--p)*1%),rgba(43,28,38,.1) 0);animation:fillring 1.3s ease-out both}
.ring:after{content:attr(data-v);position:absolute;inset:8px;border-radius:50%;background:var(--panel-solid);display:flex;align-items:center;justify-content:center;font-weight:800;font-size:19px}
@keyframes fillring{from{--p:0}}
.ringcap{color:var(--muted);font-size:12px;margin-top:9px;text-align:center}
iframe{border-radius:22px;border:1px solid var(--line)}
/* how it works: plain grid, no absolute positioning, so nothing can overlap */
.how{display:grid;grid-template-columns:repeat(auto-fit,minmax(230px,1fr));gap:16px;margin:8px 0 18px;align-items:stretch}
.how .st{background:var(--panel);border:1px solid var(--line);border-radius:20px;padding:22px 22px 24px;display:flex;flex-direction:column;gap:10px;min-width:0}
.how .n{width:34px;height:34px;border-radius:50%;background:var(--grad);color:#2a1208;font-weight:800;display:flex;align-items:center;justify-content:center}
.how h4{margin:0;font-size:17px;font-weight:700;line-height:1.3}
.how p{margin:0;color:var(--muted);font-size:14px;line-height:1.6;overflow-wrap:anywhere}
.dna{display:flex;align-items:center;gap:10px;margin:6px 0;font-size:13px}
.dna .bar{height:8px;border-radius:99px;background:var(--grad)}
.fnote{margin-top:48px;padding-top:22px;border-top:1px solid var(--line);color:var(--muted);font-size:13px;line-height:1.8;text-align:center}
@media(max-width:800px){.block-container{padding:1rem 1rem 2rem}.hero{padding:40px 24px}.status{display:none}}
@media(prefers-reduced-motion:reduce){.dot,.ring,.wrow{animation:none}}
.stApp,.stApp [data-testid="stMarkdownContainer"] *,.stApp [data-testid="stWidgetLabel"] *,.stApp [data-testid="stSlider"] div,.stApp .stRadio label *{color:var(--text)}
.stApp .stRadio label p{font-weight:600}
div[data-baseweb="popover"] ul{background:#fff}div[data-baseweb="popover"] *{color:#2b1c26}
.stDownloadButton>button{width:100%;border-radius:12px;border:1px solid var(--line);background:var(--panel);font-weight:600;min-height:40px}
.stDownloadButton>button:hover{border-color:var(--accent);background:var(--accent-soft)}
div[data-testid="stVerticalBlockBorderWrapper"]{box-shadow:0 10px 30px rgba(120,70,40,.08)}
.brandbox{display:flex;align-items:center;gap:12px}.brandbox svg{width:42px;height:42px;filter:drop-shadow(0 6px 14px rgba(232,85,58,.35))}
[data-testid="stIconMaterial"],span[class*="material"]{font-family:"Material Symbols Rounded","Material Icons"!important}
.site{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:12px 20px;background:rgba(255,255,255,.88);border:1px solid var(--line);border-radius:18px;box-shadow:0 10px 30px rgba(120,70,40,.08)}
.links{display:flex;gap:4px;flex-wrap:wrap}
.links a{color:var(--text)!important;text-decoration:none!important;font-weight:600;font-size:14px;padding:8px 14px;border-radius:999px}
.links a:hover{background:var(--accent-soft)}
.sub2{font-size:19px;font-weight:700;margin:30px 0 6px}
@media(max-width:800px){.links{display:none}}
.rep{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:16px;margin:10px 0 6px}
.rc{background:#fff;border:1px solid var(--line);border-radius:20px;padding:20px 22px 22px;box-shadow:0 10px 30px rgba(120,70,40,.08);display:flex;flex-direction:column;gap:10px;min-width:0}
.rc .l{font-size:14px;font-weight:700;color:var(--muted)}
.rc .v{font-size:42px;font-weight:800;letter-spacing:-.03em;line-height:1;background:var(--gradtxt);-webkit-background-clip:text;background-clip:text;-webkit-text-fill-color:transparent}
.rc .meter{height:7px;border-radius:99px;background:rgba(43,28,38,.09);overflow:hidden}
.rc .meter i{display:block;height:100%;border-radius:99px;background:var(--grad)}
.rc .d{font-size:13px;line-height:1.55;color:var(--muted)}
.bd{display:inline-block;align-self:flex-start;font-size:12px;font-weight:800;padding:4px 11px;border-radius:99px}
.bd.ok{background:rgba(26,157,90,.14);color:#12724a}.bd.mid{background:rgba(245,158,59,.2);color:#9a5a05}.bd.no{background:rgba(225,29,90,.12);color:#b4143f}
.rc .t{font-size:16px;font-weight:700;line-height:1.3;overflow-wrap:anywhere}
.rc .v small{font-size:14px;font-weight:600;color:var(--muted);-webkit-text-fill-color:var(--muted);margin-left:6px;letter-spacing:0}
.dnaw{display:grid;grid-template-columns:3fr 2fr;gap:16px;margin:10px 0 6px}
.dcard{background:#fff;border:1px solid var(--line);border-radius:20px;padding:22px 24px;box-shadow:0 10px 30px rgba(120,70,40,.08);min-width:0}
.dcard h5{margin:0 0 14px;font-size:14px;font-weight:700;color:var(--muted)}
.gr{display:grid;grid-template-columns:120px 1fr 44px;align-items:center;gap:12px;margin:12px 0;font-size:14px;font-weight:600}
.gr .tr{height:10px;border-radius:99px;background:rgba(43,28,38,.09);overflow:hidden}
.gr .tr i{display:block;height:100%;border-radius:99px;background:var(--grad)}
.gr .p{text-align:right;color:var(--muted);font-weight:700}
@media(max-width:800px){.dnaw{grid-template-columns:1fr}.gr{grid-template-columns:90px 1fr 40px}}
/* softer, tinted palette: less stark white */
:root{--bg:#f2e5d7;--panel:rgba(255,246,236,.72);--panel-solid:#fff4e8;--line:rgba(43,28,38,.14);--muted:#6a5560}
.stApp{background:radial-gradient(900px 520px at 95% -10%,rgba(255,122,89,.24),transparent 60%),radial-gradient(800px 520px at -10% 25%,rgba(150,110,255,.14),transparent 60%),radial-gradient(700px 500px at 60% 110%,rgba(79,209,197,.14),transparent 60%),var(--bg)}
.hero{background:#e9d8c6}
.shade{background:linear-gradient(90deg,rgba(242,229,215,.97) 0%,rgba(242,229,215,.86) 42%,rgba(242,229,215,.12) 100%)}
.site{background:rgba(255,242,228,.92)}
.status,.pill{background:rgba(255,244,232,.85)}
.stButton>button,.stDownloadButton>button{background:rgba(255,246,236,.85)}
div[data-baseweb="select"]>div{background:#fff4e8}
div[data-baseweb="popover"] ul{background:#fff4e8}
div[data-testid="stVerticalBlockBorderWrapper"]{background:linear-gradient(160deg,rgba(255,247,238,.92),rgba(255,233,216,.78))}
.rc,.dcard{background:linear-gradient(160deg,#fff1e2,#ffdfc8)}
.rc:nth-child(2),.dcard:nth-child(2){background:linear-gradient(160deg,#f5ebff,#e4d6fa)}
.rc:nth-child(3){background:linear-gradient(160deg,#e5f6ee,#cfeadc)}
.rc:nth-child(4){background:linear-gradient(160deg,#fff6d8,#fde8a8)}
.rc .meter,.gr .tr{background:rgba(43,28,38,.12)}
.chip{background:rgba(43,28,38,.09)}
.chip.g{background:rgba(232,85,58,.16)}
.ticket{display:flex;margin:14px 0 6px;background:#fbf0e1;color:#2b1c26;border:1.5px solid #2b1c26;border-radius:14px;box-shadow:6px 6px 0 rgba(43,28,38,.14)}
.tk-main{flex:1;min-width:0;padding:26px 30px}
.tk-stub{position:relative;flex:0 0 250px;padding:26px 24px;border-left:2px dashed #2b1c26;display:flex;flex-direction:column;gap:10px}
.tk-stub:before,.tk-stub:after{content:"";position:absolute;left:-13px;width:24px;height:24px;border-radius:50%;background:#f2e5d7;border:1.5px solid #2b1c26;box-sizing:border-box}
.tk-stub:before{top:-13px;clip-path:inset(50% 0 0 0)}.tk-stub:after{bottom:-13px;clip-path:inset(0 0 50% 0)}
.tk-k{font:600 11px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.16em;text-transform:uppercase;color:#6a5560}
.tk-t{font-family:Georgia,'Times New Roman',serif;font-size:38px;line-height:1.1;font-weight:700;margin:8px 0 18px;overflow-wrap:anywhere}
.tk-bar{display:flex;height:14px;border:1.5px solid #2b1c26;border-radius:4px;overflow:hidden;margin-bottom:16px}
.tk-bar i{display:block;height:100%}
.tk-r{display:flex;align-items:baseline;gap:8px;font:500 14px ui-monospace,Menlo,Consolas,monospace;padding:5px 0}
.tk-r i{width:10px;height:10px;flex:none;display:inline-block;align-self:center}
.tk-r span{white-space:nowrap}.tk-r u{flex:1;border-bottom:1.5px dotted rgba(43,28,38,.45);text-decoration:none;transform:translateY(-3px)}
.tk-r b{font-weight:700}
.tk-w{font:500 14px ui-monospace,Menlo,Consolas,monospace;line-height:1.9}
.tk-admit{margin-top:auto;font:700 12px ui-monospace,Menlo,Consolas,monospace;letter-spacing:.22em;border:1.5px solid #2b1c26;padding:6px 10px;text-align:center;border-radius:4px}
@media(max-width:800px){.ticket{flex-direction:column}.tk-stub{flex:none;border-left:0;border-top:2px dashed #2b1c26}.tk-stub:before,.tk-stub:after{display:none}.tk-t{font-size:30px}}
</style>
""", unsafe_allow_html=True)

# ---------------- helpers ----------------
_SPECIAL = {"ScienceFiction": "Science Fiction", "TVMovie": "TV Movie"}

def split_genres(v):
    s = str(v or "").strip()
    if not s:
        return []
    if "," in s:
        return [g.strip() for g in s.split(",") if g.strip()]
    s = s.replace("Science Fiction", "ScienceFiction").replace("TV Movie", "TVMovie")
    return [_SPECIAL.get(t, t) for t in s.split()]

YEARS = (pd.to_numeric(movies["release_date"].astype(str).str[:4], errors="coerce")
         if "release_date" in movies.columns else pd.Series(np.nan, index=movies.index)).to_numpy(dtype=float)

def movie_row(t):
    r = movies[movies["title"] == t]
    return None if r.empty else r.iloc[0]

def title_pos(t):
    return int(np.flatnonzero((movies["title"] == t).to_numpy())[0])

def year_of(row):
    if row is None:
        return ""
    d = str(row.get("release_date", "") or "")
    return d[:4] if len(d) >= 4 and d[:4].isdigit() else ""

def genres_of(row):
    if row is None:
        return []
    col = "genres_display" if "genres_display" in row.index else "genres"
    return split_genres(row.get(col, ""))

# ---------------- posters ----------------
poster_lookup = {}
if "poster_path" in movies.columns:
    _pd = movies[["title", "poster_path"]].dropna(subset=["title"]).drop_duplicates(subset=["title"])
    poster_lookup = dict(zip(_pd["title"], _pd["poster_path"]))

def _download(url):
    try:
        r = requests.get(url, timeout=6, headers=UA)
        if r.status_code == 200 and r.headers.get("Content-Type", "").startswith("image/") and len(r.content) > 2000:
            return r.content
    except requests.RequestException:
        pass
    return None

def _norm(x):
    return re.sub(r"[^a-z0-9]+", "", str(x).casefold())

def _itunes_url(title, year):
    try:
        data = requests.get("https://itunes.apple.com/search",
                            params={"term": title, "media": "movie", "entity": "movie", "limit": 20},
                            headers=UA, timeout=6).json()
        for it in data.get("results", []):
            if _norm(it.get("trackName", "")) != _norm(title):
                continue
            rel = str(it.get("releaseDate", ""))[:4]
            if year and rel.isdigit() and abs(int(rel) - int(year)) > 1:
                continue
            art = it.get("artworkUrl100", "")
            if art:
                return art.replace("100x100bb", "400x600bb")
    except Exception:
        pass
    return None

def _wiki_url(title, year):
    cands = ([f"{title} ({year} film)"] if year else []) + [f"{title} (film)", title]
    for c in cands:
        try:
            r = requests.get("https://en.wikipedia.org/api/rest_v1/page/summary/" + quote(c.replace(" ", "_"), safe="()_,"),
                             headers=UA, timeout=8)
            if r.status_code != 200:
                continue
            j = r.json()
            if j.get("type") == "disambiguation" or _norm(title) not in _norm(j.get("title", "")):
                continue
            text = j.get("description", "") + " " + j.get("extract", "")[:300]
            if "film" not in text.lower() and "movie" not in text.lower():
                continue
            ys = [int(v) for v in re.findall(r"\b(?:19|20)\d{2}\b", text)]
            if year and ys and min(abs(v - int(year)) for v in ys) > 1:
                continue
            src = (j.get("thumbnail") or {}).get("source")
            if src:
                return src
        except Exception:
            pass
    return None

@st.cache_data(show_spinner=False, ttl=6 * 3600, max_entries=2048)
def fetch_poster(title, year, path):
    if path:
        d = _download("https://image.tmdb.org/t/p/w342/" + path.lstrip("/"))
        if d:
            return d
    for finder in (_itunes_url, _wiki_url):
        url = finder(title, year)
        if url:
            d = _download(url)
            if d:
                return d
    return None

def get_poster(title):
    p = poster_lookup.get(title)
    return fetch_poster(title, year_of(movie_row(title)), p if isinstance(p, str) else "")

def preload(titles):
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(get_poster, titles))

def show_image(data):
    try:
        st.image(data, use_container_width=True)
    except TypeError:
        st.image(data, width="stretch")

def thumb_b64(data, w=120):
    if not data:
        return ""
    try:
        if Image is None:
            return "data:image/jpeg;base64," + base64.b64encode(data).decode()
        img = Image.open(io.BytesIO(data)).convert("RGB")
        img = img.resize((w, max(1, int(img.height * w / img.width))))
        buf = io.BytesIO()
        img.save(buf, "JPEG", quality=72)
        return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return ""

GENRE_STYLE = {"Horror": ("👻", 350), "Comedy": ("😂", 42), "Science Fiction": ("🚀", 215), "Romance": ("💕", 330),
               "Animation": ("🎨", 170), "Action": ("💥", 12), "Drama": ("🎭", 275), "Thriller": ("🔪", 0),
               "Fantasy": ("🧙", 285), "Crime": ("🕵️", 22), "Adventure": ("🧭", 150), "Family": ("🧸", 48),
               "War": ("🎖️", 100), "Music": ("🎵", 305), "History": ("🏛️", 34), "Western": ("🤠", 28),
               "Mystery": ("🔎", 245), "Documentary": ("🎥", 190), "TV Movie": ("📺", 200)}

def display_movie_visual(title):
    data = get_poster(title)
    if data:
        show_image(data)
        return
    row = movie_row(title)
    gl = genres_of(row)
    style = next((GENRE_STYLE[g] for g in gl if g in GENRE_STYLE), None)
    emoji, hue = style if style else ("🎬", sum(ord(c) for c in title) * 37 % 360)
    st.markdown(f'<div class="cover" style="--h:{hue}"><div class="big">{emoji}</div>'
                f'<div class="g">{html.escape(gl[0] if gl else "Film")}</div>'
                f'<div><div class="t">{html.escape(title)}</div><div class="y">{html.escape(year_of(row))}</div></div></div>',
                unsafe_allow_html=True)

@st.cache_data(show_spinner=False, ttl=6 * 3600)
def hero_wall():
    src = movies.sort_values("popularity", ascending=False) if "popularity" in movies.columns else movies
    pool = [str(t) for t in src["title"].dropna().unique()[:30]]
    preload(pool)
    got = []
    for t in pool:
        b = thumb_b64(get_poster(t), 150)
        if b:
            got.append(b)
        if len(got) == 18:
            break
    if len(got) < 4:
        return "", ""
    css = "".join(f".pp{i}{{background-image:url({b})}}" for i, b in enumerate(got))
    n = len(got)
    rows = ""
    for r in range(3):
        cells = "".join(f'<div class="pp pp{(i + r * 6) % n}"></div>' for i in range(n)) * 2
        rows += f'<div class="wrow">{cells}</div>'
    return f"<style>{css}</style>", f'<div class="wall">{rows}</div><div class="shade"></div>'

# ---------------- browse ----------------
@st.cache_data(show_spinner=False)
def genre_index():
    col = "genres_display" if "genres_display" in movies.columns else "genres"
    return movies[col].map(split_genres)

@st.cache_data(show_spinner=False)
def all_genres():
    counts = {}
    for gl in genre_index():
        for g in gl:
            counts[g] = counts.get(g, 0) + 1
    return [g for g, _ in sorted(counts.items(), key=lambda x: -x[1])][:14]

def browse_titles(genre, n=6):
    df = movies
    if genre != "Popular":
        df = movies[genre_index().map(lambda gl: genre in gl).to_numpy()]
    if "popularity" in df.columns:
        df = df.sort_values("popularity", ascending=False)
    return [str(t) for t in df["title"].dropna().unique()[:n]]

def add_pick(title):
    cur = list(st.session_state.get("favs", []))
    if title not in cur and len(cur) < 3:
        cur.append(title)
    st.session_state["favs"] = cur

def surprise():
    src = movies.sort_values("popularity", ascending=False) if "popularity" in movies.columns else movies
    pool = [str(t) for t in src["title"].dropna().unique()[:300]]
    st.session_state["favs"] = list(np.random.choice(pool, 3, replace=False))

def clear_picks():
    st.session_state["favs"] = []

# ---------------- ML: profile, MMR re-ranking, explanations ----------------
@st.cache_resource(show_spinner=False)
def genre_tokens():
    toks = {"science", "fiction"}
    for g in movies["genres"]:
        toks.update(t.lower() for t in str(g).replace(",", " ").split())
    return toks

def shared_terms(favs, title, k=4):
    if tfidf is None or tfidf_matrix is None:
        return []
    try:
        idxs = [title_pos(f) for f in favs]
        ri = title_pos(title)
        profile = np.asarray(tfidf_matrix[idxs].mean(axis=0)).ravel()
        contrib = profile * tfidf_matrix[ri].toarray().ravel()
        names = tfidf.get_feature_names_out()
        words = set(re.findall(r"[a-z]+", str(movies.iloc[ri].get("overview", "")).lower()))
        gt, out = genre_tokens(), []
        for j in contrib.argsort()[::-1][:80]:
            if contrib[j] <= 0:
                break
            t = names[j]
            if len(t) >= 4 and not t.isdigit() and t not in gt and t in words:
                out.append(t)
            if len(out) == k:
                break
        return out
    except Exception:
        return []

def _prof(idx):
    return np.asarray(tfidf_matrix[idx].mean(axis=0))

def get_recommendations(favs, k=5, diversity=0.0, years=None, genres=None, min_rating=0.0,
                        exclude=(), liked=(), disliked=(), friend=()):
    """Hybrid content-based engine.
    1. Taste profile = mean TF-IDF of the picks, refined by feedback (Rocchio: +liked, -disliked).
    2. Cosine similarity to every film, lightly weighted by audience rating (quality prior).
    3. Practical filters: release years, genres, minimum rating, already-watched.
    4. MMR re-ranking so the final list is relevant but not repetitive."""
    if tfidf_matrix is None:
        try:
            return recommend_movies(favs)
        except TypeError:
            return recommend_movies(favs[0])
    known = set(movies["title"].astype(str))
    idxs = [title_pos(t) for t in favs]
    li = [title_pos(t) for t in liked if t in known]
    di = [title_pos(t) for t in disliked if t in known]
    ex = [title_pos(t) for t in exclude if t in known]
    profile = _prof(idxs)
    if li:
        profile = profile + 0.6 * _prof(li)
    if di:
        profile = profile - 0.4 * _prof(di)
    rel = cosine_similarity(profile, tfidf_matrix).ravel()
    fr = [title_pos(t) for t in friend if t in known]
    if fr:  # movie night: blend the average with the least-happy person's score
        rb = cosine_similarity(_prof(fr), tfidf_matrix).ravel()
        rel = 0.5 * np.minimum(rel, rb) + 0.25 * (rel + rb)
    if "vote_average" in movies.columns:
        va = pd.to_numeric(movies["vote_average"], errors="coerce").fillna(5).to_numpy(dtype=float)
        rel = rel * (0.88 + 0.12 * np.clip(va, 0, 10) / 10)
    else:
        va = None
    rel[list(set(idxs + li + di + ex + fr))] = -1
    N = len(rel)
    masks = []
    if years is not None:
        masks.append(np.isfinite(YEARS) & ((YEARS < years[0]) | (YEARS > years[1])))
    if genres:
        masks.append(~np.array([any(g in gl for g in genres) for gl in genre_index()]))
    if va is not None and min_rating > 0:
        masks.append(va < min_rating)
    for m in masks:
        if (rel[~m] > -1).sum() > 80:
            rel[m] = -1
    cand = np.argsort(rel)[::-1][:80]
    lam, chosen = 1.0 - diversity, []
    cs = cosine_similarity(tfidf_matrix[cand])
    while len(chosen) < k and len(chosen) < len(cand):
        best, bs = None, -9
        for a in range(len(cand)):
            if a in chosen:
                continue
            pen = max((cs[a, b] for b in chosen), default=0.0)
            sc = lam * rel[cand[a]] - (1 - lam) * pen
            if sc > bs:
                best, bs = a, sc
        chosen.append(best)
    sel = cand[chosen]
    return pd.DataFrame({"title": movies["title"].iloc[sel].values, "similarity_score": np.clip(rel[sel], 0, 1)})

def engine_report(res):
    """Three numbers that tell a user whether the list is any good: (label, value, description, meter %)."""
    pos = [title_pos(str(t)) for t in res["title"]]
    out = []
    if tfidf_matrix is not None and len(pos) > 1:
        cs = cosine_similarity(tfidf_matrix[pos])
        v = int(round(100 * (1 - cs[np.triu_indices(len(pos), 1)].mean())))
        out.append(("Variety", f"{v}%", "How different the films are from each other. Higher means less repetition.", v))
    ng = len({g for p in pos for g in genre_index().iloc[p]})
    out.append(("Genres covered", str(ng), "Distinct genres across the list.", min(100, ng * 12)))
    if "popularity" in movies.columns:
        pop = pd.to_numeric(movies["popularity"], errors="coerce").fillna(0)
        cut = pop.quantile(0.9)
        k = int(sum(pop.iloc[p] < cut for p in pos))
        out.append(("Hidden gems", f"{k} of {len(pos)}", "Films outside the 10% most popular.", int(100 * k / max(len(pos), 1))))
    return out

def loo_check(favs):
    """Leave-one-out test: hide each pick, rebuild the profile from the others, see where the hidden film ranks."""
    rows = []
    if tfidf_matrix is None or len(favs) < 2:
        return rows
    for t in favs:
        rest = [title_pos(f) for f in favs if f != t]
        sim = cosine_similarity(_prof(rest), tfidf_matrix).ravel()
        sim[rest] = -1
        rows.append((t, int((sim > sim[title_pos(t)]).sum()) + 1))
    return rows

def sig_terms(favs, k=10):
    """Words that most define the user's taste (highest TF-IDF weight in the mean profile, genre names excluded)."""
    if tfidf is None or tfidf_matrix is None:
        return []
    try:
        prof = _prof([title_pos(f) for f in favs]).ravel()
        names, gt, out = tfidf.get_feature_names_out(), genre_tokens(), []
        for j in prof.argsort()[::-1][:120]:
            t = names[j]
            if prof[j] > 0 and len(t) >= 4 and not t.isdigit() and t not in gt:
                out.append(t)
            if len(out) == k:
                break
        return out
    except Exception:
        return []

def run_engine():
    favs = list(st.session_state.get("favs", []))
    if not favs:
        return
    if "res" in st.session_state and st.session_state["res"][0] != favs:
        st.session_state["liked"], st.session_state["disliked"] = [], []
    res = get_recommendations(favs, 5, st.session_state.get("dv", 0.0), st.session_state.get("yrs"),
                              st.session_state.get("gf") or None, st.session_state.get("mr", 0.0),
                              st.session_state.get("ex", []), st.session_state.get("liked", []),
                              st.session_state.get("disliked", []), st.session_state.get("friend", []))
    preload([str(t) for t in res["title"]] + favs)
    st.session_state["res"] = (favs, res)

def feedback(title, kind):
    up, dn = list(st.session_state.get("liked", [])), list(st.session_state.get("disliked", []))
    (up if kind == "up" else dn).append(title)
    st.session_state["liked"], st.session_state["disliked"] = up, dn
    run_engine()

def reset_feedback():
    st.session_state["liked"], st.session_state["disliked"] = [], []
    run_engine()

# ---------------- taste universe (3D) ----------------
def galaxy_payload(favs, results):
    n_p = len(favs)
    N = tfidf_matrix.shape[0]
    idxs = [title_pos(t) for t in favs]
    profile = np.asarray(tfidf_matrix[idxs].mean(axis=0))
    fit = cosine_similarity(profile, tfidf_matrix).ravel()
    fit[idxs] = -1
    rank = np.empty(N)
    rank[fit.argsort()] = np.arange(N)
    r = 0.10 + 0.90 * (1 - rank / max(N - 1, 1)) ** 0.55          # distance from centre = fit
    pick_sims = cosine_similarity(tfidf_matrix, tfidf_matrix[idxs])
    w = pick_sims ** 2
    sm = w.sum(axis=1, keepdims=True)
    w = np.divide(w, sm, out=np.zeros_like(w), where=sm > 1e-9)
    angles = -np.pi / 2 + 2 * np.pi * np.arange(n_p) / n_p
    mx, my = (w * np.cos(angles)).sum(1), (w * np.sin(angles)).sum(1)
    lean = np.hypot(mx, my)
    theta = np.arctan2(my, mx) + np.random.RandomState(7).uniform(-np.pi, np.pi, N) * (1 - lean) ** 1.5
    x, y = r * np.cos(theta), r * np.sin(theta)                    # direction = which pick it resembles
    cidx = w.argmax(1)
    z = np.zeros(N)                                                # depth = release year (older = deeper)
    fin = YEARS[np.isfinite(YEARS)]
    if len(fin) > 10:
        lo, hi = np.percentile(fin, 2), np.percentile(fin, 98)
        z = -np.clip((np.nan_to_num(YEARS, nan=(lo + hi) / 2) - lo) / max(hi - lo, 1) * 2 - 1, -1, 1) * 0.55

    rec_pos = [title_pos(str(t)) for t in results["title"]]
    pct_raw = [float(v) for v in results["similarity_score"]]
    top = max(max(pct_raw), 1e-9)
    skip = set(idxs) | set(rec_pos)
    sample = [int(i) for i in np.random.RandomState(3).choice(N, size=min(N, 3000), replace=False) if i not in skip]
    names = movies["title"].fillna("").astype(str).tolist()
    picked_genres = set()
    for f in favs:
        picked_genres.update(genres_of(movie_row(f)))

    picks = []
    for i, t in enumerate(favs):
        sims = pick_sims[:, i].copy()
        sims[idxs] = -1
        row = movie_row(t)
        picks.append({"title": t, "x": round(1.12 * float(np.cos(angles[i])), 3), "y": round(1.12 * float(np.sin(angles[i])), 3),
                      "z": round(float(z[idxs[i]]), 3),
                      "sub": " · ".join(v for v in [year_of(row)] + genres_of(row)[:2] if v),
                      "img": thumb_b64(get_poster(t), 100),
                      "links": [[round(float(x[j]), 3), round(float(y[j]), 3), round(float(z[j]), 3)] for j in np.argsort(sims)[::-1][:12]]})
    recs = []
    for k, (t, p) in enumerate(zip(results["title"], rec_pos)):
        t = str(t)
        row = movie_row(t)
        g = genres_of(row)
        ov = str(row.get("overview", "") or "") if row is not None else ""
        if len(ov) > 200:
            ov = ov[:200].rsplit(" ", 1)[0] + "..."
        recs.append({"title": t, "rank": k + 1, "x": round(float(x[p]), 3), "y": round(float(y[p]), 3), "z": round(float(z[p]), 3),
                     "pct": int(round(100 * pct_raw[k] / top * 0.98)), "lead": int(cidx[p]),
                     "sub": " · ".join(v for v in [year_of(row)] + g[:2] if v),
                     "tags": ([v for v in g if v in picked_genres][:2] + shared_terms(favs, t, 2))[:4],
                     "overview": ov, "img": thumb_b64(get_poster(t), 120)})
    return {"bx": [round(float(x[i]), 3) for i in sample], "by": [round(float(y[i]), 3) for i in sample],
            "bz": [round(float(z[i]), 3) for i in sample], "bc": [int(cidx[i]) for i in sample],
            "picks": picks, "recs": recs, "cols": PICK_COLORS}

GALAXY_HTML = r"""
<!doctype html><html><head><meta charset="utf-8"><style>
html,body{margin:0;height:100%;background:#0f0a14;font-family:'Inter',system-ui,sans-serif;overflow:hidden;color:#fbf3ea}
#w{position:relative;width:100%;height:100%;overflow:hidden}
canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
.bar{position:absolute;left:0;right:0;height:0;background:#000;transition:height .9s ease;z-index:4;pointer-events:none}
#b1{top:0}#b2{bottom:0}body.tour .bar{height:8.5%}
#tip{position:absolute;pointer-events:none;background:rgba(28,21,31,.97);padding:7px 10px;border-radius:10px;border:1px solid rgba(255,240,230,.18);display:none;font-size:12.5px;font-weight:600;z-index:7;max-width:220px}
#hint{position:absolute;left:16px;top:14px;color:#bba9b6;font-size:12px;line-height:1.6;max-width:310px;z-index:3}
#hint b{color:#fbf3ea}
#leg{position:absolute;left:16px;bottom:12px;right:16px;display:flex;gap:16px;flex-wrap:wrap;color:#bba9b6;font-size:12px;z-index:3}
#leg i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:6px}
body.tour #hint,body.tour #leg{display:none}
#ctl{position:absolute;right:14px;top:12px;display:flex;gap:8px;z-index:6}body.tour #ctl{top:calc(8.5% + 12px)}
.btn{background:rgba(255,255,255,.08);color:#fbf3ea;border:1px solid rgba(255,240,230,.18);padding:8px 14px;border-radius:10px;font-weight:700;font-size:12.5px;cursor:pointer}
.btn:hover{background:rgba(255,255,255,.16)}
#play{background:linear-gradient(135deg,#ff5e7e,#ff9a55,#ffcf6b);color:#2a1208;border:0;animation:glow 2.2s ease-in-out infinite}
@keyframes glow{50%{box-shadow:0 0 30px rgba(255,140,90,.6)}}
body.tour #play{display:none}#stop{display:none}body.tour #stop{display:block}
#panel{position:absolute;right:14px;top:58px;width:232px;background:rgba(28,21,31,.92);border:1px solid rgba(255,240,230,.16);border-radius:16px;padding:12px;z-index:6;display:none;box-shadow:0 20px 60px rgba(0,0,0,.6)}
body.tour #panel{top:calc(8.5% + 58px)}
#panel img{width:100%;border-radius:10px;display:block;margin-bottom:9px;aspect-ratio:2/3;object-fit:cover}
#panel .k{display:inline-block;font-size:11px;font-weight:800;padding:3px 10px;border-radius:999px;background:linear-gradient(135deg,#ff5e7e,#ffcf6b);color:#2a1208;margin-bottom:7px}
#panel .t{font-size:16px;font-weight:800;line-height:1.25}#panel .s{font-size:12px;color:#bba9b6;margin:3px 0 7px}
#panel .o{font-size:12px;line-height:1.5;color:#e0d2cc}
#panel .g{margin-top:8px;display:flex;flex-wrap:wrap;gap:5px}
#panel .g span{background:rgba(255,122,89,.18);color:#ffd2bd;border-radius:99px;padding:3px 9px;font-size:11px}
#cap{position:absolute;left:0;right:0;bottom:calc(8.5% + 20px);text-align:center;font-size:23px;font-weight:800;text-shadow:0 4px 22px #000;opacity:0;transition:opacity .5s;pointer-events:none;padding:0 22%;z-index:5}
</style></head><body><div id="w"><canvas id="c"></canvas><div class="bar" id="b1"></div><div class="bar" id="b2"></div><div id="tip"></div>
<div id="hint"><b>Your taste universe.</b> You are the glowing centre. Closer to the centre = a better fit for you. Direction = which of your picks it resembles. Depth = release year, older films sit deeper.</div>
<div id="ctl"><button class="btn" id="play">▶ Guided tour</button><button class="btn" id="fs">⛶ Full screen</button><button class="btn" id="stop">Exit tour</button></div>
<div id="panel"></div><div id="cap"></div><div id="leg"></div></div>
<script>
const D=__DATA__,C=D.cols,cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip'),panel=document.getElementById('panel'),cap=document.getElementById('cap'),play=document.getElementById('play'),sx=document.getElementById('stop');
const reduce=matchMedia('(prefers-reduced-motion: reduce)').matches;
let W,H,S,dpr=Math.min(devicePixelRatio||1,2),T=0,yaw=0,pitch=.3,dy=0,dp=0,zoom=1,tz=1,fx=0,fy=0,fz=0,tx=0,ty=0,tf=0,drag=null,hover=null,sel=null,tour=-1,steps=[],timer=null,nodes=[];
const mk=s=>{if(!s)return null;const i=new Image();i.src=s;return i};
D.picks.forEach(p=>p.im=mk(p.img));D.recs.forEach(r=>r.im=mk(r.img));
document.getElementById('leg').innerHTML=D.picks.map((p,i)=>'<span><i style="background:'+C[i]+'"></i>'+p.title+'</span>').join('')+'<span><i style="background:#ffd9a8;box-shadow:0 0 8px #ffd9a8"></i>You</span><span><i style="background:#fff;box-shadow:0 0 8px #fff"></i>Your recommendations</span>';
function size(){const b=cv.parentNode.getBoundingClientRect();W=b.width;H=b.height;cv.width=W*dpr;cv.height=H*dpr;ctx.setTransform(dpr,0,0,dpr,0,0);S=Math.min(W*.4,H*.42)}
addEventListener('resize',size);size();
function proj(x,y,z){x-=fx;y-=fy;z=(z-fz)*1.7;const a=yaw+dy,b=pitch+dp;
 const X=x*Math.cos(a)+z*Math.sin(a);let Z=-x*Math.sin(a)+z*Math.cos(a);
 const Y=y*Math.cos(b)-Z*Math.sin(b);Z=y*Math.sin(b)+Z*Math.cos(b);
 const k=1/(1+Z*.4);return[W/2+X*S*zoom*k,H/2+Y*S*zoom*k,k,Z]}
function glow(x,y,r,c,a){const g=ctx.createRadialGradient(x,y,0,x,y,r);g.addColorStop(0,c+a);g.addColorStop(1,c+'00');ctx.fillStyle=g;ctx.beginPath();ctx.arc(x,y,r,0,7);ctx.fill()}
function ring(r,al,dash,label){ctx.strokeStyle='rgba(255,240,230,'+al+')';ctx.setLineDash(dash);ctx.beginPath();
 for(let k=0;k<=72;k++){const a=k/72*6.2832,q=proj(r*Math.cos(a),r*Math.sin(a),0);k?ctx.lineTo(q[0],q[1]):ctx.moveTo(q[0],q[1])}
 ctx.stroke();ctx.setLineDash([]);const q=proj(0,-r,0);ctx.fillStyle='rgba(187,169,182,.75)';ctx.font='11px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(label,q[0],q[1]-6)}
function disc(o,x,y,R,col,lw){ctx.save();ctx.beginPath();ctx.arc(x,y,R,0,7);ctx.clip();
 if(o.im&&o.im.complete&&o.im.naturalWidth)ctx.drawImage(o.im,x-R*1.1,y-R*1.1,R*2.2,R*2.2);else{ctx.fillStyle=col;ctx.fillRect(x-R,y-R,R*2,R*2)}
 ctx.restore();ctx.strokeStyle=col;ctx.lineWidth=lw;ctx.beginPath();ctx.arc(x,y,R,0,7);ctx.stroke();ctx.lineWidth=1}
function draw(){
 T+=.016;zoom+=(tz-zoom)*.05;fx+=(tx-fx)*.05;fy+=(ty-fy)*.05;fz+=(tf-fz)*.05;
 if(!drag){if(!reduce){yaw=.5*Math.sin(T*.22);pitch=.32+.1*Math.sin(T*.17+1)}dy*=.985;dp*=.985}
 ctx.clearRect(0,0,W,H);const dim=sel||tour>=0;
 const bg=ctx.createRadialGradient(W/2,H/2,0,W/2,H/2,Math.max(W,H)*.7);bg.addColorStop(0,'#241529');bg.addColorStop(1,'#0c080f');ctx.fillStyle=bg;ctx.fillRect(0,0,W,H);
 ring(.38,.14,[],'Closest match');ring(.7,.10,[4,6],'Close');ring(1,.07,[4,8],'Worth exploring');
 const c0=proj(0,0,0);glow(c0[0],c0[1],S*.5*zoom,'#ff9a55','26');
 for(let i=0;i<D.bx.length;i++){const q=proj(D.bx[i],D.by[i],D.bz[i]);if(q[0]<-4||q[0]>W+4||q[1]<-4||q[1]>H+4)continue;
  ctx.globalAlpha=Math.max(.08,Math.min(.8,(.2+.55*q[2]*q[2])*(dim?.45:1)));ctx.fillStyle=C[D.bc[i]];const s=2.2*q[2]*Math.min(zoom,2);ctx.fillRect(q[0],q[1],s,s)}
 ctx.globalAlpha=1;
 const pq=D.picks.map(p=>proj(p.x,p.y,p.z));
 D.picks.forEach((p,i)=>{p.links.forEach(l=>{const q=proj(l[0],l[1],l[2]);ctx.strokeStyle=C[i]+'30';ctx.beginPath();ctx.moveTo(pq[i][0],pq[i][1]);ctx.lineTo(q[0],q[1]);ctx.stroke();
  ctx.fillStyle=C[i];ctx.globalAlpha=.9;ctx.beginPath();ctx.arc(q[0],q[1],2.2*q[2],0,7);ctx.fill();ctx.globalAlpha=1});
  const c=proj(0,0,0);ctx.strokeStyle=C[i]+'22';ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(pq[i][0],pq[i][1]);ctx.stroke()});
 {const k=Math.min(zoom,1.6),pr=(T*26)%34;
  ctx.strokeStyle='rgba(255,217,168,'+(.55*(1-pr/34))+')';ctx.lineWidth=2;ctx.beginPath();ctx.arc(c0[0],c0[1],(12+pr)*k,0,7);ctx.stroke();ctx.lineWidth=1;
  glow(c0[0],c0[1],38*k,'#ffd9a8','99');ctx.fillStyle='#fff';ctx.beginPath();ctx.arc(c0[0],c0[1],9*k,0,7);ctx.fill();
  ctx.fillStyle='#ff7a59';ctx.beginPath();ctx.arc(c0[0],c0[1],5*k,0,7);ctx.fill();
  ctx.fillStyle='#fff';ctx.font='800 12px Inter,sans-serif';ctx.textAlign='center';ctx.fillText('YOU',c0[0],c0[1]+28*k)}
 const items=[];
 D.recs.forEach(r=>items.push({o:r,t:'r',q:proj(r.x,r.y,r.z)}));D.picks.forEach((p,i)=>items.push({o:p,t:'p',i,q:pq[i]}));
 items.sort((a,b)=>b.q[3]-a.q[3]);nodes=[];
 items.forEach(n=>{const q=n.q,k=q[2]*Math.min(zoom,1.6);
  if(n.t==='r'){const r=n.o,on=sel===r,R=(on?19:14)*k,l=pq[r.lead];
   ctx.strokeStyle='rgba(255,217,168,'+(on?.5:.14)+')';ctx.setLineDash([3,5]);ctx.beginPath();ctx.moveTo(c0[0],c0[1]);ctx.lineTo(q[0],q[1]);ctx.stroke();ctx.setLineDash([]);
   ctx.strokeStyle=C[r.lead]+(on?'dd':'55');ctx.lineWidth=on?2:1;ctx.beginPath();ctx.moveTo(q[0],q[1]);ctx.lineTo(l[0],l[1]);ctx.stroke();ctx.lineWidth=1;
   glow(q[0],q[1],R*3,'#ffd9a8',on?'99':'55');disc(r,q[0],q[1],R,'#ffffff',on?2.5:1.5);
   ctx.fillStyle='#ff7a59';ctx.beginPath();ctx.arc(q[0]+R*.8,q[1]-R*.8,8,0,7);ctx.fill();ctx.fillStyle='#2a1208';ctx.font='800 10px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(r.rank,q[0]+R*.8,q[1]-R*.8+3.5);
   nodes.push({o:r,t:'r',x:q[0],y:q[1],R:R+4})}
  else{const p=n.o,R=19*k;glow(q[0],q[1],R*2.4,C[n.i],'66');disc(p,q[0],q[1],R,C[n.i],3);
   ctx.fillStyle=C[n.i];ctx.font='700 12px Inter,sans-serif';ctx.textAlign='center';ctx.fillText(p.title.length>24?p.title.slice(0,23)+'…':p.title,q[0],q[1]+R+16);
   nodes.push({o:p,t:'p',i:n.i,x:q[0],y:q[1],R:R+4})}});
 nodes.push({o:{title:'You: the centre of your taste'},t:'y',x:c0[0],y:c0[1],R:16});
 requestAnimationFrame(draw)}
function show(o,t){sel=(t==='r')?o:null;if(!o){panel.style.display='none';return}
 const lead=t==='r'?D.picks[o.lead].title:'';
 panel.innerHTML=(o.img?'<img src="'+o.img+'">':'')+(t==='r'?'<div class="k">#'+o.rank+' · '+o.pct+'% match</div>':'<div class="k">Your pick</div>')+'<div class="t">'+o.title+'</div><div class="s">'+o.sub+(lead?' · closest to '+lead:'')+'</div>'+(t==='r'?'<div class="o">'+o.overview+'</div><div class="g">'+o.tags.map(x=>'<span>'+x+'</span>').join('')+'</div>':'');
 panel.style.display='block'}
function look(o,z){tx=o.x*.85;ty=o.y*.85;tf=o.z;tz=z}
function pos(e){const b=cv.getBoundingClientRect();return[e.clientX-b.left,e.clientY-b.top]}
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,m:0};cv.style.cursor='grabbing';cv.setPointerCapture(e.pointerId)});
cv.addEventListener('pointermove',e=>{const[X,Y]=pos(e);
 if(drag){dy=Math.max(-1.1,Math.min(1.1,dy+(e.clientX-drag.x)*.006));dp=Math.max(-.6,Math.min(.6,dp+(e.clientY-drag.y)*.004));drag.m+=Math.abs(e.clientX-drag.x)+Math.abs(e.clientY-drag.y);drag.x=e.clientX;drag.y=e.clientY;return}
 hover=nodes.find(n=>Math.hypot(n.x-X,n.y-Y)<n.R)||null;cv.style.cursor=hover?'pointer':'grab';
 if(hover){tip.style.display='block';tip.style.left=Math.min(X+14,W-230)+'px';tip.style.top=(Y+14)+'px';tip.textContent=hover.o.title}else tip.style.display='none'});
cv.addEventListener('pointerup',e=>{const moved=drag&&drag.m>5;drag=null;cv.style.cursor='grab';if(moved||tour>=0)return;
 const[X,Y]=pos(e),n=nodes.find(n=>Math.hypot(n.x-X,n.y-Y)<n.R);
 if(n&&n.t!=='y'){show(n.o,n.t);look(n.o,1.7)}else{show(null);tx=ty=tf=0;tz=1}});
cv.addEventListener('wheel',e=>{e.preventDefault();tz=Math.max(.7,Math.min(3,tz*(e.deltaY<0?1.12:.9)))},{passive:false});
function step(){if(tour>=steps.length){stop();return}const s=steps[tour++];let o;
 if(s.k==='i'){tx=ty=tf=0;tz=1;show(null);cap.textContent='Your taste universe, with you at the centre'}
 else if(s.k==='p'){o=D.picks[s.i];look(o,1.9);show(o,'p');cap.textContent='Because you picked '+o.title}
 else{o=D.recs[s.i];look(o,2.1);show(o,'r');cap.textContent='#'+o.rank+'  '+o.title+'  ·  '+o.pct+'% match'}
 cap.style.opacity=1;timer=setTimeout(step,s.k==='i'?2600:4200)}
function stop(){clearTimeout(timer);tour=-1;document.body.classList.remove('tour');tx=ty=tf=0;tz=1;show(null);cap.style.opacity=0}
play.onclick=()=>{steps=[{k:'i'}].concat(D.picks.map((p,i)=>({k:'p',i}))).concat(D.recs.map((r,i)=>({k:'r',i})));tour=0;document.body.classList.add('tour');step()};sx.onclick=stop;document.getElementById('fs').onclick=()=>{const w=document.getElementById('w');document.fullscreenElement?document.exitFullscreen():(w.requestFullscreen&&w.requestFullscreen())};document.addEventListener('fullscreenchange',()=>setTimeout(size,120));
draw();
</script></body></html>
"""

def render_galaxy(favs, results):
    data = json.dumps(galaxy_payload(favs, results), ensure_ascii=False).replace("</", "<\\/")
    components.html(GALAXY_HTML.replace("__DATA__", data), height=760)

# ---------------- UI ----------------
total = len(movies)
st.markdown('<div class="site"><div class="brandbox"><svg viewBox="0 0 48 48" xmlns="http://www.w3.org/2000/svg"><defs><linearGradient id="lg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#ff5e7e"/><stop offset=".55" stop-color="#ff8a4c"/><stop offset="1" stop-color="#ffb347"/></linearGradient></defs><rect x="2" y="2" width="44" height="44" rx="13" fill="url(#lg)"/><rect x="9" y="12" width="30" height="5" rx="2" fill="#2b1c26" opacity=".85"/><path d="M19 22v14l12-7z" fill="#fff"/></svg><div><div class="wm">RECO<b>MIND</b></div><div class="tag">Movie discovery</div></div></div>'
            '<div class="links"><a href="#discover">Discover</a><a href="#results">Recommendations</a><a href="#map">Taste map</a><a href="#engine">Engine report</a></div>'
            f'<div class="status"><b>●</b> {total:,} films indexed</div></div>', unsafe_allow_html=True)

wall_css, wall_html = hero_wall()
st.markdown(wall_css + f'<div class="hero">{wall_html}<div class="hero-in"><div class="pill"><span class="dot"></span>Content-based recommender</div>'
            '<h1>Pick three films.<br><span>Find what to watch next.</span></h1>'
            '<p>Choose up to three films you love. RECOMIND reads their plots and genres, builds your taste profile and finds films that match it.</p></div></div>',
            unsafe_allow_html=True)

st.markdown('<div class="section-heading" id="discover">Start with a few favourites</div>'
            '<div class="section-subtitle">Search by title, or tap a film below to add it to your picks.</div>', unsafe_allow_html=True)

titles = sorted(movies["title"].dropna().astype(str).unique())
favs = st.multiselect("Your picks", titles, key="favs", max_selections=3, placeholder="Type a film title...", label_visibility="collapsed")

genre = st.radio("Browse", ["Popular"] + all_genres(), horizontal=True, label_visibility="collapsed")
bt = browse_titles(genre)
preload(bt)
cols = st.columns(len(bt))
for c, t in zip(cols, bt):
    with c:
        display_movie_visual(t)
        st.markdown(f'<div class="tcap">{html.escape(t)}</div>', unsafe_allow_html=True)
        st.button("Add", key=f"add_{t}", on_click=add_pick, args=(t,), disabled=len(favs) >= 3 and t not in favs)

_fin = YEARS[np.isfinite(YEARS)]
ymin, ymax = (int(_fin.min()), int(_fin.max())) if len(_fin) else (1900, 2026)
ymax = max(ymax, 2026)
s1, s2 = st.columns(2)
s1.slider("Familiar  ↔  Surprising", 0.0, 0.8, 0.0, 0.1, key="dv",
          help="0 = the closest matches. Higher values make the five films differ more from each other.")
s2.slider("Release years", ymin, ymax, (ymin, ymax), key="yrs", help="Only recommend films from this period.")
e1, e2, e3 = st.columns(3)
e1.multiselect("Only these genres", all_genres(), key="gf", placeholder="Any genre")
e2.multiselect("Already watched (hide)", titles, key="ex", placeholder="Films to skip")
e3.multiselect("Movie night: a friend's picks", titles, key="friend", max_selections=3,
               placeholder="Optional: films you'll both enjoy", help="Add up to 3 films your friend loves. The list is then ranked so both of you are likely to enjoy it.")
b1, b2, b3 = st.columns([4, 1, 1])
b1.button("Find my next film", type="primary", disabled=not favs, on_click=run_engine)
b2.button("Surprise me", on_click=surprise)
b3.button("Clear picks", on_click=clear_picks)

if "res" in st.session_state:
    rf, res = st.session_state["res"]
    st.markdown('<div class="section-heading" id="results" style="margin-top:34px">Your recommendations</div>', unsafe_allow_html=True)
    top = max(float(res["similarity_score"].max()), 1e-9)
    for k, (t, sc) in enumerate(zip(res["title"], res["similarity_score"])):
        t = str(t)
        row = movie_row(t)
        with st.container(border=True):
            a, b, c = st.columns([1.2, 4, 1.2])
            with a:
                display_movie_visual(t)
            with b:
                g = genres_of(row)
                tags = shared_terms(rf, t, 4)
                st.markdown(f'<span class="rank-tag">#{k+1}</span><div class="movie-title">{html.escape(t)}</div>'
                            f'<div class="movie-meta">{html.escape(" · ".join(v for v in [year_of(row)] + g[:3] if v))}</div>'
                            f'<div class="overview">{html.escape(str(row.get("overview", "") or "") if row is not None else "")}</div>'
                            + ('<div class="why">Why it fits your picks</div><div class="chips">'
                               + "".join(f'<span class="chip g">{html.escape(v)}</span>' for v in g[:3])
                               + "".join(f'<span class="chip">{html.escape(v)}</span>' for v in tags) + '</div>'
                               if g or tags else ""), unsafe_allow_html=True)
            with c:
                pct = int(round(100 * float(sc) / top * 0.98))
                st.markdown(f'<div class="ring" style="--p:{pct}" data-v="{pct}%"></div><div class="ringcap">match</div>', unsafe_allow_html=True)
                f1, f2 = st.columns(2)
                f1.button("👍", key=f"up_{k}_{t}", on_click=feedback, args=(t, "up"), help="More like this")
                f2.button("👎", key=f"dn_{k}_{t}", on_click=feedback, args=(t, "down"), help="Less like this")

    _csv = pd.DataFrame({"rank": range(1, len(res) + 1), "title": [str(t) for t in res["title"]],
                         "year": [year_of(movie_row(str(t))) for t in res["title"]],
                         "match_%": [int(round(100 * float(v) / top * 0.98)) for v in res["similarity_score"]]})
    st.download_button("Download my watchlist (CSV)", _csv.to_csv(index=False).encode("utf-8"), "recomind_watchlist.csv", "text/csv")

    if st.session_state.get("friend"):
        st.caption("Movie night mode: the list is ranked so both you and your friend are likely to enjoy it.")
    nl, nd = len(st.session_state.get("liked", [])), len(st.session_state.get("disliked", []))
    if nl or nd:
        m1, m2 = st.columns([5, 1])
        m1.caption(f"The engine is learning from your feedback: {nl} liked, {nd} disliked. Rate more results to refine the list.")
        m2.button("Reset feedback", on_click=reset_feedback)
    st.markdown('<div class="section-heading" id="engine" style="margin-top:34px">Engine report</div>'
                '<div class="section-subtitle">How good is this list? Variety is how different the films are from each other, '
                'and hidden gems are films outside the 10% most popular.</div>', unsafe_allow_html=True)
    st.markdown('<div class="rep">' + "".join(
        f'<div class="rc"><div class="l">{html.escape(l)}</div><div class="v">{html.escape(v)}</div>'
        f'<div class="meter"><i style="width:{p}%"></i></div><div class="d">{html.escape(d)}</div></div>'
        for l, v, d, p in engine_report(res)) + '</div>', unsafe_allow_html=True)
    # ---- engine check ----
    st.markdown('<div class="section-heading" style="margin-top:34px">Can it rediscover your picks?</div>'
                '<div class="section-subtitle">A fair test: each pick is hidden in turn and the engine rebuilds your taste from the others. '
                'If it ranks the hidden film near the top, the engine really understands your taste.</div>', unsafe_allow_html=True)
    rows = loo_check(rf)
    if rows:
        N_all = len(movies)
        cards = ""
        for t, rk in rows:
            pct_top = 100 * rk / N_all
            cls, word = ("ok", "Found it") if pct_top <= 1 else (("mid", "Close") if pct_top <= 5 else ("no", "Missed"))
            cards += (f'<div class="rc"><span class="bd {cls}">{word}</span><div class="t">{html.escape(t)}</div>'
                      f'<div class="v">#{rk:,}<small>of {N_all:,}</small></div>'
                      f'<div class="meter"><i style="width:{max(2, int(100 - pct_top))}%"></i></div>'
                      f'<div class="d">Ranked in the top {pct_top:.1f}% of all films without being told about it.</div></div>')
        st.markdown(f'<div class="rep">{cards}</div>', unsafe_allow_html=True)
    else:
        st.caption("Pick at least two films to run this check.")

    # ---- taste DNA: a personal "taste card", not a table ----
    def mix(ts):
        c = {}
        for t in ts:
            for g in genres_of(movie_row(str(t))):
                c[g] = c.get(g, 0) + 1
        top6 = sorted(c.items(), key=lambda x: -x[1])[:6]
        sm = sum(v for _, v in top6) or 1
        return [(g, v / sm) for g, v in top6]
    dna = mix(rf)
    if dna:
        inks = ["#d9482b", "#2b3a55", "#e0a526", "#4a7c6f", "#8a5a83", "#a8a39a"]
        title = " / ".join(g for g, _ in dna[:2])
        bar = "".join(f'<i style="width:{v * 100:.1f}%;background:{inks[i % 6]}"></i>' for i, (g, v) in enumerate(dna))
        rows = "".join(f'<div class="tk-r"><i style="background:{inks[i % 6]}"></i><span>{html.escape(g)}</span><u></u><b>{int(round(v * 100))}%</b></div>'
                       for i, (g, v) in enumerate(dna))
        words = "<br>".join(html.escape(w) for w in sig_terms(rf, 7))
        st.markdown('<div class="section-heading" style="margin-top:34px">Your taste DNA</div>'
                    f'<div class="ticket"><div class="tk-main"><div class="tk-k">Taste profile · {len(rf)} film{"s" if len(rf) != 1 else ""}</div>'
                    f'<div class="tk-t">{html.escape(title)}</div><div class="tk-bar">{bar}</div>{rows}</div>'
                    f'<div class="tk-stub"><div class="tk-k">Recurring words</div><div class="tk-w">{words or "n/a"}</div>'
                    f'<div class="tk-admit">ADMIT ONE</div></div></div>', unsafe_allow_html=True)

    if tfidf_matrix is not None:
        st.markdown('<div class="section-heading" id="map" style="margin-top:34px">Taste map</div>'
                    '<div class="section-subtitle">Every dot is a film. Distance from the centre shows how well it fits you, direction shows which pick it resembles, and depth shows its release year. '
                    'Drag to rotate, scroll to zoom, or start the guided tour.</div>', unsafe_allow_html=True)
        try:
            render_galaxy(rf, res)
        except Exception as e:
            st.info(f"The taste map could not be drawn: {e}")

st.markdown('<div class="fnote">RECOMIND · a hybrid content-based recommender with feedback learning · poster data from TMDB, iTunes and Wikipedia</div>', unsafe_allow_html=True)
