# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from io import StringIO
import math
import unicodedata

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup
from pybaseball import (
    statcast,
    statcast_pitcher,
    statcast_batter_pitch_arsenal,
    pitching_stats,
    pitching_stats_bref,
    playerid_lookup,
)

# ============================================================
# MODEL PROFESSIONAL MLB - STARTING PITCHER STRIKEOUTS
# V3.1.3 LIVE TEST
# ============================================================

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game"
SAVANT_PARK_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
ACTION_PITCHING_PROPS_URL = "https://www.actionnetwork.com/mlb/props/pitching"
TIMEOUT = 30

PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle Curve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball",
    "KN": "Knuckleball", "SV": "Slurve",
}

SWING_DESCRIPTIONS = {
    "swinging_strike", "swinging_strike_blocked", "foul", "foul_tip",
    "hit_into_play", "foul_bunt", "missed_bunt",
}
WHIFF_DESCRIPTIONS = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
CONTACT_DESCRIPTIONS = SWING_DESCRIPTIONS - WHIFF_DESCRIPTIONS
CALLED_STRIKE_DESCRIPTIONS = {"called_strike"}

# ============================================================
# PAGE / STYLE
# ============================================================

st.set_page_config(
    page_title="MLB Starting Pitcher K Model",
    page_icon="⚾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    :root{
      --blue:#4f8cff;--green:#38d996;--red:#ff6b6b;--gold:#f5c451;
      --muted:rgba(230,235,245,.62);--border:rgba(150,160,185,.16);
    }
    .block-container{max-width:1180px;padding-top:.75rem;padding-bottom:4rem}
    h1,h2,h3{letter-spacing:-.025em}
    .hero{
      position:relative;overflow:hidden;border:1px solid rgba(79,140,255,.24);
      border-radius:24px;padding:22px;background:
      radial-gradient(circle at 88% 15%,rgba(79,140,255,.20),transparent 32%),
      linear-gradient(135deg,rgba(35,42,70,.96),rgba(17,20,29,.98));
      box-shadow:0 18px 50px rgba(0,0,0,.20);margin-bottom:16px
    }
    .hero:after{content:"K";position:absolute;right:24px;top:-22px;font-size:8rem;
      font-weight:900;color:rgba(79,140,255,.055);transform:rotate(-7deg)}
    .gamecard{
      border:1px solid var(--border);border-radius:22px;padding:20px;margin:8px 0 15px;
      background:linear-gradient(145deg,rgba(25,29,39,.98),rgba(17,19,26,.96));
      box-shadow:0 10px 35px rgba(0,0,0,.12)
    }
    .finalcard{
      border:1px solid rgba(79,140,255,.56);border-radius:22px;padding:21px;margin:15px 0;
      background:radial-gradient(circle at 90% 15%,rgba(79,140,255,.22),transparent 34%),
      linear-gradient(145deg,rgba(32,55,105,.45),rgba(18,22,32,.98));
      box-shadow:0 14px 45px rgba(30,90,255,.12)
    }
    .analysis-card{
      border:1px solid var(--border);border-radius:18px;padding:16px 17px;margin:10px 0;
      background:linear-gradient(145deg,rgba(27,31,42,.90),rgba(17,19,25,.95))
    }
    .signal-positive{border-left:4px solid var(--green)}
    .signal-negative{border-left:4px solid var(--red)}
    .signal-neutral{border-left:4px solid rgba(180,190,210,.45)}
    .signal-pending{border-left:4px solid var(--gold)}
    .pill{display:inline-block;padding:5px 10px;border:1px solid rgba(150,160,185,.12);
      border-radius:999px;background:rgba(145,155,175,.10);margin:4px 4px 2px 0;font-size:.78rem}
    .section-label{font-size:.72rem;letter-spacing:.08em;font-weight:800;opacity:.55;text-transform:uppercase}
    .confidence-a{color:var(--green);font-weight:800}
    .confidence-b{color:var(--gold);font-weight:800}
    .confidence-c{color:var(--red);font-weight:800}
    div[data-testid="stMetric"]{
      border:1px solid var(--border);border-radius:17px;padding:12px;
      background:linear-gradient(145deg,rgba(28,32,42,.96),rgba(18,20,27,.96))
    }
    div[data-testid="stMetric"] label{color:var(--muted)!important}
    div[data-testid="stTabs"] button[aria-selected="true"]{color:#ff6d6d!important}
    .small-muted{color:var(--muted);font-size:.82rem}

    .slate-header{display:flex;justify-content:space-between;align-items:end;gap:12px;margin:10px 0 14px}
    .slate-count{font-size:.78rem;opacity:.58;font-weight:700}
    .match-card{border:1px solid var(--border);border-radius:20px;padding:16px;margin-bottom:12px;background:linear-gradient(145deg,rgba(27,31,42,.94),rgba(17,19,26,.98));box-shadow:0 8px 26px rgba(0,0,0,.10)}
    .match-top{display:flex;justify-content:space-between;gap:12px;align-items:center;margin-bottom:10px}
    .match-title{font-size:1.02rem;font-weight:850}
    .match-time{font-size:.78rem;opacity:.56;white-space:nowrap}
    .pitcher-row{padding:10px 0;border-top:1px solid rgba(150,160,185,.10)}
    .pitcher-name{font-weight:800;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .pitcher-sub{font-size:.76rem;opacity:.58;margin-top:2px}
    .mini-stat{display:inline-block;padding:3px 7px;border-radius:999px;margin-right:4px;margin-top:5px;font-size:.70rem;background:rgba(79,140,255,.10);border:1px solid rgba(79,140,255,.12)}
    .state-dot{display:inline-block;width:8px;height:8px;border-radius:50%;margin-right:5px}
    .dot-green{background:var(--green)} .dot-gold{background:var(--gold)}

    .board-grid{
      display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin:12px 0 18px
    }
    .board-game{
      background:linear-gradient(150deg,rgba(23,27,37,.98),rgba(13,15,21,.99));
      border:1px solid rgba(130,145,180,.18);border-radius:14px;padding:10px;
      min-height:178px;box-shadow:0 5px 16px rgba(0,0,0,.14)
    }
    .board-time{font-size:.62rem;opacity:.56;text-align:center;margin-bottom:7px}
    .board-team{
      display:grid;grid-template-columns:26px 1fr auto;align-items:center;gap:6px;
      padding:5px 0;border-bottom:1px solid rgba(140,150,175,.08)
    }
    .board-team:last-of-type{border-bottom:0}
    .board-logo{width:25px;height:25px;object-fit:contain}
    .board-abbr{font-size:.78rem;font-weight:850}
    .board-pitcher{font-size:.62rem;opacity:.62;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .board-k{font-size:.67rem;font-weight:750}
    .board-actions{display:grid;grid-template-columns:1fr 1fr;gap:5px;margin-top:7px}
    .board-link{
      text-decoration:none!important;text-align:center;padding:5px 4px;border-radius:8px;
      font-size:.61rem;font-weight:850;color:#dce7ff!important;
      background:rgba(63,116,220,.13);border:1px solid rgba(79,140,255,.22)
    }
    .board-link:hover{background:rgba(79,140,255,.22)}
    .board-legend{font-size:.70rem;opacity:.55;margin-top:-8px;margin-bottom:12px}
    @media (max-width:1100px){.board-grid{grid-template-columns:repeat(4,minmax(0,1fr))}}
    @media (max-width:760px){
      .board-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}
      .board-game{padding:7px;min-height:158px;border-radius:11px}
      .board-logo{width:21px;height:21px}
      .board-team{grid-template-columns:22px 1fr auto;gap:4px}
      .board-abbr{font-size:.69rem}
      .board-pitcher,.board-time,.board-link{font-size:.54rem}
      .board-k{font-size:.58rem}
    }

    </style>
    """,
    unsafe_allow_html=True,
)

# ============================================================
# HELPERS
# ============================================================

class DataSourceError(Exception):
    pass


def get_json(url: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(
            url, params=params, timeout=TIMEOUT,
            headers={"User-Agent": "Mozilla/5.0"},
        )
        r.raise_for_status()
        data = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataSourceError(str(exc)) from exc
    return data if isinstance(data, dict) else {}


def safe_num(v):
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(str(v).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def fmt(v, decimals=1, suffix=""):
    x = safe_num(v)
    return "N/A" if x is None else f"{x:.{decimals}f}{suffix}"


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def weighted_average(items, default=None):
    valid = [(safe_num(v), float(w)) for v, w in items if safe_num(v) is not None and w > 0]
    if not valid:
        return default
    tw = sum(w for _, w in valid)
    return sum(v*w for v, w in valid) / tw if tw else default


def normalize_name(value):
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(c for c in text if not unicodedata.combining(c))
    return " ".join(text.lower().replace(".", "").replace("-", " ").split())


def innings_decimal(value):
    if value is None:
        return None
    s = str(value)
    if "." not in s:
        return safe_num(s)
    try:
        whole, frac = s.split(".", 1)
        return int(whole) + int(frac[:1]) / 3
    except Exception:
        return None


def american_implied(odds):
    o = safe_num(odds)
    if o is None or o == 0:
        return None
    return ((-o)/((-o)+100) if o < 0 else 100/(o+100)) * 100


def fair_american(prob):
    p = safe_num(prob)
    if p is None or p <= 0 or p >= 1:
        return None
    return -100*p/(1-p) if p >= .5 else 100*(1-p)/p


def profit_per_dollar(odds):
    o = safe_num(odds)
    if o is None or o == 0:
        return None
    return o/100 if o > 0 else 100/abs(o)


def ev_per_dollar(prob, odds):
    p = safe_num(prob)
    profit = profit_per_dollar(odds)
    return None if p is None or profit is None else p*profit - (1-p)


def poisson_pmf(k, lam):
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k*math.log(lam) - math.lgamma(k+1))


def poisson_cdf(k, lam):
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k+1))


def poisson_ge(k, lam):
    return clamp(1-poisson_cdf(k-1, lam), 0, 1)


def game_cutoff(selected_date: date) -> date:
    # Fundamental anti-leak rule: never use selected game's own data.
    return selected_date - timedelta(days=1)


# ============================================================
# MLB DATA
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def pitcher_profile(player_id: int):
    d = get_json(f"{MLB_PEOPLE_URL}/{player_id}")
    people = d.get("people", [])
    return people[0] if people else {}


@st.cache_data(ttl=86400, show_spinner=False)
def team_info(team_id: int, season: int):
    d = get_json(f"{MLB_TEAMS_URL}/{team_id}", params={"season": season})
    teams = d.get("teams", [])
    return teams[0] if teams else {}


@st.cache_data(ttl=300, show_spinner=False)
def person_stats_to_date(player_id: int, season: int, group: str, end_date: str, sit_code: str | None = None):
    params = {
        "stats": "byDateRange",
        "group": group,
        "startDate": f"{season}-03-01",
        "endDate": end_date,
    }
    if sit_code:
        params["sitCodes"] = sit_code
    d = get_json(f"{MLB_PEOPLE_URL}/{player_id}/stats", params=params)
    groups = d.get("stats", [])
    if not groups:
        return {}
    splits = groups[0].get("splits", [])
    return splits[0].get("stat", {}) if splits else {}


@st.cache_data(ttl=300, show_spinner=False)
def pitcher_stats_to_date(player_id: int, season: int, end_date: str):
    stat = person_stats_to_date(player_id, season, "pitching", end_date)
    so = safe_num(stat.get("strikeOuts"))
    bb = safe_num(stat.get("baseOnBalls"))
    bf = safe_num(stat.get("battersFaced"))
    gs = safe_num(stat.get("gamesStarted"))
    ip = innings_decimal(stat.get("inningsPitched"))

    stat["calc_k_pct"] = so/bf*100 if so is not None and bf else None
    stat["calc_bb_pct"] = bb/bf*100 if bb is not None and bf else None
    stat["calc_k_minus_bb"] = (
        stat["calc_k_pct"] - stat["calc_bb_pct"]
        if stat["calc_k_pct"] is not None and stat["calc_bb_pct"] is not None else None
    )
    stat["calc_bf_start"] = bf/gs if bf is not None and gs else None
    stat["calc_ip_start"] = ip/gs if ip is not None and gs else None
    stat["calc_k_start"] = so/gs if so is not None and gs else None
    return stat


@st.cache_data(ttl=300, show_spinner=False)
def pitcher_game_log_before(player_id: int, season: int, selected_date: str):
    d = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": season},
    )
    groups = d.get("stats", [])
    if not groups:
        return []

    cutoff = pd.Timestamp(selected_date)
    out = []
    for split in groups[0].get("splits", []):
        try:
            split_date = pd.Timestamp(split.get("date"))
        except Exception:
            continue
        if split_date >= cutoff:
            continue

        s = split.get("stat", {})
        if not safe_num(s.get("gamesStarted")):
            continue
        out.append({
            "Date": split.get("date", "N/A"),
            "Opp": split.get("opponent", {}).get("name", "N/A"),
            "IP": s.get("inningsPitched", "N/A"),
            "K": s.get("strikeOuts", "N/A"),
            "BB": s.get("baseOnBalls", "N/A"),
            "ER": s.get("earnedRuns", "N/A"),
            "HR": s.get("homeRuns", "N/A"),
            "Pitches": s.get("numberOfPitches", "N/A"),
            "BF": s.get("battersFaced", "N/A"),
        })
    return out[-10:][::-1]


@st.cache_data(ttl=300, show_spinner=False)
def team_hitting_to_date(team_id: int, season: int, end_date: str, sit_code: str | None = None):
    params = {
        "stats": "byDateRange",
        "group": "hitting",
        "startDate": f"{season}-03-01",
        "endDate": end_date,
    }
    if sit_code:
        params["sitCodes"] = sit_code

    d = get_json(f"{MLB_TEAMS_URL}/{team_id}/stats", params=params)
    groups = d.get("stats", [])
    if not groups:
        return {}
    splits = groups[0].get("splits", [])
    if not splits:
        return {}
    stat = splits[0].get("stat", {})
    so = safe_num(stat.get("strikeOuts"))
    bb = safe_num(stat.get("baseOnBalls"))
    pa = safe_num(stat.get("plateAppearances"))
    stat["calc_k_pct"] = so/pa*100 if so is not None and pa else None
    stat["calc_bb_pct"] = bb/pa*100 if bb is not None and pa else None
    return stat


def game_time_label(raw):
    if not raw:
        return "TBD"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%b %d · %I:%M %p UTC").replace(" 0", " ")
    except Exception:
        return "TBD"


@st.cache_data(ttl=300, show_spinner=False)
def pitchers_for_date(selected_date: str):
    d = get_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1, "date": selected_date,
            "hydrate": "probablePitcher,team,opponents,venue",
        },
    )
    options = []
    for dg in d.get("dates", []):
        for game in dg.get("games", []):
            teams = game.get("teams", {})
            venue = game.get("venue", {})
            for side, opp_side in (("away","home"),("home","away")):
                td, od = teams.get(side, {}), teams.get(opp_side, {})
                probable = td.get("probablePitcher")
                if not isinstance(probable, dict):
                    continue
                pid, name = probable.get("id"), probable.get("fullName")
                if not pid or not name:
                    continue
                team, opp = td.get("team", {}), od.get("team", {})
                try:
                    hand = pitcher_profile(int(pid)).get("pitchHand", {}).get("description", "N/A")
                except Exception:
                    hand = "N/A"
                options.append({
                    "selection_id": f"{game.get('gamePk')}-{side}-{pid}",
                    "game_pk": game.get("gamePk"),
                    "pitcher_id": int(pid),
                    "pitcher_name": name,
                    "team": team.get("name","N/A"),
                    "team_id": team.get("id"),
                    "team_side": side,
                    "opponent": opp.get("name","N/A"),
                    "opponent_id": opp.get("id"),
                    "opponent_side": opp_side,
                    "throwing_hand": hand,
                    "venue": venue.get("name","N/A"),
                    "status": game.get("status", {}).get("detailedState","N/A"),
                    "game_time": game_time_label(game.get("gameDate")),
                })
    return options


@st.cache_data(ttl=180, show_spinner=False)
def game_feed(game_pk: int):
    return get_json(f"{MLB_GAME_FEED_URL}/{game_pk}/feed/live") if game_pk else {}


def game_context(feed):
    gd = feed.get("gameData", {})
    weather = gd.get("weather", {})
    officials = feed.get("liveData", {}).get("boxscore", {}).get("officials", [])
    umpire = next(
        (x.get("official", {}).get("fullName") for x in officials if x.get("officialType") == "Home Plate"),
        None,
    )
    return {
        "temperature": weather.get("temp"),
        "condition": weather.get("condition"),
        "wind": weather.get("wind"),
        "umpire": umpire,
    }


def confirmed_lineup(feed, side):
    box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {}).get(side, {})
    order = box.get("battingOrder", [])
    players = box.get("players", {})
    rows = []
    for i, pid in enumerate(order, 1):
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        rows.append({
            "#": i,
            "player_id": int(pid),
            "Hitter": person.get("fullName", f"Player {pid}"),
            "Bats": person.get("batSide", {}).get("code","N/A"),
        })
    return rows



def _flatten_columns(df):
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        cols = []
        for col in out.columns:
            parts = [str(x).strip() for x in col if str(x).strip() and str(x).lower() != "nan"]
            cols.append(" ".join(parts))
        out.columns = cols
    else:
        out.columns = [str(c).strip() for c in out.columns]
    return out


def _find_col(columns, *needles):
    norms = {c: normalize_name(c) for c in columns}
    for c, n in norms.items():
        if all(normalize_name(x) in n for x in needles):
            return c
    return None

def _metric_col(columns, exact_names, contains_names=()):
    norm={c:normalize_name(c) for c in columns}
    exact={normalize_name(x) for x in exact_names}
    for c,n in norm.items():
        if n in exact:
            return c
    for c,n in norm.items():
        if any(normalize_name(x) in n for x in contains_names):
            return c
    return None



@st.cache_data(ttl=21600, show_spinner=False)
def savant_team_plate_discipline(team_id: int, season: int):
    url = f"https://baseballsavant.mlb.com/team/{team_id}?season={season}"
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
    except Exception:
        return {}, "SOURCE_ERROR"

    target = None
    for raw in tables:
        t = _flatten_columns(raw)
        cols_norm = [normalize_name(c) for c in t.columns]
        if (any("pitches" in c for c in cols_norm)
            and any("chase" in c and "contact" not in c for c in cols_norm)
            and any("whiff" in c for c in cols_norm)
            and any("zone" in c and "contact" in c for c in cols_norm)):
            target = t
            break
    if target is None or target.empty:
        return {}, "TABLE_NOT_FOUND"

    pitch_col = _find_col(target.columns, "pitches")
    zone_col = next((c for c in target.columns if normalize_name(c) in ("zone", "zone %")), None)
    zc_col = _find_col(target.columns, "zone", "contact")
    chase_col = next((c for c in target.columns if "chase" in normalize_name(c) and "contact" not in normalize_name(c)), None)
    chase_contact_col = _find_col(target.columns, "chase", "contact")
    whiff_col = _find_col(target.columns, "whiff")
    swing_candidates = [c for c in target.columns if "swing" in normalize_name(c) and "zone" not in normalize_name(c) and "meatball" not in normalize_name(c) and "1st" not in normalize_name(c)]
    swing_col = swing_candidates[0] if swing_candidates else None

    def wavg(col):
        if not col or not pitch_col: return None
        vals = pd.to_numeric(target[col], errors="coerce")
        weights = pd.to_numeric(target[pitch_col], errors="coerce")
        good = vals.notna() & weights.notna() & (weights > 0)
        if not good.any(): return None
        return (vals[good]*weights[good]).sum()/weights[good].sum()

    whiff = wavg(whiff_col)
    return {
        "Pitches": safe_num(pd.to_numeric(target[pitch_col],errors="coerce").sum()) if pitch_col else None,
        "Zone%": wavg(zone_col),
        "Z-Contact%": wavg(zc_col),
        "Chase%": wavg(chase_col),
        "O-Swing%": wavg(chase_col),
        "O-Contact%": wavg(chase_contact_col),
        "Swing%": wavg(swing_col),
        "Whiff%": whiff,
        "Contact%": 100-whiff if whiff is not None else None,
    }, "SAVANT_TEAM_PAGE"


@st.cache_data(ttl=21600, show_spinner=False)
def savant_team_pitch_type(team_id: int, season: int):
    url = (
        "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
        f"?type=batter&pitchType=&year={season}&team={team_id}&min=1&csv=true"
    )
    try:
        r = requests.get(url, headers={"User-Agent":"Mozilla/5.0"}, timeout=TIMEOUT)
        r.raise_for_status()
        df = pd.read_csv(StringIO(r.text))
    except Exception:
        return pd.DataFrame(), "SOURCE_ERROR"
    if df.empty: return pd.DataFrame(), "EMPTY"

    df = _flatten_columns(df)
    cols = list(df.columns)
    def col(*names): return _find_col(cols,*names)

    pitch_col=col("pitch"); pitches_col=col("pitches"); pa_col=col("pa")
    if not pitch_col or not pitches_col: return pd.DataFrame(),"COLUMNS_NOT_FOUND"

    mapping = {
        "BA":_metric_col(cols,["ba","batting average"],["ba"]),
        "SLG":_metric_col(cols,["slg","slugging"],["slg"]),
        "wOBA":_metric_col(cols,["woba"],["woba"]),
        "Whiff%":_metric_col(cols,["whiff","whiff %","whiff_percent"],["whiff"]),
        "K%":_metric_col(cols,["k%","k_percent","strikeout %"],["k percent","strikeout"]),
        "PutAway%":_metric_col(cols,["putaway%","put away %","put_away"],["put away","putaway"]),
        "xBA":_metric_col(cols,["xba","estimated ba","estimated_ba"],["xba","estimated ba"]),
        "xSLG":_metric_col(cols,["xslg","estimated slg","estimated_slg"],["xslg","estimated slg"]),
        "xwOBA":_metric_col(cols,["xwoba","estimated woba","estimated_woba"],["xwoba","estimated woba"]),
        "HardHit%":_metric_col(cols,["hardhit%","hard hit %","hard_hit_percent"],["hard hit","hardhit"]),
        "RV100":_metric_col(cols,["rv100","rv/100","run value / 100 pitches"],["rv 100","run value 100"])
    }
    rv_col=col("run","value")

    rows=[]
    for pitch,g in df.groupby(pitch_col,dropna=True):
        wp=pd.to_numeric(g[pitches_col],errors="coerce").fillna(0)
        wpa=pd.to_numeric(g[pa_col],errors="coerce").fillna(0) if pa_col else wp
        def weighted(c,w):
            if not c:return None
            vals=pd.to_numeric(g[c],errors="coerce")
            good=vals.notna() & (w>0)
            if not good.any():return None
            return (vals[good]*w[good]).sum()/w[good].sum()
        row={"Pitch":str(pitch),"Pitches":wp.sum(),"PA":wpa.sum() if pa_col else None}
        for label,cname in mapping.items():
            row[label]=weighted(cname, wp if label in ("Whiff%","PutAway%","RV100") else wpa)
        row["Run Value"]=pd.to_numeric(g[rv_col],errors="coerce").sum() if rv_col else None
        rows.append(row)
    out=pd.DataFrame(rows)
    return (out.sort_values("Pitches",ascending=False),"SAVANT_PITCH_ARSENAL") if not out.empty else (out,"EMPTY_AGG")

# ============================================================
# STATCAST ADVANCED METRICS
# ============================================================

@st.cache_data(ttl=3600, show_spinner=False)
def pitcher_statcast(player_id: int, start_dt: str, end_dt: str):
    try:
        df = statcast_pitcher(start_dt, end_dt, player_id)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def team_statcast(team_abbr: str, start_dt: str, end_dt: str):
    try:
        df = statcast(start_dt, end_dt, team=team_abbr, verbose=False, parallel=True)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def offensive_team_rows(df: pd.DataFrame, team_abbr: str):
    if df.empty or not {"home_team","away_team","inning_topbot"}.issubset(df.columns):
        return pd.DataFrame()
    home_offense = df["home_team"].astype(str).eq(team_abbr) & df["inning_topbot"].astype(str).str.lower().eq("bot")
    away_offense = df["away_team"].astype(str).eq(team_abbr) & df["inning_topbot"].astype(str).str.lower().eq("top")
    return df[home_offense | away_offense].copy()


def plate_discipline(df: pd.DataFrame):
    result = {
        "Pitches": None, "SwStr%": None, "Whiff%": None, "CSW%": None,
        "Contact%": None, "O-Contact%": None, "Z-Contact%": None,
        "Chase%": None, "O-Swing%": None, "Swing%": None, "Zone%": None,
        "Ball%": None,
    }
    if df.empty or "description" not in df.columns:
        return result

    d = df.copy()
    desc = d["description"].astype(str)
    swings = desc.isin(SWING_DESCRIPTIONS)
    whiffs = desc.isin(WHIFF_DESCRIPTIONS)
    contacts = desc.isin(CONTACT_DESCRIPTIONS)
    called = desc.isin(CALLED_STRIKE_DESCRIPTIONS)
    total = len(d)

    if "zone" in d.columns:
        zone_num = pd.to_numeric(d["zone"], errors="coerce")
        in_zone = zone_num.between(1,9)
        out_zone = zone_num.notna() & ~in_zone
    else:
        in_zone = pd.Series(False, index=d.index)
        out_zone = pd.Series(False, index=d.index)

    swing_n = swings.sum()
    whiff_n = whiffs.sum()
    contact_n = contacts.sum()
    out_swings = (swings & out_zone).sum()
    zone_swings = (swings & in_zone).sum()

    result["Pitches"] = total
    result["SwStr%"] = whiff_n/total*100 if total else None
    result["Whiff%"] = whiff_n/swing_n*100 if swing_n else None
    result["CSW%"] = (whiff_n + called.sum())/total*100 if total else None
    result["Contact%"] = contact_n/swing_n*100 if swing_n else None
    result["O-Contact%"] = (contacts & out_zone).sum()/out_swings*100 if out_swings else None
    result["Z-Contact%"] = (contacts & in_zone).sum()/zone_swings*100 if zone_swings else None
    result["O-Swing%"] = out_swings/out_zone.sum()*100 if out_zone.sum() else None
    result["Chase%"] = result["O-Swing%"]
    result["Swing%"] = swing_n/total*100 if total else None
    result["Zone%"] = in_zone.sum()/(in_zone.sum()+out_zone.sum())*100 if (in_zone.sum()+out_zone.sum()) else None

    ball_like = desc.isin({"ball","blocked_ball","pitchout"})
    result["Ball%"] = ball_like.sum()/total*100 if total else None
    return result


def pa_rates(df: pd.DataFrame):
    out = {"PA": None, "K%": None, "BB%": None, "K-BB%": None}
    if df.empty or "events" not in df.columns:
        return out
    pa = df[df["events"].notna()].copy()
    if pa.empty:
        return out
    e = pa["events"].astype(str)
    k = e.str.contains("strikeout", case=False, na=False).sum()
    bb = e.isin(["walk","intent_walk"]).sum()
    n = len(pa)
    out["PA"] = n
    out["K%"] = k/n*100
    out["BB%"] = bb/n*100
    out["K-BB%"] = out["K%"] - out["BB%"]
    return out


def arsenal_table(df: pd.DataFrame):
    if df.empty or "pitch_type" not in df.columns:
        return pd.DataFrame()

    rows = []
    for pitch, g in df.groupby("pitch_type"):
        disc = plate_discipline(g)
        pa = g[g["events"].notna()] if "events" in g.columns else pd.DataFrame()

        pa_n = len(pa)
        k_n = (
            pa["events"].astype(str).str.contains("strikeout",case=False,na=False).sum()
            if pa_n else 0
        )
        k_pct = k_n/pa_n*100 if pa_n else None

        # PutAway: two-strike pitches that end in a strikeout / all two-strike pitches.
        strikes = pd.to_numeric(g["strikes"], errors="coerce") if "strikes" in g.columns else pd.Series(index=g.index,dtype=float)
        two_strike = strikes.eq(2)
        k_event = (
            g["events"].astype(str).str.contains("strikeout",case=False,na=False)
            if "events" in g.columns else pd.Series(False,index=g.index)
        )
        putaway = (two_strike & k_event).sum()/two_strike.sum()*100 if two_strike.sum() else None

        velo = pd.to_numeric(g["release_speed"],errors="coerce").mean() if "release_speed" in g.columns else None
        spin = pd.to_numeric(g["release_spin_rate"],errors="coerce").mean() if "release_spin_rate" in g.columns else None
        mov_x = pd.to_numeric(g["pfx_x"],errors="coerce").mean() if "pfx_x" in g.columns else None
        mov_z = pd.to_numeric(g["pfx_z"],errors="coerce").mean() if "pfx_z" in g.columns else None
        xwoba = pd.to_numeric(g["estimated_woba_using_speedangle"],errors="coerce").mean() if "estimated_woba_using_speedangle" in g.columns else None
        drv = pd.to_numeric(g["delta_run_exp"],errors="coerce").sum() if "delta_run_exp" in g.columns else None
        # Pitcher-positive run value: negative batter run expectancy change becomes positive.
        run_value = -drv if drv is not None and pd.notna(drv) else None

        rows.append({
            "Pitch": PITCH_NAMES.get(str(pitch), str(pitch)),
            "Code": str(pitch),
            "Pitches": len(g),
            "Usage%": len(g)/len(df)*100,
            "Velo": velo,
            "Spin": spin,
            "Whiff%": disc["Whiff%"],
            "K%": k_pct,
            "PutAway%": putaway,
            "xwOBA": xwoba,
            "Run Value": run_value,
            "pfx_x": mov_x,
            "pfx_z": mov_z,
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Usage%",ascending=False) if not result.empty else result


def opponent_pitch_type_table(team_offense: pd.DataFrame):
    if team_offense.empty or "pitch_type" not in team_offense.columns:
        return pd.DataFrame()

    rows = []
    for pitch, g in team_offense.groupby("pitch_type"):
        disc = plate_discipline(g)
        pa = g[g["events"].notna()] if "events" in g.columns else pd.DataFrame()
        pa_rates_row = pa_rates(g)

        ab_events = {"single","double","triple","home_run","field_out","force_out","grounded_into_double_play","field_error","fielders_choice","fielders_choice_out"}
        ab = pa[pa["events"].astype(str).isin(ab_events)] if not pa.empty else pd.DataFrame()
        hits = pa["events"].astype(str).isin(["single","double","triple","home_run"]).sum() if not pa.empty else 0
        total_bases = 0
        if not pa.empty:
            ev = pa["events"].astype(str)
            total_bases = ev.eq("single").sum()+2*ev.eq("double").sum()+3*ev.eq("triple").sum()+4*ev.eq("home_run").sum()

        ba = hits/len(ab) if len(ab) else None
        slg = total_bases/len(ab) if len(ab) else None

        woba_val = pd.to_numeric(g["woba_value"],errors="coerce") if "woba_value" in g.columns else pd.Series(dtype=float)
        woba_den = pd.to_numeric(g["woba_denom"],errors="coerce") if "woba_denom" in g.columns else pd.Series(dtype=float)
        woba = woba_val.sum()/woba_den.sum() if len(woba_den) and woba_den.sum() else None

        xba = pd.to_numeric(g["estimated_ba_using_speedangle"],errors="coerce").mean() if "estimated_ba_using_speedangle" in g.columns else None
        xwoba = pd.to_numeric(g["estimated_woba_using_speedangle"],errors="coerce").mean() if "estimated_woba_using_speedangle" in g.columns else None
        xslg = pd.to_numeric(g["estimated_slg_using_speedangle"],errors="coerce").mean() if "estimated_slg_using_speedangle" in g.columns else None

        launch = pd.to_numeric(g["launch_speed"],errors="coerce") if "launch_speed" in g.columns else pd.Series(dtype=float)
        hardhit = (launch >= 95).sum()/launch.notna().sum()*100 if launch.notna().sum() else None

        drv = pd.to_numeric(g["delta_run_exp"],errors="coerce").sum() if "delta_run_exp" in g.columns else None
        rv100 = drv/len(g)*100 if drv is not None and len(g) else None

        rows.append({
            "Pitch": PITCH_NAMES.get(str(pitch),str(pitch)),
            "Code": str(pitch),
            "Pitches": len(g),
            "PA": pa_rates_row["PA"],
            "BA": ba,
            "SLG": slg,
            "wOBA": woba,
            "Whiff%": disc["Whiff%"],
            "K%": pa_rates_row["K%"],
            "PutAway%": None,
            "xBA": xba,
            "xSLG": xslg,
            "xwOBA": xwoba,
            "HardHit%": hardhit,
            "RV100": rv100,
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Pitches",ascending=False) if not result.empty else result




PITCH_CANON = {
    "ff":"ff","4 seam":"ff","4 seam fastball":"ff","four seam":"ff","four seam fastball":"ff",
    "si":"si","sinker":"si",
    "sl":"sl","slider":"sl",
    "st":"st","sweeper":"st",
    "fc":"fc","cutter":"fc",
    "ch":"ch","change":"ch","changeup":"ch",
    "cu":"cu","curve":"cu","curveball":"cu",
    "kc":"kc","knuckle curve":"kc","knuckle curveball":"kc",
    "fs":"fs","splitter":"fs","split finger":"fs","split fingered":"fs",
    "sv":"sv","slurve":"sv"
}
def canonical_pitch(v):
    return PITCH_CANON.get(normalize_name(v), normalize_name(v))

def merge_pitch_type_fallback(primary: pd.DataFrame, fallback: pd.DataFrame):
    """
    Fill missing Savant aggregate fields from cutoff-safe raw Statcast.
    Pitch code/name matching is normalized so FF/4-Seam, ST/Sweeper, etc. join correctly.
    """
    if primary is None or primary.empty:
        return fallback.copy() if isinstance(fallback,pd.DataFrame) else pd.DataFrame()
    if fallback is None or fallback.empty:
        return primary.copy()

    p=primary.copy()
    f=fallback.copy()

    def canon(v):
        return canonical_pitch(v)

    p["_join"]=p["Code"].map(canon) if "Code" in p.columns else p["Pitch"].map(canon)
    f["_join"]=f["Code"].map(canon) if "Code" in f.columns else f["Pitch"].map(canon)

    desired=["Pitches","PA","BA","SLG","wOBA","Whiff%","K%","PutAway%","xBA","xSLG","xwOBA","HardHit%","RV100","Run Value"]
    fmap={r["_join"]:r for _,r in f.iterrows()}
    for i,row in p.iterrows():
        fr=fmap.get(row["_join"])
        if fr is None:
            continue
        for col in desired:
            if col not in p.columns:
                p[col]=None
            cur=p.at[i,col]
            missing = cur is None
            if not missing:
                try: missing = pd.isna(cur)
                except Exception: pass
            if missing and col in fr.index:
                val=fr.get(col)
                if val is not None:
                    try:
                        if pd.isna(val): continue
                    except Exception: pass
                    p.at[i,col]=val
    return p.drop(columns=["_join"],errors="ignore")


def split_metrics(df: pd.DataFrame, side: str):
    if df.empty or "stand" not in df.columns:
        return {}
    sub = df[df["stand"].astype(str).eq(side)]
    return {**pa_rates(sub), **plate_discipline(sub)}


def statcast_recent_games(df: pd.DataFrame, n=10):
    if df.empty or "game_date" not in df.columns:
        return pd.DataFrame()

    rows = []
    dates = pd.to_datetime(df["game_date"], errors="coerce")
    d = df.assign(_date=dates)
    unique_dates = sorted(d["_date"].dropna().dt.date.unique(), reverse=True)[:n]

    for gd in unique_dates:
        g = d[d["_date"].dt.date.eq(gd)]
        pa = pa_rates(g)
        disc = plate_discipline(g)
        velo = pd.to_numeric(g["release_speed"],errors="coerce").mean() if "release_speed" in g.columns else None
        spin = pd.to_numeric(g["release_spin_rate"],errors="coerce").mean() if "release_spin_rate" in g.columns else None
        px = pd.to_numeric(g["pfx_x"],errors="coerce").mean() if "pfx_x" in g.columns else None
        pz = pd.to_numeric(g["pfx_z"],errors="coerce").mean() if "pfx_z" in g.columns else None
        usage = g["pitch_type"].value_counts(normalize=True).head(3).mul(100).to_dict() if "pitch_type" in g.columns else {}

        rows.append({
            "Date": str(gd),
            "K%": pa["K%"],
            "K-BB%": pa["K-BB%"],
            "Whiff%": disc["Whiff%"],
            "CSW%": disc["CSW%"],
            "Ball%": disc["Ball%"],
            "Velo": velo,
            "Spin": spin,
            "pfx_x": px,
            "pfx_z": pz,
            "Top Usage": ", ".join(f"{k} {v:.0f}%" for k,v in usage.items()),
        })
    return pd.DataFrame(rows)


# ============================================================
# PARK FACTOR
# ============================================================

@st.cache_data(ttl=86400, show_spinner=False)
def savant_park_factors(year: int):
    params = {"type":"year","year":year,"condition":"All","parks":"mlb","rolling":3}
    try:
        r = requests.get(SAVANT_PARK_URL,params=params,headers={"User-Agent":"Mozilla/5.0"},timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        for t in tables:
            if isinstance(t.columns,pd.MultiIndex):
                t.columns = [str(c[-1]).strip() for c in t.columns]
            else:
                t.columns = [str(c).strip() for c in t.columns]
            if "Venue" in t.columns and "SO" in t.columns:
                return t
    except Exception:
        pass
    return pd.DataFrame()


SAVANT_SO_VERIFIED_CACHE = {(2026,"wrigley field"):102.0}


def park_so_factor(venue, year):
    target = normalize_name(venue)
    aliases={
        "comerica park":["comerica park","comerica"],
        "wrigley field":["wrigley field","wrigley"],
        "oracle park":["oracle park","oracle"],
        "rogers centre":["rogers centre","rogers center"],
        "rogers center":["rogers centre","rogers center"],
    }
    targets=set(aliases.get(target,[target]))
    targets.add(target)

    df = savant_park_factors(year)
    if not df.empty:
        venue_col=next((c for c in df.columns if normalize_name(c) in ("venue","park","stadium","name")),None)
        so_col=next((c for c in df.columns if normalize_name(c) in ("so","strikeout","strikeouts","k")),None)
        if venue_col and so_col:
            vn=df[venue_col].astype(str).map(normalize_name)
            mask=vn.apply(lambda x:any(t in x or x in t for t in targets if t))
            hit=df[mask]
            if not hit.empty:
                x=safe_num(hit.iloc[0].get(so_col))
                if x is not None:
                    return x,"Baseball Savant live"

    cached = SAVANT_SO_VERIFIED_CACHE.get((year,target))
    if cached is not None:
        return cached, "Savant verified cache"
    return None, "SOURCE RETRY NEEDED"


# ============================================================
# FANGRAPHS / BASEBALL REFERENCE - VALIDATION ONLY
# ============================================================

@st.cache_data(ttl=21600,show_spinner=False)
def fangraphs_pitchers(season):
    try:
        try:
            df = pitching_stats(season,season,qual=0)
        except TypeError:
            df = pitching_stats(season,season)
        return (df,"OK") if isinstance(df,pd.DataFrame) and not df.empty else (pd.DataFrame(),"EMPTY")
    except Exception as exc:
        return pd.DataFrame(), ("BLOCKED_403" if "403" in str(exc) else f"ERROR: {str(exc)[:70]}")


@st.cache_data(ttl=21600,show_spinner=False)
def bref_pitchers(season):
    try:
        return pitching_stats_bref(season)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400,show_spinner=False)
def player_crosswalk(name):
    parts = str(name).split()
    if len(parts)<2:
        return {}
    try:
        df = playerid_lookup(parts[-1]," ".join(parts[:-1]),fuzzy=True)
        return df.iloc[0].to_dict() if isinstance(df,pd.DataFrame) and not df.empty else {}
    except Exception:
        return {}


def match_pitcher_row(df,name,mlbam_id=None,crosswalk=None):
    if df.empty:
        return {}
    crosswalk = crosswalk or {}
    for col in ("mlbID","MLBAMID","mlb_ID","key_mlbam"):
        if col in df.columns and mlbam_id is not None:
            ids = pd.to_numeric(df[col],errors="coerce")
            hit = df[ids.eq(int(mlbam_id))]
            if not hit.empty:
                return hit.iloc[0].to_dict()
    fgid = safe_num(crosswalk.get("key_fangraphs"))
    if fgid is not None:
        for col in ("IDfg","key_fangraphs"):
            if col in df.columns:
                ids = pd.to_numeric(df[col],errors="coerce")
                hit = df[ids.eq(int(fgid))]
                if not hit.empty:
                    return hit.iloc[0].to_dict()
    name_col = next((c for c in ("Name","NameASCII","name_common") if c in df.columns),None)
    if name_col:
        hit = df[df[name_col].astype(str).map(normalize_name).eq(normalize_name(name))]
        if not hit.empty:
            return hit.iloc[0].to_dict()
    return {}



@st.cache_data(ttl=21600,show_spinner=False)
def mlb_people_info(player_ids):
    ids=[str(int(x)) for x in player_ids if safe_num(x) is not None]
    if not ids:return {}
    out={}
    # MLB Stats API supports comma-separated personIds in hydrate-style endpoint poorly,
    # so batch individual lightweight calls; cache prevents repeated network cost.
    for pid in ids:
        try:
            js=get_json(f"https://statsapi.mlb.com/api/v1/people/{pid}")
            person=(js.get("people") or [{}])[0]
            out[int(pid)]={
                "Bats":((person.get("batSide") or {}).get("code") or (person.get("batSide") or {}).get("description")),
                "FullName":person.get("fullName")
            }
        except Exception:
            out[int(pid)]={"Bats":None,"FullName":None}
    return out

# ============================================================
# LINEUP
# ============================================================

def hitter_k_profile(pid,season,end_date,pitcher_hand):
    sit = "vl" if pitcher_hand.lower().startswith("left") else "vr"
    try:
        stat = person_stats_to_date(pid,season,"hitting",end_date,sit)
        source = "MLB split"
    except Exception:
        stat,source = {},"N/A"
    if not stat:
        try:
            stat = person_stats_to_date(pid,season,"hitting",end_date)
            source = "MLB overall"
        except Exception:
            stat = {}
    so,pa = safe_num(stat.get("strikeOuts")),safe_num(stat.get("plateAppearances"))
    return {
        "K% vs hand": so/pa*100 if so is not None and pa else None,
        "PA": pa, "Contact%": None, "Whiff%": None, "Source": source if stat else "N/A",
    }


def enrich_lineup(lineup,season,end_date,pitcher_hand):
    return [{**r,**hitter_k_profile(r["player_id"],season,end_date,pitcher_hand)} for r in lineup]



def enrich_lineup_rows(lineup_rows, season, cutoff_str):
    if not lineup_rows:return lineup_rows
    ids=[r.get("PlayerID") or r.get("player_id") or r.get("id") for r in lineup_rows]
    people=mlb_people_info(ids)
    for r in lineup_rows:
        pid=safe_num(r.get("PlayerID") or r.get("player_id") or r.get("id"))
        if pid is not None:
            pinfo=people.get(int(pid),{})
            if not r.get("Bats") or str(r.get("Bats")).upper() in ("N/A","NONE",""):
                r["Bats"]=pinfo.get("Bats")
        # Never expose raw None in the UI.
        for c in ("Contact%","Whiff%"):
            if r.get(c) is None:
                r[c]="—"
    return lineup_rows

def lineup_quality(lineup_rows):
    if len(lineup_rows)<9:
        return False,["lineup no confirmado"]
    issues=[]
    if sum(1 for r in lineup_rows if str(r.get("Bats") or "").upper() in ("L","R","S"))<9:
        issues.append("handedness incompleto")
    k_valid=sum(1 for r in lineup_rows if safe_num(r.get("K% vs hand")) is not None)
    if k_valid<7:issues.append("K% vs hand insuficiente")
    return len(issues)==0,issues

def lineup_k_pct(lineup):
    vals = [safe_num(r.get("K% vs hand")) for r in lineup]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None


# ============================================================
# VOLUME / RECENT
# ============================================================

def recent_summary(log):
    rows = log[:10]
    if not rows:
        return {}
    def vals(key):
        out = [safe_num(r.get(key)) for r in rows]
        return [x for x in out if x is not None]
    p = vals("Pitches"); bf = vals("BF"); k = vals("K")
    ips = [innings_decimal(r.get("IP")) for r in rows]
    ips = [x for x in ips if x is not None]

    return {
        "avg_pitches": sum(p)/len(p) if p else None,
        "max_pitches": max(p) if p else None,
        "avg_bf": sum(bf)/len(bf) if bf else None,
        "avg_k": sum(k)/len(k) if k else None,
        "avg_ip": sum(ips)/len(ips) if ips else None,
        "freq_90": sum(x>=90 for x in p)/len(p)*100 if p else None,
        "freq_100": sum(x>=100 for x in p)/len(p)*100 if p else None,
        "pitches_per_bf": (sum(p)/sum(bf)) if p and bf and sum(bf) else None,
    }


def days_rest(log, selected_date):
    if not log:
        return None
    try:
        prev = pd.Timestamp(log[0]["Date"]).date()
        return (selected_date-prev).days
    except Exception:
        return None



def normalize_wind_direction(text):
    t=normalize_name(text)
    if not t:return None
    if "out" in t:
        if "left" in t or "lf" in t:return "OUT TO LF"
        if "right" in t or "rf" in t:return "OUT TO RF"
        if "center" in t or "cf" in t:return "OUT TO CF"
        return "OUT"
    if "in" in t:
        if "left" in t or "lf" in t:return "IN FROM LF"
        if "right" in t or "rf" in t:return "IN FROM RF"
        if "center" in t or "cf" in t:return "IN FROM CF"
        return "IN"
    if "left to right" in t:return "LEFT → RIGHT"
    if "right to left" in t:return "RIGHT → LEFT"
    return str(text)

# ============================================================
# ACTION NETWORK PRO VALIDATION
# ============================================================

@st.cache_data(ttl=300,show_spinner=False)
def action_public_status():
    try:
        r = requests.get(ACTION_PITCHING_PROPS_URL,headers={"User-Agent":"Mozilla/5.0"},timeout=TIMEOUT)
        r.raise_for_status()
        soup = BeautifulSoup(r.text,"html.parser")
        text = soup.get_text(" ",strip=True)
        if "Pitching" in text or "MLB" in text:
            return "PAGE_OK"
        return "PAGE_REACHED"
    except Exception as exc:
        return f"UNAVAILABLE: {str(exc)[:70]}"



def market_hit(k_value,market,line):
    k=safe_num(k_value);ln=safe_num(line)
    if k is None or ln is None:return None
    m=str(market).lower().strip()
    if m.startswith("over"):return k>ln
    if m.startswith("under"):return k<ln
    if "alt" in m or "+" in m:return k>=int(round(ln))
    return None

def line_history_stats(log_rows,market,line,n):
    outs=[]
    for r in log_rows[:n]:
        h=market_hit(r.get("K"),market,line)
        if h is not None:outs.append(bool(h))
    return ((sum(outs)/len(outs)*100,len(outs)) if outs else (None,0))

def append_line_history(market_df,log_rows):
    if market_df.empty:return market_df
    out=market_df.copy();l5=[];l10=[]
    for _,r in out.iterrows():
        h5,_=line_history_stats(log_rows,r["Market"],r["Line"],5)
        h10,_=line_history_stats(log_rows,r["Market"],r["Line"],10)
        l5.append(h5);l10.append(h10)
    out["Hit L5%"]=l5;out["Hit L10%"]=l10
    return out

# ============================================================
# PROJECTION
# ============================================================

def build_projection(mlb,pdisc,team_general,team_split,lineup,recent,park_so):
    # Main weighted architecture:
    # M1 20, M2 20, M3 10, M4 20, M5 15, M6 5, M7 5, M8 5
    pitcher_k = safe_num(mlb.get("calc_k_pct"))
    opp_k = safe_num(team_split.get("calc_k_pct")) or safe_num(team_general.get("calc_k_pct"))
    lineup_k = lineup_k_pct(lineup)

    base_k = weighted_average([
        (pitcher_k,.50),
        (opp_k,.30),
        (lineup_k,.20),
    ],default=pitcher_k or opp_k or 22.0)

    whiff = safe_num(pdisc.get("Whiff%"))
    csw = safe_num(pdisc.get("CSW%"))
    contact = safe_num(pdisc.get("Contact%"))
    chase = safe_num(pdisc.get("Chase%"))

    ability_adj = 0.0
    if whiff is not None: ability_adj += clamp((whiff-24)*.06,-.8,.8)
    if csw is not None: ability_adj += clamp((csw-27)*.08,-.8,.8)
    if contact is not None: ability_adj += clamp((76-contact)*.04,-.5,.5)
    if chase is not None: ability_adj += clamp((chase-29)*.03,-.35,.35)

    bf = weighted_average([
        (mlb.get("calc_bf_start"),.60),
        (recent.get("avg_bf"),.40),
    ],default=22.0)
    bf = clamp(bf,14,32)

    season_k_start = safe_num(mlb.get("calc_k_start"))
    recent_k = safe_num(recent.get("avg_k"))
    form_adj = clamp((recent_k-season_k_start)*.22,-.6,.6) if recent_k is not None and season_k_start is not None else 0

    park_adj = clamp(((park_so or 100)-100)*.05,-.8,.8)

    projected_k_pct = clamp(base_k+ability_adj+park_adj,8,45)
    central = clamp(bf*projected_k_pct/100 + form_adj,.5,14)
    sigma = max(1.35,math.sqrt(central)*.80)

    return {
        "central":central,
        "low":max(0,central-1.35*sigma),
        "high":central+1.35*sigma,
        "bf":bf,
        "k_pct":projected_k_pct,
        "base_k":base_k,
        "ability_adj":ability_adj,
        "form_adj":form_adj,
        "park_adj":park_adj,
    }


def grade_bet(ev,edge,modules):
    if ev is None or edge is None:
        return "NO BET"
    if modules <= 6:
        return "PASS"
    evp = ev*100
    raw = "PASS"
    if evp>=10 and edge>=6: raw="A"
    elif evp>=6 and edge>=4: raw="B"
    elif evp>=3 and edge>=2: raw="C"
    if modules<=7 and raw in ("A","B"): return "C"
    if modules==8 and raw=="A": return "B"
    return raw


# ============================================================
# TECHNICAL ANALYSIS
# ============================================================

def direction(value, neutral, lower_good=False, band=1.0):
    x = safe_num(value)
    if x is None:
        return "N/A"
    delta = neutral-x if lower_good else x-neutral
    if delta > band: return "POSITIVO"
    if delta < -band: return "NEGATIVO"
    return "NEUTRAL"



def clean_display_frame(df, missing="—"):
    """Display-only cleanup: replace NaN/None/inf without changing model calculations."""
    if df is None:
        return pd.DataFrame()
    out = df.copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA)
    return out.fillna(missing)

def technical_analysis(mlb,pdisc,split_l,split_r,opp_disc,team_general,team_split,arsenal,opp_pitch,recent,park_so,lineup,proj,manual,log):
    items=[]
    k_pct=safe_num(mlb.get("calc_k_pct"));k9=safe_num(mlb.get("strikeoutsPer9Inn"))
    whiff=safe_num(pdisc.get("Whiff%"));swstr=safe_num(pdisc.get("SwStr%"));csw=safe_num(pdisc.get("CSW%"))
    contact=safe_num(pdisc.get("Contact%"));kbb=safe_num(mlb.get("calc_k_minus_bb"))
    s1=direction(k_pct,22.5,False,1.5)
    t1=(f"El perfil base es {fmt(k_pct,1,'%')} K%, {fmt(k9,2)} K/9 y {fmt(kbb,1,'%')} K-BB%. "
        f"Los indicadores que respaldan ese resultado muestran {fmt(whiff,1,'%')} Whiff%, {fmt(swstr,1,'%')} SwStr%, "
        f"{fmt(csw,1,'%')} CSW% y {fmt(contact,1,'%')} Contact%. ")
    t1+=("La combinación respalda capacidad real de swing-and-miss y eleva el techo de K." if s1=="POSITIVO"
         else "El swing-and-miss no respalda un perfil dominante, así que M1 limita el techo de K." if s1=="NEGATIVO"
         else "El conjunto es cercano a neutral y no justifica mover agresivamente la expectativa.")
    items.append(("M1 · Capacidad real de K",s1,t1))

    ap=safe_num(recent.get("avg_pitches"));abf=safe_num(recent.get("avg_bf"));aip=safe_num(recent.get("avg_ip"))
    ak=safe_num(recent.get("avg_k"));f90=safe_num(recent.get("freq_90"));f100=safe_num(recent.get("freq_100"))
    s2="POSITIVO" if ap is not None and ap>=90 else "NEUTRAL"
    if manual.get("pitch_limit") or manual.get("injury_return") or manual.get("opener"):s2="NEGATIVO"
    t2=(f"En sus aperturas recientes promedia {fmt(ap,1)} pitcheos, {fmt(aip,2)} IP y {fmt(abf,1)} BF; "
        f"{fmt(f90,0,'%')} llegaron a 90+ y {fmt(f100,0,'%')} a 100+. El modelo proyecta {proj['bf']:.1f} BF. ")
    t2+=("Ese leash permite volumen suficiente para acumular strikeouts." if s2=="POSITIVO"
         else "Una restricción de rol/leash reduce oportunidades y obliga a recortar el techo." if s2=="NEGATIVO"
         else "El volumen es moderado; necesita eficiencia para alcanzar un techo alto.")
    items.append(("M2 · Volumen / Leash",s2,t2))

    l=safe_num(split_l.get("K%"));r=safe_num(split_r.get("K%"));lw=safe_num(split_l.get("Whiff%"));rw=safe_num(split_r.get("Whiff%"))
    gap=abs(l-r) if l is not None and r is not None else None
    s3="MATCHUP-DEPENDENT" if gap is not None and gap>=5 else "NEUTRAL"
    t3=(f"Registra {fmt(l,1,'%')} K% vs LHB y {fmt(r,1,'%')} vs RHB, con Whiff% de {fmt(lw,1,'%')} y {fmt(rw,1,'%')}. ")
    if s3=="MATCHUP-DEPENDENT":
        better="zurdos" if (l or 0)>(r or 0) else "derechos"
        t3+=f"La brecha es importante: cuantos más bateadores {better} aparezcan en el lineup, mejor será el matchup de K."
    else:t3+="La diferencia por handedness es limitada, por lo que la composición L/R/S mueve menos la proyección."
    items.append(("M3 · Splits",s3,t3))

    ok=safe_num(team_split.get("calc_k_pct")) or safe_num(team_general.get("calc_k_pct"))
    ow=safe_num(opp_disc.get("Whiff%"));oc=safe_num(opp_disc.get("Contact%"));och=safe_num(opp_disc.get("Chase%"));oz=safe_num(opp_disc.get("Zone%"))
    s4=direction(ok,22.0,False,1.2)
    t4=(f"El rival tiene {fmt(ok,1,'%')} K% contra la mano del abridor. Savant añade {fmt(ow,1,'%')} Whiff%, "
        f"{fmt(oc,1,'%')} Contact%, {fmt(och,1,'%')} Chase% y {fmt(oz,1,'%')} Zone%. ")
    t4+=("La ofensiva ofrece más oportunidades de K que una referencia MLB cercana al 22%." if s4=="POSITIVO"
         else "La ofensiva protege el plato mejor que el promedio y reduce oportunidades de K." if s4=="NEGATIVO"
         else "El rival es cercano al entorno MLB; aquí el arsenal específico gana importancia.")
    items.append(("M4 · Rival / disciplina",s4,t4))

    s5="PENDIENTE";t5=""
    if isinstance(arsenal,pd.DataFrame) and not arsenal.empty:
        top=arsenal.head(3);pieces=[];matches=[];vals=[]
        for _,rr in top.iterrows():
            pn=str(rr.get("Pitch"));pieces.append(f"{pn} {fmt(rr.get('Usage%'),1,'%')} uso / {fmt(rr.get('Whiff%'),1,'%')} Whiff")
            if isinstance(opp_pitch,pd.DataFrame) and not opp_pitch.empty:
                aliases={"4 seam":"ff","4 seam fastball":"ff","sweeper":"st","cutter":"fc","sinker":"si","changeup":"ch","slider":"sl","curveball":"cu","splitter":"fs"}
                pkey=aliases.get(normalize_name(pn),normalize_name(pn))
                keys=opp_pitch["Code"].astype(str).str.lower() if "Code" in opp_pitch.columns else opp_pitch["Pitch"].astype(str).map(lambda x: aliases.get(normalize_name(x),normalize_name(x)))
                hit=opp_pitch[keys.eq(pkey)]
                if not hit.empty:
                    rh=hit.iloc[0];matches.append(f"vs {pn}: {fmt(rh.get('Whiff%'),1,'%')} Whiff, {fmt(rh.get('K%'),1,'%')} K, {fmt(rh.get('xwOBA'),3)} xwOBA")
                    if safe_num(rh.get("Whiff%")) is not None:vals.append(safe_num(rh.get("Whiff%")))
        t5="Las tres armas principales son "+("; ".join(pieces))+". "
        if matches:
            t5+="El rival responde así: "+("; ".join(matches))+". "
            avg=sum(vals)/len(vals) if vals else None
            s5="POSITIVO" if avg is not None and avg>=27 else ("NEGATIVO" if avg is not None and avg<21 else "NEUTRAL")
            t5+=("Hay una coincidencia favorable entre las armas de mayor uso y las debilidades del rival." if s5=="POSITIVO"
                 else "El rival hace buen contacto contra las armas principales, lo que recorta el potencial de K." if s5=="NEGATIVO"
                 else "La coincidencia arsenal-rival es mixta y no produce un ajuste fuerte.")
        else:t5+="No hubo coincidencia suficiente entre nombres de pitcheos para interpretar el matchup; M5 queda pendiente."
    else:t5="No se pudo construir el arsenal del pitcher; M5 queda pendiente."
    items.append(("M5 · Arsenal vs Matchup",s5,t5))

    sk=safe_num(mlb.get("calc_k_start"));s6="NEUTRAL"
    if ak is not None and sk is not None:
        if ak>=sk+.7:s6="POSITIVO"
        elif ak<=sk-.7:s6="NEGATIVO"
    t6=(f"Las últimas aperturas promedian {fmt(ak,1)} K frente a {fmt(sk,1)} K/start de temporada. "
        "También se comparan Whiff%, CSW%, Ball%, velocidad, spin, movimiento y cambios de uso salida por salida. ")
    t6+=("La forma reciente está por encima de su línea base." if s6=="POSITIVO"
         else "La producción reciente está por debajo de su línea base." if s6=="NEGATIVO"
         else "No hay evidencia suficiente para declarar una tendencia nueva.")
    items.append(("M6 · Forma reciente",s6,t6))

    s7="PENDIENTE" if park_so is None else ("POSITIVO" if park_so>101 else ("NEGATIVO" if park_so<99 else "NEUTRAL"))
    t7=(f"El SO Park Factor es {fmt(park_so,0)} cuando está disponible, con 100 como neutral. "
        "Temperatura, velocidad y dirección del viento se registran como contexto; por ahora mantienen peso bajo. "
        "El umpire queda pendiente hasta confirmación de MLB.")
    items.append(("M7 · Contexto",s7,t7))

    lk=lineup_k_pct(lineup)
    if len(lineup)>=9:
        s8=direction(lk,22.0,False,1.2)
        hi=[r["Hitter"] for r in lineup if safe_num(r.get("K% vs hand")) is not None and r["K% vs hand"]>=27]
        lo=[r["Hitter"] for r in lineup if safe_num(r.get("K% vs hand")) is not None and r["K% vs hand"]<=17]
        t8=(f"El lineup confirmado tiene {fmt(lk,1,'%')} K% agregado vs la mano del pitcher. "
            f"High-K: {', '.join(hi) if hi else 'ninguno'}; low-K: {', '.join(lo) if lo else 'ninguno'}. "
            "Este bloque reemplaza la aproximación del equipo por los nueve bateadores reales.")
    else:
        s8="PENDIENTE";t8=("El lineup todavía no está confirmado. La proyección es PROVISIONAL porque los splits del pitcher "
                            "pueden cambiar materialmente según la composición L/R/S y los K% individuales de los nueve bateadores.")
    items.append(("M8 · Lineup confirmado",s8,t8))
    return items



def slate_games(options):
    grouped={}
    for opt in options:
        grouped.setdefault(opt.get("game_pk"),[]).append(opt)
    return list(grouped.values())

@st.cache_data(ttl=300,show_spinner=False)
def quick_pitcher_snapshot(player_id:int,season:int,cutoff_date:str):
    try:
        s=pitcher_stats_to_date(player_id,season,cutoff_date)
    except Exception:
        s={}
    return {"K%":safe_num(s.get("calc_k_pct")),"K/start":safe_num(s.get("calc_k_start")),"BF/start":safe_num(s.get("calc_bf_start"))}

def slate_status(option,selected_date):
    snap=quick_pitcher_snapshot(option["pitcher_id"],selected_date.year,game_cutoff(selected_date).isoformat())
    complete=all(snap.get(x) is not None for x in ("K%","K/start","BF/start"))
    return ("LISTO","dot-green",snap) if complete else ("PARCIAL","dot-gold",snap)

# ============================================================
# APP LOAD
# ============================================================


st.markdown(
    """
    <div class="hero">
      <div class="section-label">MODELO PROFESIONAL MLB · STARTING PITCHER STRIKEOUTS</div>
      <div style="font-size:2.05rem;font-weight:880;margin-top:3px">Starting Pitcher Strikeout Lab</div>
      <div style="opacity:.70;margin-top:6px">V3.1.3 LIVE TEST · Daily Board → Pitcher → Full Matchup Lab</div>
    </div>
    """, unsafe_allow_html=True
)

if "view_mode" not in st.session_state:
    st.session_state["view_mode"]="slate"
if "selected_pitcher_id" not in st.session_state:
    st.session_state["selected_pitcher_id"]=None

date_col,_=st.columns([1.15,2.0])
with date_col:
    game_date=st.date_input("Fecha",value=date.today(),min_value=date(2015,1,1),key="slate_date")

try:
    options=pitchers_for_date(game_date.isoformat())
except Exception as exc:
    st.error(f"No se pudo cargar MLB: {exc}")
    st.stop()
if not options:
    st.warning("No hay abridores probables disponibles para esta fecha.")
    st.stop()

by_id={x["selection_id"]:x for x in options}
if st.session_state.get("selected_pitcher_id") not in by_id:
    st.session_state["selected_pitcher_id"]=None
    st.session_state["view_mode"]="slate"

# Query-param navigation allows the daily board to be pure compact HTML.
qp_pitch = st.query_params.get("pitcher")
if qp_pitch and qp_pitch in by_id:
    st.session_state["selected_pitcher_id"]=qp_pitch
    st.session_state["view_mode"]="analysis"

if st.session_state["view_mode"]=="slate":
    games=slate_games(options)
    st.markdown(
        f"""
        <div class="slate-header">
          <div>
            <div class="section-label">DAILY PITCHER BOARD</div>
            <div style="font-size:1.38rem;font-weight:850">{game_date.strftime('%A · %B %d').upper()}</div>
          </div>
          <div class="slate-count">{len(games)} GAMES · {len(options)} STARTERS</div>
        </div>
        """, unsafe_allow_html=True
    )

    cards=[]
    for game_opts in games:
        first=game_opts[0]
        by_side={x.get("team_side"):x for x in game_opts}
        away=by_side.get("away",game_opts[0] if game_opts else None)
        home=by_side.get("home",game_opts[1] if len(game_opts)>1 else None)
        if away is None or home is None:
            continue

        away_status,_,away_snap=slate_status(away,game_date)
        home_status,_,home_snap=slate_status(home,game_date)
        away_logo=f"https://www.mlbstatic.com/team-logos/{away['team_id']}.svg"
        home_logo=f"https://www.mlbstatic.com/team-logos/{home['team_id']}.svg"

        def abbr(name):
            words=str(name).replace("D-backs","Diamondbacks").split()
            return "".join(w[0] for w in words[-2:]).upper() if len(words)>=2 else str(name)[:3].upper()

        def pshort(opt):
            parts=opt["pitcher_name"].split()
            return (parts[0][0]+". "+parts[-1]) if len(parts)>1 else opt["pitcher_name"]

        cards.append(
            f'<div class="board-game">'
            f'<div class="board-time">{first.get("game_time","TBD")} · {first.get("venue","")}</div>'
            f'<div class="board-team">'
            f'<img class="board-logo" src="{away_logo}">'
            f'<div><div class="board-abbr">{abbr(away["team"])}</div><div class="board-pitcher">{pshort(away)} ({away["throwing_hand"][:1]})</div></div>'
            f'<div class="board-k">{fmt(away_snap.get("K%"),1,"%")}</div>'
            f'</div>'
            f'<div class="board-team">'
            f'<img class="board-logo" src="{home_logo}">'
            f'<div><div class="board-abbr">{abbr(home["team"])}</div><div class="board-pitcher">{pshort(home)} ({home["throwing_hand"][:1]})</div></div>'
            f'<div class="board-k">{fmt(home_snap.get("K%"),1,"%")}</div>'
            f'</div>'
            f'<div class="board-actions">'
            f'<a class="board-link" href="?pitcher={away["selection_id"]}">ANALIZAR {abbr(away["team"])}</a>'
            f'<a class="board-link" href="?pitcher={home["selection_id"]}">ANALIZAR {abbr(home["team"])}</a>'
            f'</div></div>'
        )

    board_html='<div class="board-grid">'+''.join(cards)+'</div>'
    st.markdown(board_html,unsafe_allow_html=True)
    st.markdown('<div class="board-legend">K% rápido = perfil base pregame. El board NO emite apuestas; abre un pitcher para M1–M9.</div>',unsafe_allow_html=True)
    st.stop()

selected_id=st.session_state["selected_pitcher_id"]
p=by_id[selected_id]
back_col,title_col=st.columns([.65,3.35])
with back_col:
    if st.button("← SLATE",use_container_width=True):
        st.session_state["view_mode"]="slate"
        st.query_params.clear()
        st.rerun()

cutoff=game_cutoff(game_date)
cutoff_str=cutoff.isoformat()
with title_col:
    st.markdown(
        f"""
        <div class="gamecard">
          <div class="section-label">MATCHUP LAB</div>
          <div style="font-size:1.55rem;font-weight:850;margin-top:3px">{p['pitcher_name']} <span style="opacity:.48">vs {p['opponent']}</span></div>
          <span class="pill">{p['team']}</span><span class="pill">{p['throwing_hand']}</span>
          <span class="pill">{p['venue']}</span><span class="pill">{p['game_time']}</span><span class="pill">{p['status']}</span>
          <div class="small-muted" style="margin-top:8px">Pregame cutoff: {cutoff_str} · el juego seleccionado nunca entra en sus propios inputs.</div>
        </div>
        """, unsafe_allow_html=True
    )

with st.spinner("Cargando y cruzando fuentes pregame..."):
    try: mlb = pitcher_stats_to_date(p["pitcher_id"],game_date.year,cutoff_str)
    except Exception: mlb={}
    try: log = pitcher_game_log_before(p["pitcher_id"],game_date.year,game_date.isoformat())
    except Exception: log=[]

    sit="vl" if p["throwing_hand"].lower().startswith("left") else "vr"
    try: team_general=team_hitting_to_date(p["opponent_id"],game_date.year,cutoff_str)
    except Exception: team_general={}
    try: team_split=team_hitting_to_date(p["opponent_id"],game_date.year,cutoff_str,sit)
    except Exception: team_split={}

    try: feed=game_feed(p["game_pk"])
    except Exception: feed={}
    context=game_context(feed)

    sc_start=f"{game_date.year}-03-01"
    sc_pitcher=pitcher_statcast(p["pitcher_id"],sc_start,cutoff_str)
    pdisc=plate_discipline(sc_pitcher)
    split_l=split_metrics(sc_pitcher,"L")
    split_r=split_metrics(sc_pitcher,"R")
    arsenal=arsenal_table(sc_pitcher)
    recent_sc=statcast_recent_games(sc_pitcher,10)

    try:
        opp_info=team_info(p["opponent_id"],game_date.year)
        opp_abbr=opp_info.get("abbreviation","")
    except Exception:
        opp_abbr=""

    opp_disc, opp_disc_source = savant_team_plate_discipline(p["opponent_id"], game_date.year)
    opp_pitch, opp_pitch_source = savant_team_pitch_type(p["opponent_id"], game_date.year)

    opp_sc_all = pd.DataFrame()
    opp_off = pd.DataFrame()
    try:
        # Raw Statcast is cutoff-safe and is also used to fill fields that the
        # Savant aggregate table omits (notably expected metrics).
        opp_sc_all = team_statcast(opp_abbr,sc_start,cutoff_str) if opp_abbr else pd.DataFrame()
        opp_off = offensive_team_rows(opp_sc_all,opp_abbr) if opp_abbr else pd.DataFrame()
        if not opp_disc and not opp_off.empty:
            opp_disc = plate_discipline(opp_off)
            opp_disc_source="RAW_STATCAST_FALLBACK"
        if not opp_off.empty:
            raw_pitch = opponent_pitch_type_table(opp_off)
            if opp_pitch.empty:
                opp_pitch = raw_pitch
                opp_pitch_source="RAW_STATCAST_FALLBACK"
            else:
                before_missing = int(opp_pitch.isna().sum().sum())
                opp_pitch = merge_pitch_type_fallback(opp_pitch, raw_pitch)
                after_missing = int(opp_pitch.isna().sum().sum())
                if after_missing < before_missing:
                    opp_pitch_source = f"{opp_pitch_source} + RAW_STATCAST_FILL"
    except Exception:
        pass

    park_so,park_source=park_so_factor(p["venue"],game_date.year)

    raw_lineup=confirmed_lineup(feed,p["opponent_side"])
    lineup=enrich_lineup(raw_lineup,game_date.year,cutoff_str,p["throwing_hand"]) if raw_lineup else []

    recent=recent_summary(log)
    rest_days=days_rest(log,game_date)

    fg_df,fg_status=fangraphs_pitchers(game_date.year)
    br_df=bref_pitchers(game_date.year)
    cross=player_crosswalk(p["pitcher_name"])
    fg=match_pitcher_row(fg_df,p["pitcher_name"],p["pitcher_id"],cross)
    br=match_pitcher_row(br_df,p["pitcher_name"],p["pitcher_id"],cross)

proj=build_projection(mlb,pdisc,team_general,team_split,lineup,recent,park_so)

# ============================================================
# TABS
# ============================================================

tab_summary,tab_modules,tab_analysis,tab_market,tab_sources=st.tabs(
    ["Resumen","M1–M8","Análisis","Mercado","Fuentes"]
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
with tab_summary:
    a,b,c,d=st.columns(4)
    a.metric("BF proyectados",fmt(proj["bf"],1))
    b.metric("K% proyectado",fmt(proj["k_pct"],1,"%"))
    c.metric("Strikeouts proyectados",fmt(proj["central"],2))
    d.metric("Rango probable",f"{proj['low']:.1f}–{proj['high']:.1f}")

    st.subheader("Probabilidad por umbral")
    dist=[]
    for k in range(3,11):
        prob=poisson_ge(k,proj["central"])
        dist.append({"Línea":f"{k}+","Probabilidad":prob*100,"Fair Odds":fair_american(prob)})
    dist_df=pd.DataFrame(dist)
    dist_df["Probabilidad"]=dist_df["Probabilidad"].round(1)
    dist_df["Fair Odds"]=dist_df["Fair Odds"].round(0)
    st.dataframe(dist_df,hide_index=True,use_container_width=True)

    st.info("Esta es la proyección estadística pregame. La selección de apuesta ocurre solamente después de comparar TODAS las líneas y precios disponibles.")

# ------------------------------------------------------------
# MODULES 1-8
# ------------------------------------------------------------
with tab_modules:
    with st.expander("M1 · Capacidad real de strikeout — 20%",expanded=True):
        a,b,c,d=st.columns(4)
        a.metric("K%",fmt(mlb.get("calc_k_pct"),1,"%"))
        a.metric("K/9",mlb.get("strikeoutsPer9Inn","N/A"))
        a.metric("K-BB%",fmt(mlb.get("calc_k_minus_bb"),1,"%"))
        a.metric("BB%",fmt(mlb.get("calc_bb_pct"),1,"%"))
        b.metric("SwStr%",fmt(pdisc.get("SwStr%"),1,"%"))
        b.metric("Whiff%",fmt(pdisc.get("Whiff%"),1,"%"))
        b.metric("CSW%",fmt(pdisc.get("CSW%"),1,"%"))
        b.metric("Contact%",fmt(pdisc.get("Contact%"),1,"%"))
        c.metric("O-Contact%",fmt(pdisc.get("O-Contact%"),1,"%"))
        c.metric("Z-Contact%",fmt(pdisc.get("Z-Contact%"),1,"%"))
        c.metric("Chase / O-Swing%",fmt(pdisc.get("O-Swing%"),1,"%"))
        c.metric("Zone%",fmt(pdisc.get("Zone%"),1,"%"))
        d.metric("IP",mlb.get("inningsPitched","N/A"))
        d.metric("BF",mlb.get("battersFaced","N/A"))
        d.metric("Fastball Velo",fmt(
            arsenal[arsenal["Code"].isin(["FF","SI","FC"])]["Velo"].dropna().iloc[0]
            if not arsenal.empty and not arsenal[arsenal["Code"].isin(["FF","SI","FC"])]["Velo"].dropna().empty else None,
            1," mph"
        ))
        d.metric("Sample pitches",fmt(pdisc.get("Pitches"),0))

    with st.expander("M2 · Volumen / Leash — 20%"):
        a,b,c,d=st.columns(4)
        a.metric("IP/start",fmt(mlb.get("calc_ip_start"),2))
        a.metric("BF/start",fmt(mlb.get("calc_bf_start"),1))
        a.metric("K/start",fmt(mlb.get("calc_k_start"),1))
        b.metric("Pitches L10",fmt(recent.get("avg_pitches"),1))
        b.metric("Max pitches L10",fmt(recent.get("max_pitches"),0))
        b.metric("90+ frequency",fmt(recent.get("freq_90"),0,"%"))
        c.metric("100+ frequency",fmt(recent.get("freq_100"),0,"%"))
        c.metric("Days rest",fmt(rest_days,0))
        c.metric("Pitches/BF",fmt(recent.get("pitches_per_bf"),2))
        d.metric("Avg BF L10",fmt(recent.get("avg_bf"),1))
        d.metric("Avg IP L10",fmt(recent.get("avg_ip"),2))
        d.metric("Avg K L10",fmt(recent.get("avg_k"),1))
        if log:
            st.dataframe(pd.DataFrame(log),hide_index=True,use_container_width=True)

        st.markdown("**Manual / news context**")
        m21,m22,m23=st.columns(3)
        with m21:
            injury_return=st.checkbox("Regreso de lesión")
            pitch_limit=st.checkbox("Posible pitch limit")
        with m22:
            opener=st.checkbox("Opener / bulk pitcher")
            manager_quick_hook=st.checkbox("Manager con hook corto esperado")
        with m23:
            manual_pitch_limit=st.number_input("Pitch limit estimado (0 = desconocido)",0,130,0,5)
        leash_notes=st.text_area("Notas de leash / manager / lesión",key="leash_notes")

    with st.expander("M3 · Splits del pitcher — 10%"):
        split_df=pd.DataFrame([
            {"Split":"vs RHB",**{k:split_r.get(k) for k in ("PA","K%","BB%","K-BB%","Whiff%","Contact%")}},
            {"Split":"vs LHB",**{k:split_l.get(k) for k in ("PA","K%","BB%","K-BB%","Whiff%","Contact%")}},
        ])
        st.dataframe(split_df,hide_index=True,use_container_width=True)
        if lineup:
            counts=pd.Series([r.get("Bats","N/A") for r in lineup]).value_counts().to_dict()
            st.write(f"**Lineup esperado/confirmado:** L {counts.get('L',0)} · R {counts.get('R',0)} · S {counts.get('S',0)}")
        else:
            st.info("Composición final pendiente hasta que MLB publique lineup.")

    with st.expander("M4 · Rival / propensión a strikeout — 20%"):
        a,b,c,d=st.columns(4)
        a.metric("Team K%",fmt(team_general.get("calc_k_pct"),1,"%"))
        a.metric("K% vs hand",fmt(team_split.get("calc_k_pct"),1,"%"))
        a.metric("PA vs hand",team_split.get("plateAppearances","N/A"))
        b.metric("BB%",fmt(team_split.get("calc_bb_pct") or team_general.get("calc_bb_pct"),1,"%"))
        b.metric("Contact%",fmt(opp_disc.get("Contact%"),1,"%"))
        b.metric("O-Contact%",fmt(opp_disc.get("O-Contact%"),1,"%"))
        c.metric("Z-Contact%",fmt(opp_disc.get("Z-Contact%"),1,"%"))
        c.metric("Whiff%",fmt(opp_disc.get("Whiff%"),1,"%"))
        c.metric("Chase%",fmt(opp_disc.get("Chase%"),1,"%"))
        d.metric("Swing%",fmt(opp_disc.get("Swing%"),1,"%"))
        d.metric("Zone%",fmt(opp_disc.get("Zone%"),1,"%"))
        opp_k_cmp=safe_num(team_split.get("calc_k_pct")) or safe_num(team_general.get("calc_k_pct"))
        d.metric("K% vs MLB reference",fmt(opp_k_cmp-22.0 if opp_k_cmp is not None else None,1," pp"))
        st.caption(f"Plate discipline: {opp_disc_source} · Baseball Savant team batting page.")

    with st.expander("M5 · Arsenal vs Matchup — 15%"):
        st.markdown("**Pitcher arsenal**")
        if not arsenal.empty:
            view=arsenal[["Pitch","Pitches","Usage%","Velo","Spin","Whiff%","K%","PutAway%","xwOBA","Run Value"]].copy()
            st.dataframe(view.round(3),hide_index=True,use_container_width=True)
        else:
            st.info("No arsenal Statcast.")

        st.markdown("**Rival vs pitch type**")
        if not opp_pitch.empty:
            show_cols=[c for c in ["Pitch","Pitches","PA","BA","SLG","wOBA","Whiff%","K%","PutAway%","xBA","xSLG","xwOBA","HardHit%","RV100","Run Value"] if c in opp_pitch.columns]
            st.dataframe(clean_display_frame(opp_pitch[show_cols].round(3),"—"),hide_index=True,use_container_width=True)
            st.caption(f"Fuente rival vs pitch type: {opp_pitch_source}. Expected stats se exigen antes de marcar M5 completo.")
        else:
            st.error("ERROR DE FUENTE: Savant no devolvió el perfil rival por tipo de pitcheo.")

    with st.expander("M6 · Forma y cambios recientes — 5%"):
        if not recent_sc.empty:
            st.dataframe(recent_sc.round(2),hide_index=True,use_container_width=True)
        else:
            st.info("No recent Statcast trend table.")
        st.caption("Incluye K%, K-BB%, Whiff%, CSW%, Ball%, velocidad, spin, movimiento y cambios de uso. xFIP/SIERA permanecen como cross-check de FanGraphs cuando la fuente responde.")

    with st.expander("M7 · Contexto — 5%"):
        a,b,c,d=st.columns(4)
        a.metric("Local / Visitante","Home" if p["team_side"]=="home" else "Away")
        a.metric("Stadium",p["venue"])
        b.metric("SO Park Factor",fmt(park_so,0))
        b.caption(park_source)
        display_temp=(f"{action_temp}°F · Action" if 'action_temp' in locals() and action_temp>0
                      else (f"{context['temperature']}°F · MLB" if context.get("temperature") is not None else "PENDIENTE"))
        mlb_wind=context.get("wind")
        display_wind=(f"{action_wind_mph} mph · {action_wind_dir} · Action"
                      if 'action_wind_mph' in locals() and action_wind_mph>0 and action_wind_dir!="PENDIENTE"
                      else (normalize_wind_direction(mlb_wind) if mlb_wind else "PENDIENTE"))
        c.metric("Temperature",display_temp)
        c.metric("Wind",display_wind)
        d.metric("Umpire",context.get("umpire") or "PENDIENTE")
        d.metric("Umpire K tendency","PENDIENTE" if not context.get("umpire") else "POR VALIDAR")
        st.caption("Weather and umpire stay low-weight until validation supports stronger use.")

    with st.expander("M8 · Lineup confirmado — 5%"):
        if lineup:
            ldf=pd.DataFrame(lineup)
            st.dataframe(ldf[["#","Hitter","Bats","K% vs hand","PA","Contact%","Whiff%","Source"]],hide_index=True,use_container_width=True)
            high=[r["Hitter"] for r in lineup if safe_num(r.get("K% vs hand")) is not None and r["K% vs hand"]>=27]
            low=[r["Hitter"] for r in lineup if safe_num(r.get("K% vs hand")) is not None and r["K% vs hand"]<=17]
            st.write("**High-K hitters:** "+(", ".join(high) if high else "None flagged"))
            st.write("**Low-K hitters:** "+(", ".join(low) if low else "None flagged"))
        else:
            st.warning("NO cerrar análisis definitivo: lineup real todavía no está confirmado.")
        lineup_notes=st.text_area("Ausencias / sustituciones / diferencias vs lineup habitual",key="lineup_notes")

# ------------------------------------------------------------
# MARKET / ALL LINES
# ------------------------------------------------------------
with tab_market:
    st.subheader("M9 · TODAS las líneas disponibles")
    st.caption("Añade solamente las líneas reales que tengas en DraftKings y FanDuel. El modelo evalúa cada precio y NO fuerza la línea principal.")

    default_market=pd.DataFrame([
        {"Sportsbook":"DraftKings","Market":"Over","Line":4.5,"Odds":-110},
        {"Sportsbook":"DraftKings","Market":"Under","Line":4.5,"Odds":-110},
        {"Sportsbook":"DraftKings","Market":"Over","Line":5.5,"Odds":-110},
        {"Sportsbook":"DraftKings","Market":"Under","Line":5.5,"Odds":-110},
        {"Sportsbook":"FanDuel","Market":"Over","Line":4.5,"Odds":-110},
        {"Sportsbook":"FanDuel","Market":"Under","Line":4.5,"Odds":-110},
        {"Sportsbook":"FanDuel","Market":"Over","Line":5.5,"Odds":-110},
        {"Sportsbook":"FanDuel","Market":"Under","Line":5.5,"Odds":-110},
        {"Sportsbook":"DraftKings","Market":"Alt 4+","Line":4.0,"Odds":100},
        {"Sportsbook":"FanDuel","Market":"Alt 5+","Line":5.0,"Odds":100},
    ])

    market_input=st.data_editor(
        default_market,num_rows="dynamic",hide_index=True,use_container_width=True,
        column_config={
            "Sportsbook":st.column_config.SelectboxColumn(options=["DraftKings","FanDuel"]),
            "Market":st.column_config.TextColumn(help="Examples: Over, Under, Alt 4+, Alt 5+, Alt 6+"),
            "Line":st.column_config.NumberColumn(step=.5),
            "Odds":st.column_config.NumberColumn(step=5),
        },
        key="all_market_lines"
    )

    rows=[]
    for _,r in market_input.iterrows():
        book=str(r.get("Sportsbook",""))
        market=str(r.get("Market","")).strip()
        line=safe_num(r.get("Line")); odds=safe_num(r.get("Odds"))
        if not book or not market or line is None or odds is None:
            continue

        low_market=market.lower()
        if low_market.startswith("over"):
            threshold=math.floor(line)+1
            prob=poisson_ge(threshold,proj["central"])
        elif low_market.startswith("under"):
            maxk=math.floor(line)
            prob=poisson_cdf(maxk,proj["central"])
        elif "alt" in low_market or "+" in low_market:
            # line itself is treated as the integer threshold for alt K+
            threshold=int(round(line))
            prob=poisson_ge(threshold,proj["central"])
        else:
            continue

        implied=american_implied(odds)
        edge=prob*100-implied if implied is not None else None
        ev=ev_per_dollar(prob,odds)
        rows.append({
            "Sportsbook":book,"Market":market,"Line":line,"Odds":int(odds),
            "Model%":prob*100,"Implied%":implied,"Fair":fair_american(prob),
            "Edge pp":edge,"EV%":ev*100 if ev is not None else None
        })

    market_df=pd.DataFrame(rows)
    if not market_df.empty:
        market_df=append_line_history(market_df,log)
        for col in ("Model%","Implied%","Fair","Edge pp","EV%","Hit L5%","Hit L10%"):
            market_df[col]=pd.to_numeric(market_df[col],errors="coerce").round(1)
        st.dataframe(market_df.sort_values("EV%",ascending=False),hide_index=True,use_container_width=True)
        st.caption("Hit L5/L10 = historial descriptivo del pitcher contra esa línea; no sustituye la probabilidad del modelo.")

    st.divider()
    st.subheader("Action Network PRO / B.A.R.T.O.L.O. validation")
    st.markdown(f"[Abrir Action Network MLB Pitching Props]({ACTION_PITCHING_PROPS_URL})")
    st.caption(f"Public page status: {action_public_status()}. PRO authentication remains manual/semi-manual.")

    ac1,ac2,ac3=st.columns(3)
    with ac1:
        bartolo_projection=st.number_input("B.A.R.T.O.L.O. projection K (0 = N/A)",0.0,15.0,0.0,.1)
        bets_pct=st.number_input("Action % Bets",0,100,50,1)
    with ac2:
        money_pct=st.number_input("Action % Money",0,100,50,1)
        sharp_action=st.checkbox("Sharp Action")
    with ac3:
        opening_line=st.number_input("Opening K line (0 = N/A)",0.0,15.5,0.0,.5)
        current_line=st.number_input("Current K line (0 = N/A)",0.0,15.5,0.0,.5)

    st.markdown("**Action Weather / Ballpark**")
    aw1,aw2,aw3=st.columns(3)
    with aw1:
        action_temp=st.number_input("Action temperature °F (0 = N/A)",0,120,0,1)
    with aw2:
        action_wind_mph=st.number_input("Action wind mph (0 = N/A)",0,50,0,1)
    with aw3:
        action_wind_dir=st.selectbox("Dirección del viento",[
            "PENDIENTE","OUT TO CF","OUT TO LF","OUT TO RF","IN FROM CF","IN FROM LF","IN FROM RF",
            "LEFT → RIGHT","RIGHT → LEFT","CALM"])
    action_notes=st.text_area("Action PRO notes / screenshot transcription")

# ------------------------------------------------------------
# ANALYSIS + FINAL CONCLUSION
# ------------------------------------------------------------
manual={
    "injury_return":locals().get("injury_return",False),
    "pitch_limit":locals().get("pitch_limit",False) or (locals().get("manual_pitch_limit",0)>0),
    "opener":locals().get("opener",False),
    "manager_quick_hook":locals().get("manager_quick_hook",False),
}

analysis_items=technical_analysis(
    mlb,pdisc,split_l,split_r,opp_disc,team_general,team_split,
    arsenal,opp_pitch,recent,park_so,lineup,proj,manual,log
)

# Data completeness is based on actual model modules, not validation websites.

def pitch_matchup_quality(df):
    if not isinstance(df,pd.DataFrame) or df.empty:return False
    required=["Whiff%","K%","xBA","xSLG","xwOBA"]
    present=0
    for c in required:
        if c in df.columns and pd.to_numeric(df[c],errors="coerce").notna().any():
            present+=1
    return present>=4

status={
    "M1":bool(mlb and not sc_pitcher.empty),
    "M2":bool(mlb and log),
    "M3":bool(not sc_pitcher.empty),
    "M4":bool(team_general and opp_disc and all(safe_num(opp_disc.get(x)) is not None for x in ("Whiff%","Contact%","Chase%","Zone%"))),
    "M5":bool(not arsenal.empty and pitch_matchup_quality(opp_pitch)),
    "M6":bool(not recent_sc.empty),
    "M7":bool(park_so is not None),
    "M8":bool(lineup_quality(lineup)[0]),
    "M9":bool('market_df' in locals() and not market_df.empty),
}
modules=sum(status.values())

best=None
if 'market_df' in locals() and not market_df.empty:
    eligible=market_df[pd.to_numeric(market_df["EV%"],errors="coerce").notna()]
    if not eligible.empty:
        best=eligible.sort_values("EV%",ascending=False).iloc[0].to_dict()

with tab_analysis:
    st.subheader("Qué nos dicen los datos")
    for title,signal,text in analysis_items:
        sig=str(signal).upper()
        cls=("signal-positive" if sig=="POSITIVO" else "signal-negative" if sig=="NEGATIVO"
             else "signal-pending" if sig in ("PENDIENTE","MATCHUP-DEPENDENT") else "signal-neutral")
        st.markdown(
            f"""<div class="analysis-card {cls}">
            <div class="section-label">{title}</div>
            <div style="font-size:.9rem;font-weight:800;margin:4px 0 8px">{signal}</div>
            <div style="opacity:.82;line-height:1.58">{text}</div>
            </div>""",unsafe_allow_html=True)

    st.subheader("Factores a favor / riesgos")
    positives=[title for title,signal,_ in analysis_items if signal=="POSITIVO"]
    negatives=[title for title,signal,_ in analysis_items if signal=="NEGATIVO"]
    x1,x2=st.columns(2)
    with x1:
        st.success("**A favor:** "+(", ".join(positives) if positives else "No hay señales fuertes aisladas."))
    with x2:
        risk_text=(", ".join(negatives) if negatives else "No hay señales negativas fuertes en módulos automáticos.")
        if manual.get("pitch_limit"): risk_text += " · Pitch limit / leash concern."
        if manual.get("injury_return"): risk_text += " · Injury return."
        if manual.get("opener"): risk_text += " · Opener/bulk role."
        st.warning("**Riesgos:** "+risk_text)

    st.subheader("Historial del pitcher contra las líneas")
    if 'market_df' in locals() and not market_df.empty:
        hcols=[c for c in ["Sportsbook","Market","Line","Odds","Hit L5%","Hit L10%"] if c in market_df.columns]
        st.dataframe(market_df[hcols].drop_duplicates(subset=["Market","Line"]).sort_values(["Line","Market"]),
                     hide_index=True,use_container_width=True)
        st.caption("El historial L5/L10 es descriptivo; no reemplaza matchup, precio ni probabilidad del modelo.")
    else:
        st.info("Carga las líneas reales en Mercado / Edge para construir el historial L5/L10.")

    st.subheader("Conclusión final")
    conf="A" if modules==9 else ("B" if modules==8 else "C")
    critical_ready=all(status.get(x,False) for x in ("M5","M7","M8"))
    provisional=not critical_ready
    state_label="DATA INCOMPLETE / PROVISIONAL" if provisional else "FINAL PREGAME"
    conf_class="confidence-a" if conf=="A" else ("confidence-b" if conf=="B" else "confidence-c")
    st.markdown(f'**Estado:** {state_label} · **Calidad:** <span class="{conf_class}">{conf}</span> · {modules}/9 módulos completos',
                unsafe_allow_html=True)
    st.write(f"**BF proyectados:** {proj['bf']:.1f}")
    st.write(f"**K% proyectado:** {proj['k_pct']:.1f}%")
    st.write(f"**Strikeouts proyectados:** {proj['central']:.2f}")
    st.write(f"**Rango probable:** {proj['low']:.1f}–{proj['high']:.1f} K")

    if best:
        ev_val=safe_num(best.get("EV%"))/100 if safe_num(best.get("EV%")) is not None else None
        edge_val=safe_num(best.get("Edge pp"))
        grade=grade_bet(ev_val,edge_val,modules)
        decision="PASS / NO BET"
        if (not provisional) and grade in ("A","B","C") and safe_num(best.get("EV%")) is not None and best["EV%"]>=3:
            decision=f"{best['Market']} {best['Line']} · {best['Sportsbook']} {int(best['Odds']):+d}"
        elif provisional:
            decision="ESPERAR DATOS CRÍTICOS · PROYECCIÓN PROVISIONAL"

        st.markdown(
            f"""
            <div class="finalcard">
              <div style="font-size:.78rem;opacity:.7">DICTAMEN DEL MODELO</div>
              <div style="font-size:1.7rem;font-weight:850;margin:5px 0">{decision}</div>
              <div>
                Model <b>{best['Model%']:.1f}%</b> · Fair <b>{best['Fair']:+.0f}</b> ·
                Edge <b>{best['Edge pp']:.1f} pp</b> · EV <b>{best['EV%']:.1f}%</b> ·
                Grade <b>{grade}</b>
              </div>
            </div>
            """,unsafe_allow_html=True,
        )

        if bartolo_projection>0:
            diff=proj["central"]-bartolo_projection
            st.info(
                f"B.A.R.T.O.L.O. comparison: Action {bartolo_projection:.1f} K vs model {proj['central']:.2f} K "
                f"(difference {diff:+.2f} K). Se usa como validación externa, no como sustituto."
            )

        st.caption("Una calificación A NO obliga a apostar. Sin edge/EV suficiente = NO BET.")
    else:
        st.warning("No hay líneas válidas cargadas. Dictamen: PASS hasta introducir precios reales.")

    st.progress(modules/9)
    st.caption(" · ".join(f"{k} {'✅' if v else '⏳'}" for k,v in status.items()))

# ------------------------------------------------------------
# SOURCES
# ------------------------------------------------------------
with tab_sources:
    st.subheader("Fuentes y estado")
    src=pd.DataFrame([
        {"Fuente":"MLB Stats API","Uso":"Schedule, probables, season-to-date, logs, lineup, weather, umpire","Estado":"OK" if mlb else "Partial"},
        {"Fuente":"Baseball Savant / Statcast","Uso":"Pitcher discipline, arsenal, movement, spin","Estado":"OK" if not sc_pitcher.empty else "ERROR"},
        {"Fuente":"Savant Team Page","Uso":"Opponent Contact, Chase, Zone, Whiff","Estado":opp_disc_source},
        {"Fuente":"Savant Pitch Arsenal","Uso":"Opponent vs pitch type","Estado":opp_pitch_source},
        {"Fuente":"Savant SO Park Factor","Uso":"M7","Estado":f"{fmt(park_so,0)} · {park_source}"},
        {"Fuente":"Baseball-Reference","Uso":"Cross-check + fallback cuando la métrica existe públicamente","Estado":"OK" if br else "No match"},
        {"Fuente":"FanGraphs","Uso":"Validation / xFIP / SIERA when reachable","Estado":fg_status},
        {"Fuente":"Action Network PRO","Uso":"B.A.R.T.O.L.O., % Bets, % Money, sharp, movement","Estado":"Manual PRO validation"},
        {"Fuente":"DraftKings / FanDuel","Uso":"Official model prices in current phase","Estado":"Manual line/odds entry"},
    ])
    st.dataframe(src,hide_index=True,use_container_width=True)

    st.warning(
        "Important: full-season leaderboards are never allowed to leak future data into historical pregame projections. "
        "Core projection inputs remain cutoff-safe. Fallbacks only fill a metric when the source is compatible with the selected pregame cutoff."
    )

st.caption("V3.1.3 LIVE TEST · Data-quality gate · expected-stat fallback · lineup handedness · cutoff-safe pregame model.")
