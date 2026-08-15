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
    pitching_stats,
    pitching_stats_bref,
    playerid_lookup,
)

# ============================================================
# MODEL PROFESSIONAL MLB - STARTING PITCHER STRIKEOUTS
# V2.0 LIVE TEST
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
    page_icon="â¾",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width:1180px;padding-top:.7rem;padding-bottom:4rem}
    h1,h2,h3 {letter-spacing:-.025em}
    .hero {
      border:1px solid rgba(128,128,128,.18);
      border-radius:22px;padding:20px;
      background:linear-gradient(135deg,rgba(26,93,255,.16),rgba(126,78,220,.07));
      margin-bottom:14px
    }
    .gamecard {
      border:1px solid rgba(128,128,128,.18);
      border-radius:20px;padding:18px;margin:8px 0 14px;
      background:rgba(128,128,128,.035)
    }
    .finalcard {
      border:2px solid rgba(26,93,255,.36);
      border-radius:22px;padding:20px;margin:14px 0;
      background:linear-gradient(145deg,rgba(26,93,255,.18),rgba(26,93,255,.05))
    }
    .analysis-card {
      border:1px solid rgba(128,128,128,.16);
      border-radius:16px;padding:14px 16px;margin:8px 0;
      background:rgba(128,128,128,.025)
    }
    .pill {
      display:inline-block;padding:5px 9px;border-radius:999px;
      background:rgba(128,128,128,.12);margin:3px;font-size:.78rem
    }
    div[data-testid="stMetric"] {
      border:1px solid rgba(128,128,128,.14);border-radius:15px;
      padding:11px;background:rgba(128,128,128,.028)
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
        return dt.astimezone(timezone.utc).strftime("%b %d Â· %I:%M %p UTC").replace(" 0", " ")
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

        # xSLG is not a direct Statcast pitch-level field; leave explicit N/A.
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
            "xSLG": None,
            "xwOBA": xwoba,
            "HardHit%": hardhit,
            "RV100": rv100,
        })
    result = pd.DataFrame(rows)
    return result.sort_values("Pitches",ascending=False) if not result.empty else result


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
    df = savant_park_factors(year)
    if not df.empty:
        vn = df["Venue"].astype(str).map(normalize_name)
        hit = df[vn.eq(target)]
        if not hit.empty:
            x = safe_num(hit.iloc[0].get("SO"))
            if x is not None:
                return x, "Savant live"
    cached = SAVANT_SO_VERIFIED_CACHE.get((year,target))
    if cached is not None:
        return cached, "Savant verified cache"
    return None, "Unavailable"


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


def technical_analysis(mlb,pdisc,split_l,split_r,opp_disc,team_general,team_split,arsenal,recent,park_so,lineup,proj,manual):
    items = []

    k_pct = safe_num(mlb.get("calc_k_pct"))
    whiff = safe_num(pdisc.get("Whiff%"))
    csw = safe_num(pdisc.get("CSW%"))
    contact = safe_num(pdisc.get("Contact%"))
    m1 = direction(k_pct,22.5,False,1.5)
    items.append(("M1 Â· Capacidad",m1,
        f"K% {fmt(k_pct,1,'%')}, Whiff% {fmt(whiff,1,'%')}, CSW% {fmt(csw,1,'%')} y Contact% {fmt(contact,1,'%')}. "
        "La combinaciÃ³n mide dominio real: ponches, swings fallidos y capacidad de evitar contacto."))

    m2 = "POSITIVO" if safe_num(recent.get("avg_pitches")) and recent["avg_pitches"]>=90 else "NEUTRAL"
    if manual.get("pitch_limit") or manual.get("injury_return") or manual.get("opener"):
        m2 = "NEGATIVO"
    items.append(("M2 Â· Volumen / Leash",m2,
        f"BF proyectados {fmt(proj['bf'],1)}, pitches L10 {fmt(recent.get('avg_pitches'),1)}, "
        f"90+ {fmt(recent.get('freq_90'),0,'%')}, 100+ {fmt(recent.get('freq_100'),0,'%')}. "
        "El techo de K depende de cuÃ¡ntos bateadores realmente puede enfrentar."))

    l = safe_num(split_l.get("K%")); r = safe_num(split_r.get("K%"))
    m3 = "NEUTRAL"
    if l is not None and r is not None and abs(l-r)>=5: m3="MATCHUP-DEPENDENT"
    items.append(("M3 Â· Splits",m3,
        f"K% vs LHB {fmt(l,1,'%')} y vs RHB {fmt(r,1,'%')}. "
        "Una diferencia grande hace que la composiciÃ³n L/R/S del lineup tenga mÃ¡s importancia."))

    ok = safe_num(team_split.get("calc_k_pct")) or safe_num(team_general.get("calc_k_pct"))
    m4 = direction(ok,22.5,False,1.5)
    items.append(("M4 Â· Rival",m4,
        f"Opponent K% vs hand {fmt(ok,1,'%')}; Whiff% {fmt(opp_disc.get('Whiff%'),1,'%')}; "
        f"Contact% {fmt(opp_disc.get('Contact%'),1,'%')}. Un rival con mÃ¡s K y menos contacto eleva oportunidades."))

    top_pitch = arsenal.iloc[0]["Pitch"] if isinstance(arsenal,pd.DataFrame) and not arsenal.empty else "N/A"
    top_whiff = arsenal.iloc[0]["Whiff%"] if isinstance(arsenal,pd.DataFrame) and not arsenal.empty else None
    m5 = "POSITIVO" if safe_num(top_whiff) and safe_num(top_whiff)>=25 else "NEUTRAL"
    items.append(("M5 Â· Arsenal",m5,
        f"Lanzamiento principal: {top_pitch}, Whiff% {fmt(top_whiff,1,'%')}. "
        "El mÃ³dulo compara las armas de mayor uso con cÃ³mo responde la ofensiva a esos tipos de pitcheo."))

    m6 = "POSITIVO" if safe_num(recent.get("avg_k")) and safe_num(mlb.get("calc_k_start")) and recent["avg_k"]>mlb["calc_k_start"]+.5 else "NEUTRAL"
    items.append(("M6 Â· Forma reciente",m6,
        f"K L10 {fmt(recent.get('avg_k'),1)} vs K/start temporada {fmt(mlb.get('calc_k_start'),1)}. "
        "Velocidad, spin, movimiento, uso y disciplina recientes sirven para detectar cambios reales."))

    m7 = "POSITIVO" if safe_num(park_so) and park_so>101 else ("NEGATIVO" if safe_num(park_so) and park_so<99 else "NEUTRAL")
    items.append(("M7 Â· Contexto",m7,
        f"SO Park Factor {fmt(park_so,0)} (100 neutral). Clima y umpire se muestran, pero mantienen peso bajo hasta validaciÃ³n."))

    lk = lineup_k_pct(lineup)
    m8 = direction(lk,22.5,False,1.5) if lineup else "PENDIENTE"
    items.append(("M8 Â· Lineup",m8,
        f"Lineup K% agregado {fmt(lk,1,'%')}. El anÃ¡lisis definitivo mejora cuando los 9 bateadores confirmados estÃ¡n disponibles."))

    return items


# ============================================================
# APP LOAD
# ============================================================

st.markdown(
    """
    <div class="hero">
      <div style="font-size:.78rem;opacity:.68">MODELO PROFESIONAL MLB</div>
      <div style="font-size:2rem;font-weight:850">Starting Pitcher Strikeout Lab</div>
      <div style="opacity:.72;margin-top:5px">V2.0 LIVE TEST Â· Pregame-only data Â· Multi-line EV Â· Technical Analysis</div>
    </div>
    """,unsafe_allow_html=True,
)

c1,c2 = st.columns([1,2.3])
with c1:
    game_date = st.date_input("Fecha",value=date.today(),min_value=date(2015,1,1))

try:
    options = pitchers_for_date(game_date.isoformat())
except Exception as exc:
    st.error(f"No se pudo cargar MLB: {exc}")
    st.stop()

if not options:
    st.warning("No hay abridores probables disponibles para esta fecha.")
    st.stop()

by_id={x["selection_id"]:x for x in options}
with c2:
    selected=st.selectbox(
        "Abridor",list(by_id),
        format_func=lambda x:f"{by_id[x]['pitcher_name']} Â· {by_id[x]['team']} vs {by_id[x]['opponent']}"
    )
p=by_id[selected]

cutoff = game_cutoff(game_date)
cutoff_str = cutoff.isoformat()

st.markdown(
    f"""
    <div class="gamecard">
      <div style="font-size:1.6rem;font-weight:800">{p['pitcher_name']} <span style="opacity:.5">vs {p['opponent']}</span></div>
      <span class="pill">{p['team']}</span><span class="pill">{p['throwing_hand']}</span>
      <span class="pill">{p['venue']}</span><span class="pill">{p['game_time']}</span>
      <span class="pill">{p['status']}</span>
      <div style="margin-top:9px;font-size:.8rem;opacity:.72">Pregame cutoff: datos hasta {cutoff_str}. El juego seleccionado nunca se usa en sus propios inputs.</div>
    </div>
    """,unsafe_allow_html=True,
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

    opp_sc_all=team_statcast(opp_abbr,sc_start,cutoff_str) if opp_abbr else pd.DataFrame()
    opp_off=offensive_team_rows(opp_sc_all,opp_abbr) if opp_abbr else pd.DataFrame()
    opp_disc=plate_discipline(opp_off)
    opp_pitch=opponent_pitch_type_table(opp_off)

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
    ["Resumen","MÃ³dulos 1â8","AnÃ¡lisis TÃ©cnico","Mercado / Edge","Fuentes"]
)

# ------------------------------------------------------------
# SUMMARY
# ------------------------------------------------------------
with tab_summary:
    a,b,c,d=st.columns(4)
    a.metric("BF proyectados",fmt(proj["bf"],1))
    b.metric("K% proyectado",fmt(proj["k_pct"],1,"%"))
    c.metric("Strikeouts proyectados",fmt(proj["central"],2))
    d.metric("Rango probable",f"{proj['low']:.1f}â{proj['high']:.1f}")

    st.subheader("Probabilidad por umbral")
    dist=[]
    for k in range(3,11):
        prob=poisson_ge(k,proj["central"])
        dist.append({"LÃ­nea":f"{k}+","Probabilidad":prob*100,"Fair Odds":fair_american(prob)})
    dist_df=pd.DataFrame(dist)
    dist_df["Probabilidad"]=dist_df["Probabilidad"].round(1)
    dist_df["Fair Odds"]=dist_df["Fair Odds"].round(0)
    st.dataframe(dist_df,hide_index=True,use_container_width=True)

    st.info("Esta es la proyecciÃ³n estadÃ­stica pregame. La selecciÃ³n de apuesta ocurre solamente despuÃ©s de comparar TODAS las lÃ­neas y precios disponibles.")

# ------------------------------------------------------------
# MODULES 1-8
# ------------------------------------------------------------
with tab_modules:
    with st.expander("M1 Â· Capacidad real de strikeout â 20%",expanded=True):
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

    with st.expander("M2 Â· Volumen / Leash â 20%"):
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
            injury_return=st.checkbox("Regreso de lesiÃ³n")
            pitch_limit=st.checkbox("Posible pitch limit")
        with m22:
            opener=st.checkbox("Opener / bulk pitcher")
            manager_quick_hook=st.checkbox("Manager con hook corto esperado")
        with m23:
            manual_pitch_limit=st.number_input("Pitch limit estimado (0 = desconocido)",0,130,0,5)
        leash_notes=st.text_area("Notas de leash / manager / lesiÃ³n",key="leash_notes")

    with st.expander("M3 Â· Splits del pitcher â 10%"):
        split_df=pd.DataFrame([
            {"Split":"vs RHB",**{k:split_r.get(k) for k in ("PA","K%","BB%","K-BB%","Whiff%","Contact%")}},
            {"Split":"vs LHB",**{k:split_l.get(k) for k in ("PA","K%","BB%","K-BB%","Whiff%","Contact%")}},
        ])
        st.dataframe(split_df,hide_index=True,use_container_width=True)
        if lineup:
            counts=pd.Series([r.get("Bats","N/A") for r in lineup]).value_counts().to_dict()
            st.write(f"**Lineup esperado/confirmado:** L {counts.get('L',0)} Â· R {counts.get('R',0)} Â· S {counts.get('S',0)}")
        else:
            st.info("ComposiciÃ³n final pendiente hasta que MLB publique lineup.")

    with st.expander("M4 Â· Rival / propensiÃ³n a strikeout â 20%"):
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
        d.metric("MLB Avg comparison","N/A")
        st.caption("MLB-average comparison remains visible but N/A until a reliable current-season league plate-discipline feed is validated.")

    with st.expander("M5 Â· Arsenal vs Matchup â 15%"):
        st.markdown("**Pitcher arsenal**")
        if not arsenal.empty:
            view=arsenal[["Pitch","Pitches","Usage%","Velo","Spin","Whiff%","K%","PutAway%","xwOBA","Run Value"]].copy()
            st.dataframe(view.round(3),hide_index=True,use_container_width=True)
        else:
            st.info("No arsenal Statcast.")

        st.markdown("**Rival vs pitch type**")
        if not opp_pitch.empty:
            st.dataframe(
                opp_pitch[["Pitch","Pitches","PA","BA","SLG","wOBA","Whiff%","K%","PutAway%","xBA","xSLG","xwOBA","HardHit%","RV100"]].round(3),
                hide_index=True,use_container_width=True
            )
            st.caption("xSLG and opponent PutAway are displayed as N/A when Statcast does not provide a defensible direct calculation.")
        else:
            st.info("No se pudo construir el perfil de la ofensiva por tipo de pitcheo.")

    with st.expander("M6 Â· Forma y cambios recientes â 5%"):
        if not recent_sc.empty:
            st.dataframe(recent_sc.round(2),hide_index=True,use_container_width=True)
        else:
            st.info("No recent Statcast trend table.")
        st.caption("Incluye K%, K-BB%, Whiff%, CSW%, Ball%, velocidad, spin, movimiento y cambios de uso. xFIP/SIERA permanecen como cross-check de FanGraphs cuando la fuente responde.")

    with st.expander("M7 Â· Contexto â 5%"):
        a,b,c,d=st.columns(4)
        a.metric("Local / Visitante","Home" if p["team_side"]=="home" else "Away")
        a.metric("Stadium",p["venue"])
        b.metric("SO Park Factor",fmt(park_so,0))
        b.caption(park_source)
        c.metric("Temperature",f"{context['temperature']}Â°F" if context.get("temperature") is not None else "N/A")
        c.metric("Wind",context.get("wind") or "N/A")
        d.metric("Umpire",context.get("umpire") or "N/A")
        d.metric("Umpire K tendency","N/A")
        st.caption("Weather and umpire stay low-weight until validation supports stronger use.")

    with st.expander("M8 Â· Lineup confirmado â 5%"):
        if lineup:
            ldf=pd.DataFrame(lineup)
            st.dataframe(ldf[["#","Hitter","Bats","K% vs hand","PA","Contact%","Whiff%","Source"]],hide_index=True,use_container_width=True)
            high=[r["Hitter"] for r in lineup if safe_num(r.get("K% vs hand")) is not None and r["K% vs hand"]>=27]
            low=[r["Hitter"] for r in lineup if safe_num(r.get("K% vs hand")) is not None and r["K% vs hand"]<=17]
            st.write("**High-K hitters:** "+(", ".join(high) if high else "None flagged"))
            st.write("**Low-K hitters:** "+(", ".join(low) if low else "None flagged"))
        else:
            st.warning("NO cerrar anÃ¡lisis definitivo: lineup real todavÃ­a no estÃ¡ confirmado.")
        lineup_notes=st.text_area("Ausencias / sustituciones / diferencias vs lineup habitual",key="lineup_notes")

# ------------------------------------------------------------
# MARKET / ALL LINES
# ------------------------------------------------------------
with tab_market:
    st.subheader("M9 Â· TODAS las lÃ­neas disponibles")
    st.caption("AÃ±ade solamente las lÃ­neas reales que tengas en DraftKings y FanDuel. El modelo evalÃºa cada precio y NO fuerza la lÃ­nea principal.")

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
        for col in ("Model%","Implied%","Fair","Edge pp","EV%"):
            market_df[col]=pd.to_numeric(market_df[col],errors="coerce").round(1)
        st.dataframe(market_df.sort_values("EV%",ascending=False),hide_index=True,use_container_width=True)

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
    arsenal,recent,park_so,lineup,proj,manual
)

# Data completeness is based on actual model modules, not validation websites.
status={
    "M1":bool(mlb and not sc_pitcher.empty),
    "M2":bool(mlb and log),
    "M3":bool(not sc_pitcher.empty),
    "M4":bool(team_general),
    "M5":bool(not arsenal.empty),
    "M6":bool(not recent_sc.empty),
    "M7":bool(park_so is not None),
    "M8":bool(lineup),
    "M9":bool('market_df' in locals() and not market_df.empty),
}
modules=sum(status.values())

best=None
if 'market_df' in locals() and not market_df.empty:
    eligible=market_df[pd.to_numeric(market_df["EV%"],errors="coerce").notna()]
    if not eligible.empty:
        best=eligible.sort_values("EV%",ascending=False).iloc[0].to_dict()

with tab_analysis:
    st.subheader("QuÃ© nos dicen los datos")
    for title,signal,text in analysis_items:
        st.markdown(
            f"""<div class="analysis-card"><b>{title}</b> Â· <b>{signal}</b><br><span style="opacity:.78">{text}</span></div>""",
            unsafe_allow_html=True,
        )

    st.subheader("Factores a favor / riesgos")
    positives=[title for title,signal,_ in analysis_items if signal=="POSITIVO"]
    negatives=[title for title,signal,_ in analysis_items if signal=="NEGATIVO"]
    x1,x2=st.columns(2)
    with x1:
        st.success("**A favor:** "+(", ".join(positives) if positives else "No hay seÃ±ales fuertes aisladas."))
    with x2:
        risk_text=(", ".join(negatives) if negatives else "No hay seÃ±ales negativas fuertes en mÃ³dulos automÃ¡ticos.")
        if manual.get("pitch_limit"): risk_text += " Â· Pitch limit / leash concern."
        if manual.get("injury_return"): risk_text += " Â· Injury return."
        if manual.get("opener"): risk_text += " Â· Opener/bulk role."
        st.warning("**Riesgos:** "+risk_text)

    st.subheader("ConclusiÃ³n final")
    conf="A" if modules==9 else ("B" if modules==8 else "C")
    st.write(f"**Calidad del anÃ¡lisis:** {conf} Â· {modules}/9 mÃ³dulos completos")
    st.write(f"**BF proyectados:** {proj['bf']:.1f}")
    st.write(f"**K% proyectado:** {proj['k_pct']:.1f}%")
    st.write(f"**Strikeouts proyectados:** {proj['central']:.2f}")
    st.write(f"**Rango probable:** {proj['low']:.1f}â{proj['high']:.1f} K")

    if best:
        ev_val=safe_num(best.get("EV%"))/100 if safe_num(best.get("EV%")) is not None else None
        edge_val=safe_num(best.get("Edge pp"))
        grade=grade_bet(ev_val,edge_val,modules)
        decision="PASS / NO BET"
        if grade in ("A","B","C") and safe_num(best.get("EV%")) is not None and best["EV%"]>=3:
            decision=f"{best['Market']} {best['Line']} Â· {best['Sportsbook']} {int(best['Odds']):+d}"

        st.markdown(
            f"""
            <div class="finalcard">
              <div style="font-size:.78rem;opacity:.7">DICTAMEN DEL MODELO</div>
              <div style="font-size:1.7rem;font-weight:850;margin:5px 0">{decision}</div>
              <div>
                Model <b>{best['Model%']:.1f}%</b> Â· Fair <b>{best['Fair']:+.0f}</b> Â·
                Edge <b>{best['Edge pp']:.1f} pp</b> Â· EV <b>{best['EV%']:.1f}%</b> Â·
                Grade <b>{grade}</b>
              </div>
            </div>
            """,unsafe_allow_html=True,
        )

        if bartolo_projection>0:
            diff=proj["central"]-bartolo_projection
            st.info(
                f"B.A.R.T.O.L.O. comparison: Action {bartolo_projection:.1f} K vs model {proj['central']:.2f} K "
                f"(difference {diff:+.2f} K). Se usa como validaciÃ³n externa, no como sustituto."
            )

        st.caption("Una calificaciÃ³n A NO obliga a apostar. Sin edge/EV suficiente = NO BET.")
    else:
        st.warning("No hay lÃ­neas vÃ¡lidas cargadas. Dictamen: PASS hasta introducir precios reales.")

    st.progress(modules/9)
    st.caption(" Â· ".join(f"{k} {'â' if v else 'â³'}" for k,v in status.items()))

# ------------------------------------------------------------
# SOURCES
# ------------------------------------------------------------
with tab_sources:
    st.subheader("Fuentes y estado")
    src=pd.DataFrame([
        {"Fuente":"MLB Stats API","Uso":"Schedule, probables, season-to-date, logs, lineup, weather, umpire","Estado":"OK" if mlb else "Partial"},
        {"Fuente":"Baseball Savant / Statcast","Uso":"Plate discipline, arsenal, pitch movement, spin, pitch-type matchup","Estado":"OK" if not sc_pitcher.empty else "Unavailable"},
        {"Fuente":"Savant SO Park Factor","Uso":"M7","Estado":f"{fmt(park_so,0)} Â· {park_source}"},
        {"Fuente":"Baseball-Reference","Uso":"Validation cross-check","Estado":"OK" if br else "No match"},
        {"Fuente":"FanGraphs","Uso":"Validation / xFIP / SIERA when reachable","Estado":fg_status},
        {"Fuente":"Action Network PRO","Uso":"B.A.R.T.O.L.O., % Bets, % Money, sharp, movement","Estado":"Manual PRO validation"},
        {"Fuente":"DraftKings / FanDuel","Uso":"Official model prices in current phase","Estado":"Manual line/odds entry"},
    ])
    st.dataframe(src,hide_index=True,use_container_width=True)

    st.warning(
        "Important: FanGraphs/BBRef full-season leaderboards are validation only and are NOT fed into historical pregame projections, "
        "preventing future-data leakage. Core projection inputs use cutoff-safe MLB + Statcast data."
    )

st.caption("V2.0 LIVE TEST Â· Structure frozen. Next phase: real-game logging, calibration and evidence-based adjustments.")
