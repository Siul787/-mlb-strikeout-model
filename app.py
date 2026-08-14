# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date, datetime, timezone
from io import StringIO
from urllib.parse import urlencode
import math
import unicodedata

import pandas as pd
import requests
import streamlit as st
from pybaseball import (
    statcast_pitcher,
    pitching_stats,
    pitching_stats_bref,
    playerid_lookup,
)

# =========================================================
# CONFIG
# =========================================================

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game"
SAVANT_PARK_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"

TIMEOUT = 25

PITCH_NAMES = {
    "FF": "4-Seam", "SI": "Sinker", "FC": "Cutter", "SL": "Slider",
    "ST": "Sweeper", "CU": "Curveball", "KC": "Knuckle Curve",
    "CH": "Changeup", "FS": "Splitter", "FO": "Forkball",
    "KN": "Knuckleball", "SV": "Slurve",
}

# =========================================================
# STYLE
# =========================================================

st.set_page_config(
    page_title="MLB K Model",
    page_icon="\u26be",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
    <style>
    .block-container {max-width: 1180px; padding-top: 1.2rem; padding-bottom: 3rem;}
    h1, h2, h3 {letter-spacing: -0.02em;}
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,.18);
        border-radius: 16px;
        padding: 14px 14px 10px 14px;
        background: rgba(128,128,128,.045);
    }
    div[data-testid="stMetricLabel"] {font-size: .82rem;}
    .hero {
        border: 1px solid rgba(128,128,128,.22);
        border-radius: 20px;
        padding: 20px;
        margin: 2px 0 18px 0;
        background: linear-gradient(145deg, rgba(55,110,255,.10), rgba(128,128,128,.03));
    }
    .bet-card {
        border: 2px solid rgba(55,110,255,.32);
        border-radius: 20px;
        padding: 18px;
        margin-top: 10px;
        background: rgba(55,110,255,.07);
    }
    .muted {opacity:.72; font-size:.9rem;}
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================================================
# HELPERS
# =========================================================

class DataSourceError(Exception):
    pass


def get_json(url: str, params: dict | None = None) -> dict:
    try:
        r = requests.get(url, params=params, timeout=TIMEOUT)
        r.raise_for_status()
        payload = r.json()
    except (requests.RequestException, ValueError) as exc:
        raise DataSourceError(str(exc)) from exc
    return payload if isinstance(payload, dict) else {}


def safe_num(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).replace("%", "").replace(",", ""))
    except (TypeError, ValueError):
        return None


def fmt(value, decimals=1, suffix=""):
    v = safe_num(value)
    return "N/A" if v is None else f"{v:.{decimals}f}{suffix}"


def normalize_name(value: str) -> str:
    text = unicodedata.normalize("NFKD", str(value or ""))
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return " ".join(text.lower().replace(".", "").replace("-", " ").split())


def weighted_average(items, default=None):
    valid = [(safe_num(v), float(w)) for v, w in items if safe_num(v) is not None and w > 0]
    if not valid:
        return default
    weight = sum(w for _, w in valid)
    return sum(v * w for v, w in valid) / weight if weight else default


def clamp(value, low, high):
    return max(low, min(high, value))


def american_implied(odds):
    o = safe_num(odds)
    if o is None or o == 0:
        return None
    return ((-o) / ((-o) + 100) if o < 0 else 100 / (o + 100)) * 100


def fair_american(prob):
    p = safe_num(prob)
    if p is None or p <= 0 or p >= 1:
        return None
    return -100 * p / (1 - p) if p >= .5 else 100 * (1 - p) / p


def profit_per_dollar(odds):
    o = safe_num(odds)
    if o is None or o == 0:
        return None
    return o / 100 if o > 0 else 100 / abs(o)


def ev_per_dollar(prob, odds):
    p = safe_num(prob)
    profit = profit_per_dollar(odds)
    if p is None or profit is None:
        return None
    return p * profit - (1 - p)


def poisson_pmf(k, lam):
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1)) if lam > 0 else (1.0 if k == 0 else 0.0)


def poisson_cdf(k, lam):
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def poisson_ge(k, lam):
    return clamp(1 - poisson_cdf(k - 1, lam), 0.0, 1.0)


def innings_to_decimal(value):
    if value is None:
        return None
    text = str(value)
    if "." not in text:
        return safe_num(text)
    try:
        whole, frac = text.split(".", 1)
        return int(whole) + int(frac[:1]) / 3
    except Exception:
        return None


# =========================================================
# MLB API
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def pitcher_profile(player_id: int):
    data = get_json(f"{MLB_PEOPLE_URL}/{player_id}")
    people = data.get("people", [])
    return people[0] if people else {}


@st.cache_data(ttl=300, show_spinner=False)
def person_stats(player_id: int, season: int, group: str, sit_code: str | None = None):
    params = {"stats": "season", "group": group, "season": season}
    if sit_code:
        params["sitCodes"] = sit_code
    data = get_json(f"{MLB_PEOPLE_URL}/{player_id}/stats", params=params)
    groups = data.get("stats", [])
    if not groups:
        return {}
    splits = groups[0].get("splits", [])
    return splits[0].get("stat", {}) if splits else {}


@st.cache_data(ttl=300, show_spinner=False)
def pitcher_season_stats(player_id: int, season: int):
    stat = person_stats(player_id, season, "pitching")
    so = safe_num(stat.get("strikeOuts"))
    bb = safe_num(stat.get("baseOnBalls"))
    bf = safe_num(stat.get("battersFaced"))
    gs = safe_num(stat.get("gamesStarted"))
    ip = innings_to_decimal(stat.get("inningsPitched"))

    stat["calc_k_pct"] = so / bf * 100 if so is not None and bf else None
    stat["calc_bb_pct"] = bb / bf * 100 if bb is not None and bf else None
    stat["calc_k_minus_bb"] = (
        stat["calc_k_pct"] - stat["calc_bb_pct"]
        if stat["calc_k_pct"] is not None and stat["calc_bb_pct"] is not None else None
    )
    stat["calc_bf_start"] = bf / gs if bf is not None and gs else None
    stat["calc_ip_start"] = ip / gs if ip is not None and gs else None
    stat["calc_k_start"] = so / gs if so is not None and gs else None
    return stat


@st.cache_data(ttl=300, show_spinner=False)
def pitcher_game_log(player_id: int, season: int):
    data = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": season},
    )
    groups = data.get("stats", [])
    if not groups:
        return []

    out = []
    for split in groups[0].get("splits", []):
        stat = split.get("stat", {})
        if not safe_num(stat.get("gamesStarted")):
            continue
        out.append({
            "Date": split.get("date", "N/A"),
            "Opp": split.get("opponent", {}).get("name", "N/A"),
            "IP": stat.get("inningsPitched", "N/A"),
            "K": stat.get("strikeOuts", "N/A"),
            "BB": stat.get("baseOnBalls", "N/A"),
            "ER": stat.get("earnedRuns", "N/A"),
            "HR": stat.get("homeRuns", "N/A"),
            "Pitches": stat.get("numberOfPitches", "N/A"),
            "BF": stat.get("battersFaced", "N/A"),
        })
    return out[-10:][::-1]


@st.cache_data(ttl=300, show_spinner=False)
def team_hitting_stats(team_id: int, season: int, sit_code: str | None = None):
    params = {"stats": "season", "group": "hitting", "season": season}
    if sit_code:
        params["sitCodes"] = sit_code
    data = get_json(f"{MLB_TEAMS_URL}/{team_id}/stats", params=params)
    groups = data.get("stats", [])
    if not groups:
        return {}
    splits = groups[0].get("splits", [])
    if not splits:
        return {}
    stat = splits[0].get("stat", {})
    so, pa = safe_num(stat.get("strikeOuts")), safe_num(stat.get("plateAppearances"))
    stat["calc_k_pct"] = so / pa * 100 if so is not None and pa else None
    return stat


def game_time_label(raw):
    if not raw:
        return "TBD"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        return dt.astimezone(timezone.utc).strftime("%b %d \u00b7 %I:%M %p UTC").replace(" 0", " ")
    except ValueError:
        return "TBD"


@st.cache_data(ttl=300, show_spinner=False)
def pitchers_for_date(selected_date: str):
    data = get_json(
        MLB_SCHEDULE_URL,
        params={"sportId": 1, "date": selected_date, "hydrate": "probablePitcher,team,opponents,venue"},
    )
    options = []
    for date_group in data.get("dates", []):
        for game in date_group.get("games", []):
            teams = game.get("teams", {})
            venue = game.get("venue", {})
            for side, opp_side in (("away", "home"), ("home", "away")):
                td, od = teams.get(side, {}), teams.get(opp_side, {})
                probable = td.get("probablePitcher")
                if not isinstance(probable, dict):
                    continue
                pid, pname = probable.get("id"), probable.get("fullName")
                if not pid or not pname:
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
                    "pitcher_name": pname,
                    "team": team.get("name", "N/A"),
                    "team_id": team.get("id"),
                    "team_side": side,
                    "opponent": opp.get("name", "N/A"),
                    "opponent_id": opp.get("id"),
                    "opponent_side": opp_side,
                    "throwing_hand": hand,
                    "venue": venue.get("name", "N/A"),
                    "status": game.get("status", {}).get("detailedState", "N/A"),
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
        (r.get("official", {}).get("fullName") for r in officials if r.get("officialType") == "Home Plate"),
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
    result = []
    for i, pid in enumerate(order, 1):
        p = players.get(f"ID{pid}", {})
        person = p.get("person", {})
        result.append({
            "#": i,
            "player_id": int(pid),
            "Hitter": person.get("fullName", f"Player {pid}"),
            "Bats": person.get("batSide", {}).get("code", "N/A"),
        })
    return result


# =========================================================
# BASEBALL SAVANT / STATCAST
# =========================================================

@st.cache_data(ttl=3600, show_spinner=False)
def statcast_data(player_id, start_dt, end_dt):
    try:
        df = statcast_pitcher(start_dt, end_dt, player_id)
        return df if isinstance(df, pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


def statcast_metrics(df):
    out = {
        "whiff_pct": None, "csw_pct": None, "arsenal": [],
        "vs_l": None, "vs_r": None, "home": None, "away": None,
        "fastball_velo": None,
    }
    if df.empty:
        return out

    desc = df["description"].astype(str) if "description" in df else pd.Series(index=df.index, dtype=str)
    swings = {"swinging_strike","swinging_strike_blocked","foul","foul_tip","hit_into_play","foul_bunt","missed_bunt"}
    whiffs = {"swinging_strike","swinging_strike_blocked","missed_bunt"}
    swing_n = desc.isin(swings).sum()
    whiff_n = desc.isin(whiffs).sum()
    called_n = desc.eq("called_strike").sum()
    out["whiff_pct"] = whiff_n / swing_n * 100 if swing_n else None
    out["csw_pct"] = (whiff_n + called_n) / len(df) * 100 if len(df) else None

    def k_rate(mask):
        if "events" not in df:
            return None
        pa = df[mask & df["events"].notna()]
        if pa.empty:
            return None
        k = pa["events"].astype(str).str.contains("strikeout", case=False, na=False).sum()
        return k / len(pa) * 100

    if "stand" in df:
        out["vs_l"] = k_rate(df["stand"].astype(str).eq("L"))
        out["vs_r"] = k_rate(df["stand"].astype(str).eq("R"))

    if {"inning_topbot","home_team","away_team"}.issubset(df.columns):
        out["home"] = k_rate(df["inning_topbot"].astype(str).str.lower().eq("top"))
        out["away"] = k_rate(df["inning_topbot"].astype(str).str.lower().eq("bot"))

    if "pitch_type" in df:
        rows = []
        for pitch, g in df.groupby("pitch_type"):
            gd = g["description"].astype(str) if "description" in g else pd.Series(index=g.index, dtype=str)
            sn, wn = gd.isin(swings).sum(), gd.isin(whiffs).sum()
            velo = pd.to_numeric(g["release_speed"], errors="coerce").mean() if "release_speed" in g else None
            rows.append({
                "Pitch": PITCH_NAMES.get(str(pitch), str(pitch)),
                "Code": str(pitch),
                "Usage%": len(g) / len(df) * 100,
                "Velo": float(velo) if pd.notna(velo) else None,
                "Whiff%": wn / sn * 100 if sn else None,
            })
        rows.sort(key=lambda r: r["Usage%"], reverse=True)
        out["arsenal"] = rows
        for code in ("FF","SI","FC"):
            row = next((r for r in rows if r["Code"] == code and r["Velo"] is not None), None)
            if row:
                out["fastball_velo"] = row["Velo"]
                break

    return out


@st.cache_data(ttl=86400, show_spinner=False)
def savant_park_factors(year: int):
    """
    Baseball Savant Statcast Park Factors.
    SO = 100 is neutral. We try Savant CSV output first, then the HTML table.
    """
    base_params = {
        "type": "year",
        "year": year,
        "condition": "All",
        "parks": "mlb",
        "rolling": 1,
        "stat": "index_wOBA",
    }
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) "
            "AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1"
        )
    }

    # Savant leaderboards commonly support csv=true.
    try:
        csv_params = {**base_params, "csv": "true"}
        r = requests.get(SAVANT_PARK_URL, params=csv_params, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        text = r.text.lstrip()
        if text and not text.lower().startswith("<!doctype") and not text.lower().startswith("<html"):
            df = pd.read_csv(StringIO(text))
            df.columns = [str(c).strip() for c in df.columns]
            if "Venue" in df.columns and "SO" in df.columns:
                return df
    except Exception:
        pass

    # HTML fallback.
    try:
        r = requests.get(SAVANT_PARK_URL, params=base_params, headers=headers, timeout=TIMEOUT)
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        for table in tables:
            # Flatten MultiIndex columns when necessary.
            if isinstance(table.columns, pd.MultiIndex):
                flat = []
                for col in table.columns:
                    parts = [str(x).strip() for x in col if str(x).strip() and str(x) != "nan"]
                    flat.append(parts[-1] if parts else "")
                table.columns = flat
            else:
                table.columns = [str(c).strip() for c in table.columns]

            if "Venue" in table.columns and "SO" in table.columns:
                return table
    except Exception:
        pass

    return pd.DataFrame()


def park_so_factor(venue: str, year: int):
    df = savant_park_factors(year)
    if df.empty or not venue:
        return None
    target = normalize_name(venue)
    venue_norm = df["Venue"].astype(str).map(normalize_name)
    matches = df[venue_norm.eq(target)]
    if matches.empty:
        # relaxed match for renamed / formatted venues
        matches = df[venue_norm.map(lambda x: target in x or x in target)]
    if matches.empty:
        return None
    return safe_num(matches.iloc[0].get("SO"))


# =========================================================
# FANGRAPHS + BASEBALL REFERENCE
# =========================================================

@st.cache_data(ttl=21600, show_spinner=False)
def fangraphs_pitchers(season: int):
    """
    pybaseball season leaderboard backed by FanGraphs.
    FanGraphs can occasionally return HTTP 403 to automated requests; we expose
    that state instead of silently pretending the value exists.
    """
    try:
        try:
            df = pitching_stats(season, season, qual=0)
        except TypeError:
            df = pitching_stats(season, season)
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df, "OK"
        return pd.DataFrame(), "EMPTY"
    except Exception as exc:
        msg = str(exc)
        return pd.DataFrame(), ("BLOCKED_403" if "403" in msg else f"ERROR: {msg[:90]}")


@st.cache_data(ttl=21600, show_spinner=False)
def bref_pitchers(season: int):
    try:
        return pitching_stats_bref(season)
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=86400, show_spinner=False)
def player_crosswalk(name: str):
    """
    Resolve MLBAM -> FanGraphs / Baseball-Reference identifiers via Chadwick data
    exposed by pybaseball.
    """
    parts = str(name).strip().split()
    if len(parts) < 2:
        return {}
    first = " ".join(parts[:-1])
    last = parts[-1]
    try:
        df = playerid_lookup(last, first, fuzzy=True)
        if isinstance(df, pd.DataFrame) and not df.empty:
            # Prefer exact normalized full-name match when possible.
            target = normalize_name(name)
            for _, row in df.iterrows():
                full = f"{row.get('name_first','')} {row.get('name_last','')}"
                if normalize_name(full) == target:
                    return row.to_dict()
            return df.iloc[0].to_dict()
    except Exception:
        pass
    return {}


def match_pitcher_row(df: pd.DataFrame, name: str, mlbam_id: int | None = None, crosswalk: dict | None = None):
    if df.empty:
        return {}

    crosswalk = crosswalk or {}

    # Direct MLBAM columns (common for Baseball-Reference outputs).
    if mlbam_id is not None:
        for col in ("mlbID", "MLBAMID", "mlb_ID", "key_mlbam"):
            if col in df.columns:
                ids = pd.to_numeric(df[col], errors="coerce")
                hit = df[ids.eq(int(mlbam_id))]
                if not hit.empty:
                    return hit.iloc[0].to_dict()

    # FanGraphs ID match.
    fg_id = safe_num(crosswalk.get("key_fangraphs"))
    if fg_id is not None:
        for col in ("IDfg", "key_fangraphs"):
            if col in df.columns:
                ids = pd.to_numeric(df[col], errors="coerce")
                hit = df[ids.eq(int(fg_id))]
                if not hit.empty:
                    return hit.iloc[0].to_dict()

    # Baseball-Reference ID match.
    br_id = str(crosswalk.get("key_bbref") or "").strip()
    if br_id:
        for col in ("player_ID", "key_bbref"):
            if col in df.columns:
                hit = df[df[col].astype(str).str.strip().eq(br_id)]
                if not hit.empty:
                    return hit.iloc[0].to_dict()

    # Exact normalized name.
    name_col = next((c for c in ("Name", "NameASCII", "name_common") if c in df.columns), None)
    if name_col:
        target = normalize_name(name)
        hit = df[df[name_col].astype(str).map(normalize_name).eq(target)]
        if not hit.empty:
            return hit.iloc[0].to_dict()

    return {}


def source_validation(mlb, fg, br):
    result = []
    for label, row, mapping in (
        ("MLB", mlb, {"ERA":"era","WHIP":"whip","K/9":"strikeoutsPer9Inn"}),
        ("FanGraphs", fg, {"ERA":"ERA","WHIP":"WHIP","K/9":"K/9","FIP":"FIP","xFIP":"xFIP","SIERA":"SIERA","SwStr%":"SwStr%","CSW%":"CSW%"}),
        ("Baseball-Reference", br, {"ERA":"ERA","WHIP":"WHIP","K/9":"SO9","SO/W":"SO/W"}),
    ):
        for stat_name, key in mapping.items():
            value = row.get(key) if isinstance(row, dict) else None
            if value is not None and str(value) != "nan":
                result.append({"Source": label, "Stat": stat_name, "Value": value})
    return pd.DataFrame(result)


# =========================================================
# LINEUP
# =========================================================

def hitter_k_profile(pid: int, season: int, pitcher_hand: str):
    sit = "vl" if pitcher_hand.lower().startswith("left") else "vr"
    stat = {}
    source = "MLB split"
    try:
        stat = person_stats(pid, season, "hitting", sit)
    except Exception:
        pass
    if not stat:
        source = "MLB season"
        try:
            stat = person_stats(pid, season, "hitting")
        except Exception:
            stat = {}
    so, pa = safe_num(stat.get("strikeOuts")), safe_num(stat.get("plateAppearances"))
    return {
        "K% vs hand": so / pa * 100 if so is not None and pa else None,
        "PA": pa,
        "Source": source if stat else "N/A",
    }


def enrich_lineup(lineup, season, pitcher_hand):
    out = []
    for row in lineup:
        out.append({**row, **hitter_k_profile(row["player_id"], season, pitcher_hand)})
    return out


def lineup_k_pct(lineup):
    vals = [safe_num(r.get("K% vs hand")) for r in lineup]
    vals = [v for v in vals if v is not None]
    return sum(vals) / len(vals) if vals else None


# =========================================================
# RECENT FORM + MODEL
# =========================================================

def recent_summary(log):
    rows = log[:5]
    if not rows:
        return {}
    def avg(key):
        vals = [safe_num(r.get(key)) for r in rows]
        vals = [v for v in vals if v is not None]
        return sum(vals)/len(vals) if vals else None
    ips = [innings_to_decimal(r.get("IP")) for r in rows]
    ips = [v for v in ips if v is not None]
    pitches = [safe_num(r.get("Pitches")) for r in rows]
    pitches = [v for v in pitches if v is not None]
    return {
        "avg_k": avg("K"),
        "avg_bf": avg("BF"),
        "avg_ip": sum(ips)/len(ips) if ips else None,
        "avg_pitches": sum(pitches)/len(pitches) if pitches else None,
        "max_pitches": max(pitches) if pitches else None,
    }


def build_projection(mlb, fg, sm, team_general, team_split, lineup, recent, park_so, context):
    # M1 ability: MLB + FanGraphs/Statcast validation
    season_k = safe_num(mlb.get("calc_k_pct"))
    fg_k = safe_num(fg.get("K%"))
    if fg_k is not None and fg_k <= 1:
        fg_k *= 100

    ability_k = weighted_average([
        (season_k, .55),
        (fg_k, .25),
        (sm.get("vs_l"), .10),
        (sm.get("vs_r"), .10),
    ], default=season_k or fg_k or 22.0)

    # M2 leash
    bf = weighted_average([
        (mlb.get("calc_bf_start"), .60),
        (recent.get("avg_bf"), .40),
    ], default=22.0)
    bf = clamp(bf, 14, 32)

    # M4 opponent / M8 lineup
    opp_general = safe_num(team_general.get("calc_k_pct"))
    opp_split = safe_num(team_split.get("calc_k_pct"))
    lineup_rate = lineup_k_pct(lineup)

    matchup_k = weighted_average([
        (ability_k, .40),
        (opp_split, .25),
        (lineup_rate, .20),
        (opp_general, .15),
    ], default=ability_k)

    # M5 arsenal adjustment (conservative)
    whiff = safe_num(sm.get("whiff_pct"))
    csw = safe_num(sm.get("csw_pct"))
    arsenal_adj = 0.0
    if whiff is not None:
        arsenal_adj += clamp((whiff - 24) * .08, -1.2, 1.2)
    if csw is not None:
        arsenal_adj += clamp((csw - 27) * .10, -1.0, 1.0)

    # M6 recent form
    recent_k = safe_num(recent.get("avg_k"))
    season_k_start = safe_num(mlb.get("calc_k_start"))
    form_adj = clamp((recent_k - season_k_start) * .25, -.6, .6) if recent_k is not None and season_k_start is not None else 0.0

    # M7 park factor: SO factor 100 = neutral. Keep bounded.
    park_adj_pct = clamp(((park_so or 100) - 100) * .06, -1.2, 1.2)

    # Weather / umpire shown but not scored until validated.
    final_k_pct = clamp(matchup_k + arsenal_adj + park_adj_pct, 8, 45)
    central = clamp(bf * final_k_pct / 100 + form_adj, .5, 14)

    sigma = max(1.35, math.sqrt(central) * .80)
    return {
        "central": central,
        "low": max(0, central - 1.35*sigma),
        "high": central + 1.35*sigma,
        "bf": bf,
        "k_pct": final_k_pct,
        "ability_k": ability_k,
        "matchup_k": matchup_k,
        "arsenal_adj": arsenal_adj,
        "park_adj": park_adj_pct,
        "form_adj": form_adj,
    }


def grade_bet(ev, edge_pp, completeness):
    if ev is None or edge_pp is None:
        return "NO BET"
    if completeness < .70:
        return "PASS"
    ev_pct = ev * 100
    if ev_pct >= 10 and edge_pp >= 6:
        return "A"
    if ev_pct >= 6 and edge_pp >= 4:
        return "B"
    if ev_pct >= 3 and edge_pp >= 2:
        return "C"
    return "PASS"


# =========================================================
# APP
# =========================================================

st.markdown(
    """
    <div class="hero">
      <h1 style="margin:0">\u26be MLB Starting Pitcher K Model</h1>
      <div class="muted">V1.2 \u00b7 MLB + Baseball Savant + FanGraphs + Baseball-Reference \u00b7 DraftKings + FanDuel</div>
    </div>
    """,
    unsafe_allow_html=True,
)

top1, top2 = st.columns([1, 2])
with top1:
    game_date = st.date_input("Fecha", value=date.today(), min_value=date(2008,1,1))

try:
    options = pitchers_for_date(game_date.isoformat())
except Exception as exc:
    st.error(f"No se pudo cargar el calendario MLB: {exc}")
    st.stop()

if not options:
    st.warning("MLB todav\u00eda no tiene abridores probables disponibles para esta fecha.")
    st.stop()

by_id = {p["selection_id"]: p for p in options}
with top2:
    selected_id = st.selectbox(
        "Abridor",
        list(by_id),
        format_func=lambda x: f"{by_id[x]['pitcher_name']} \u00b7 {by_id[x]['team']} vs {by_id[x]['opponent']}",
    )

p = by_id[selected_id]

st.markdown(
    f"""
    <div class="hero">
      <h2 style="margin:0">{p['pitcher_name']} <span style="opacity:.55">vs {p['opponent']}</span></h2>
      <div class="muted">{p['team']} \u00b7 {p['throwing_hand']} \u00b7 {p['venue']} \u00b7 {p['game_time']} \u00b7 {p['status']}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.spinner("Cargando fuentes..."):
    try:
        mlb = pitcher_season_stats(p["pitcher_id"], game_date.year)
    except Exception:
        mlb = {}

    try:
        log = pitcher_game_log(p["pitcher_id"], game_date.year)
    except Exception:
        log = []

    try:
        team_general = team_hitting_stats(p["opponent_id"], game_date.year)
    except Exception:
        team_general = {}

    sit = "vl" if p["throwing_hand"].lower().startswith("left") else "vr"
    try:
        team_split = team_hitting_stats(p["opponent_id"], game_date.year, sit)
    except Exception:
        team_split = {}

    try:
        feed = game_feed(p["game_pk"])
    except Exception:
        feed = {}

    sc_df = statcast_data(p["pitcher_id"], f"{game_date.year}-03-01", game_date.isoformat())
    sm = statcast_metrics(sc_df)

    fg_df, fg_status = fangraphs_pitchers(game_date.year)
    br_df = bref_pitchers(game_date.year)
    crosswalk = player_crosswalk(p["pitcher_name"])
    fg = match_pitcher_row(fg_df, p["pitcher_name"], p["pitcher_id"], crosswalk)
    br = match_pitcher_row(br_df, p["pitcher_name"], p["pitcher_id"], crosswalk)

    park_so = park_so_factor(p["venue"], game_date.year)
    context = game_context(feed)

    lineup_raw = confirmed_lineup(feed, p["opponent_side"])
    lineup = enrich_lineup(lineup_raw, game_date.year, p["throwing_hand"]) if lineup_raw else []
    recent = recent_summary(log)

proj = build_projection(mlb, fg, sm, team_general, team_split, lineup, recent, park_so, context)

# MAIN SUMMARY
s1, s2, s3, s4, s5 = st.columns(5)
s1.metric("Proyecci\u00f3n", fmt(proj["central"], 2, " K"))
s2.metric("Rango", f"{proj['low']:.1f}\u2013{proj['high']:.1f}")
s3.metric("Projected BF", fmt(proj["bf"], 1))
s4.metric("Projected K%", fmt(proj["k_pct"], 1, "%"))
s5.metric("SO Park Factor", fmt(park_so, 0))

# MODULES
with st.expander("M1 \u00b7 Capacidad real de K \u2014 20%", expanded=True):
    a,b,c,d = st.columns(4)
    a.metric("K%", fmt(mlb.get("calc_k_pct"),1,"%"))
    a.metric("K-BB%", fmt(mlb.get("calc_k_minus_bb"),1,"%"))
    b.metric("K/9", mlb.get("strikeoutsPer9Inn","N/A"))
    b.metric("WHIP", mlb.get("whip","N/A"))
    c.metric("Whiff%", fmt(sm.get("whiff_pct"),1,"%"))
    c.metric("CSW%", fmt(sm.get("csw_pct"),1,"%"))
    d.metric("Fastball Velo", fmt(sm.get("fastball_velo"),1," mph"))
    fg_k_display = (safe_num(fg.get("K%"))*100 if safe_num(fg.get("K%")) is not None and safe_num(fg.get("K%")) <= 1 else fg.get("K%"))
    d.metric("FanGraphs K%", fmt(fg_k_display, 1, "%"))
    if fg_status != "OK":
        d.caption("FanGraphs source: " + fg_status)

with st.expander("M2 \u00b7 Volumen / Leash \u2014 20%"):
    a,b,c,d = st.columns(4)
    a.metric("GS", mlb.get("gamesStarted","N/A"))
    a.metric("IP/start", fmt(mlb.get("calc_ip_start"),2))
    b.metric("BF/start", fmt(mlb.get("calc_bf_start"),1))
    b.metric("Avg BF L5", fmt(recent.get("avg_bf"),1))
    c.metric("Avg Pitches L5", fmt(recent.get("avg_pitches"),1))
    c.metric("Max Pitches L5", fmt(recent.get("max_pitches"),0))
    d.metric("Avg K L5", fmt(recent.get("avg_k"),1))
    d.metric("Avg IP L5", fmt(recent.get("avg_ip"),2))

with st.expander("M3 \u00b7 Splits \u2014 10%"):
    a,b,c,d = st.columns(4)
    a.metric("K% vs LHB", fmt(sm.get("vs_l"),1,"%"))
    b.metric("K% vs RHB", fmt(sm.get("vs_r"),1,"%"))
    c.metric("K% Home", fmt(sm.get("home"),1,"%"))
    d.metric("K% Away", fmt(sm.get("away"),1,"%"))

with st.expander("M4 \u00b7 Propensi\u00f3n del rival a poncharse \u2014 20%"):
    a,b,c = st.columns(3)
    a.metric("Opponent K%", fmt(team_general.get("calc_k_pct"),1,"%"))
    b.metric("Opponent K% vs hand", fmt(team_split.get("calc_k_pct"),1,"%"))
    c.metric("Confirmed lineup K%", fmt(lineup_k_pct(lineup),1,"%"))

with st.expander("M5 \u00b7 Arsenal vs Matchup \u2014 15%"):
    if sm["arsenal"]:
        arsenal = pd.DataFrame(sm["arsenal"])
        for col in ("Usage%","Velo","Whiff%"):
            arsenal[col] = pd.to_numeric(arsenal[col], errors="coerce").round(1)
        st.dataframe(arsenal[["Pitch","Usage%","Velo","Whiff%"]], hide_index=True, use_container_width=True)
    else:
        st.info("Statcast no devolvi\u00f3 arsenal.")

with st.expander("M6 \u00b7 Forma / Cambios recientes \u2014 5%"):
    if log:
        st.dataframe(pd.DataFrame(log[:5]), hide_index=True, use_container_width=True)
    else:
        st.info("Sin game log reciente.")

with st.expander("M7 \u00b7 Contexto \u2014 5%"):
    a,b,c,d = st.columns(4)
    a.metric("SO Park Factor", fmt(park_so,0))
    b.metric("Temp", f"{context['temperature']}\u00b0F" if context.get("temperature") is not None else "N/A")
    c.metric("Weather", context.get("condition") or "N/A")
    d.metric("Umpire", context.get("umpire") or "N/A")
    if park_so is None:
        st.warning("Savant SO Park Factor no pudo cargarse en este intento. El modelo lo trata como neutral y NO inventa un valor.")
    st.caption("Savant SO Park Factor: 100 = neutral. Weather y umpire se muestran, pero no reciben peso fuerte hasta validacion.")

with st.expander("M8 \u00b7 Lineup confirmado \u2014 5%"):
    if lineup:
        view = pd.DataFrame(lineup)
        view["K% vs hand"] = pd.to_numeric(view["K% vs hand"], errors="coerce").round(1)
        st.dataframe(view[["#","Hitter","Bats","K% vs hand","PA","Source"]], hide_index=True, use_container_width=True)
    else:
        st.info("MLB todav\u00eda no public\u00f3 el batting order confirmado.")

with st.expander("Fuentes de validaci\u00f3n \u00b7 FanGraphs / Baseball-Reference"):
    validation = source_validation(mlb, fg, br)
    if not validation.empty:
        st.dataframe(validation, hide_index=True, use_container_width=True)
    else:
        st.info("FanGraphs/Baseball-Reference no devolvieron una fila coincidente para este pitcher.")

# MARKET
st.header("M9 \u00b7 Mercado \u00b7 DraftKings + FanDuel")
st.caption("Introduce las l\u00edneas/odds actuales. La app compara autom\u00e1ticamente qu\u00e9 book ofrece mejor EV.")

books = {}
for book in ("DraftKings","FanDuel"):
    st.subheader(book)
    x,y,z = st.columns(3)
    with x:
        line = st.number_input(f"{book} \u00b7 L\u00ednea K", .5, 15.5, 5.5, 1.0, key=f"{book}_line")
    with y:
        over = st.number_input(f"{book} \u00b7 Over odds", -1000, 2000, -110, 5, key=f"{book}_over")
    with z:
        under = st.number_input(f"{book} \u00b7 Under odds", -1000, 2000, -110, 5, key=f"{book}_under")
    books[book] = {"line": line, "over": over, "under": under}

market_rows = []
for book, q in books.items():
    line = q["line"]
    over_threshold = math.floor(line) + 1
    under_max = math.floor(line)
    p_over = poisson_ge(over_threshold, proj["central"])
    p_under = poisson_cdf(under_max, proj["central"])

    for side, prob, odds in (("OVER",p_over,q["over"]),("UNDER",p_under,q["under"])):
        implied = american_implied(odds)
        edge = prob*100 - implied if implied is not None else None
        ev = ev_per_dollar(prob, odds)
        market_rows.append({
            "Book": book,
            "Side": side,
            "Line": line,
            "Odds": odds,
            "Model%": prob*100,
            "Fair": fair_american(prob),
            "Edge pp": edge,
            "EV%": ev*100 if ev is not None else None,
        })

market_df = pd.DataFrame(market_rows)
for c in ("Model%","Fair","Edge pp","EV%"):
    market_df[c] = pd.to_numeric(market_df[c], errors="coerce").round(1)

st.dataframe(market_df, hide_index=True, use_container_width=True)

# DISTRIBUTION
st.header("Distribuci\u00f3n P(4+\u20269+)")
dist = []
for k in range(4,10):
    prob = poisson_ge(k, proj["central"])
    dist.append({"K":f"{k}+","Probability":f"{prob*100:.1f}%","Fair Odds":f"{fair_american(prob):+.0f}"})
st.dataframe(pd.DataFrame(dist), hide_index=True, use_container_width=True)

# COMPLETENESS + BEST BET
status = {
    "M1": bool(mlb and not sc_df.empty),
    "M2": bool(mlb and log),
    "M3": bool(not sc_df.empty),
    "M4": bool(team_general),
    "M5": bool(sm["arsenal"]),
    "M6": bool(log),
    "M7": park_so is not None,
    "M8": bool(lineup),
    "M9": True,
}
completeness = sum(status.values())/9

best = market_df.sort_values("EV%", ascending=False).iloc[0].to_dict()
grade = grade_bet(safe_num(best.get("EV%"))/100 if safe_num(best.get("EV%")) is not None else None, safe_num(best.get("Edge pp")), completeness)

st.markdown(
    f"""
    <div class="bet-card">
      <h2 style="margin-top:0">Mejor precio actual</h2>
      <h1 style="margin:.2rem 0">{best['Side']} {best['Line']} K \u00b7 {best['Book']} {int(best['Odds']):+d}</h1>
      <div>Modelo <b>{best['Model%']:.1f}%</b> \u00b7 Fair <b>{best['Fair']:+.0f}</b> \u00b7 Edge <b>{best['Edge pp']:.1f} pp</b> \u00b7 EV <b>{best['EV%']:.1f}%</b> \u00b7 Grade <b>{grade}</b></div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.subheader("Estado de datos")
st.progress(completeness)
st.write(f"**{sum(status.values())}/9 m\u00f3dulos con datos disponibles**")
st.caption(" \u00b7 ".join(f"{k} {'\u2705' if v else '\u23f3'}" for k,v in status.items()))

st.info(
    "V1.2 corrige encoding, fortalece Savant Park Factor y mejora el matching de FanGraphs/Baseball-Reference. "
    "La siguiente fase ya no requiere reconstruir modulos: se enfoca en backtesting/calibracion de pesos y thresholds."
)
