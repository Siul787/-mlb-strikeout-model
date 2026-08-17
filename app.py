# -*- coding: utf-8 -*-
from __future__ import annotations

from datetime import date, datetime, timezone, timedelta
from zoneinfo import ZoneInfo
from io import StringIO
from pathlib import Path
import json
import math
import re
import unicodedata

import pandas as pd
import requests
import streamlit as st
from bs4 import BeautifulSoup

try:
    from openai import OpenAI
except Exception:
    OpenAI = None
from pybaseball import (
    statcast,
    statcast_pitcher,
    statcast_batter,
    statcast_batter_pitch_arsenal,
    pitching_stats,
    pitching_stats_bref,
    playerid_lookup,
)

# ============================================================
# MODEL PROFESSIONAL MLB - STARTING PITCHER STRIKEOUTS
# V3.2.15 LIVE BOARD VALIDATION
# ============================================================

MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game"
SAVANT_PARK_URL = "https://baseballsavant.mlb.com/leaderboard/statcast-park-factors"
ACTION_PITCHING_PROPS_URL = "https://www.actionnetwork.com/mlb/props/pitching"
TIMEOUT = 30

# M8 sample-size protection: individual hitter K% is regressed toward
# the opponent team K% vs the pitcher hand so tiny samples cannot dominate.
LINEUP_K_PRIOR_PA = 100.0
LINEUP_K_DEFAULT_PRIOR = 22.0

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
      display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:9px;margin:12px 0 18px;align-items:stretch
    }
    .board-game{
      background:linear-gradient(150deg,rgba(23,27,37,.98),rgba(13,15,21,.99));
      border:1px solid rgba(130,145,180,.18);border-radius:14px;padding:10px;
      height:168px;box-sizing:border-box;box-shadow:0 5px 16px rgba(0,0,0,.14);
      display:grid;grid-template-rows:30px 43px 43px 36px;overflow:hidden
    }
    .board-time{font-size:.62rem;opacity:.62;text-align:center;margin:0;line-height:1.2;display:flex;align-items:center;justify-content:center;overflow:hidden;white-space:normal}
    .board-team{
      display:grid;grid-template-columns:26px 1fr 24px;align-items:center;gap:6px;
      padding:4px 0;border-bottom:1px solid rgba(140,150,175,.08);min-height:0
    }
    .board-team:last-of-type{border-bottom:0}
    .board-logo{width:25px;height:25px;object-fit:contain}
    .board-abbr{font-size:.78rem;font-weight:850}
    .board-pitcher{font-size:.62rem;opacity:.62;white-space:nowrap;overflow:hidden;text-overflow:ellipsis}
    .board-pitcher-link{color:inherit!important;text-decoration:none!important;border-bottom:1px dotted rgba(190,205,235,.35)}
    .board-pitcher-link:hover{color:#dce7ff!important;border-bottom-color:#9fc1ff}
    .board-score{font-size:.95rem;font-weight:950;text-align:right;line-height:1;min-width:18px;color:#f2f5fb}
    .board-actions{display:grid;grid-template-columns:1fr 1fr;gap:5px;padding-top:4px;min-height:0}
    .board-metric{padding:5px 4px;border-radius:8px;font-size:.56rem;font-weight:850;text-align:center;line-height:1.35;background:rgba(63,116,220,.10);border:1px solid rgba(79,140,255,.18);min-height:34px;display:flex;flex-direction:column;align-items:center;justify-content:center;box-sizing:border-box}
    .board-metric .proj{color:#9fc1ff}
    .board-metric .actual{color:#bff8df}
    .board-metric .finalk{color:#f7e29a}
    .board-metric .muted{opacity:.38}
    .board-legend{font-size:.70rem;opacity:.55;margin-top:-8px;margin-bottom:12px}
    .board-live{margin:-1px -1px 7px;padding:5px 7px;border-radius:8px;font-size:.62rem;font-weight:900;text-align:center;letter-spacing:.02em;color:#bff8df;background:rgba(56,217,150,.12);border:1px solid rgba(56,217,150,.22)}
    .board-final{color:#dfe6f5;background:rgba(150,160,185,.10);border-color:rgba(150,160,185,.18)}
    .board-track{font-size:.58rem;line-height:1.35;margin-top:4px;color:#9fc0ff;font-weight:750;white-space:normal}
    .board-track-final{color:#bff8df}
    @media (max-width:1100px){.board-grid{grid-template-columns:repeat(4,minmax(0,1fr))}}
    @media (max-width:760px){
      .board-grid{grid-template-columns:repeat(2,minmax(0,1fr));gap:7px}
      .board-game{padding:7px;height:160px;border-radius:11px;grid-template-rows:28px 40px 40px 34px}
      .board-logo{width:21px;height:21px}
      .board-team{grid-template-columns:22px 1fr auto;gap:4px}
      .board-abbr{font-size:.69rem}
      .board-pitcher,.board-time{font-size:.54rem}
      .board-score{font-size:.82rem}
      .board-metric{font-size:.50rem}
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
    """MLB gameDate is UTC; display the slate in U.S. Eastern Time.

    America/New_York handles EDT/EST automatically, so the board stays correct
    across daylight-saving changes.
    """
    if not raw:
        return "TBD"
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        et = dt.astimezone(ZoneInfo("America/New_York"))
        tz_label = et.tzname() or "ET"
        return et.strftime(f"%b %d · %I:%M %p {tz_label}").replace(" 0", " ")
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
                    "game_date_raw": game.get("gameDate"),
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


@st.cache_data(ttl=300, show_spinner=False)
def team_roster_player_ids(team_id: int, season: int, roster_date: str):
    """Return the official MLB roster IDs for the expected opponent team.

    M8 is fail-closed: if we cannot independently verify the team roster,
    the lineup is not allowed into the projection.
    """
    if not team_id:
        return set(), "NO_TEAM_ID"

    # Active is the preferred source. 40-man is a conservative fallback for
    # edge cases around same-day activations while still preventing cross-team mixes.
    for roster_type in ("active", "40Man"):
        try:
            d = get_json(
                f"{MLB_TEAMS_URL}/{int(team_id)}/roster",
                params={
                    "rosterType": roster_type,
                    "season": int(season),
                    "date": str(roster_date),
                },
            )
            ids = {
                int(r.get("person", {}).get("id"))
                for r in (d.get("roster", []) or [])
                if r.get("person", {}).get("id")
            }
            if len(ids) >= 9:
                return ids, f"MLB_{roster_type.upper()}_ROSTER"
        except Exception:
            continue

    return set(), "ROSTER_UNAVAILABLE"


def confirmed_lineup(feed, side, expected_team_id, season, roster_date):
    """Read and validate the nine-man batting order for the expected opponent.

    Guard rails:
    1) the boxscore side must be the expected MLB team,
    2) battingOrder must contain exactly nine distinct players,
    3) every player must exist in that side's boxscore player dictionary,
    4) every player must also appear on MLB's roster for the expected team/date.

    Any failure returns an empty lineup so M8 cannot contaminate M3/M8/projection.
    """
    teams_box = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
    box = teams_box.get(side, {}) if isinstance(teams_box, dict) else {}

    # Cross-check team identity in two independent parts of the game feed.
    box_team_id = safe_num((box.get("team") or {}).get("id"))
    game_team_id = safe_num(
        (((feed.get("gameData", {}) or {}).get("teams", {}) or {}).get(side, {}) or {}).get("id")
    )
    expected = safe_num(expected_team_id)

    observed_ids = [x for x in (box_team_id, game_team_id) if x is not None]
    if expected is None:
        return [], "LINEUP GUARD: expected opponent Team ID missing"
    if observed_ids and any(int(x) != int(expected) for x in observed_ids):
        return [], f"LINEUP GUARD: team mismatch (expected {int(expected)})"
    if not observed_ids:
        return [], "LINEUP GUARD: game feed team identity unavailable"

    order = box.get("battingOrder", []) or []
    players = box.get("players", {}) or {}

    try:
        order_ids = [int(pid) for pid in order]
    except Exception:
        return [], "LINEUP GUARD: invalid battingOrder IDs"

    if len(order_ids) != 9:
        return [], f"LINEUP NOT CONFIRMED: expected 9 hitters, found {len(order_ids)}"
    if len(set(order_ids)) != 9:
        return [], "LINEUP GUARD: duplicate hitter IDs detected"

    roster_ids, roster_source = team_roster_player_ids(
        int(expected), int(season), str(roster_date)
    )
    if not roster_ids:
        return [], f"LINEUP GUARD: official opponent roster unavailable ({roster_source})"

    rows = []
    invalid = []
    for i, pid in enumerate(order_ids, 1):
        player_obj = players.get(f"ID{pid}", {}) or {}
        person = player_obj.get("person", {}) or {}
        name = person.get("fullName")

        if not name:
            invalid.append(f"ID{pid}: missing from {side} boxscore players")
            continue
        if pid not in roster_ids:
            invalid.append(f"{name} ({pid}): not on expected team roster")
            continue

        rows.append({
            "#": i,
            "player_id": pid,
            "Hitter": name,
            "Bats": (person.get("batSide") or {}).get("code", "N/A"),
        })

    if invalid or len(rows) != 9:
        detail = "; ".join(invalid[:3])
        if len(invalid) > 3:
            detail += f"; +{len(invalid)-3} more"
        return [], f"LINEUP GUARD REJECTED: {detail or 'validation incomplete'}"

    return rows, f"VERIFIED · Team {int(expected)} · 9/9 · {roster_source}"



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

@st.cache_data(ttl=21600, show_spinner=False)
def savant_expected_pitch_type_html(team_id: int, season: int):
    """Second Savant path for xBA/xSLG/xwOBA when the CSV export omits them."""
    frames=[]
    for code in ("FF","SI","FC","SL","ST","CU","KC","CH","FS"):
        url=("https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
             f"?type=batter&pitchType={code}&year={season}&team={team_id}&min=1")
        try:
            r=requests.get(url,headers={"User-Agent":"Mozilla/5.0"},timeout=TIMEOUT)
            r.raise_for_status()
            tables=pd.read_html(StringIO(r.text))
        except Exception:
            continue
        for raw in tables:
            t=_flatten_columns(raw)
            cols=list(t.columns)
            xba=_metric_col(cols,["xba","xBA"],["xba","expected batting average"])
            xslg=_metric_col(cols,["xslg","xSLG"],["xslg","expected slugging"])
            xw=_metric_col(cols,["xwoba","xwOBA"],["xwoba","expected weighted"])
            pa=_metric_col(cols,["pa","PA"],["plate appearances"])
            if not (xba and xslg and xw):
                continue
            w=pd.to_numeric(t[pa],errors="coerce").fillna(1) if pa else pd.Series(1,index=t.index,dtype=float)
            def avg(c):
                v=pd.to_numeric(t[c],errors="coerce"); good=v.notna() & (w>0)
                return (v[good]*w[good]).sum()/w[good].sum() if good.any() else None
            frames.append({"Pitch":PITCH_NAMES.get(code,code),"Code":code,"xBA":avg(xba),"xSLG":avg(xslg),"xwOBA":avg(xw)})
            break
    return pd.DataFrame(frames)


@st.cache_data(ttl=21600, show_spinner=False)
def savant_pitch_type_detail_csv(team_id: int, season: int):
    """
    Robust Savant fallback for M5 expected stats.
    Query each pitch family separately with csv=true, then aggregate the rows
    returned for the selected opponent team. This route exposes xBA/xSLG/xwOBA
    and RV/100 on Savant even when the all-pitches aggregate omits them.
    """
    pitch_codes=("FF","SI","FC","SL","ST","CU","KC","CH","FS","FO","SV","KN")
    rows=[]

    def metric_col(columns, names):
        norm={c:normalize_name(c) for c in columns}
        wanted=[normalize_name(x) for x in names]
        # exact first
        for c,n in norm.items():
            if n in wanted:
                return c
        # then tolerant containment
        for c,n in norm.items():
            if any(w in n or n in w for w in wanted if w):
                return c
        return None

    for pitch_code in pitch_codes:
        url=(
            "https://baseballsavant.mlb.com/leaderboard/pitch-arsenal-stats"
            f"?type=batter&pitchType={pitch_code}&year={season}"
            f"&team={team_id}&min=1&minPitches=1&csv=true"
        )
        try:
            r=requests.get(
                url,
                headers={
                    "User-Agent":"Mozilla/5.0",
                    "Accept":"text/csv,text/plain,*/*",
                    "Referer":"https://baseballsavant.mlb.com/",
                },
                timeout=TIMEOUT,
            )
            r.raise_for_status()
            df=pd.read_csv(StringIO(r.text))
        except Exception:
            continue

        if df.empty:
            continue

        df=_flatten_columns(df)
        cols=list(df.columns)

        pitches_col=metric_col(cols,["pitches","pitch_count","total pitches"])
        pa_col=metric_col(cols,["pa","plate appearances","plate_appearances"])
        rv_col=metric_col(cols,["run value","run_value","runs"])
        rv100_col=metric_col(cols,["rv/100","rv100","run value / 100 pitches","run_value_per_100"])
        ba_col=metric_col(cols,["ba","batting average"])
        slg_col=metric_col(cols,["slg","slugging"])
        woba_col=metric_col(cols,["woba"])
        whiff_col=metric_col(cols,["whiff %","whiff%","whiff_percent","whiff rate"])
        k_col=metric_col(cols,["k%","k_percent","strikeout %","strikeout rate"])
        putaway_col=metric_col(cols,["put away %","putaway%","put_away_percent"])
        xba_col=metric_col(cols,["xba","estimated ba","estimated_ba"])
        xslg_col=metric_col(cols,["xslg","estimated slg","estimated_slg"])
        xwoba_col=metric_col(cols,["xwoba","estimated woba","estimated_woba"])
        hh_col=metric_col(cols,["hard hit %","hardhit%","hard_hit_percent"])

        if pitches_col is None:
            continue

        wp=pd.to_numeric(df[pitches_col],errors="coerce").fillna(0)
        wpa=pd.to_numeric(df[pa_col],errors="coerce").fillna(0) if pa_col else wp

        def weighted(col, weights):
            if col is None:
                return None
            vals=pd.to_numeric(df[col],errors="coerce")
            good=vals.notna() & weights.notna() & (weights>0)
            if not good.any():
                return None
            return float((vals[good]*weights[good]).sum()/weights[good].sum())

        run_value=None
        if rv_col:
            vals=pd.to_numeric(df[rv_col],errors="coerce")
            if vals.notna().any():
                run_value=float(vals.sum())

        rv100=weighted(rv100_col,wp)
        total_pitches=float(wp.sum()) if wp.notna().any() else None
        if rv100 is None and run_value is not None and total_pitches:
            rv100=run_value/total_pitches*100

        rows.append({
            "Pitch":PITCH_NAMES.get(pitch_code,pitch_code),
            "Code":pitch_code,
            "Pitches":total_pitches,
            "PA":float(wpa.sum()) if pa_col else None,
            "BA":weighted(ba_col,wpa),
            "SLG":weighted(slg_col,wpa),
            "wOBA":weighted(woba_col,wpa),
            "Whiff%":weighted(whiff_col,wp),
            "K%":weighted(k_col,wpa),
            "PutAway%":weighted(putaway_col,wp),
            "xBA":weighted(xba_col,wpa),
            "xSLG":weighted(xslg_col,wpa),
            "xwOBA":weighted(xwoba_col,wpa),
            "HardHit%":weighted(hh_col,wpa),
            "RV100":rv100,
            "Run Value":run_value,
        })

    out=pd.DataFrame(rows)
    return out.sort_values("Pitches",ascending=False) if not out.empty else out


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

        drv_series = pd.to_numeric(g["delta_run_exp"],errors="coerce") if "delta_run_exp" in g.columns else pd.Series(dtype=float)
        drv = drv_series.sum(min_count=1) if not drv_series.empty else None
        rv100 = (drv/len(g)*100) if drv is not None and not pd.isna(drv) and len(g) else None

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
    "ff":"ff","4 seam":"ff","4 seamer":"ff","4 seam fastball":"ff","four seam":"ff","four seamer":"ff","four seam fastball":"ff","four seamer fastball":"ff",
    "si":"si","sinker":"si","two seam":"si","2 seam":"si","two seamer":"si","2 seamer":"si",
    "sl":"sl","slider":"sl",
    "st":"st","sweeper":"st",
    "fc":"fc","cutter":"fc","cut fastball":"fc",
    "ch":"ch","change":"ch","change up":"ch","changeup":"ch",
    "cu":"cu","curve":"cu","curveball":"cu",
    "kc":"kc","knuckle curve":"kc","knuckle curveball":"kc",
    "fs":"fs","splitter":"fs","split finger":"fs","split fingered":"fs","split finger fastball":"fs",
    "fo":"fo","forkball":"fo",
    "sv":"sv","slurve":"sv",
    "kn":"kn","knuckleball":"kn"
}
def canonical_pitch(v):
    n=normalize_name(v)
    if n in PITCH_CANON:
        return PITCH_CANON[n]
    # Savant labels can vary (e.g. "4-Seamer", "Four-Seam Fastball").
    tests=(
        (("4 seam","4 seamer","four seam","four seamer"),"ff"),
        (("sinker","two seam","2 seam"),"si"),
        (("sweeper",),"st"),
        (("slider",),"sl"),
        (("cutter","cut fastball"),"fc"),
        (("changeup","change up","change"),"ch"),
        (("knuckle curve",),"kc"),
        (("curveball","curve"),"cu"),
        (("splitter","split finger"),"fs"),
        (("forkball",),"fo"),
        (("slurve",),"sv"),
        (("knuckleball",),"kn"),
    )
    for needles,code in tests:
        if any(x in n for x in needles):
            return code
    return n

def pitch_expected_coverage(df: pd.DataFrame):
    fields=("xBA","xSLG","xwOBA","RV100")
    out={}
    if not isinstance(df,pd.DataFrame) or df.empty:
        return {f:0 for f in fields}
    for f in fields:
        out[f]=int(pd.to_numeric(df[f],errors="coerce").notna().sum()) if f in df.columns else 0
    return out

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
def savant_park_factors(year: int, rolling: int = 3):
    # Baseball Savant Park Factors. B.A.R.T.O.L.O. uses a 3-year rolling view,
    # so that is our primary context too.
    params = {
        "type":"year",
        "year":year,
        "condition":"All",
        "parks":"mlb",
        "rolling":rolling,
        "stat":"index_wOBA",
        "batSide":"",
    }
    try:
        r = requests.get(
            SAVANT_PARK_URL,
            params=params,
            headers={
                "User-Agent":"Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Safari/604.1",
                "Accept":"text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            },
            timeout=TIMEOUT,
        )
        r.raise_for_status()
        tables = pd.read_html(StringIO(r.text))
        for t in tables:
            if isinstance(t.columns,pd.MultiIndex):
                t.columns = [str(c[-1]).strip() for c in t.columns]
            else:
                t.columns = [str(c).strip() for c in t.columns]
            norm={normalize_name(c):c for c in t.columns}
            venue_col=next((norm[k] for k in norm if k in ("venue","park","stadium","name")),None)
            so_col=next((norm[k] for k in norm if k in ("so","strikeout","strikeouts","k")),None)
            if venue_col and so_col:
                return t
    except Exception:
        pass
    return pd.DataFrame()


# Verified Baseball Savant 3-year rolling SO factors.
# These are used only when Savant's page cannot be parsed server-side.
# 100 = neutral. Values are source-labelled in the UI instead of being invented.
SAVANT_SO_VERIFIED_CACHE = {
    # Baseball Savant 2024-2026 overall SO factors, verified Aug 2026.
    # Sutter Health Park uses the current-year factor because a full 3Y
    # sample is not yet available.
    (2026,"coors field"):90.0,
    (2026,"oriole park at camden yards"):99.0,
    (2026,"chase field"):90.0,
    (2026,"target field"):97.0,
    (2026,"great american ball park"):103.0,
    (2026,"nationals park"):94.0,
    (2026,"fenway park"):98.0,
    (2026,"citizens bank park"):104.0,
    (2026,"uniqlo field at dodger stadium"):101.0,
    (2026,"dodger stadium"):101.0,
    (2026,"rogers centre"):98.0,
    (2026,"rogers center"):98.0,
    (2026,"kauffman stadium"):90.0,
    (2026,"yankee stadium"):102.0,
    (2026,"angel stadium"):105.0,
    (2026,"loandepot park"):97.0,
    (2026,"comerica park"):99.0,
    (2026,"pnc park"):99.0,
    (2026,"daikin park"):107.0,
    (2026,"minute maid park"):107.0,
    (2026,"truist park"):104.0,
    (2026,"citi field"):103.0,
    (2026,"rate field"):97.0,
    (2026,"guaranteed rate field"):97.0,
    (2026,"progressive field"):104.0,
    (2026,"petco park"):102.0,
    (2026,"tropicana field"):102.0,
    (2026,"busch stadium"):90.0,
    (2026,"oracle park"):97.0,
    (2026,"american family field"):110.0,
    (2026,"wrigley field"):102.0,
    (2026,"globe life field"):101.0,
    (2026,"t mobile park"):118.0,
    (2026,"t-mobile park"):118.0,
    (2026,"sutter health park"):91.0,
}

def park_so_factor(venue, year):
    target = normalize_name(venue)
    aliases={
        "comerica park":["comerica park","comerica"],
        "wrigley field":["wrigley field","wrigley"],
        "oracle park":["oracle park","oracle"],
        "rogers centre":["rogers centre","rogers center"],
        "rogers center":["rogers centre","rogers center"],
        "uniqlo field at dodger stadium":["uniqlo field at dodger stadium","dodger stadium"],
        "rate field":["rate field","guaranteed rate field"],
        "daikin park":["daikin park","minute maid park"],
    }
    targets=set(aliases.get(target,[target]))
    targets.add(target)

    # Primary: 3-year rolling, matching the contextual view used by Savant/BARTOLO.
    for rolling in (3,1):
        df = savant_park_factors(year, rolling=rolling)
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
                        label="Baseball Savant live · 3Y rolling" if rolling==3 else "Baseball Savant live · current year"
                        return x,label

    # Stable fallback: Park Factor is known before game time and must not
    # remain blank just because Savant's HTML parser changes.
    lookup_names=[target]
    for alias_key,alias_values in aliases.items():
        if target==alias_key or target in alias_values:
            lookup_names.extend(alias_values)
            lookup_names.append(alias_key)
    for nm in lookup_names:
        cached=SAVANT_SO_VERIFIED_CACHE.get((year,normalize_name(nm)))
        if cached is not None:
            period="current year" if normalize_name(nm)=="sutter health park" else "3Y rolling"
            return cached, f"Baseball Savant verified cache · {period}"
    return None, "PARK NOT MAPPED"


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

@st.cache_data(ttl=21600, show_spinner=False)
def hitter_statcast(pid:int,start_dt:str,end_dt:str):
    try:
        df=statcast_batter(start_dt,end_dt,int(pid))
        return df if isinstance(df,pd.DataFrame) else pd.DataFrame()
    except Exception:
        return pd.DataFrame()

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
    sc=hitter_statcast(pid,f"{season}-03-01",end_date)
    disc=plate_discipline(sc) if not sc.empty else {}
    return {
        "K% vs hand": so/pa*100 if so is not None and pa else None,
        "PA": pa, "Contact%": disc.get("Contact%"), "Whiff%": disc.get("Whiff%"),
        "Source": (source + " + Statcast") if stat and not sc.empty else (source if stat else ("Statcast" if not sc.empty else "N/A")),
    }


def enrich_lineup(lineup,season,end_date,pitcher_hand):
    people=mlb_people_info([r.get("player_id") for r in lineup])
    out=[]
    for r in lineup:
        rr=dict(r); info=people.get(int(rr.get("player_id")),{}) if safe_num(rr.get("player_id")) is not None else {}
        if str(rr.get("Bats") or "").upper() not in ("L","R","S"):
            rr["Bats"]=info.get("Bats") or "N/A"
        out.append({**rr,**hitter_k_profile(rr["player_id"],season,end_date,pitcher_hand)})
    return out


def sample_size_regressed_k(raw_k, pa, prior_k, prior_pa=LINEUP_K_PRIOR_PA):
    """Empirical-Bayes style shrinkage for hitter K%.

    A hitter with a tiny split sample is pulled strongly toward the opponent
    team K% vs the same pitcher hand. As PA grows, the hitter's own rate
    increasingly controls the estimate.
    """
    prior=safe_num(prior_k)
    if prior is None:
        prior=LINEUP_K_DEFAULT_PRIOR
    raw=safe_num(raw_k)
    n=safe_num(pa)

    if raw is None or n is None or n <= 0:
        return clamp(prior,0,60)

    strength=max(1.0,float(prior_pa))
    value=(raw*n + prior*strength)/(n+strength)
    return clamp(value,0,60)


def apply_lineup_sample_size_protection(lineup, prior_k, prior_pa=LINEUP_K_PRIOR_PA):
    """Attach raw + stabilized K rates to every hitter without deleting raw data."""
    out=[]
    prior=safe_num(prior_k)
    if prior is None:
        prior=LINEUP_K_DEFAULT_PRIOR

    for row in lineup or []:
        rr=dict(row)
        raw=safe_num(rr.get("K% vs hand"))
        pa=safe_num(rr.get("PA"))
        adjusted=sample_size_regressed_k(raw,pa,prior,prior_pa)
        sample_weight=(pa/(pa+prior_pa)*100) if pa is not None and pa>0 else 0.0

        if pa is None or pa < 50:
            sample_tier="LOW"
        elif pa < 150:
            sample_tier="MEDIUM"
        else:
            sample_tier="HIGH"

        rr["K% raw vs hand"]=raw
        rr["K% model vs hand"]=adjusted
        rr["K prior%"]=prior
        rr["Sample weight%"]=sample_weight
        rr["K sample"]=sample_tier
        out.append(rr)

    return out


def lineup_model_k(row):
    adjusted=safe_num(row.get("K% model vs hand"))
    if adjusted is not None:
        return adjusted
    return safe_num(row.get("K% vs hand"))


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
    vals = [lineup_model_k(r) for r in lineup]
    vals = [v for v in vals if v is not None]
    return sum(vals)/len(vals) if vals else None



# ============================================================
# AUTOMATIC LEASH INTELLIGENCE + OPTIONAL AI ANALYST
# ============================================================

@st.cache_data(ttl=1800, show_spinner=False)
def player_transactions(player_id:int, start_date:str, end_date:str):
    try:
        d=get_json(
            "https://statsapi.mlb.com/api/v1/transactions",
            params={"playerId":int(player_id),"startDate":start_date,"endDate":end_date},
        )
        out=[]
        for t in d.get("transactions",[]) or []:
            out.append({
                "date":t.get("date") or t.get("effectiveDate"),
                "type":t.get("typeDesc") or t.get("typeCode") or "",
                "description":t.get("description") or "",
            })
        return out
    except Exception:
        return []


def automatic_leash_intelligence(player_id:int, selected_date:date, log:list, recent:dict):
    pitches=[safe_num(r.get("Pitches")) for r in (log or [])[:10]]
    pitches=[x for x in pitches if x is not None]
    ips=[innings_decimal(r.get("IP")) for r in (log or [])[:10]]
    ips=[x for x in ips if x is not None]
    bf=[safe_num(r.get("BF")) for r in (log or [])[:10]]
    bf=[x for x in bf if x is not None]

    l3=pitches[:3]
    l5=pitches[:5]
    avg3=sum(l3)/len(l3) if l3 else None
    avg5=sum(l5)/len(l5) if l5 else None
    avg10=sum(pitches)/len(pitches) if pitches else None
    max5=max(l5) if l5 else None
    under90=(sum(x<90 for x in pitches)/len(pitches)*100) if pitches else None

    start=(selected_date-timedelta(days=120)).isoformat()
    end=(selected_date-timedelta(days=1)).isoformat()
    tx=player_transactions(player_id,start,end)
    recent_activation=False
    recent_il=False
    for t in tx:
        txt=f"{t.get('type','')} {t.get('description','')}".lower()
        try:
            dt=pd.Timestamp(t.get("date")).date()
            age=(selected_date-dt).days
        except Exception:
            age=999
        if age<=35 and any(x in txt for x in ("reinstated","activated","returned from injured","recalled from rehab")):
            recent_activation=True
        if age<=60 and any(x in txt for x in ("injured list","15-day il","60-day il","10-day il","rehab assignment")):
            recent_il=True

    gap_days=None
    long_gap=False
    if len(log)>=2:
        try:
            d0=pd.Timestamp(log[0]["Date"]).date()
            d1=pd.Timestamp(log[1]["Date"]).date()
            gap_days=(d0-d1).days
            long_gap=gap_days>=18
        except Exception:
            pass

    ramp=False
    if len(l3)>=3:
        seq=list(reversed(l3))
        ramp=(seq[-1]-seq[0]>=8 and all(seq[i] <= seq[i+1]+2 for i in range(len(seq)-1)))

    injury_return=bool(recent_activation and recent_il)

    pitch_limit=False
    if injury_return and max5 is not None and max5<90:
        pitch_limit=True
    elif long_gap and avg3 is not None and avg3<85:
        pitch_limit=True
    elif ramp and max5 is not None and max5<90:
        pitch_limit=True

    avg_ip=sum(ips[:5])/len(ips[:5]) if ips[:5] else None
    avg_bf=sum(bf[:5])/len(bf[:5]) if bf[:5] else None
    opener_bulk=bool((avg_ip is not None and avg_ip<3.6) or (avg_bf is not None and avg_bf<16.5))

    quick_hook=bool(
        len(pitches)>=5 and avg10 is not None and avg10<88 and
        under90 is not None and under90>=60
    )

    ceiling=None
    if pitches:
        base=max(l5) if l5 else max(pitches)
        if pitch_limit:
            ceiling=round(min(max((pitches[0] if pitches else base)+8, base),92))
        else:
            ceiling=round(min(max(base,90),110))

    role=("SHORT START / BULK RISK" if opener_bulk
          else "STARTER · LEASH RESTRINGIDO" if pitch_limit
          else "TRADITIONAL STARTER")
    hook=("SHORT" if quick_hook else "LONG" if avg10 is not None and avg10>=94 else "NORMAL")
    confidence=("HIGH" if len(pitches)>=8 else "MEDIUM" if len(pitches)>=5 else "LOW")

    reasons=[]
    if injury_return: reasons.append("IL/activation reciente detectada")
    if long_gap and not injury_return: reasons.append(f"pausa de {gap_days} días detectada")
    if ramp: reasons.append("ramp-up reciente de pitch count")
    if pitch_limit: reasons.append("historial compatible con límite/restricción")
    if opener_bulk: reasons.append("volumen reciente compatible con short-start/bulk")
    if quick_hook: reasons.append("alta frecuencia de salidas antes de 90 pitches")
    if not reasons: reasons.append("leash reciente estable")

    return {
        "injury_return":injury_return,
        "pitch_limit":pitch_limit,
        "opener":opener_bulk,
        "manager_quick_hook":quick_hook,
        "projected_pitch_ceiling":ceiling,
        "role":role,
        "hook":hook,
        "confidence":confidence,
        "avg_pitches_l3":avg3,
        "avg_pitches_l5":avg5,
        "under90_pct":under90,
        "gap_days":gap_days,
        "reason":" · ".join(reasons),
        "transactions":tx,
    }


def apply_leash_adjustment(proj:dict, recent:dict, leash:dict):
    out=dict(proj)
    bf=safe_num(out.get("bf"))
    if bf is None:
        return out
    factor=1.0
    if leash.get("opener"):
        factor=min(factor,.76)
    elif leash.get("pitch_limit"):
        ceiling=safe_num(leash.get("projected_pitch_ceiling"))
        avgp=safe_num(recent.get("avg_pitches"))
        if ceiling is not None and avgp is not None and avgp>0:
            factor=min(factor,clamp(ceiling/avgp,.82,1.0))
        else:
            factor=min(factor,.90)
    if leash.get("manager_quick_hook"):
        factor=min(factor,.94)

    if factor<.999:
        old_bf=bf
        out["bf"]=clamp(old_bf*factor,12,32)
        k_pct=safe_num(out.get("k_pct")) or 22.0
        form_adj=safe_num(out.get("form_adj")) or 0.0
        out["central"]=clamp(out["bf"]*k_pct/100 + form_adj,.5,14)
        sigma=max(1.35,math.sqrt(out["central"])*.80)
        out["low"]=max(0,out["central"]-1.35*sigma)
        out["high"]=out["central"]+1.35*sigma
        out["leash_factor"]=factor
        out["bf_pre_leash"]=old_bf
    else:
        out["leash_factor"]=1.0
        out["bf_pre_leash"]=bf
    return out


def ai_api_key():
    try:
        return st.secrets.get("OPENAI_API_KEY")
    except Exception:
        return None


def ai_status():
    if OpenAI is None:
        return "SDK NO INSTALADO"
    return "IA CONECTADA" if ai_api_key() else "IA NO CONECTADA"


def build_ai_payload(p,mlb,pdisc,split_l,split_r,team_general,team_split,opp_disc,
                     arsenal,opp_pitch,recent,park_so,lineup,proj,leash,market_df):
    top_arsenal=[]
    if isinstance(arsenal,pd.DataFrame) and not arsenal.empty:
        for _,r in arsenal.head(4).iterrows():
            top_arsenal.append({
                "pitch":r.get("Pitch"),"usage":safe_num(r.get("Usage%")),
                "whiff":safe_num(r.get("Whiff%")),"k":safe_num(r.get("K%")),
                "putaway":safe_num(r.get("PutAway%")),"xwoba":safe_num(r.get("xwOBA")),
            })
    matchup=[]
    if isinstance(opp_pitch,pd.DataFrame) and not opp_pitch.empty:
        for _,r in opp_pitch.head(8).iterrows():
            matchup.append({
                "pitch":r.get("Pitch"),"pitches":safe_num(r.get("Pitches")),
                "whiff":safe_num(r.get("Whiff%")),"k":safe_num(r.get("K%")),
                "xba":safe_num(r.get("xBA")),"xslg":safe_num(r.get("xSLG")),
                "xwoba":safe_num(r.get("xwOBA")),"rv100":safe_num(r.get("RV100")),
            })
    hitters=[]
    for r in (lineup or [])[:9]:
        hitters.append({
            "order":r.get("#"),"name":r.get("Hitter"),"bats":r.get("Bats"),
            "k_raw_vs_hand":safe_num(r.get("K% vs hand")),
            "k_model_vs_hand":lineup_model_k(r),
            "pa":safe_num(r.get("PA")),
            "sample_tier":r.get("K sample"),
            "sample_weight":safe_num(r.get("Sample weight%")),
            "contact":safe_num(r.get("Contact%")),"whiff":safe_num(r.get("Whiff%")),
        })
    markets=[]
    if isinstance(market_df,pd.DataFrame) and not market_df.empty:
        for _,r in market_df.head(12).iterrows():
            markets.append({
                "book":r.get("Sportsbook"),"market":r.get("Market"),"line":safe_num(r.get("Line")),
                "odds":safe_num(r.get("Odds")),"model":safe_num(r.get("Model%")),
                "implied":safe_num(r.get("Implied%")),"edge":safe_num(r.get("Edge pp")),
                "ev":safe_num(r.get("EV%")),"l5":safe_num(r.get("Hit L5%")),"l10":safe_num(r.get("Hit L10%")),
            })
    return {
        "pitcher":p.get("pitcher_name"),"team":p.get("team"),"opponent":p.get("opponent"),
        "hand":p.get("throwing_hand"),"venue":p.get("venue"),
        "M1":{
            "K%":safe_num(mlb.get("calc_k_pct")),"K9":safe_num(mlb.get("strikeoutsPer9Inn")),
            "K-BB%":safe_num(mlb.get("calc_k_minus_bb")),"Whiff%":safe_num(pdisc.get("Whiff%")),
            "SwStr%":safe_num(pdisc.get("SwStr%")),"CSW%":safe_num(pdisc.get("CSW%")),
            "Contact%":safe_num(pdisc.get("Contact%")),
        },
        "M2":{"recent":recent,"leash":leash},
        "M3":{"vsL":split_l,"vsR":split_r},
        "M4":{"teamK":safe_num(team_general.get("calc_k_pct")),
              "teamKvsHand":safe_num(team_split.get("calc_k_pct")),"discipline":opp_disc},
        "M5":{"arsenal":top_arsenal,"opponent_vs_pitch":matchup},
        "M7":{"parkSO":park_so},
        "M8":{"lineup":hitters,"sample_size_prior_pa":LINEUP_K_PRIOR_PA},
        "projection":{"BF":proj.get("bf"),"K%":proj.get("k_pct"),
                      "Ks":proj.get("central"),"low":proj.get("low"),"high":proj.get("high")},
        "M9":{"markets":markets},
    }


def run_ai_analyst(payload):
    key=ai_api_key()
    if OpenAI is None:
        return None,"Instala el paquete openai en requirements.txt."
    if not key:
        return None,"Añade OPENAI_API_KEY en Streamlit Secrets para activar el Analista IA."

    import json
    client=OpenAI(api_key=key)
    instructions=(
        "Eres el Analista IA de un modelo MLB de strikeouts de pitcher abridor. "
        "El motor cuantitativo ya calculó BF, K%, Ks, probabilidades, fair odds, edge y EV. "
        "NO cambies esos números ni inventes datos. Analiza solamente la evidencia recibida. "
        "Explica en español, en párrafos claros y específicos: "
        "1) tesis del matchup, 2) qué datos la respaldan, 3) contradicciones entre módulos, "
        "4) principales riesgos de fallo, 5) lectura del leash/volumen, "
        "6) lectura arsenal-vs-rival, 7) lectura del lineup confirmado si existe, "
        "8) lectura del mercado si hay líneas, 9) confianza analítica. "
        "Si falta un dato dilo explícitamente. No recomiendes aumentar stake ni persigas pérdidas."
    )
    response=client.responses.create(
        model="gpt-5.6",
        instructions=instructions,
        input=json.dumps(payload,ensure_ascii=False,default=str),
    )
    return response.output_text,None

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





SPORTSBOOK_ALIASES = {
    "draftkings": "DraftKings", "dk": "DraftKings",
    "fanduel": "FanDuel", "fd": "FanDuel",
    "betmgm": "BetMGM", "mgm": "BetMGM",
    "caesars": "Caesars", "fanatics": "Fanatics",
    "bet365": "bet365", "betrivers": "BetRivers",
}

def _detect_book(text):
    t=normalize_name(text)
    for k,v in SPORTSBOOK_ALIASES.items():
        if k in t:
            return v
    return None

def _american_from_any(v):
    if isinstance(v,(int,float)):
        x=float(v)
        if abs(x)>=100 and abs(x)<=5000:
            return int(round(x))
        return None
    if isinstance(v,str):
        m=re.search(r'(?<!\d)([+-]\d{2,4})(?!\d)',v.replace('−','-'))
        if m:
            try:return int(m.group(1))
            except Exception:return None
    return None

def _line_from_any(v):
    if isinstance(v,(int,float)):
        x=float(v)
        if 0.5 <= x <= 20:
            return x
        return None
    if isinstance(v,str):
        m=re.search(r'(?<!\d)(\d{1,2}(?:\.5)?)(?!\d)',v)
        if m:
            try:
                x=float(m.group(1)); return x if .5<=x<=20 else None
            except Exception:return None
    return None

def _market_from_text(text,line=None):
    t=str(text or '').lower().replace('strikeouts',' ').replace('strikeout',' ')
    if re.search(r'\bunder\b|\bu\s*\d',t): return 'Under'
    if re.search(r'\bover\b|\bo\s*\d',t): return 'Over'
    m=re.search(r'(?<!\d)(\d{1,2})\s*\+',t)
    if m:return f'Alt {int(m.group(1))}+'
    return None

def _dedupe_market_rows(rows):
    out=[];seen=set()
    for r in rows:
        book=str(r.get('Sportsbook') or 'Action Network')
        market=str(r.get('Market') or '').strip()
        line=safe_num(r.get('Line'));odds=safe_num(r.get('Odds'))
        if not market or line is None or odds is None:continue
        key=(book.lower(),market.lower(),round(line,2),int(odds))
        if key in seen:continue
        seen.add(key)
        out.append({'Sportsbook':book,'Market':market,'Line':float(line),'Odds':int(odds),'Source':r.get('Source','Action Network')})
    return out

def _extract_action_from_text_block(text,book=None):
    rows=[]
    clean=' '.join(str(text or '').replace('−','-').split())
    book=book or _detect_book(clean) or 'Action Network'
    pats=[
        (r'(?i)\bover\s*(\d{1,2}(?:\.5)?)\s*([+-]\d{2,4})','Over'),
        (r'(?i)\bo\s*(\d{1,2}(?:\.5)?)\s*([+-]\d{2,4})','Over'),
        (r'(?i)\bunder\s*(\d{1,2}(?:\.5)?)\s*([+-]\d{2,4})','Under'),
        (r'(?i)\bu\s*(\d{1,2}(?:\.5)?)\s*([+-]\d{2,4})','Under'),
    ]
    for pat,market in pats:
        for m in re.finditer(pat,clean):
            rows.append({'Sportsbook':book,'Market':market,'Line':float(m.group(1)),'Odds':int(m.group(2)),'Source':'Action Network public props'})
    for m in re.finditer(r'(?<!\d)(\d{1,2})\s*\+\s*([+-]\d{2,4})',clean):
        rows.append({'Sportsbook':book,'Market':f'Alt {int(m.group(1))}+','Line':float(m.group(1)),'Odds':int(m.group(2)),'Source':'Action Network public props'})
    return rows

def _walk_action_json(obj,pitcher_norm,inherited=False,rows=None):
    if rows is None:rows=[]
    if isinstance(obj,dict):
        scalar=' '.join(str(v) for v in obj.values() if isinstance(v,(str,int,float,bool)) and v is not None)
        here=inherited or (pitcher_norm and pitcher_norm in normalize_name(scalar))
        if here:
            # First try textual patterns from the local object.
            rows.extend(_extract_action_from_text_block(scalar,_detect_book(scalar)))
            low={str(k).lower():v for k,v in obj.items()}
            odds=None;line=None;market=None;book=None
            for k,v in low.items():
                if odds is None and ('odd' in k or 'price' in k):odds=_american_from_any(v)
                if line is None and any(x in k for x in ('line','handicap','threshold','total')):line=_line_from_any(v)
                if market is None and any(x in k for x in ('market','side','selection','label','name')):market=_market_from_text(v,line)
                if book is None and any(x in k for x in ('book','sportsbook','operator')):book=_detect_book(v)
            if odds is not None and line is not None and market:
                rows.append({'Sportsbook':book or 'Action Network','Market':market,'Line':line,'Odds':odds,'Source':'Action Network embedded data'})
        for v in obj.values():
            if isinstance(v,(dict,list)):_walk_action_json(v,pitcher_norm,here,rows)
    elif isinstance(obj,list):
        for v in obj:_walk_action_json(v,pitcher_norm,inherited,rows)
    return rows

@st.cache_data(ttl=60,show_spinner=False)
def action_auto_k_odds(pitcher_name):
    """Best-effort automatic strikeout odds from Action Network's public pitching-props page.
    Never fabricates prices: if a real American price cannot be parsed, returns an empty table.
    """
    cols=['Sportsbook','Market','Line','Odds','Source']
    try:
        r=requests.get(ACTION_PITCHING_PROPS_URL,headers={
            'User-Agent':'Mozilla/5.0 (iPhone; CPU iPhone OS 18_0 like Mac OS X) AppleWebKit/605.1.15 Version/18.0 Mobile/15E148 Safari/604.1',
            'Accept-Language':'en-US,en;q=0.9',
        },timeout=TIMEOUT)
        r.raise_for_status()
        soup=BeautifulSoup(r.text,'html.parser')
        pn=normalize_name(pitcher_name)
        last=pn.split()[-1] if pn else ''
        rows=[]

        # 1) Server-rendered DOM: inspect local containers containing the pitcher.
        for node in soup.find_all(string=True):
            txt=' '.join(str(node).split())
            nt=normalize_name(txt)
            if not txt or not last or (pn not in nt and last not in nt):continue
            cur=node.parent
            for _ in range(5):
                if cur is None:break
                block=' '.join(cur.stripped_strings)
                if len(block)<=2500:
                    rows.extend(_extract_action_from_text_block(block,_detect_book(block)))
                cur=cur.parent

        # 2) Embedded JSON / Next.js payloads.
        for sc in soup.find_all('script'):
            raw=sc.string or sc.get_text() or ''
            raw=raw.strip()
            if not raw or (pn not in normalize_name(raw) and last not in normalize_name(raw)):
                continue
            if raw.startswith('{') or raw.startswith('['):
                try:
                    data=json.loads(raw)
                    rows.extend(_walk_action_json(data,pn))
                except Exception:
                    pass
            # Also inspect the script as text for compact odds strings.
            idx=normalize_name(raw).find(last) if last else -1
            if idx>=0:
                # Raw and normalized offsets differ; use broad raw chunks instead.
                rows.extend(_extract_action_from_text_block(raw[:120000],_detect_book(raw[:120000])))

        rows=_dedupe_market_rows(rows)
        # Prefer identifiable books, but keep Action best/consensus if that is all the public page exposes.
        df=pd.DataFrame(rows,columns=cols)
        if df.empty:return df
        # Remove obviously malformed duplicate combinations and sort main O/U before alts.
        df=df[(df['Odds'].abs()>=100)&(df['Odds'].abs()<=5000)&(df['Line']>=0.5)&(df['Line']<=20)].copy()
        return df.drop_duplicates(subset=['Sportsbook','Market','Line','Odds']).reset_index(drop=True)
    except Exception:
        return pd.DataFrame(columns=cols)


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
        hi=[r["Hitter"] for r in lineup if lineup_model_k(r) is not None and lineup_model_k(r)>=27]
        lo=[r["Hitter"] for r in lineup if lineup_model_k(r) is not None and lineup_model_k(r)<=17]
        low_samples=sum(1 for r in lineup if r.get("K sample")=="LOW")
        t8=(f"El lineup confirmado tiene {fmt(lk,1,'%')} K% agregado AJUSTADO por tamaño de muestra vs la mano del pitcher. "
            f"High-K: {', '.join(hi) if hi else 'ninguno'}; low-K: {', '.join(lo) if lo else 'ninguno'}. "
            f"{low_samples} bateadores tienen muestra LOW. "
            f"Las tasas individuales se estabilizan hacia el K% del rival contra esa mano usando un prior de {LINEUP_K_PRIOR_PA:.0f} PA, "
            "evitando que una muestra pequeña domine M8. El raw K% se conserva para auditoría.")
    else:
        s8="PENDIENTE";t8=("El lineup todavía no está confirmado. La proyección es PROVISIONAL porque los splits del pitcher "
                            "pueden cambiar materialmente según la composición L/R/S y los K% individuales de los nueve bateadores.")
    items.append(("M8 · Lineup confirmado",s8,t8))
    return items



VALIDATION_SNAPSHOT_FILE = Path("/tmp/mlb_k_validation_snapshots.json")


def _load_validation_snapshots():
    try:
        if VALIDATION_SNAPSHOT_FILE.exists():
            data=json.loads(VALIDATION_SNAPSHOT_FILE.read_text())
            return data if isinstance(data,dict) else {}
    except Exception:
        pass
    return {}


@st.cache_resource
def validation_snapshots():
    return _load_validation_snapshots()


def _persist_validation_snapshots():
    try:
        VALIDATION_SNAPSHOT_FILE.write_text(
            json.dumps(validation_snapshots(),ensure_ascii=False,indent=2,default=str)
        )
    except Exception:
        pass


def save_projection_snapshot(pitcher,proj,state=None,selected_date=None):
    """Freeze the pregame projection used for validation.

    A full pitcher-page analysis may replace an AUTO provisional snapshot while
    the game is still pregame. Once the game is live/final, the snapshot is immutable.
    """
    store=validation_snapshots()
    key=f"{pitcher.get('game_pk')}:{pitcher.get('pitcher_id')}"
    state=state or {}
    timing=("FINAL" if state.get("is_final") else ("LIVE" if state.get("is_live") else "PREGAME"))
    existing=store.get(key)
    # Full page analysis is also forbidden from becoming an official validation
    # snapshot until the official lineup is verified and M1-M8 are 100% complete.
    if not (proj.get("lineup_confirmed") and proj.get("analysis_ready")):
        return existing
    if timing!="PREGAME":
        return existing
    may_replace=(
        existing is None or
        (timing=="PREGAME" and existing.get("snapshot_timing")=="PREGAME" and existing.get("snapshot_source")=="AUTO")
    )
    if may_replace:
        store[key]={
            "game_pk":pitcher.get("game_pk"),
            "pitcher_id":pitcher.get("pitcher_id"),
            "pitcher_name":pitcher.get("pitcher_name"),
            "team":pitcher.get("team"),
            "team_id":pitcher.get("team_id"),
            "opponent":pitcher.get("opponent"),
            "projected_k":safe_num(proj.get("central")),
            "projected_bf":safe_num(proj.get("bf")),
            "projected_k_pct":safe_num(proj.get("k_pct")),
            "projected_low":safe_num(proj.get("low")),
            "projected_high":safe_num(proj.get("high")),
            "snapshot_timing":timing,
            "snapshot_source":"FULL_ANALYSIS",
            "lineup_confirmed":True if proj.get("lineup_confirmed") else False,
            "analysis_ready":True if proj.get("analysis_ready") else False,
            "module_status":proj.get("module_status",{}),
            "model_version":"V3.2.15",
            "game_date":str(selected_date) if selected_date is not None else None,
            "captured_at_utc":datetime.now(timezone.utc).isoformat(),
        }
        _persist_validation_snapshots()
    return store[key]


def projection_snapshot(game_pk,pitcher_id):
    return validation_snapshots().get(f"{game_pk}:{pitcher_id}")


def official_projection_snapshot(game_pk,pitcher_id):
    """Return only validation-eligible snapshots.

    V3.2.13 could persist provisional AUTO projections before the official lineup.
    Those rows must never appear as MODEL K or enter Records after the 100% gate.
    """
    snap=projection_snapshot(game_pk,pitcher_id)
    if not snap:
        return None
    if not (snap.get("lineup_confirmed") and snap.get("analysis_ready")):
        return None
    if str(snap.get("snapshot_timing") or "").upper()!="PREGAME":
        return None
    return snap


@st.cache_data(ttl=15,show_spinner=False)
def live_game_state(game_pk:int):
    try:
        feed=get_json(f"{MLB_GAME_FEED_URL}/{game_pk}/feed/live")
    except Exception:
        return {}
    gd=feed.get("gameData",{})
    ld=feed.get("liveData",{})
    status=gd.get("status",{})
    linescore=ld.get("linescore",{})
    box=ld.get("boxscore",{}).get("teams",{})
    teams=gd.get("teams",{})

    def runs(side):
        v=linescore.get("teams",{}).get(side,{}).get("runs")
        if v is None:
            v=box.get(side,{}).get("teamStats",{}).get("batting",{}).get("runs")
        try:return int(v or 0)
        except Exception:return 0

    pitcher_ks={}
    for side in ("away","home"):
        for pdata in box.get(side,{}).get("players",{}).values():
            pid=pdata.get("person",{}).get("id")
            pst=pdata.get("stats",{}).get("pitching",{})
            if pid and pst:
                try:pitcher_ks[int(pid)]=int(pst.get("strikeOuts",0) or 0)
                except Exception:pitcher_ks[int(pid)]=0

    abstract=status.get("abstractGameState") or "Preview"
    coded=status.get("codedGameState")
    is_final=abstract=="Final" or coded in {"F","O"}
    is_live=abstract=="Live" and not is_final
    inning=linescore.get("currentInning")
    inning_state=linescore.get("inningState") or ""
    inning_text=f"{inning_state} {inning}".strip() if inning else ""
    return {
        "is_live":is_live,"is_final":is_final,
        "detailed":status.get("detailedState") or abstract,
        "away_name":teams.get("away",{}).get("name","Away"),
        "home_name":teams.get("home",{}).get("name","Home"),
        "away_runs":runs("away"),"home_runs":runs("home"),
        "inning_text":inning_text,"pitcher_ks":pitcher_ks,
    }


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
# OFFICIAL PROJECTION READINESS GATE
# ============================================================

def projection_module_readiness(mlb,log,sc_pitcher,split_l,split_r,team_general,opp_disc,arsenal,opp_pitch,recent_sc,park_so,lineup,lineup_confirmed):
    """M1-M8 must be complete before MODEL K can become official.

    M9 is the betting/price layer and does not change the strikeout projection itself,
    so MODEL K freezes once the quantitative projection modules M1-M8 are 100% ready
    and MLB's official 9-man opponent lineup passes Team Guard.
    """
    def m5_quality(df):
        if not isinstance(df,pd.DataFrame) or df.empty:
            return False
        required=("Whiff%","K%","xBA","xSLG","xwOBA","RV100")
        for c in required:
            if c not in df.columns or not pd.to_numeric(df[c],errors="coerce").notna().any():
                return False
        return True

    split_ok=(safe_num((split_l or {}).get("PA")) or 0)>0 and (safe_num((split_r or {}).get("PA")) or 0)>0
    opp_ok=bool(team_general and opp_disc and all(safe_num(opp_disc.get(x)) is not None for x in ("Whiff%","Contact%","Chase%","Zone%")))
    lineup_ok=bool(lineup_confirmed and lineup_quality(lineup)[0])
    status={
        "M1":bool(mlb and isinstance(sc_pitcher,pd.DataFrame) and not sc_pitcher.empty),
        "M2":bool(mlb and log),
        "M3":bool(isinstance(sc_pitcher,pd.DataFrame) and not sc_pitcher.empty and split_ok),
        "M4":opp_ok,
        "M5":bool(isinstance(arsenal,pd.DataFrame) and not arsenal.empty and m5_quality(opp_pitch)),
        "M6":bool(isinstance(recent_sc,pd.DataFrame) and not recent_sc.empty),
        "M7":bool(park_so is not None),
        "M8":lineup_ok,
    }
    return status, all(status.values())


@st.cache_data(ttl=30,show_spinner=False)
def lineup_gate_for_option(game_pk:int,opponent_side:str,opponent_id:int,season:int,selected_date_iso:str):
    try:
        feed=game_feed(int(game_pk))
        raw,status=confirmed_lineup(feed,opponent_side,opponent_id,season,selected_date_iso)
        return bool(len(raw)==9 and str(status).startswith("VERIFIED")), status
    except Exception:
        return False, "LINEUP PENDING"


# ============================================================
# AUTOMATIC SLATE PROJECTIONS + RECORDS
# ============================================================

@st.cache_data(ttl=300, show_spinner=False)
def automatic_projection_for_option(option_json:str, selected_date_iso:str):
    """Compute an OFFICIAL automatic MODEL K only after MLB confirms the lineup.

    Before the official 9-man lineup exists, this function returns immediately and
    intentionally does not calculate or persist a projection. Once the lineup is
    confirmed, M1-M8 must all pass the same data-quality gate before MODEL K is valid.
    """
    opt=json.loads(option_json)
    selected_date=date.fromisoformat(selected_date_iso)
    cutoff_str=game_cutoff(selected_date).isoformat()

    # First gate: do not spend time generating a projection before MLB publishes
    # and Team Guard verifies the opponent's official 9-man lineup.
    try:
        feed=game_feed(int(opt["game_pk"]))
    except Exception:
        feed={}
    raw_lineup, lineup_guard_status = confirmed_lineup(
        feed,opt.get("opponent_side"),opt.get("opponent_id"),selected_date.year,selected_date.isoformat()
    )
    lineup_confirmed=bool(len(raw_lineup)==9 and str(lineup_guard_status).startswith("VERIFIED"))
    if not lineup_confirmed:
        return {
            "lineup_confirmed":False,"analysis_ready":False,"module_status":{},
            "lineup_guard_status":lineup_guard_status,"waiting_reason":"WAITING_LINEUP"
        }

    try: mlb=pitcher_stats_to_date(int(opt["pitcher_id"]),selected_date.year,cutoff_str)
    except Exception: mlb={}
    try: log=pitcher_game_log_before(int(opt["pitcher_id"]),selected_date.year,selected_date.isoformat())
    except Exception: log=[]

    sit="vl" if str(opt.get("throwing_hand","")).lower().startswith("left") else "vr"
    try: team_general=team_hitting_to_date(int(opt["opponent_id"]),selected_date.year,cutoff_str)
    except Exception: team_general={}
    try: team_split=team_hitting_to_date(int(opt["opponent_id"]),selected_date.year,cutoff_str,sit)
    except Exception: team_split={}

    sc_start=f"{selected_date.year}-03-01"
    try: sc_pitcher=pitcher_statcast(int(opt["pitcher_id"]),sc_start,cutoff_str)
    except Exception: sc_pitcher=pd.DataFrame()
    pdisc=plate_discipline(sc_pitcher) if isinstance(sc_pitcher,pd.DataFrame) else {}
    split_l=split_metrics(sc_pitcher,"L") if isinstance(sc_pitcher,pd.DataFrame) else {}
    split_r=split_metrics(sc_pitcher,"R") if isinstance(sc_pitcher,pd.DataFrame) else {}
    arsenal=arsenal_table(sc_pitcher) if isinstance(sc_pitcher,pd.DataFrame) else pd.DataFrame()
    recent_sc=statcast_recent_games(sc_pitcher,10) if isinstance(sc_pitcher,pd.DataFrame) else pd.DataFrame()

    try: opp_disc,_=savant_team_plate_discipline(int(opt["opponent_id"]),selected_date.year)
    except Exception: opp_disc={}
    try: opp_pitch,_=savant_team_pitch_type(int(opt["opponent_id"]),selected_date.year)
    except Exception: opp_pitch=pd.DataFrame()

    # Fill M5 expected metrics from Savant detail when the aggregate route omits them.
    try:
        if isinstance(opp_pitch,pd.DataFrame) and not opp_pitch.empty:
            cov=pitch_expected_coverage(opp_pitch)
            if min(cov.values())==0:
                detail=savant_pitch_type_detail_csv(int(opt["opponent_id"]),selected_date.year)
                if isinstance(detail,pd.DataFrame) and not detail.empty:
                    opp_pitch=merge_pitch_type_fallback(opp_pitch,detail)
    except Exception:
        pass

    try: lineup=enrich_lineup(raw_lineup,selected_date.year,cutoff_str,opt.get("throwing_hand"))
    except Exception: lineup=[]
    lineup_prior_k=(safe_num(team_split.get("calc_k_pct")) or safe_num(team_general.get("calc_k_pct")) or LINEUP_K_DEFAULT_PRIOR)
    lineup=apply_lineup_sample_size_protection(lineup,lineup_prior_k)

    recent=recent_summary(log)
    try: park_so,_=park_so_factor(opt.get("venue"),selected_date.year)
    except Exception: park_so=None
    auto_leash=automatic_leash_intelligence(int(opt["pitcher_id"]),selected_date,log,recent)

    module_status,analysis_ready=projection_module_readiness(
        mlb,log,sc_pitcher,split_l,split_r,team_general,opp_disc,arsenal,opp_pitch,recent_sc,park_so,lineup,lineup_confirmed
    )
    if not analysis_ready:
        return {
            "lineup_confirmed":True,"analysis_ready":False,"module_status":module_status,
            "lineup_guard_status":lineup_guard_status,"waiting_reason":"WAITING_100_PERCENT"
        }

    proj=build_projection(mlb,pdisc,team_general,team_split,lineup,recent,park_so)
    proj=apply_leash_adjustment(proj,recent,auto_leash)
    proj["lineup_confirmed"]=True
    proj["analysis_ready"]=True
    proj["module_status"]=module_status
    proj["lineup_guard_status"]=lineup_guard_status
    return proj


def save_auto_projection_snapshot(pitcher,proj,state=None,selected_date=None):
    """Create/update AUTO projection only while the game is still pregame.

    Provisional snapshots may improve when the official 9-man lineup becomes
    available. They never get created or changed after first pitch.
    """
    state=state or {}
    if state.get("is_live") or state.get("is_final"):
        return projection_snapshot(pitcher.get("game_pk"),pitcher.get("pitcher_id"))

    store=validation_snapshots()
    key=f"{pitcher.get('game_pk')}:{pitcher.get('pitcher_id')}"
    existing=store.get(key)
    if existing and existing.get("snapshot_source") not in (None,"AUTO"):
        return existing

    new_confirmed=bool(proj.get("lineup_confirmed"))
    new_ready=bool(proj.get("analysis_ready"))
    old_confirmed=bool(existing.get("lineup_confirmed")) if existing else False
    old_ready=bool(existing.get("analysis_ready")) if existing else False

    # Never create an official MODEL K until lineup + M1-M8 are 100% ready.
    if not (new_confirmed and new_ready):
        return existing
    # Once an official pregame AUTO snapshot is frozen, never move it.
    if existing and old_confirmed and old_ready:
        return existing

    store[key]={
        "game_pk":pitcher.get("game_pk"),
        "pitcher_id":pitcher.get("pitcher_id"),
        "pitcher_name":pitcher.get("pitcher_name"),
        "team":pitcher.get("team"),
        "team_id":pitcher.get("team_id"),
        "opponent":pitcher.get("opponent"),
        "projected_k":safe_num(proj.get("central")),
        "projected_bf":safe_num(proj.get("bf")),
        "projected_k_pct":safe_num(proj.get("k_pct")),
        "projected_low":safe_num(proj.get("low")),
        "projected_high":safe_num(proj.get("high")),
        "snapshot_timing":"PREGAME",
        "snapshot_source":"AUTO",
        "lineup_confirmed":new_confirmed,
        "analysis_ready":new_ready,
        "module_status":proj.get("module_status",{}),
        "lineup_guard_status":proj.get("lineup_guard_status"),
        "model_version":"V3.2.15",
        "game_date":str(selected_date) if selected_date is not None else None,
        "captured_at_utc":datetime.now(timezone.utc).isoformat(),
    }
    _persist_validation_snapshots()
    return store[key]


def ensure_automatic_slate_projections(options, selected_date):
    """Populate MODEL K for all starters automatically without postgame leakage."""
    eligible=[]
    for opt in options:
        state=live_game_state(opt.get("game_pk"))
        if state.get("is_live") or state.get("is_final"):
            continue
        snap=projection_snapshot(opt.get("game_pk"),opt.get("pitcher_id"))
        # Keep checking until an OFFICIAL lineup-confirmed, 100%-ready snapshot exists.
        if snap is None or not (snap.get("lineup_confirmed") and snap.get("analysis_ready")):
            eligible.append(opt)

    if not eligible:
        return 0,0

    done=0
    failed=0
    progress=st.progress(0,text=f"Generando proyecciones automáticas 0/{len(eligible)}")
    for i,opt in enumerate(eligible,1):
        try:
            payload=json.dumps(opt,sort_keys=True,default=str)
            proj=automatic_projection_for_option(payload,selected_date.isoformat())
            save_auto_projection_snapshot(opt,proj,live_game_state(opt.get("game_pk")),selected_date.isoformat())
            done+=1
        except Exception:
            failed+=1
        progress.progress(i/len(eligible),text=f"Generando proyecciones automáticas {i}/{len(eligible)}")
    progress.empty()
    return done,failed


def purge_provisional_snapshots():
    """Remove legacy provisional snapshots created before the official 100% gate."""
    store=validation_snapshots()
    bad=[k for k,v in list(store.items()) if not (v.get("lineup_confirmed") and v.get("analysis_ready"))]
    if not bad:
        return 0
    for k in bad:
        store.pop(k,None)
    _persist_validation_snapshots()
    return len(bad)


def validation_record_rows():
    """Build historical projection record and update actual K from MLB when available."""
    rows=[]
    for snap in validation_snapshots().values():
        # Records only include official pregame projections made with verified lineup
        # and all quantitative modules M1-M8 complete. Older provisional snapshots
        # are intentionally excluded from the official record.
        if not (snap.get("lineup_confirmed") and snap.get("analysis_ready")):
            continue
        gp=safe_num(snap.get("game_pk")); pid=safe_num(snap.get("pitcher_id"))
        if gp is None or pid is None:
            continue
        state=live_game_state(int(gp))
        actual=(state.get("pitcher_ks") or {}).get(int(pid))
        proj=safe_num(snap.get("projected_k"))
        low=safe_num(snap.get("projected_low")); high=safe_num(snap.get("projected_high"))
        result="PENDING"
        if state.get("is_final") and actual is not None:
            if low is not None and high is not None:
                result="HIT" if low <= actual <= high else "MISS"
            elif proj is not None:
                result="HIT" if abs(actual-proj) <= 1.5 else "MISS"
        rows.append({
            "Date":snap.get("game_date"),
            "Pitcher":snap.get("pitcher_name"),
            "Matchup":f"{snap.get('team','')} vs {snap.get('opponent','')}",
            "Model K":round(proj,2) if proj is not None else None,
            "Range":f"{low:.1f}–{high:.1f}" if low is not None and high is not None else "—",
            "Actual K":actual,
            "Error K":round(actual-proj,2) if actual is not None and proj is not None else None,
            "Result":result,
            "Lineup":"CONFIRMED · 100%",
            "Source":snap.get("snapshot_source","—"),
            "Version":snap.get("model_version","—"),
        })
    return rows


def render_records_section():
    st.markdown("## 📊 Récord de proyecciones")
    rows=validation_record_rows()
    if not rows:
        st.info("Todavía no hay proyecciones guardadas.")
        return
    df=pd.DataFrame(rows)
    final=df[df["Result"].isin(["HIT","MISS"])].copy()
    hits=int((final["Result"]=="HIT").sum()) if not final.empty else 0
    misses=int((final["Result"]=="MISS").sum()) if not final.empty else 0
    total=hits+misses
    pending=int((df["Result"]=="PENDING").sum())
    mae=None
    if not final.empty:
        vals=pd.to_numeric(final["Error K"],errors="coerce").dropna().abs()
        if not vals.empty: mae=float(vals.mean())

    a,b,c,d=st.columns(4)
    a.metric("Récord",f"{hits}-{misses}")
    b.metric("Hit rate",f"{(hits/total*100):.1f}%" if total else "N/A")
    c.metric("MAE",f"{mae:.2f} K" if mae is not None else "N/A")
    d.metric("Pendientes",pending)

    st.caption("Solo cuentan proyecciones oficiales con lineup confirmado + M1-M8 al 100%. HIT = los K reales terminaron dentro del rango probable pregame del modelo. MISS = terminaron fuera del rango. El error conserva además la diferencia exacta contra la proyección central.")
    show=df.sort_values(["Date","Pitcher"],ascending=[False,True])
    st.dataframe(show,hide_index=True,use_container_width=True)
    st.download_button(
        "Descargar récord CSV",
        data=show.to_csv(index=False).encode("utf-8"),
        file_name="mlb_strikeout_model_records.csv",
        mime="text/csv",
        use_container_width=True,
    )

# ============================================================
# APP LOAD
# ============================================================


st.markdown(
    """
    <div class="hero">
      <div class="section-label">MODELO PROFESIONAL MLB · STARTING PITCHER STRIKEOUTS</div>
      <div style="font-size:2.05rem;font-weight:880;margin-top:3px">Starting Pitcher Strikeout Lab</div>
      <div style="opacity:.70;margin-top:6px">V3.2.15 LIVE VALIDATION · Confirmed-Lineup 100% Gate · Official Snapshot Cleanup · Records Fix · Clean Uniform Score Cards · ET Game Times · Clickable Pitcher Names · In-Card Score/K Tracker · Lineup Team Guard · Sample-Size Protection · Automatic Leash Intelligence · AI Analyst</div>
    </div>
    """, unsafe_allow_html=True
)

if "view_mode" not in st.session_state:
    st.session_state["view_mode"]="slate"
if "selected_pitcher_id" not in st.session_state:
    st.session_state["selected_pitcher_id"]=None
if not st.session_state.get("official_snapshot_cleanup_v3215"):
    purge_provisional_snapshots()
    st.session_state["official_snapshot_cleanup_v3215"]=True

date_col,records_col,_=st.columns([1.15,.85,1.15])
with date_col:
    game_date=st.date_input("Fecha",value=date.today(),min_value=date(2015,1,1),key="slate_date")
with records_col:
    st.write("")
    st.write("")
    if st.button("📊 RÉCORDS",use_container_width=True):
        st.session_state["view_mode"]="records"
        st.query_params.clear()

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
    # Do not kick the user out of Records just because a previously selected
    # pitcher belongs to another date/slate. Only analysis requires a valid ID.
    if st.session_state.get("view_mode")=="analysis":
        st.session_state["view_mode"]="slate"

# Query-param navigation allows the daily board to be pure compact HTML.
qp_pitch = st.query_params.get("pitcher")
if qp_pitch and qp_pitch in by_id:
    st.session_state["selected_pitcher_id"]=qp_pitch
    st.session_state["view_mode"]="analysis"

if st.session_state["view_mode"]=="records":
    if st.button("← SLATE",use_container_width=False):
        st.session_state["view_mode"]="slate"
        st.rerun()
    render_records_section()
    st.stop()

if st.session_state["view_mode"]=="slate":
    # Automatically generate/freeze pregame MODEL K for every starter. Missing
    # projections are never backfilled once a game has started.
    auto_done,auto_failed=ensure_automatic_slate_projections(options,game_date)
    if auto_failed:
        st.caption(f"Auto projection: {auto_done} creadas/actualizadas · {auto_failed} no disponibles todavía.")
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
        live_state=live_game_state(first.get("game_pk"))
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

        # No separate LIVE/PREGAME header: the score itself communicates game state.
        # Pregame = dash; once the game starts MLB supplies numeric runs (including 0-0).

        def board_time_text(opt):
            # Re-format directly from MLB gameDate at render time to avoid stale/UTC labels.
            raw=opt.get("game_date_raw")
            if raw:
                try:
                    dt=datetime.fromisoformat(str(raw).replace("Z","+00:00"))
                    et=dt.astimezone(ZoneInfo("America/New_York"))
                    return et.strftime("%-I:%M %p ET")
                except Exception:
                    try:
                        return game_time_label(raw).replace("EDT","ET").replace("EST","ET")
                    except Exception:
                        pass
            label=str(opt.get("game_time") or "TBD")
            # Last-resort protection: never intentionally label a board time as UTC.
            return label.replace(" UTC"," ET")

        def team_score(opt):
            if live_state.get("is_live") or live_state.get("is_final"):
                side=opt.get("team_side")
                runs=live_state.get("away_runs",0) if side=="away" else live_state.get("home_runs",0)
                return str(int(runs or 0))
            return "—"

        def pitcher_metric_html(opt):
            """Bottom-of-card pitcher projection + live/final Ks.

            The pitcher name itself is the analysis link. This strip is display-only.
            """
            snap=official_projection_snapshot(opt.get("game_pk"),opt.get("pitcher_id"))
            proj_k=safe_num(snap.get("projected_k")) if snap else None
            actual=live_state.get("pitcher_ks",{}).get(int(opt.get("pitcher_id"))) if live_state else None

            pieces=[]
            if proj_k is not None:
                pieces.append(f'<span class="proj">MODEL {proj_k:.2f} K</span>')
            else:
                if live_state.get("is_live") or live_state.get("is_final"):
                    pieces.append('<span class="muted">NO PROJ</span>')
                else:
                    lineup_ok,_=lineup_gate_for_option(
                        int(opt.get("game_pk")),opt.get("opponent_side"),int(opt.get("opponent_id")),
                        game_date.year,game_date.isoformat()
                    )
                    pieces.append('<span class="muted">WAIT 100%</span>' if lineup_ok else '<span class="muted">WAIT LINEUP</span>')

            if actual is not None and live_state.get("is_live"):
                pieces.append(f'<span class="actual">LIVE {int(actual)} K</span>')
            elif actual is not None and live_state.get("is_final"):
                pieces.append(f'<span class="finalk">FINAL {int(actual)} K</span>')
            elif live_state.get("is_live") or live_state.get("is_final"):
                pieces.append('<span class="muted">K —</span>')

            return '<br>'.join(pieces)

        def pitcher_name_link(opt):
            return (f'<a class="board-pitcher-link" href="?pitcher={opt["selection_id"]}">'
                    f'{pshort(opt)} ({opt["throwing_hand"][:1]})</a>')

        cards.append(
            f'<div class="board-game">'
            f'<div class="board-time">{board_time_text(first)} · {first.get("venue","")}</div>'
            f'<div class="board-team">'
            f'<img class="board-logo" src="{away_logo}">'
            f'<div><div class="board-abbr">{abbr(away["team"])}</div><div class="board-pitcher">{pitcher_name_link(away)}</div></div>'
            f'<div class="board-score">{team_score(away)}</div>'
            f'</div>'
            f'<div class="board-team">'
            f'<img class="board-logo" src="{home_logo}">'
            f'<div><div class="board-abbr">{abbr(home["team"])}</div><div class="board-pitcher">{pitcher_name_link(home)}</div></div>'
            f'<div class="board-score">{team_score(home)}</div>'
            f'</div>'
            f'<div class="board-actions">'
            f'<div class="board-metric">{pitcher_metric_html(away)}</div>'
            f'<div class="board-metric">{pitcher_metric_html(home)}</div>'
            f'</div></div>'
        )

    board_html='<div class="board-grid">'+''.join(cards)+'</div>'
    st.markdown(board_html,unsafe_allow_html=True)
    st.markdown('<div class="board-legend">Horario en ET. Antes del juego el score aparece como —; al comenzar cambia automáticamente a 0-0 o al marcador real. MODEL K solo aparece cuando MLB confirma los 9 bateadores y M1-M8 están 100% completos; entonces queda congelado. LIVE/FINAL K llega de MLB. Toca el nombre del pitcher para abrir el análisis.</div>',unsafe_allow_html=True)

    if any(live_game_state(g[0].get("game_pk")).get("is_live") for g in games if g):
        st.markdown('<meta http-equiv="refresh" content="30">',unsafe_allow_html=True)
        st.caption("Live board: actualización automática cada 30 segundos mientras haya juegos en vivo.")
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
                before_cov = pitch_expected_coverage(opp_pitch)
                opp_pitch = merge_pitch_type_fallback(opp_pitch, raw_pitch)
                after_cov = pitch_expected_coverage(opp_pitch)
                if sum(after_cov.values()) > sum(before_cov.values()):
                    opp_pitch_source = f"{opp_pitch_source} + RAW_STATCAST_FILL"
                # If a Savant naming variant still prevented a join, force a canonical
                # code merge one more time using inferred pitch family codes.
                if sum(after_cov.values()) == sum(before_cov.values()):
                    p2=opp_pitch.copy(); r2=raw_pitch.copy()
                    p2["Code"]=[canonical_pitch(x) for x in (p2["Code"] if "Code" in p2.columns else p2["Pitch"])]
                    r2["Code"]=[canonical_pitch(x) for x in (r2["Code"] if "Code" in r2.columns else r2["Pitch"])]
                    opp_pitch=merge_pitch_type_fallback(p2,r2)
                    after2=pitch_expected_coverage(opp_pitch)
                    if sum(after2.values()) > sum(after_cov.values()):
                        opp_pitch_source=f"{opp_pitch_source} + RAW_STATCAST_ALIAS_FILL"
    except Exception:
        pass

    # Savant HTML fallback: the public leaderboard displays xBA/xSLG/xwOBA even
    # when its CSV route omits those columns. Only fill missing values.
    try:
        needed=(not opp_pitch.empty and any(c not in opp_pitch.columns or opp_pitch[c].isna().any() for c in ("xBA","xSLG","xwOBA")))
        if needed:
            exp=savant_expected_pitch_type_html(p["opponent_id"],game_date.year)
            if not exp.empty:
                before=int(opp_pitch[[c for c in ("xBA","xSLG","xwOBA") if c in opp_pitch.columns]].isna().sum().sum())
                opp_pitch=merge_pitch_type_fallback(opp_pitch,exp)
                after=int(opp_pitch[[c for c in ("xBA","xSLG","xwOBA") if c in opp_pitch.columns]].isna().sum().sum())
                if after<before: opp_pitch_source=f"{opp_pitch_source} + SAVANT_HTML_EXPECTED"
    except Exception:
        pass

    # Final M5 fallback: query Savant's CSV one pitch family at a time.
    # This is the route that explicitly exposes xBA/xSLG/xwOBA and RV/100.
    try:
        cov=pitch_expected_coverage(opp_pitch)
        if not opp_pitch.empty and min(cov.values())==0:
            detail=savant_pitch_type_detail_csv(p["opponent_id"],game_date.year)
            if not detail.empty:
                before_cov=pitch_expected_coverage(opp_pitch)
                opp_pitch=merge_pitch_type_fallback(opp_pitch,detail)
                after_cov=pitch_expected_coverage(opp_pitch)
                if sum(after_cov.values())>sum(before_cov.values()):
                    opp_pitch_source=f"{opp_pitch_source} + SAVANT_DETAIL_CSV"
    except Exception:
        pass

    park_so,park_source=park_so_factor(p["venue"],game_date.year)

    raw_lineup, lineup_guard_status = confirmed_lineup(
        feed,
        p["opponent_side"],
        p["opponent_id"],
        game_date.year,
        game_date.isoformat(),
    )
    lineup=enrich_lineup(raw_lineup,game_date.year,cutoff_str,p["throwing_hand"]) if raw_lineup else []
    lineup_prior_k=(
        safe_num(team_split.get("calc_k_pct"))
        or safe_num(team_general.get("calc_k_pct"))
        or LINEUP_K_DEFAULT_PRIOR
    )
    lineup=apply_lineup_sample_size_protection(lineup,lineup_prior_k)

    recent=recent_summary(log)
    rest_days=days_rest(log,game_date)
    auto_leash=automatic_leash_intelligence(p["pitcher_id"],game_date,log,recent)

    fg_df,fg_status=fangraphs_pitchers(game_date.year)
    br_df=bref_pitchers(game_date.year)
    cross=player_crosswalk(p["pitcher_name"])
    fg=match_pitcher_row(fg_df,p["pitcher_name"],p["pitcher_id"],cross)
    br=match_pitcher_row(br_df,p["pitcher_name"],p["pitcher_id"],cross)

proj=build_projection(mlb,pdisc,team_general,team_split,lineup,recent,park_so)
proj=apply_leash_adjustment(proj,recent,auto_leash)
proj["lineup_confirmed"]=bool(len(raw_lineup)==9 and str(lineup_guard_status).startswith("VERIFIED"))
full_module_status,full_analysis_ready=projection_module_readiness(
    mlb,log,sc_pitcher,split_l,split_r,team_general,opp_disc,arsenal,opp_pitch,recent_sc,park_so,lineup,proj["lineup_confirmed"]
)
proj["analysis_ready"]=full_analysis_ready
proj["module_status"]=full_module_status

# Freeze only an OFFICIAL pregame projection: verified lineup + M1-M8 100%.
validation_state=live_game_state(p.get("game_pk"))
projection_snapshot_saved=save_projection_snapshot(p,proj,validation_state,game_date.isoformat())

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
    if projection_snapshot_saved and projection_snapshot_saved.get("analysis_ready"):
        st.caption(f"OFFICIAL MODEL K: {projection_snapshot_saved.get('projected_k',0):.2f} K · lineup confirmado · M1-M8 100% · {projection_snapshot_saved.get('model_version','V3.2.15')}")
    elif not proj.get("lineup_confirmed"):
        st.warning("PROYECCIÓN NO OFICIAL · Esperando lineup confirmado de MLB. No se guarda en RÉCORDS.")
    elif not proj.get("analysis_ready"):
        missing=[k for k,v in proj.get("module_status",{}).items() if not v]
        st.warning("PROYECCIÓN NO OFICIAL · Lineup confirmado, pero el análisis todavía no está 100% completo"+(f" ({', '.join(missing)} pendiente)." if missing else ".")+" No se guarda en RÉCORDS.")

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

        st.markdown("**Leash Intelligence · automático**")
        li1,li2,li3,li4=st.columns(4)
        li1.metric("Injury return","YES" if auto_leash.get("injury_return") else "NO")
        li1.metric("Role",auto_leash.get("role","N/A"))
        li2.metric("Projected ceiling",fmt(auto_leash.get("projected_pitch_ceiling"),0," pitches"))
        li2.metric("Pitch-limit risk","YES" if auto_leash.get("pitch_limit") else "NO")
        li3.metric("Hook tendency",auto_leash.get("hook","N/A"))
        li3.metric("<90 pitches",fmt(auto_leash.get("under90_pct"),0,"%"))
        li4.metric("Inference confidence",auto_leash.get("confidence","N/A"))
        li4.metric("L3 pitch avg",fmt(auto_leash.get("avg_pitches_l3"),1))
        st.caption("Detección automática: "+auto_leash.get("reason",""))

        with st.expander("Advanced override · solo si hay noticia oficial que el sistema no captó"):
            override_injury=st.checkbox("Override: regreso de lesión",value=False)
            override_pitch_limit=st.checkbox("Override: pitch limit oficial",value=False)
            override_opener=st.checkbox("Override: opener / bulk",value=False)
            override_hook=st.checkbox("Override: hook corto",value=False)
            override_ceiling=st.number_input("Override pitch ceiling (0 = automático)",0,130,0,5)
            override_notes=st.text_area("Fuente / nota oficial del override",key="override_leash_notes")

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
            cov=pitch_expected_coverage(opp_pitch)
            st.caption(
                f"Fuente rival vs pitch type: {opp_pitch_source} · "
                f"Cobertura xBA {cov['xBA']} / xSLG {cov['xSLG']} / "
                f"xwOBA {cov['xwOBA']} / RV100 {cov['RV100']} pitch types."
            )
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
            st.success(f"Lineup Team Guard: {lineup_guard_status}")
            ldf=pd.DataFrame(lineup)
            display_cols=[
                "#","Hitter","Bats","K% raw vs hand","K% model vs hand","PA",
                "K sample","Sample weight%","Contact%","Whiff%","Source"
            ]
            m8_view=ldf[[c for c in display_cols if c in ldf.columns]].copy()
            m8_view=m8_view.rename(columns={
                "K% raw vs hand":"Raw K% vs hand",
                "K% model vs hand":"Model K% vs hand",
            })
            st.dataframe(m8_view.round(2),hide_index=True,use_container_width=True)
            raw_vals=[safe_num(r.get("K% raw vs hand")) for r in lineup]
            raw_vals=[v for v in raw_vals if v is not None]
            lineup_raw_k=(sum(raw_vals)/len(raw_vals)) if raw_vals else None
            lineup_adj_k=lineup_k_pct(lineup)
            delta_k=(lineup_adj_k-lineup_raw_k) if lineup_raw_k is not None and lineup_adj_k is not None else None
            k1,k2,k3=st.columns(3)
            k1.metric("Lineup K% raw",fmt(lineup_raw_k,1,"%"))
            k2.metric("Lineup K% adjusted",fmt(lineup_adj_k,1,"%"))
            k3.metric("Sample-size adjustment",fmt(delta_k,1," pp"))
            st.caption(
                f"Sample-Size Protection: Model K% estabiliza cada tasa individual hacia "
                f"{lineup_prior_k:.1f}% (K% del rival vs la mano del pitcher) con un prior de "
                f"{LINEUP_K_PRIOR_PA:.0f} PA. Raw K% nunca se borra; muestras pequeñas reciben menos peso."
            )
            high=[r["Hitter"] for r in lineup if lineup_model_k(r) is not None and lineup_model_k(r)>=27]
            low=[r["Hitter"] for r in lineup if lineup_model_k(r) is not None and lineup_model_k(r)<=17]
            st.write("**High-K hitters (Model K%):** "+(", ".join(high) if high else "None flagged"))
            st.write("**Low-K hitters (Model K%):** "+(", ".join(low) if low else "None flagged"))
        else:
            st.warning(f"NO cerrar análisis definitivo: {lineup_guard_status}")
        lineup_notes=st.text_area("Ausencias / sustituciones / diferencias vs lineup habitual",key="lineup_notes")

# ------------------------------------------------------------
# MARKET / ALL LINES
# ------------------------------------------------------------
with tab_market:
    st.subheader("M9 · CUOTAS REALES AUTOMÁTICAS")
    st.caption("El modelo intenta leer automáticamente las líneas reales de strikeouts publicadas por Action Network. Nunca inventa precios: si no puede verificar una cuota, M9 queda pendiente y el dictamen es NO BET.")

    auto_market=action_auto_k_odds(p["pitcher_name"])
    if st.button("↻ Actualizar cuotas ahora",key="refresh_auto_odds"):
        action_auto_k_odds.clear()
        st.rerun()

    if auto_market.empty:
        st.warning("M9 DATA UNAVAILABLE · No pude verificar cuotas públicas de strikeouts para este pitcher ahora mismo. NO BET hasta que una fuente automática devuelva precios reales.")
        market_df=pd.DataFrame()
    else:
        st.success(f"AUTO ODDS OK · {len(auto_market)} precios reales detectados para {p['pitcher_name']}.")
        st.dataframe(auto_market,hide_index=True,use_container_width=True)
        rows=[]
        for _,r in auto_market.iterrows():
            book=str(r.get("Sportsbook","")).strip() or "Action Network"
            market=str(r.get("Market","")).strip()
            line=safe_num(r.get("Line")); odds=safe_num(r.get("Odds"))
            if not market or line is None or odds is None:continue
            low_market=market.lower()
            if low_market.startswith("over"):
                threshold=math.floor(line)+1
                prob=poisson_ge(threshold,proj["central"])
            elif low_market.startswith("under"):
                maxk=math.floor(line)
                prob=poisson_cdf(maxk,proj["central"])
            elif "alt" in low_market or "+" in low_market:
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
                "Edge pp":edge,"EV%":ev*100 if ev is not None else None,
                "Source":r.get("Source","Action Network public props")
            })
        market_df=pd.DataFrame(rows)
        if not market_df.empty:
            market_df=append_line_history(market_df,log)
            for col in ("Model%","Implied%","Fair","Edge pp","EV%","Hit L5%","Hit L10%"):
                market_df[col]=pd.to_numeric(market_df[col],errors="coerce").round(1)
            st.markdown("**Evaluación automática de todas las cuotas verificadas**")
            st.dataframe(market_df.sort_values("EV%",ascending=False),hide_index=True,use_container_width=True)
            st.caption("Las cuotas se vuelven a consultar cada 60 s. Hit L5/L10 es descriptivo. Si Action cambia su página o bloquea el acceso, M9 se cierra en NO BET en vez de usar un precio ficticio.")
        else:
            st.warning("Se detectó información de mercado, pero ninguna línea de K pudo verificarse con formato completo (mercado + línea + American odds). NO BET.")

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
if locals().get("override_ceiling",0)>0:
    auto_leash["projected_pitch_ceiling"]=locals().get("override_ceiling")
    auto_leash["pitch_limit"]=True
    proj=apply_leash_adjustment(
        build_projection(mlb,pdisc,team_general,team_split,lineup,recent,park_so),
        recent,auto_leash
    )
manual={
    "injury_return":bool(auto_leash.get("injury_return") or locals().get("override_injury",False)),
    "pitch_limit":bool(auto_leash.get("pitch_limit") or locals().get("override_pitch_limit",False) or locals().get("override_ceiling",0)>0),
    "opener":bool(auto_leash.get("opener") or locals().get("override_opener",False)),
    "manager_quick_hook":bool(auto_leash.get("manager_quick_hook") or locals().get("override_hook",False)),
}

analysis_items=technical_analysis(
    mlb,pdisc,split_l,split_r,opp_disc,team_general,team_split,
    arsenal,opp_pitch,recent,park_so,lineup,proj,manual,log
)

# Data completeness is based on actual model modules, not validation websites.

def pitch_matchup_quality(df):
    if not isinstance(df,pd.DataFrame) or df.empty:return False
    required=["Whiff%","K%","xBA","xSLG","xwOBA","RV100"]
    for c in required:
        if c not in df.columns:
            return False
        if not pd.to_numeric(df[c],errors="coerce").notna().any():
            return False
    return True

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
    st.subheader("Analista IA · lectura conjunta")
    ai_state=ai_status()
    state_icon="🟢" if ai_state=="IA CONECTADA" else "🟡"
    st.markdown(f"**{state_icon} {ai_state}**")
    st.caption("La IA interpreta M1–M9 y explica interacciones, contradicciones y riesgos. No cambia la matemática del motor.")

    ai_payload=build_ai_payload(
        p,mlb,pdisc,split_l,split_r,team_general,team_split,opp_disc,
        arsenal,opp_pitch,recent,park_so,lineup,proj,auto_leash,
        market_df if 'market_df' in locals() else pd.DataFrame()
    )

    ai_key=f"ai_analysis_{p['selection_id']}_{game_date.isoformat()}"
    if ai_state=="IA CONECTADA":
        if st.button("GENERAR ANÁLISIS IA",type="primary",key=f"run_{ai_key}"):
            with st.spinner("La IA está leyendo M1–M9..."):
                txt,err=run_ai_analyst(ai_payload)
                if err:
                    st.session_state[ai_key]=f"ERROR: {err}"
                else:
                    st.session_state[ai_key]=txt
        if st.session_state.get(ai_key):
            st.markdown(st.session_state[ai_key])
    else:
        st.info("El bloque ya está instalado. Para activarlo añade OPENAI_API_KEY en Streamlit Secrets. Mientras tanto el motor cuantitativo sigue funcionando completo.")

    st.divider()
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
        st.warning("M9 PENDIENTE · No hay cuotas reales cargadas. Dictamen: NO BET hasta introducir el precio actual del sportsbook.")

    st.progress(modules/9)
    st.caption(" · ".join(f"{k} {'✅' if v else '⏳'}" for k,v in status.items()))

# ------------------------------------------------------------
# SOURCES
# ------------------------------------------------------------
with tab_sources:
    st.subheader("Fuentes y estado")
    src=pd.DataFrame([
        {"Fuente":"MLB Stats API","Uso":"Schedule, probables, season-to-date, logs, lineup, weather, umpire","Estado":"OK" if mlb else "Partial"},
        {"Fuente":"MLB Lineup Team Guard","Uso":"M8: Team ID + exactly 9 hitters + official opponent roster verification","Estado":lineup_guard_status},
        {"Fuente":"M8 Sample-Size Protection","Uso":f"Shrinkage de K% individual · prior {LINEUP_K_PRIOR_PA:.0f} PA · prior dinámico = rival vs mano","Estado":f"ACTIVE · prior {lineup_prior_k:.1f}%"},
        {"Fuente":"Baseball Savant / Statcast","Uso":"Pitcher discipline, arsenal, movement, spin","Estado":"OK" if not sc_pitcher.empty else "ERROR"},
        {"Fuente":"Savant Team Page","Uso":"Opponent Contact, Chase, Zone, Whiff","Estado":opp_disc_source},
        {"Fuente":"Savant Pitch Arsenal","Uso":"Opponent vs pitch type","Estado":opp_pitch_source},
        {"Fuente":"Savant SO Park Factor","Uso":"M7","Estado":f"{fmt(park_so,0)} · {park_source}"},
        {"Fuente":"Baseball-Reference","Uso":"Cross-check + fallback cuando la métrica existe públicamente","Estado":"OK" if br else "No match"},
        {"Fuente":"FanGraphs","Uso":"Validation / xFIP / SIERA when reachable","Estado":fg_status},
        {"Fuente":"Action Network PRO","Uso":"Automatic public K odds + B.A.R.T.O.L.O., % Bets, % Money, sharp, movement","Estado":"AUTO ODDS + Manual PRO validation"},
        {"Fuente":"Sportsbook K Odds","Uso":"Automatic real strikeout prices parsed from Action Network public props","Estado":"AUTO · fails closed to NO BET"},
        {"Fuente":"OpenAI Responses API","Uso":"Analista IA explicativo; no modifica la proyección cuantitativa","Estado":ai_status()},
    ])
    st.dataframe(src,hide_index=True,use_container_width=True)

    st.warning(
        "Important: full-season leaderboards are never allowed to leak future data into historical pregame projections. "
        "Core projection inputs remain cutoff-safe. Fallbacks only fill a metric when the source is compatible with the selected pregame cutoff."
    )

st.caption("V3.2.15 LIVE VALIDATION · Automatic Real Odds · Action Network Public Props · No Fabricated Prices · Clean Uniform Score Cards · ET Game Times · Clickable Pitcher Names · In-Card Score/K Tracker · Lineup Team Guard · Sample-Size Protection · Automatic Leash Intelligence · AI Analyst · cutoff-safe quantitative engine.")
