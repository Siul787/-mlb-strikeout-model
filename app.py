from datetime import date, datetime, timezone
from urllib.parse import urlencode

import math

import pandas as pd
import requests
import streamlit as st
from pybaseball import statcast_pitcher


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
MLB_GAME_FEED_URL = "https://statsapi.mlb.com/api/v1.1/game"

REQUEST_TIMEOUT_SECONDS = 20

PITCH_NAMES = {
    "FF": "4-Seam",
    "SI": "Sinker",
    "FC": "Cutter",
    "SL": "Slider",
    "ST": "Sweeper",
    "CU": "Curveball",
    "KC": "Knuckle Curve",
    "CH": "Changeup",
    "FS": "Splitter",
    "FO": "Forkball",
    "SC": "Screwball",
    "KN": "Knuckleball",
    "SV": "Slurve",
}


class MLBApiError(Exception):
    pass


def get_json(url: str, params: dict | None = None) -> dict:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
        payload = response.json()
    except (requests.RequestException, ValueError) as exc:
        raise MLBApiError(f"Could not load MLB data: {exc}") from exc

    if not isinstance(payload, dict):
        raise MLBApiError("MLB returned an unexpected response.")

    return payload


def safe_num(value):
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, decimals=1, suffix=""):
    number = safe_num(value)
    if number is None:
        return "N/A"
    return f"{number:.{decimals}f}{suffix}"


def american_implied_probability(odds):
    odds = safe_num(odds)
    if odds is None or odds == 0:
        return None
    if odds < 0:
        return (-odds) / ((-odds) + 100) * 100
    return 100 / (odds + 100) * 100


def innings_to_outs(value):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    if "." not in text:
        try:
            return int(text) * 3
        except ValueError:
            return None
    whole, frac = text.split(".", 1)
    try:
        innings = int(whole)
        partial = int(frac[:1] or "0")
    except ValueError:
        return None
    if partial not in (0, 1, 2):
        return None
    return innings * 3 + partial


def outs_to_innings_decimal(outs):
    if outs is None:
        return None
    return outs / 3


def stat_k_metrics(stat: dict) -> dict:
    strikeouts = safe_num(stat.get("strikeOuts"))
    walks = safe_num(stat.get("baseOnBalls"))
    batters_faced = safe_num(stat.get("battersFaced"))
    starts = safe_num(stat.get("gamesStarted"))
    innings_outs = innings_to_outs(stat.get("inningsPitched"))

    k_pct = None
    bb_pct = None
    k_minus_bb = None
    bf_start = None
    ip_start = None

    if strikeouts is not None and batters_faced and batters_faced > 0:
        k_pct = strikeouts / batters_faced * 100
    if walks is not None and batters_faced and batters_faced > 0:
        bb_pct = walks / batters_faced * 100
    if k_pct is not None and bb_pct is not None:
        k_minus_bb = k_pct - bb_pct
    if starts and starts > 0:
        if batters_faced is not None:
            bf_start = batters_faced / starts
        if innings_outs is not None:
            ip_start = outs_to_innings_decimal(innings_outs) / starts

    return {
        "k_pct": k_pct,
        "bb_pct": bb_pct,
        "k_minus_bb": k_minus_bb,
        "bf_start": bf_start,
        "ip_start": ip_start,
    }


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_profile(player_id: int) -> dict:
    payload = get_json(f"{MLB_PEOPLE_URL}/{player_id}")
    people = payload.get("people", [])
    return people[0] if people else {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_person_season_stats(
    player_id: int,
    season: int,
    group: str,
    sit_code: str | None = None,
) -> dict:
    params = {
        "stats": "season",
        "group": group,
        "season": season,
    }
    if sit_code:
        params["sitCodes"] = sit_code

    payload = get_json(f"{MLB_PEOPLE_URL}/{player_id}/stats", params=params)
    groups = payload.get("stats", [])
    if not groups:
        return {}
    splits = groups[0].get("splits", [])
    if not splits:
        return {}
    stat = splits[0].get("stat", {})
    return stat if isinstance(stat, dict) else {}


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_season_stats(player_id: int, season: int) -> dict:
    stat = fetch_person_season_stats(player_id, season, "pitching")
    stat.update({f"calc_{k}": v for k, v in stat_k_metrics(stat).items()})
    return stat


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_game_log(player_id: int, season: int) -> list[dict]:
    payload = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={"stats": "gameLog", "group": "pitching", "season": season},
    )
    groups = payload.get("stats", [])
    if not groups:
        return []

    rows = []
    for split in groups[0].get("splits", []):
        stat = split.get("stat", {})
        if not safe_num(stat.get("gamesStarted")):
            continue
        opponent = split.get("opponent", {})
        rows.append(
            {
                "date": split.get("date", "N/A"),
                "opponent": opponent.get("name", "N/A"),
                "IP": stat.get("inningsPitched", "N/A"),
                "K": stat.get("strikeOuts", "N/A"),
                "BB": stat.get("baseOnBalls", "N/A"),
                "ER": stat.get("earnedRuns", "N/A"),
                "HR": stat.get("homeRuns", "N/A"),
                "Pitches": stat.get("numberOfPitches", "N/A"),
                "BF": stat.get("battersFaced", "N/A"),
            }
        )
    return rows[-10:][::-1]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_team_hitting_stats(
    team_id: int,
    season: int,
    sit_code: str | None = None,
) -> dict:
    params = {"stats": "season", "group": "hitting", "season": season}
    if sit_code:
        params["sitCodes"] = sit_code

    payload = get_json(f"{MLB_TEAMS_URL}/{team_id}/stats", params=params)
    groups = payload.get("stats", [])
    if not groups:
        return {}
    splits = groups[0].get("splits", [])
    if not splits:
        return {}

    stat = splits[0].get("stat", {})
    if not isinstance(stat, dict):
        return {}

    strikeouts = safe_num(stat.get("strikeOuts"))
    pa = safe_num(stat.get("plateAppearances"))
    stat["calcKPercent"] = strikeouts / pa * 100 if strikeouts is not None and pa else None
    return stat


def format_game_time(game_date: str | None) -> str:
    if not game_date:
        return "Time TBD"
    try:
        parsed = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
    except ValueError:
        return "Time TBD"
    label = parsed.astimezone(timezone.utc).strftime("%I:%M %p").lstrip("0")
    return f"{parsed.astimezone(timezone.utc).strftime('%B')} {parsed.day}, {label} UTC"


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitchers_for_date(selected_date: str) -> dict:
    payload = get_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": selected_date,
            "hydrate": "probablePitcher,team,opponents,venue",
        },
    )

    options = []
    games_count = 0

    for date_group in payload.get("dates", []):
        games = date_group.get("games", [])
        games_count += len(games)

        for game in games:
            teams = game.get("teams", {})
            venue = game.get("venue", {})

            for side, opponent_side in (("away", "home"), ("home", "away")):
                team_data = teams.get(side, {})
                opponent_data = teams.get(opponent_side, {})
                probable = team_data.get("probablePitcher")

                if not isinstance(probable, dict):
                    continue

                pitcher_id = probable.get("id")
                pitcher_name = probable.get("fullName")
                if not pitcher_id or not pitcher_name:
                    continue

                team = team_data.get("team", {})
                opponent = opponent_data.get("team", {})
                throwing_hand = "N/A"

                try:
                    profile = fetch_pitcher_profile(int(pitcher_id))
                    throwing_hand = (
                        profile.get("pitchHand", {}).get("description") or "N/A"
                    )
                except MLBApiError:
                    pass

                options.append(
                    {
                        "selection_id": f"{game.get('gamePk')}-{side}-{pitcher_id}",
                        "game_pk": game.get("gamePk"),
                        "pitcher_id": int(pitcher_id),
                        "pitcher_name": pitcher_name,
                        "team": team.get("name", "N/A"),
                        "team_id": team.get("id"),
                        "team_side": side,
                        "opponent": opponent.get("name", "N/A"),
                        "opponent_id": opponent.get("id"),
                        "opponent_side": opponent_side,
                        "throwing_hand": throwing_hand,
                        "venue": venue.get("name", "N/A"),
                        "status": game.get("status", {}).get("detailedState", "N/A"),
                        "game_time": format_game_time(game.get("gameDate")),
                    }
                )

    return {"pitchers": options, "scheduled_games": games_count}


@st.cache_data(ttl=180, show_spinner=False)
def fetch_game_feed(game_pk: int) -> dict:
    if not game_pk:
        return {}
    return get_json(f"{MLB_GAME_FEED_URL}/{game_pk}/feed/live")


def parse_game_context(feed: dict) -> dict:
    game_data = feed.get("gameData", {})
    live_data = feed.get("liveData", {})
    weather = game_data.get("weather", {})
    venue = game_data.get("venue", {})
    officials = live_data.get("boxscore", {}).get("officials", [])

    umpire = None
    for row in officials:
        if row.get("officialType") == "Home Plate":
            umpire = row.get("official", {}).get("fullName")
            break

    return {
        "venue": venue.get("name"),
        "temperature": weather.get("temp"),
        "condition": weather.get("condition"),
        "wind": weather.get("wind"),
        "umpire": umpire,
    }


def parse_confirmed_lineup(feed: dict, opponent_side: str) -> list[dict]:
    box_teams = feed.get("liveData", {}).get("boxscore", {}).get("teams", {})
    side_data = box_teams.get(opponent_side, {})
    batting_order = side_data.get("battingOrder", [])
    players = side_data.get("players", {})

    lineup = []
    for index, player_id in enumerate(batting_order, start=1):
        player = players.get(f"ID{player_id}", {})
        person = player.get("person", {})
        lineup.append(
            {
                "order": index,
                "player_id": int(player_id),
                "name": person.get("fullName", f"Player {player_id}"),
                "bat_side": player.get("battingStats", {}).get("battingSide")
                or person.get("batSide", {}).get("code")
                or "N/A",
            }
        )
    return lineup


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_statcast_data(player_id: int, start_date: str, end_date: str):
    try:
        df = statcast_pitcher(start_date, end_date, player_id)
    except Exception:
        return pd.DataFrame()
    return df if isinstance(df, pd.DataFrame) else pd.DataFrame()


def calculate_pa_split(df: pd.DataFrame, mask: pd.Series) -> dict:
    if df.empty or "events" not in df.columns:
        return {"K": None, "PA": None, "K%": None}

    pa_rows = df[mask & df["events"].notna()].copy()
    if pa_rows.empty:
        return {"K": 0, "PA": 0, "K%": None}

    events = pa_rows["events"].astype(str)
    strikeouts = events.str.contains("strikeout", case=False, na=False).sum()
    pa = len(pa_rows)
    return {"K": int(strikeouts), "PA": int(pa), "K%": strikeouts / pa * 100 if pa else None}


def calculate_statcast_metrics(df: pd.DataFrame) -> dict:
    result = {
        "pitches": None,
        "whiff_pct": None,
        "csw_pct": None,
        "fastball_velo": None,
        "arsenal": [],
        "vs_l": {"K": None, "PA": None, "K%": None},
        "vs_r": {"K": None, "PA": None, "K%": None},
        "home": {"K": None, "PA": None, "K%": None},
        "away": {"K": None, "PA": None, "K%": None},
    }

    if df.empty:
        return result

    total_pitches = len(df)
    result["pitches"] = total_pitches

    descriptions = (
        df["description"].astype(str)
        if "description" in df.columns
        else pd.Series(index=df.index, dtype=str)
    )

    swing_events = {
        "swinging_strike",
        "swinging_strike_blocked",
        "foul",
        "foul_tip",
        "hit_into_play",
        "foul_bunt",
        "missed_bunt",
    }
    whiff_events = {"swinging_strike", "swinging_strike_blocked", "missed_bunt"}
    called_strike_events = {"called_strike"}

    swings = descriptions.isin(swing_events).sum()
    whiffs = descriptions.isin(whiff_events).sum()
    called_strikes = descriptions.isin(called_strike_events).sum()

    result["whiff_pct"] = whiffs / swings * 100 if swings else None
    result["csw_pct"] = (whiffs + called_strikes) / total_pitches * 100 if total_pitches else None

    if "stand" in df.columns:
        result["vs_l"] = calculate_pa_split(df, df["stand"].astype(str).eq("L"))
        result["vs_r"] = calculate_pa_split(df, df["stand"].astype(str).eq("R"))

    if "inning_topbot" in df.columns:
        top = df["inning_topbot"].astype(str).str.lower().eq("top")
        bottom = df["inning_topbot"].astype(str).str.lower().eq("bot")
        result["home"] = calculate_pa_split(df, top)
        result["away"] = calculate_pa_split(df, bottom)

    if "pitch_type" in df.columns:
        pitch_rows = []

        for pitch_type, group in df.groupby("pitch_type"):
            usage = len(group) / total_pitches * 100
            velocity = (
                pd.to_numeric(group["release_speed"], errors="coerce").mean()
                if "release_speed" in group.columns
                else None
            )

            group_desc = (
                group["description"].astype(str)
                if "description" in group.columns
                else pd.Series(index=group.index, dtype=str)
            )
            pitch_swings = group_desc.isin(swing_events).sum()
            pitch_whiffs = group_desc.isin(whiff_events).sum()
            pitch_whiff_pct = pitch_whiffs / pitch_swings * 100 if pitch_swings else None

            pitch_rows.append(
                {
                    "Pitch": PITCH_NAMES.get(str(pitch_type), str(pitch_type)),
                    "Code": str(pitch_type),
                    "Usage%": round(usage, 1),
                    "Velo": round(float(velocity), 1) if pd.notna(velocity) else None,
                    "Whiff%": round(pitch_whiff_pct, 1) if pitch_whiff_pct is not None else None,
                }
            )

        pitch_rows.sort(key=lambda row: row["Usage%"], reverse=True)
        result["arsenal"] = pitch_rows

        preferred_fastballs = ["FF", "SI", "FC"]
        for code in preferred_fastballs:
            row = next((r for r in pitch_rows if r["Code"] == code), None)
            if row and row["Velo"] is not None:
                result["fastball_velo"] = row["Velo"]
                break

    return result


def recent_start_summary(recent_starts: list[dict]) -> dict:
    last5 = recent_starts[:5]
    if not last5:
        return {}

    ks = [safe_num(row["K"]) for row in last5]
    pitches = [safe_num(row["Pitches"]) for row in last5]
    bfs = [safe_num(row["BF"]) for row in last5]
    outs = [innings_to_outs(row["IP"]) for row in last5]

    ks = [v for v in ks if v is not None]
    pitches = [v for v in pitches if v is not None]
    bfs = [v for v in bfs if v is not None]
    outs = [v for v in outs if v is not None]

    return {
        "avg_k": sum(ks) / len(ks) if ks else None,
        "avg_pitches": sum(pitches) / len(pitches) if pitches else None,
        "avg_bf": sum(bfs) / len(bfs) if bfs else None,
        "avg_ip": (sum(outs) / len(outs) / 3) if outs else None,
        "max_pitches": max(pitches) if pitches else None,
    }


def hitter_k_rate(player_id: int, season: int, pitcher_hand: str) -> dict:
    sit_code = "vl" if pitcher_hand.lower().startswith("left") else "vr"

    try:
        split_stat = fetch_person_season_stats(player_id, season, "hitting", sit_code)
    except MLBApiError:
        split_stat = {}

    source = "split"
    stat = split_stat

    # If MLB does not return the handedness split, fall back to full-season rate.
    if not stat:
        try:
            stat = fetch_person_season_stats(player_id, season, "hitting")
            source = "season"
        except MLBApiError:
            stat = {}

    strikeouts = safe_num(stat.get("strikeOuts"))
    pa = safe_num(stat.get("plateAppearances"))
    k_pct = strikeouts / pa * 100 if strikeouts is not None and pa else None

    return {
        "K": strikeouts,
        "PA": pa,
        "K%": k_pct,
        "source": source if k_pct is not None else "N/A",
    }


def enrich_lineup(lineup: list[dict], season: int, pitcher_hand: str) -> list[dict]:
    enriched = []
    for hitter in lineup:
        metrics = hitter_k_rate(hitter["player_id"], season, pitcher_hand)
        enriched.append({**hitter, **metrics})
    return enriched


def lineup_weighted_k_pct(lineup: list[dict]):
    valid = [row["K%"] for row in lineup if safe_num(row.get("K%")) is not None]
    if not valid:
        return None
    return sum(valid) / len(valid)



# =========================================================
# PREDICTION ENGINE
# =========================================================

def clamp(value, low, high):
    return max(low, min(high, value))


def weighted_average(items, default=None):
    valid = [(safe_num(v), w) for v, w in items if safe_num(v) is not None and w > 0]
    if not valid:
        return default
    total_weight = sum(w for _, w in valid)
    if total_weight <= 0:
        return default
    return sum(v * w for v, w in valid) / total_weight


def poisson_pmf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0 if k == 0 else 0.0
    return math.exp(-lam + k * math.log(lam) - math.lgamma(k + 1))


def poisson_cdf(k: int, lam: float) -> float:
    if k < 0:
        return 0.0
    return sum(poisson_pmf(i, lam) for i in range(k + 1))


def poisson_prob_ge(threshold: int, lam: float) -> float:
    return clamp(1.0 - poisson_cdf(threshold - 1, lam), 0.0, 1.0)


def fair_american_odds(probability: float):
    p = safe_num(probability)
    if p is None or p <= 0 or p >= 1:
        return None
    if p >= 0.5:
        return -100 * p / (1 - p)
    return 100 * (1 - p) / p


def american_profit_per_unit(odds: float):
    odds = safe_num(odds)
    if odds is None or odds == 0:
        return None
    if odds > 0:
        return odds / 100
    return 100 / abs(odds)


def ev_per_unit(probability: float, odds: float):
    p = safe_num(probability)
    profit = american_profit_per_unit(odds)
    if p is None or profit is None:
        return None
    return p * profit - (1 - p)


def probability_edge(model_probability: float, market_odds: float):
    market_prob = american_implied_probability(market_odds)
    if market_prob is None:
        return None
    return model_probability * 100 - market_prob


def matchup_pitcher_split_k_pct(statcast_metrics: dict, lineup_enriched: list[dict]):
    if not lineup_enriched:
        return None
    l_count = sum(1 for r in lineup_enriched if str(r.get('bat_side', '')).upper() == 'L')
    r_count = sum(1 for r in lineup_enriched if str(r.get('bat_side', '')).upper() == 'R')
    s_count = sum(1 for r in lineup_enriched if str(r.get('bat_side', '')).upper() == 'S')
    if l_count + r_count + s_count == 0:
        return None
    vs_l = safe_num(statcast_metrics.get('vs_l', {}).get('K%'))
    vs_r = safe_num(statcast_metrics.get('vs_r', {}).get('K%'))
    switch_rate = None
    if vs_l is not None and vs_r is not None:
        switch_rate = (vs_l + vs_r) / 2
    elif vs_l is not None:
        switch_rate = vs_l
    elif vs_r is not None:
        switch_rate = vs_r
    pieces = []
    if l_count and vs_l is not None:
        pieces.append((vs_l, l_count))
    if r_count and vs_r is not None:
        pieces.append((vs_r, r_count))
    if s_count and switch_rate is not None:
        pieces.append((switch_rate, s_count))
    return weighted_average(pieces)


def arsenal_quality_adjustment(statcast_metrics: dict):
    whiff = safe_num(statcast_metrics.get('whiff_pct'))
    csw = safe_num(statcast_metrics.get('csw_pct'))
    adjustment = 0.0
    if whiff is not None:
        adjustment += clamp((whiff - 24.0) * 0.10, -1.5, 1.5)
    if csw is not None:
        adjustment += clamp((csw - 27.0) * 0.12, -1.2, 1.2)
    return clamp(adjustment, -2.0, 2.0)


def recent_form_adjustment(recent_summary: dict, season_k_per_start):
    recent_k = safe_num(recent_summary.get('avg_k'))
    season_k = safe_num(season_k_per_start)
    if recent_k is None or season_k is None:
        return 0.0
    return clamp((recent_k - season_k) * 0.35, -0.8, 0.8)


def context_adjustment(game_context: dict):
    temp = safe_num(game_context.get('temperature'))
    wind = str(game_context.get('wind') or '').lower()
    adj = 0.0
    if temp is not None:
        if temp <= 50:
            adj += 0.15
        elif temp >= 90:
            adj -= 0.10
    if 'in from' in wind:
        adj += 0.05
    elif 'out to' in wind:
        adj -= 0.05
    return clamp(adj, -0.2, 0.2)


def build_projection(pitcher_stats, recent_summary, statcast_metrics, opponent_stats, opponent_split_stats, lineup_enriched, game_context):
    season_k_pct = safe_num(pitcher_stats.get('calc_k_pct'))
    season_bf_start = safe_num(pitcher_stats.get('calc_bf_start'))
    season_k_total = safe_num(pitcher_stats.get('strikeOuts'))
    season_starts = safe_num(pitcher_stats.get('gamesStarted'))
    season_k_per_start = None
    if season_k_total is not None and season_starts and season_starts > 0:
        season_k_per_start = season_k_total / season_starts
    recent_bf = safe_num(recent_summary.get('avg_bf'))
    projected_bf = weighted_average([(season_bf_start, 0.60), (recent_bf, 0.40)], default=season_bf_start or recent_bf or 22.0)
    projected_bf = clamp(projected_bf, 14.0, 32.0)
    opponent_general_k = safe_num(opponent_stats.get('calcKPercent'))
    opponent_split_k = safe_num(opponent_split_stats.get('calcKPercent'))
    lineup_k = lineup_weighted_k_pct(lineup_enriched)
    pitcher_lineup_split_k = matchup_pitcher_split_k_pct(statcast_metrics, lineup_enriched)
    matchup_k_pct = weighted_average([
        (season_k_pct, 0.38),
        (opponent_split_k, 0.22),
        (lineup_k, 0.16),
        (pitcher_lineup_split_k, 0.14),
        (opponent_general_k, 0.10),
    ], default=season_k_pct or opponent_split_k or opponent_general_k or 22.0)
    arsenal_adj = arsenal_quality_adjustment(statcast_metrics)
    adjusted_k_pct = clamp(matchup_k_pct + arsenal_adj, 8.0, 45.0)
    base_projection = projected_bf * adjusted_k_pct / 100.0
    form_adj = recent_form_adjustment(recent_summary, season_k_per_start)
    game_adj = context_adjustment(game_context)
    central = clamp(base_projection + form_adj + game_adj, 0.5, 14.0)
    sigma = max(1.35, math.sqrt(central) * 0.78)
    low = max(0.0, central - 1.35 * sigma)
    high = central + 1.35 * sigma
    return {
        'central': central,
        'low': low,
        'high': high,
        'projected_bf': projected_bf,
        'projected_k_pct': adjusted_k_pct,
        'base_matchup_k_pct': matchup_k_pct,
        'arsenal_adj_pct_points': arsenal_adj,
        'form_adj_k': form_adj,
        'game_adj_k': game_adj,
    }


def model_grade(ev_value, edge_points, data_completeness):
    if ev_value is None or edge_points is None:
        return 'NO BET', 'Falta mercado'
    ev_pct = ev_value * 100
    if data_completeness < 0.70:
        return 'PASS', 'Datos incompletos'
    if ev_pct >= 12 and edge_points >= 7:
        return 'A', 'Premium'
    if ev_pct >= 7 and edge_points >= 4:
        return 'B', 'Fuerte'
    if ev_pct >= 3 and edge_points >= 2:
        return 'C', 'Ligera'
    if ev_pct > 0:
        return 'D', 'Edge pequeÃ±o'
    return 'PASS', 'Sin edge'


st.set_page_config(
    page_title="MLB Strikeout Predictor",
    page_icon="â¾",
    layout="centered",
)

st.title("MLB Starting Pitcher Strikeout Predictor")
st.caption("V0.4 â 9 Module Research Dashboard Â· MLB Stats API + Baseball Savant/Statcast")

game_date = st.date_input(
    "Game date",
    value=date.today(),
    min_value=date(2000, 1, 1),
)

source_url = (
    f"{MLB_SCHEDULE_URL}?"
    f"{urlencode({'sportId': 1, 'date': game_date.isoformat()})}"
)
st.caption(f"Live schedule source: [MLB Stats API]({source_url})")

try:
    matchup_data = fetch_pitchers_for_date(game_date.isoformat())
except MLBApiError as exc:
    st.error(str(exc))
    st.stop()

pitchers = matchup_data["pitchers"]
if not pitchers:
    st.warning("No probable starting pitchers are currently available for this date.")
    st.stop()

pitcher_by_id = {p["selection_id"]: p for p in pitchers}
selected_id = st.selectbox(
    "Probable starting pitcher",
    options=list(pitcher_by_id),
    format_func=lambda sid: (
        f"{pitcher_by_id[sid]['pitcher_name']} â "
        f"{pitcher_by_id[sid]['team']} vs {pitcher_by_id[sid]['opponent']}"
    ),
)
selected_pitcher = pitcher_by_id[selected_id]

st.subheader("Selected Matchup")
st.write(f"**Pitcher:** {selected_pitcher['pitcher_name']}")
st.write(f"**Team:** {selected_pitcher['team']}")
st.write(f"**Opponent:** {selected_pitcher['opponent']}")
st.write(f"**Throwing hand:** {selected_pitcher['throwing_hand']}")
st.write(f"**Game time:** {selected_pitcher['game_time']}")
st.write(f"**Venue:** {selected_pitcher['venue']}")

pitcher_stats = {}
recent_starts = []
opponent_stats = {}
opponent_split_stats = {}
game_feed = {}

try:
    pitcher_stats = fetch_pitcher_season_stats(selected_pitcher["pitcher_id"], game_date.year)
    recent_starts = fetch_pitcher_game_log(selected_pitcher["pitcher_id"], game_date.year)

    if selected_pitcher.get("opponent_id"):
        opponent_stats = fetch_team_hitting_stats(selected_pitcher["opponent_id"], game_date.year)

        split_code = (
            "vl"
            if selected_pitcher["throwing_hand"].lower().startswith("left")
            else "vr"
        )
        opponent_split_stats = fetch_team_hitting_stats(
            selected_pitcher["opponent_id"],
            game_date.year,
            split_code,
        )
except MLBApiError:
    pass

try:
    game_feed = fetch_game_feed(selected_pitcher["game_pk"])
except MLBApiError:
    game_feed = {}

statcast_start = date(game_date.year, 3, 1)
with st.spinner("Loading Baseball Savant / Statcast..."):
    statcast_df = fetch_statcast_data(
        selected_pitcher["pitcher_id"],
        statcast_start.isoformat(),
        game_date.isoformat(),
    )

statcast_metrics = calculate_statcast_metrics(statcast_df)
recent_summary = recent_start_summary(recent_starts)
game_context = parse_game_context(game_feed)

lineup = parse_confirmed_lineup(game_feed, selected_pitcher["opponent_side"])
lineup_enriched = []
if lineup:
    with st.spinner("Loading confirmed-lineup strikeout profiles..."):
        lineup_enriched = enrich_lineup(
            lineup,
            game_date.year,
            selected_pitcher["throwing_hand"],
        )

st.divider()

# M1
st.header("M1 Â· Capacidad real de K â 20%")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("K/9", pitcher_stats.get("strikeoutsPer9Inn", "N/A"))
    st.metric("K%", fmt(pitcher_stats.get("calc_k_pct"), 1, "%"))
    st.metric("Whiff%", fmt(statcast_metrics["whiff_pct"], 1, "%"))

with c2:
    st.metric("BB/9", pitcher_stats.get("walksPer9Inn", "N/A"))
    st.metric("BB%", fmt(pitcher_stats.get("calc_bb_pct"), 1, "%"))
    st.metric("CSW%", fmt(statcast_metrics["csw_pct"], 1, "%"))

with c3:
    st.metric("K-BB%", fmt(pitcher_stats.get("calc_k_minus_bb"), 1, "%"))
    st.metric("WHIP", pitcher_stats.get("whip", "N/A"))
    st.metric("Fastball velo", fmt(statcast_metrics["fastball_velo"], 1, " mph"))

st.caption("Whiff%, CSW% and pitch velocity are calculated from Baseball Savant/Statcast.")

# M2
st.header("M2 Â· Volumen / Leash â 20%")
c1, c2, c3 = st.columns(3)

with c1:
    st.metric("GS", pitcher_stats.get("gamesStarted", "N/A"))
    st.metric("IP", pitcher_stats.get("inningsPitched", "N/A"))
    st.metric("Avg K â last 5", fmt(recent_summary.get("avg_k"), 1))

with c2:
    st.metric("BF", pitcher_stats.get("battersFaced", "N/A"))
    st.metric("BF/start", fmt(pitcher_stats.get("calc_bf_start"), 1))
    st.metric("Avg BF â last 5", fmt(recent_summary.get("avg_bf"), 1))

with c3:
    st.metric("IP/start", fmt(pitcher_stats.get("calc_ip_start"), 2))
    st.metric("Avg pitches â last 5", fmt(recent_summary.get("avg_pitches"), 1))
    st.metric("Max pitches â last 5", fmt(recent_summary.get("max_pitches"), 0))

# M3
st.header("M3 Â· Splits â 10%")
c1, c2 = st.columns(2)

with c1:
    st.metric("K% vs LHB", fmt(statcast_metrics["vs_l"]["K%"], 1, "%"))
    st.caption(f"{statcast_metrics['vs_l']['K']} K / {statcast_metrics['vs_l']['PA']} PA")
    st.metric("K% Home", fmt(statcast_metrics["home"]["K%"], 1, "%"))

with c2:
    st.metric("K% vs RHB", fmt(statcast_metrics["vs_r"]["K%"], 1, "%"))
    st.caption(f"{statcast_metrics['vs_r']['K']} K / {statcast_metrics['vs_r']['PA']} PA")
    st.metric("K% Away", fmt(statcast_metrics["away"]["K%"], 1, "%"))

st.caption("L/R and Home/Away splits are calculated from Statcast plate-appearance outcomes.")

# M4
st.header("M4 Â· PropensiÃ³n del rival a poncharse â 20%")
general_k = opponent_stats.get("calcKPercent")
split_k = opponent_split_stats.get("calcKPercent")
split_label = (
    "Opponent K% vs LHP"
    if selected_pitcher["throwing_hand"].lower().startswith("left")
    else "Opponent K% vs RHP"
)

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Opponent K%", fmt(general_k, 1, "%"))
with c2:
    st.metric(split_label, fmt(split_k, 1, "%"))
with c3:
    st.metric("Confirmed lineup K%", fmt(lineup_weighted_k_pct(lineup_enriched), 1, "%"))

if split_k is None:
    st.caption("Team handedness split was not returned by MLB for this matchup; general team K% remains available.")

# M5
st.header("M5 Â· Arsenal vs Matchup â 15%")
if statcast_metrics["arsenal"]:
    arsenal_df = pd.DataFrame(statcast_metrics["arsenal"])
    st.dataframe(
        arsenal_df[["Pitch", "Usage%", "Velo", "Whiff%"]],
        use_container_width=True,
        hide_index=True,
    )
else:
    st.info("No Statcast arsenal data available.")

if lineup_enriched:
    bat_counts = pd.Series([row.get("bat_side", "N/A") for row in lineup_enriched]).value_counts().to_dict()
    st.caption(
        "Confirmed lineup handedness: "
        f"L {bat_counts.get('L', 0)} Â· R {bat_counts.get('R', 0)} Â· "
        f"S {bat_counts.get('S', 0)} Â· Other {bat_counts.get('N/A', 0)}"
    )

# M6
st.header("M6 Â· Forma / Cambios recientes â 5%")
if recent_starts:
    for start in recent_starts[:5]:
        st.write(
            f"**{start['date']} vs {start['opponent']}** â "
            f"{start['IP']} IP Â· {start['K']} K Â· {start['BB']} BB Â· "
            f"{start['ER']} ER Â· {start['HR']} HR Â· {start['Pitches']} pitches"
        )
else:
    st.info("No recent starts available.")

# M7
st.header("M7 Â· Contexto del juego â 5%")
c1, c2 = st.columns(2)

with c1:
    st.metric("Venue", game_context.get("venue") or selected_pitcher["venue"])
    st.metric("Temperature", f"{game_context['temperature']}Â°F" if game_context.get("temperature") is not None else "N/A")
    st.metric("Umpire", game_context.get("umpire") or "N/A")

with c2:
    st.metric("Status", selected_pitcher["status"])
    st.metric("Weather", game_context.get("condition") or "N/A")
    st.metric("Wind", game_context.get("wind") or "N/A")

st.caption("Park strikeout factor is intentionally left N/A until a reliable park-factor source is connected.")

# M8
st.header("M8 Â· Lineup confirmado â 5%")
if lineup_enriched:
    lineup_df = pd.DataFrame(
        [
            {
                "#": row["order"],
                "Hitter": row["name"],
                "Bats": row["bat_side"],
                "K% vs hand": round(row["K%"], 1) if row["K%"] is not None else None,
                "PA": int(row["PA"]) if row["PA"] is not None else None,
                "Source": row["source"],
            }
            for row in lineup_enriched
        ]
    )
    st.dataframe(lineup_df, use_container_width=True, hide_index=True)
    st.success("Confirmed batting order is available from the MLB game feed.")
else:
    st.info("MLB has not published a confirmed batting order for this opponent yet.")

# M9
st.header("M9 Â· Mercado / LÃ­neas / Edge")
st.caption("Introduce las cuotas de Action Network Pro. El modelo no inventa mercado.")

prop_line = st.number_input("Strikeout line", min_value=0.5, max_value=15.5, value=5.5, step=1.0)
c1, c2 = st.columns(2)
with c1:
    over_odds = st.number_input("Over odds (American)", min_value=-1000, max_value=2000, value=-110, step=5)
with c2:
    under_odds = st.number_input("Under odds (American)", min_value=-1000, max_value=2000, value=-110, step=5)

market_notes = st.text_area(
    "Action Network notes (optional)",
    placeholder="Sharp Money, % bets, % money, line movement, alternate K lines...",
)

projection = build_projection(
    pitcher_stats,
    recent_summary,
    statcast_metrics,
    opponent_stats,
    opponent_split_stats,
    lineup_enriched,
    game_context,
)

module_status = {
    "M1": bool(pitcher_stats and not statcast_df.empty),
    "M2": bool(pitcher_stats and recent_starts),
    "M3": bool(not statcast_df.empty),
    "M4": bool(opponent_stats),
    "M5": bool(statcast_metrics["arsenal"]),
    "M6": bool(recent_starts),
    "M7": bool(game_context),
    "M8": bool(lineup_enriched),
    "M9": True,
}
completed = sum(module_status.values())
data_completeness = completed / 9
lam = projection["central"]

prob_table = []
for threshold in range(4, 10):
    p = poisson_prob_ge(threshold, lam)
    fair = fair_american_odds(p)
    prob_table.append({
        "K threshold": f"{threshold}+",
        "Probability": f"{p * 100:.1f}%",
        "Fair odds": f"{fair:+.0f}" if fair is not None else "N/A",
    })

over_threshold = math.floor(prop_line) + 1
under_max = math.floor(prop_line)
over_prob = poisson_prob_ge(over_threshold, lam)
under_prob = poisson_cdf(under_max, lam)
over_fair = fair_american_odds(over_prob)
under_fair = fair_american_odds(under_prob)
over_edge = probability_edge(over_prob, over_odds)
under_edge = probability_edge(under_prob, under_odds)
over_ev = ev_per_unit(over_prob, over_odds)
under_ev = ev_per_unit(under_prob, under_odds)

best_side = "OVER" if (over_ev if over_ev is not None else -999) >= (under_ev if under_ev is not None else -999) else "UNDER"
best_prob = over_prob if best_side == "OVER" else under_prob
best_edge = over_edge if best_side == "OVER" else under_edge
best_ev = over_ev if best_side == "OVER" else under_ev
best_odds = over_odds if best_side == "OVER" else under_odds
best_fair = over_fair if best_side == "OVER" else under_fair
grade, grade_label = model_grade(best_ev, best_edge, data_completeness)

st.divider()
st.header("PredicciÃ³n FINAL")
c1, c2, c3 = st.columns(3)
with c1:
    st.metric("ProyecciÃ³n central", fmt(projection["central"], 2, " K"))
    st.metric("Projected BF", fmt(projection["projected_bf"], 1))
with c2:
    st.metric("Rango probable", f"{projection['low']:.1f} â {projection['high']:.1f} K")
    st.metric("Projected K%", fmt(projection["projected_k_pct"], 1, "%"))
with c3:
    st.metric("Data completeness", f"{completed}/9")
    st.metric("Model grade", f"{grade} Â· {grade_label}")

st.subheader("DistribuciÃ³n de strikeouts")
st.dataframe(pd.DataFrame(prob_table), use_container_width=True, hide_index=True)

st.subheader(f"Mercado principal Â· {prop_line:.1f} K")
c1, c2 = st.columns(2)
with c1:
    st.markdown("### OVER")
    st.metric("Model probability", f"{over_prob * 100:.1f}%")
    st.metric("Fair odds", f"{over_fair:+.0f}" if over_fair is not None else "N/A")
    st.metric("Market odds", f"{over_odds:+d}")
    st.metric("Edge", fmt(over_edge, 1, " pp"))
    st.metric("EV / $1", fmt(over_ev, 3))
with c2:
    st.markdown("### UNDER")
    st.metric("Model probability", f"{under_prob * 100:.1f}%")
    st.metric("Fair odds", f"{under_fair:+.0f}" if under_fair is not None else "N/A")
    st.metric("Market odds", f"{under_odds:+d}")
    st.metric("Edge", fmt(under_edge, 1, " pp"))
    st.metric("EV / $1", fmt(under_ev, 3))

st.subheader("Dictamen")
if grade in ("A", "B", "C"):
    st.success(
        f"**{best_side} {prop_line:.1f} K Â· {best_odds:+d}** â "
        f"Modelo {best_prob * 100:.1f}% Â· Fair {best_fair:+.0f} Â· "
        f"Edge {best_edge:.1f} pp Â· EV {best_ev * 100:.1f}% Â· "
        f"Grade {grade} ({grade_label})"
    )
else:
    st.warning(
        f"**{grade}** â mejor lado actual: {best_side}, pero no supera el filtro del modelo con suficiente fuerza."
    )

with st.expander("Ver cÃ³mo se construyÃ³ la proyecciÃ³n"):
    st.write({
        "Base matchup K%": round(projection["base_matchup_k_pct"], 2),
        "Arsenal adjustment (K%-points)": round(projection["arsenal_adj_pct_points"], 2),
        "Recent form adjustment (K)": round(projection["form_adj_k"], 2),
        "Game context adjustment (K)": round(projection["game_adj_k"], 2),
        "Projected BF": round(projection["projected_bf"], 2),
        "Final projected K%": round(projection["projected_k_pct"], 2),
    })
    st.caption(
        "Los coeficientes son transparentes y conservadores. V1.0 es funcional, pero deben calibrarse mediante backtesting antes de considerarla estadÃ­sticamente validada."
    )

st.divider()
st.subheader("Estado del modelo")
st.progress(data_completeness)
st.write(f"**MÃ³dulos disponibles: {completed}/9**")
st.caption(" Â· ".join(f"{name} {'â' if ok else 'â³'}" for name, ok in module_status.items()))
if not lineup_enriched:
    st.info("M8 lineup confirmado aÃºn no estÃ¡ disponible. La proyecciÃ³n funciona y se vuelve mÃ¡s especÃ­fica cuando MLB publique el batting order.")
if not opponent_split_stats:
    st.info("El split de equipo vs mano del pitcher no fue devuelto por MLB; el modelo repondera automÃ¡ticamente los otros inputs disponibles.")
st.warning(
    "V1.0 FUNCIONAL: mÃ³dulos, proyecciÃ³n, rango, P(4+â¦9+), fair odds, edge, EV y calificaciÃ³n estÃ¡n activos. La siguiente fase es validar y calibrar; no reconstruir la app."
)
