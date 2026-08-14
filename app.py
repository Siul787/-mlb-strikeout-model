from datetime import date, datetime, timezone, timedelta
from urllib.parse import urlencode

import pandas as pd
import requests
import streamlit as st

from pybaseball import statcast_pitcher


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
MLB_GAME_URL = "https://statsapi.mlb.com/api/v1.1/game"

REQUEST_TIMEOUT_SECONDS = 20


# =========================================================
# GENERAL HELPERS
# =========================================================

class MLBApiError(Exception):
    pass


def get_json(url: str, params: dict | None = None) -> dict:
    try:
        response = requests.get(
            url,
            params=params,
            timeout=REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MLBApiError(
            "The MLB Stats API could not be reached."
        ) from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise MLBApiError(
            "The MLB Stats API returned invalid JSON."
        ) from exc

    if not isinstance(payload, dict):
        raise MLBApiError(
            "Unexpected response from MLB Stats API."
        )

    return payload


def safe_num(value):
    if isinstance(value, (int, float)):
        return value

    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def fmt(value, decimals=1, suffix=""):
    number = safe_num(value)

    if number is None:
        return "N/A"

    return f"{number:.{decimals}f}{suffix}"


def safe_mean(series):
    if series is None:
        return None

    values = pd.to_numeric(series, errors="coerce").dropna()

    if values.empty:
        return None

    return float(values.mean())


# =========================================================
# MLB PLAYER / SCHEDULE
# =========================================================

@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_profile(player_id: int) -> dict:
    payload = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}"
    )

    people = payload.get("people", [])

    if not people:
        raise MLBApiError(
            "MLB did not return this pitcher profile."
        )

    return people[0]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_season_stats(
    player_id: int,
    season: int,
) -> dict:

    payload = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={
            "stats": "season",
            "group": "pitching",
            "season": season,
        },
    )

    groups = payload.get("stats", [])

    if not groups:
        return {}

    splits = groups[0].get("splits", [])

    if not splits:
        return {}

    stat = splits[0].get("stat", {})

    if not isinstance(stat, dict):
        return {}

    strikeouts = safe_num(
        stat.get("strikeOuts")
    )

    walks = safe_num(
        stat.get("baseOnBalls")
    )

    batters_faced = safe_num(
        stat.get("battersFaced")
    )

    innings = safe_num(
        stat.get("inningsPitched")
    )

    starts = safe_num(
        stat.get("gamesStarted")
    )

    k_pct = None
    bb_pct = None
    k_minus_bb = None
    bf_start = None
    ip_start = None

    if (
        strikeouts is not None
        and batters_faced
        and batters_faced > 0
    ):
        k_pct = (
            strikeouts / batters_faced
        ) * 100

    if (
        walks is not None
        and batters_faced
        and batters_faced > 0
    ):
        bb_pct = (
            walks / batters_faced
        ) * 100

    if (
        k_pct is not None
        and bb_pct is not None
    ):
        k_minus_bb = (
            k_pct - bb_pct
        )

    if starts and starts > 0:

        if batters_faced is not None:
            bf_start = (
                batters_faced / starts
            )

        if innings is not None:
            ip_start = (
                innings / starts
            )

    stat["calcKPercent"] = k_pct
    stat["calcBBPercent"] = bb_pct
    stat["calcKMinusBB"] = k_minus_bb
    stat["calcBFStart"] = bf_start
    stat["calcIPStart"] = ip_start

    return stat


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_game_log(
    player_id: int,
    season: int,
) -> list[dict]:

    payload = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
        },
    )

    groups = payload.get("stats", [])

    if not groups:
        return []

    splits = groups[0].get(
        "splits",
        [],
    )

    starts = []

    for split in splits:

        stat = split.get(
            "stat",
            {},
        )

        if not stat.get(
            "gamesStarted"
        ):
            continue

        opponent = split.get(
            "opponent",
            {},
        )

        starts.append(
            {
                "date":
                    split.get(
                        "date",
                        "N/A",
                    ),

                "opponent":
                    opponent.get(
                        "name",
                        "N/A",
                    ),

                "IP":
                    stat.get(
                        "inningsPitched",
                        "N/A",
                    ),

                "K":
                    stat.get(
                        "strikeOuts",
                        "N/A",
                    ),

                "BB":
                    stat.get(
                        "baseOnBalls",
                        "N/A",
                    ),

                "ER":
                    stat.get(
                        "earnedRuns",
                        "N/A",
                    ),

                "HR":
                    stat.get(
                        "homeRuns",
                        "N/A",
                    ),

                "Pitches":
                    stat.get(
                        "numberOfPitches",
                        "N/A",
                    ),

                "BF":
                    stat.get(
                        "battersFaced",
                        "N/A",
                    ),
            }
        )

    return starts[-10:][::-1]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_team_hitting_stats(
    team_id: int,
    season: int,
) -> dict:

    payload = get_json(
        f"{MLB_TEAMS_URL}/{team_id}/stats",
        params={
            "stats": "season",
            "group": "hitting",
            "season": season,
        },
    )

    groups = payload.get("stats", [])

    if not groups:
        return {}

    splits = groups[0].get(
        "splits",
        [],
    )

    if not splits:
        return {}

    stat = splits[0].get(
        "stat",
        {},
    )

    strikeouts = safe_num(
        stat.get("strikeOuts")
    )

    pa = safe_num(
        stat.get("plateAppearances")
    )

    k_pct = None

    if (
        strikeouts is not None
        and pa
        and pa > 0
    ):
        k_pct = (
            strikeouts / pa
        ) * 100

    stat["calcKPercent"] = k_pct

    return stat


def format_game_time(
    game_date: str | None,
):

    if not game_date:
        return "Time TBD"

    try:
        parsed = datetime.fromisoformat(
            game_date.replace(
                "Z",
                "+00:00",
            )
        )

    except ValueError:
        return "Time TBD"

    label = (
        parsed
        .astimezone(
            timezone.utc
        )
        .strftime(
            "%I:%M %p"
        )
        .lstrip("0")
    )

    return (
        f"{parsed.strftime('%B')} "
        f"{parsed.day}, "
        f"{label} UTC"
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitchers_for_date(
    selected_date: str,
):

    payload = get_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": selected_date,
            "hydrate":
                "probablePitcher,"
                "team,"
                "opponents",
        },
    )

    options = []
    games_count = 0

    for date_group in payload.get(
        "dates",
        [],
    ):

        games = date_group.get(
            "games",
            [],
        )

        games_count += len(games)

        for game in games:

            teams = game.get(
                "teams",
                {},
            )

            venue = game.get(
                "venue",
                {},
            )

            for (
                side,
                opponent_side,
            ) in (
                ("away", "home"),
                ("home", "away"),
            ):

                team_data = teams.get(
                    side,
                    {},
                )

                opponent_data = teams.get(
                    opponent_side,
                    {},
                )

                probable = team_data.get(
                    "probablePitcher"
                )

                if not isinstance(
                    probable,
                    dict,
                ):
                    continue

                pitcher_id = probable.get(
                    "id"
                )

                pitcher_name = probable.get(
                    "fullName"
                )

                if (
                    not pitcher_id
                    or not pitcher_name
                ):
                    continue

                team = team_data.get(
                    "team",
                    {},
                )

                opponent = opponent_data.get(
                    "team",
                    {},
                )

                throwing_hand = "N/A"

                try:

                    profile = (
                        fetch_pitcher_profile(
                            int(
                                pitcher_id
                            )
                        )
                    )

                    pitch_hand = (
                        profile.get(
                            "pitchHand",
                            {},
                        )
                    )

                    throwing_hand = (
                        pitch_hand.get(
                            "description",
                            "N/A",
                        )
                    )

                except MLBApiError:
                    pass

                options.append(
                    {
                        "selection_id":
                            (
                                f"{game.get('gamePk')}-"
                                f"{side}-"
                                f"{pitcher_id}"
                            ),

                        "game_pk":
                            game.get(
                                "gamePk"
                            ),

                        "pitcher_id":
                            int(
                                pitcher_id
                            ),

                        "pitcher_name":
                            pitcher_name,

                        "team":
                            team.get(
                                "name",
                                "N/A",
                            ),

                        "team_id":
                            team.get(
                                "id"
                            ),

                        "opponent":
                            opponent.get(
                                "name",
                                "N/A",
                            ),

                        "opponent_id":
                            opponent.get(
                                "id"
                            ),

                        "throwing_hand":
                            throwing_hand,

                        "venue":
                            venue.get(
                                "name",
                                "N/A",
                            ),

                        "status":
                            game.get(
                                "status",
                                {},
                            ).get(
                                "detailedState",
                                "N/A",
                            ),

                        "game_time":
                            format_game_time(
                                game.get(
                                    "gameDate"
                                )
                            ),
                    }
                )

    return {
        "pitchers": options,
        "scheduled_games": games_count,
    }


# =========================================================
# STATCAST / BASEBALL SAVANT
# =========================================================

@st.cache_data(
    ttl=3600,
    show_spinner=False,
)
def fetch_statcast_data(
    player_id: int,
    start_date: str,
    end_date: str,
):

    try:

        df = statcast_pitcher(
            start_date,
            end_date,
            player_id,
        )

    except Exception:
        return pd.DataFrame()

    if df is None:
        return pd.DataFrame()

    return df


def calculate_statcast_metrics(
    df: pd.DataFrame,
):

    result = {
        "pitches": None,
        "velocity": None,
        "whiff_pct": None,
        "csw_pct": None,
        "arsenal": [],
        "vs_l_k": None,
        "vs_l_pa": None,
        "vs_l_k_pct": None,
        "vs_r_k": None,
        "vs_r_pa": None,
        "vs_r_k_pct": None,
    }

    if df.empty:
        return result

    total_pitches = len(df)

    result["pitches"] = total_pitches

    if "release_speed" in df.columns:
        result["velocity"] = safe_mean(
            df["release_speed"]
        )

    descriptions = (
        df["description"]
        if "description" in df.columns
        else pd.Series(dtype=str)
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

    whiff_events = {
        "swinging_strike",
        "swinging_strike_blocked",
        "missed_bunt",
    }

    called_strike_events = {
        "called_strike",
    }

    swings = descriptions.isin(
        swing_events
    ).sum()

    whiffs = descriptions.isin(
        whiff_events
    ).sum()

    called_strikes = descriptions.isin(
        called_strike_events
    ).sum()

    if swings > 0:

        result["whiff_pct"] = (
            whiffs / swings
        ) * 100

    if total_pitches > 0:

        result["csw_pct"] = (
            (
                whiffs
                + called_strikes
            )
            / total_pitches
        ) * 100

    if "pitch_type" in df.columns:

        pitch_rows = []

        for pitch_type, group in df.groupby(
            "pitch_type"
        ):

            usage = (
                len(group)
                / total_pitches
                * 100
            )

            velocity = None

            if (
                "release_speed"
                in group.columns
            ):
                velocity = safe_mean(
                    group[
                        "release_speed"
                    ]
                )

            descriptions_group = (
                group["description"]
                if "description"
                in group.columns
                else pd.Series(dtype=str)
            )

            pitch_swings = (
                descriptions_group.isin(
                    swing_events
                ).sum()
            )

            pitch_whiffs = (
                descriptions_group.isin(
                    whiff_events
                ).sum()
            )

            pitch_whiff_pct = None

            if pitch_swings > 0:

                pitch_whiff_pct = (
                    pitch_whiffs
                    / pitch_swings
                ) * 100

            pitch_rows.append(
                {
                    "Pitch":
                        pitch_type,

                    "Usage%":
                        round(
                            usage,
                            1,
                        ),

                    "Velo":
                        (
                            round(
                                velocity,
                                1,
                            )
                            if velocity
                            is not None
                            else None
                        ),

                    "Whiff%":
                        (
                            round(
                                pitch_whiff_pct,
                                1,
                            )
                            if pitch_whiff_pct
                            is not None
                            else None
                        ),
                }
            )

        pitch_rows.sort(
            key=lambda x:
                x["Usage%"],
            reverse=True,
        )

        result["arsenal"] = pitch_rows


    # =====================================================
    # SPLITS BY BATTER SIDE
    # =====================================================

    if (
        "events" in df.columns
        and "stand" in df.columns
    ):

        pa_rows = df[
            df["events"].notna()
        ].copy()

        for side in (
            "L",
            "R",
        ):

            side_df = pa_rows[
                pa_rows["stand"]
                == side
            ]

            pa = len(side_df)

            strikeouts = (
                side_df[
                    "events"
                ]
                .astype(str)
                .str.contains(
                    "strikeout",
                    case=False,
                    na=False,
                )
                .sum()
            )

            k_pct = None

            if pa > 0:

                k_pct = (
                    strikeouts
                    / pa
                ) * 100

            if side == "L":

                result["vs_l_k"] = (
                    int(strikeouts)
                )

                result["vs_l_pa"] = pa

                result["vs_l_k_pct"] = (
                    k_pct
                )

            else:

                result["vs_r_k"] = (
                    int(strikeouts)
                )

                result["vs_r_pa"] = pa

                result["vs_r_k_pct"] = (
                    k_pct
                )

    return result


# =========================================================
# STREAMLIT UI
# =========================================================

st.set_page_config(
    page_title=
        "MLB Strikeout Predictor",
    page_icon="⚾",
    layout="centered",
)


st.title(
    "MLB Starting Pitcher "
    "Strikeout Predictor"
)

st.caption(
    "V0.3 — MLB + Statcast "
    "9 Module Research Dashboard"
)


game_date = st.date_input(
    "Game date",
    value=date.today(),
    min_value=date(
        2000,
        1,
        1,
    ),
)


source_url = (
    f"{MLB_SCHEDULE_URL}?"
    f"{urlencode({
        'sportId': 1,
        'date': game_date.isoformat()
    })}"
)

st.caption(
    f"Live source: "
    f"[MLB Stats API]"
    f"({source_url})"
)


try:

    matchup_data = (
        fetch_pitchers_for_date(
            game_date.isoformat()
        )
    )

except MLBApiError as exc:

    st.error(str(exc))
    st.stop()


pitchers = (
    matchup_data[
        "pitchers"
    ]
)


if not pitchers:

    st.warning(
        "No probable starting pitchers "
        "are currently available "
        "for this date."
    )

    st.stop()


pitcher_by_id = {
    p["selection_id"]: p
    for p in pitchers
}


selected_id = st.selectbox(
    "Probable starting pitcher",
    options=list(
        pitcher_by_id
    ),
    format_func=lambda sid: (
        f"{pitcher_by_id[sid]['pitcher_name']} — "
        f"{pitcher_by_id[sid]['team']} vs "
        f"{pitcher_by_id[sid]['opponent']}"
    ),
)


selected_pitcher = (
    pitcher_by_id[
        selected_id
    ]
)


st.subheader(
    "Selected Matchup"
)

st.write(
    f"**Pitcher:** "
    f"{selected_pitcher['pitcher_name']}"
)

st.write(
    f"**Team:** "
    f"{selected_pitcher['team']}"
)

st.write(
    f"**Opponent:** "
    f"{selected_pitcher['opponent']}"
)

st.write(
    f"**Throwing hand:** "
    f"{selected_pitcher['throwing_hand']}"
)

st.write(
    f"**Game time:** "
    f"{selected_pitcher['game_time']}"
)

st.write(
    f"**Venue:** "
    f"{selected_pitcher['venue']}"
)


# =========================================================
# LOAD DATA
# =========================================================

pitcher_stats = {}
recent_starts = []
opponent_stats = {}


try:

    pitcher_stats = (
        fetch_pitcher_season_stats(
            selected_pitcher[
                "pitcher_id"
            ],
            game_date.year,
        )
    )

    recent_starts = (
        fetch_pitcher_game_log(
            selected_pitcher[
                "pitcher_id"
            ],
            game_date.year,
        )
    )

    if selected_pitcher.get(
        "opponent_id"
    ):

        opponent_stats = (
            fetch_team_hitting_stats(
                selected_pitcher[
                    "opponent_id"
                ],
                game_date.year,
            )
        )

except MLBApiError:
    pass


statcast_start = date(
    game_date.year,
    3,
    1,
)

statcast_end = game_date


with st.spinner(
    "Loading Baseball Savant / Statcast..."
):

    statcast_df = (
        fetch_statcast_data(
            selected_pitcher[
                "pitcher_id"
            ],
            statcast_start.isoformat(),
            statcast_end.isoformat(),
        )
    )


statcast_metrics = (
    calculate_statcast_metrics(
        statcast_df
    )
)


st.divider()


# =========================================================
# MODULE 1
# =========================================================

st.header(
    "M1 · Capacidad real de K — 20%"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "K/9",
        pitcher_stats.get(
            "strikeoutsPer9Inn",
            "N/A",
        ),
    )

    st.metric(
        "K%",
        fmt(
            pitcher_stats.get(
                "calcKPercent"
            ),
            1,
            "%",
        ),
    )

    st.metric(
        "Whiff%",
        fmt(
            statcast_metrics[
                "whiff_pct"
            ],
            1,
            "%",
        ),
    )


with c2:

    st.metric(
        "BB/9",
        pitcher_stats.get(
            "walksPer9Inn",
            "N/A",
        ),
    )

    st.metric(
        "BB%",
        fmt(
            pitcher_stats.get(
                "calcBBPercent"
            ),
            1,
            "%",
        ),
    )

    st.metric(
        "CSW%",
        fmt(
            statcast_metrics[
                "csw_pct"
            ],
            1,
            "%",
        ),
    )


with c3:

    st.metric(
        "K-BB%",
        fmt(
            pitcher_stats.get(
                "calcKMinusBB"
            ),
            1,
            "%",
        ),
    )

    st.metric(
        "WHIP",
        pitcher_stats.get(
            "whip",
            "N/A",
        ),
    )

    st.metric(
        "Avg Velo",
        fmt(
            statcast_metrics[
                "velocity"
            ],
            1,
            " mph",
        ),
    )


# =========================================================
# MODULE 2
# =========================================================

st.header(
    "M2 · Volumen / Leash — 20%"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "GS",
        pitcher_stats.get(
            "gamesStarted",
            "N/A",
        ),
    )

    st.metric(
        "IP",
        pitcher_stats.get(
            "inningsPitched",
            "N/A",
        ),
    )


with c2:

    st.metric(
        "BF",
        pitcher_stats.get(
            "battersFaced",
            "N/A",
        ),
    )

    st.metric(
        "BF/start",
        fmt(
            pitcher_stats.get(
                "calcBFStart"
            ),
            1,
        ),
    )


with c3:

    st.metric(
        "IP/start",
        fmt(
            pitcher_stats.get(
                "calcIPStart"
            ),
            2,
        ),
    )


if recent_starts:

    pitch_values = [
        safe_num(
            start["Pitches"]
        )
        for start in recent_starts[:5]
    ]

    pitch_values = [
        value
        for value in pitch_values
        if value is not None
    ]

    if pitch_values:

        st.metric(
            "Avg pitches — last 5",
            fmt(
                sum(pitch_values)
                / len(pitch_values),
                1,
            ),
        )


# =========================================================
# MODULE 3
# =========================================================

st.header(
    "M3 · Splits — 10%"
)


c1, c2 = st.columns(2)


with c1:

    st.metric(
        "K% vs LHB",
        fmt(
            statcast_metrics[
                "vs_l_k_pct"
            ],
            1,
            "%",
        ),
    )

    st.caption(
        f"{statcast_metrics['vs_l_k']} K / "
        f"{statcast_metrics['vs_l_pa']} PA"
    )


with c2:

    st.metric(
        "K% vs RHB",
        fmt(
            statcast_metrics[
                "vs_r_k_pct"
            ],
            1,
            "%",
        ),
    )

    st.caption(
        f"{statcast_metrics['vs_r_k']} K / "
        f"{statcast_metrics['vs_r_pa']} PA"
    )


st.caption(
    "Splits calculated from "
    "Statcast PA outcomes."
)


# =========================================================
# MODULE 4
# =========================================================

st.header(
    "M4 · Propensión del rival "
    "a poncharse — 20%"
)


c1, c2, c3 = st.columns(3)


with c1:

    st.metric(
        "Opponent K",
        opponent_stats.get(
            "strikeOuts",
            "N/A",
        ),
    )


with c2:

    st.metric(
        "PA",
        opponent_stats.get(
            "plateAppearances",
            "N/A",
        ),
    )


with c3:

    st.metric(
        "Opponent K%",
        fmt(
            opponent_stats.get(
                "calcKPercent"
            ),
            1,
            "%",
        ),
    )


# =========================================================
# MODULE 5
# =========================================================

st.header(
    "M5 · Arsenal vs Matchup — 15%"
)


if statcast_metrics[
    "arsenal"
]:

    arsenal_df = pd.DataFrame(
        statcast_metrics[
            "arsenal"
        ]
    )

    st.dataframe(
        arsenal_df,
        use_container_width=True,
        hide_index=True,
    )

else:

    st.info(
        "No Statcast arsenal data available."
    )


# =========================================================
# MODULE 6
# =========================================================

st.header(
    "M6 · Forma / Cambios recientes — 5%"
)


if recent_starts:

    for start in recent_starts[:5]:

        st.write(
            f"**{start['date']} "
            f"vs {start['opponent']}** — "
            f"{start['IP']} IP · "
            f"{start['K']} K · "
            f"{start['BB']} BB · "
            f"{start['ER']} ER · "
            f"{start['HR']} HR · "
            f"{start['Pitches']} pitches"
        )

else:

    st.info(
        "No recent starts available."
    )


# =========================================================
# MODULE 7
# =========================================================

st.header(
    "M7 · Contexto del juego — 5%"
)


c1, c2 = st.columns(2)


with c1:

    st.metric(
        "Venue",
        selected_pitcher[
            "venue"
        ],
    )


with c2:

    st.metric(
        "Status",
        selected_pitcher[
            "status"
        ],
    )


st.caption(
    "Weather, umpire and park "
    "strikeout factors still pending."
)


# =========================================================
# MODULE 8
# =========================================================

st.header(
    "M8 · Lineup confirmado — 5%"
)


st.info(
    "Automatic confirmed lineup "
    "integration is the next step."
)


# =========================================================
# MODULE 9
# =========================================================

st.header(
    "M9 · Mercado / Líneas / Edge"
)


prop_line = st.number_input(
    "Strikeout line",
    min_value=0.5,
    max_value=15.5,
    value=5.5,
    step=1.0,
)


american_odds = st.number_input(
    "American odds",
    min_value=-500,
    max_value=1000,
    value=-110,
    step=5,
)


st.caption(
    "For now this can be entered "
    "from Action Network Pro. "
    "Automatic odds integration "
    "will come later."
)


# =========================================================
# MODEL STATUS
# =========================================================

st.divider()


st.subheader(
    "Estado del modelo"
)


st.warning(
    "Predicción FINAL todavía desactivada. "
    "V0.3 now adds Baseball Savant / Statcast "
    "for Whiff%, CSW%, velocity, arsenal "
    "and batter-handedness splits. "
    "Next we complete lineup, opponent split "
    "and game context before activating "
    "projection, probabilities, fair odds, "
    "edge and EV."
)
