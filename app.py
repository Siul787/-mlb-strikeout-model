from datetime import date, datetime, timezone
from urllib.parse import urlencode

import requests
import streamlit as st


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
MLB_TEAMS_URL = "https://statsapi.mlb.com/api/v1/teams"
REQUEST_TIMEOUT_SECONDS = 20


class MLBApiError(Exception):
    """Raised when the official MLB Stats API cannot provide valid data."""


def get_json(url: str, params: dict[str, object] | None = None) -> dict:
    try:
        response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise MLBApiError("The MLB Stats API could not be reached.") from exc

    try:
        payload = response.json()
    except ValueError as exc:
        raise MLBApiError("The MLB Stats API returned an invalid response.") from exc

    if not isinstance(payload, dict):
        raise MLBApiError("The MLB Stats API returned an unexpected response.")

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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_profile(player_id: int) -> dict:
    payload = get_json(f"{MLB_PEOPLE_URL}/{player_id}")
    people = payload.get("people", [])
    if not people or not isinstance(people[0], dict):
        raise MLBApiError("The MLB Stats API did not return this pitcher's profile.")
    return people[0]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_season_stats(player_id: int, season: int) -> dict:
    payload = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={
            "stats": "season",
            "group": "pitching",
            "season": season,
        },
    )

    stats_groups = payload.get("stats", [])
    if not stats_groups:
        return {}

    splits = stats_groups[0].get("splits", [])
    if not splits:
        return {}

    stat = splits[0].get("stat", {})
    if not isinstance(stat, dict):
        return {}

    strikeouts = safe_num(stat.get("strikeOuts"))
    walks = safe_num(stat.get("baseOnBalls"))
    batters_faced = safe_num(stat.get("battersFaced"))
    innings_pitched = safe_num(stat.get("inningsPitched"))
    games_started = safe_num(stat.get("gamesStarted"))

    k_pct = None
    bb_pct = None
    k_minus_bb_pct = None
    bf_per_start = None
    ip_per_start = None

    if strikeouts is not None and batters_faced and batters_faced > 0:
        k_pct = (strikeouts / batters_faced) * 100

    if walks is not None and batters_faced and batters_faced > 0:
        bb_pct = (walks / batters_faced) * 100

    if k_pct is not None and bb_pct is not None:
        k_minus_bb_pct = k_pct - bb_pct

    if games_started and games_started > 0:
        if batters_faced is not None:
            bf_per_start = batters_faced / games_started
        if innings_pitched is not None:
            ip_per_start = innings_pitched / games_started

    stat["calculatedKPercentage"] = k_pct
    stat["calculatedBBPercentage"] = bb_pct
    stat["calculatedKMinusBBPercentage"] = k_minus_bb_pct
    stat["calculatedBFPerStart"] = bf_per_start
    stat["calculatedIPPerStart"] = ip_per_start

    return stat


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_game_log(player_id: int, season: int) -> list[dict]:
    payload = get_json(
        f"{MLB_PEOPLE_URL}/{player_id}/stats",
        params={
            "stats": "gameLog",
            "group": "pitching",
            "season": season,
        },
    )

    stats_groups = payload.get("stats", [])
    if not stats_groups:
        return []

    splits = stats_groups[0].get("splits", [])
    if not isinstance(splits, list):
        return []

    starts = []

    for split in splits:
        if not isinstance(split, dict):
            continue

        stat = split.get("stat", {})
        if not isinstance(stat, dict):
            continue

        if not stat.get("gamesStarted"):
            continue

        opponent = split.get("opponent", {})

        starts.append(
            {
                "date": split.get("date", "N/A"),
                "opponent": (
                    opponent.get("name", "N/A")
                    if isinstance(opponent, dict)
                    else "N/A"
                ),
                "innings": stat.get("inningsPitched", "N/A"),
                "strikeouts": stat.get("strikeOuts", "N/A"),
                "walks": stat.get("baseOnBalls", "N/A"),
                "hits": stat.get("hits", "N/A"),
                "earned_runs": stat.get("earnedRuns", "N/A"),
                "home_runs": stat.get("homeRuns", "N/A"),
                "pitches": stat.get("numberOfPitches", "N/A"),
                "batters_faced": stat.get("battersFaced", "N/A"),
            }
        )

    return starts[-5:][::-1]


@st.cache_data(ttl=300, show_spinner=False)
def fetch_team_hitting_stats(team_id: int, season: int) -> dict:
    payload = get_json(
        f"{MLB_TEAMS_URL}/{team_id}/stats",
        params={
            "stats": "season",
            "group": "hitting",
            "season": season,
        },
    )

    stats_groups = payload.get("stats", [])
    if not stats_groups:
        return {}

    splits = stats_groups[0].get("splits", [])
    if not splits:
        return {}

    stat = splits[0].get("stat", {})
    if not isinstance(stat, dict):
        return {}

    strikeouts = safe_num(stat.get("strikeOuts"))
    plate_appearances = safe_num(stat.get("plateAppearances"))
    at_bats = safe_num(stat.get("atBats"))

    opponent_k_pct = None
    denominator = plate_appearances or at_bats

    if strikeouts is not None and denominator and denominator > 0:
        opponent_k_pct = (strikeouts / denominator) * 100

    stat["calculatedStrikeoutRate"] = opponent_k_pct

    return stat


def format_game_time(game_date: str | None) -> str:
    if not game_date:
        return "Time TBD"

    try:
        parsed = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
    except ValueError:
        return "Time TBD"

    time_label = (
        parsed.astimezone(timezone.utc)
        .strftime("%I:%M %p")
        .lstrip("0")
    )

    return (
        f"{parsed.astimezone(timezone.utc).strftime('%B')} "
        f"{parsed.day}, {time_label} UTC"
    )


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitchers_for_date(selected_date: str) -> dict:
    payload = get_json(
        MLB_SCHEDULE_URL,
        params={
            "sportId": 1,
            "date": selected_date,
            "hydrate": "probablePitcher,team,opponents",
        },
    )

    pitcher_options = []
    scheduled_games = 0
    profile_errors = 0

    for date_group in payload.get("dates", []):
        if not isinstance(date_group, dict):
            continue

        games = date_group.get("games", [])
        scheduled_games += len(games)

        for game in games:
            if not isinstance(game, dict):
                continue

            game_teams = game.get("teams", {})
            venue = game.get("venue", {})

            if not isinstance(game_teams, dict):
                continue

            for side, opponent_side in (
                ("away", "home"),
                ("home", "away"),
            ):
                team_data = game_teams.get(side, {})
                opponent_data = game_teams.get(opponent_side, {})

                probable_pitcher = (
                    team_data.get("probablePitcher")
                    if isinstance(team_data, dict)
                    else None
                )

                if not isinstance(probable_pitcher, dict):
                    continue

                pitcher_id = probable_pitcher.get("id")
                pitcher_name = probable_pitcher.get("fullName")

                team = (
                    team_data.get("team", {})
                    if isinstance(team_data, dict)
                    else {}
                )

                opponent = (
                    opponent_data.get("team", {})
                    if isinstance(opponent_data, dict)
                    else {}
                )

                if not pitcher_id or not pitcher_name:
                    continue

                throwing_hand = "Not listed by MLB"

                try:
                    profile = fetch_pitcher_profile(int(pitcher_id))
                    pitch_hand = profile.get("pitchHand", {})

                    if isinstance(pitch_hand, dict):
                        throwing_hand = (
                            pitch_hand.get("description")
                            or "Not listed by MLB"
                        )

                except MLBApiError:
                    profile_errors += 1

                pitcher_options.append(
                    {
                        "selection_id":
                            f"{game.get('gamePk')}-{side}-{pitcher_id}",

                        "pitcher_id":
                            int(pitcher_id),

                        "pitcher_name":
                            pitcher_name,

                        "team":
                            team.get("name", "Unknown team"),

                        "team_id":
                            team.get("id"),

                        "opponent":
                            (
                                opponent.get(
                                    "name",
                                    "Unknown opponent",
                                )
                                if isinstance(opponent, dict)
                                else "Unknown opponent"
                            ),

                        "opponent_id":
                            (
                                opponent.get("id")
                                if isinstance(opponent, dict)
                                else None
                            ),

                        "throwing_hand":
                            throwing_hand,

                        "game_time":
                            format_game_time(
                                game.get("gameDate")
                            ),

                        "venue":
                            (
                                venue.get("name", "N/A")
                                if isinstance(venue, dict)
                                else "N/A"
                            ),

                        "status":
                            (
                                game.get("status", {}).get(
                                    "detailedState",
                                    "N/A",
                                )
                                if isinstance(
                                    game.get("status"),
                                    dict,
                                )
                                else "N/A"
                            ),
                    }
                )

    return {
        "pitchers": pitcher_options,
        "scheduled_games": scheduled_games,
        "profile_errors": profile_errors,
    }


st.set_page_config(
    page_title="MLB Strikeout Predictor",
    page_icon="⚾",
    layout="centered",
)


st.title("MLB Starting Pitcher Strikeout Predictor")
st.caption("V0.2 — 9 Module Research Dashboard")


game_date = st.date_input(
    "Game date",
    value=date.today(),
    min_value=date(2000, 1, 1),
)


source_url = (
    f"{MLB_SCHEDULE_URL}?"
    f"{urlencode({'sportId': 1, 'date': game_date.isoformat()})}"
)

st.caption(
    f"Live data source: [MLB Stats API]({source_url})"
)


try:

    with st.spinner(
        "Loading MLB schedule and probable pitchers..."
    ):
        matchup_data = fetch_pitchers_for_date(
            game_date.isoformat()
        )

except MLBApiError as exc:

    st.error(str(exc))
    st.stop()


pitchers = matchup_data["pitchers"]
scheduled_games = matchup_data["scheduled_games"]


if not scheduled_games:

    st.info(
        "No MLB games are scheduled for this date."
    )
    st.stop()


if not pitchers:

    st.warning(
        "MLB games are scheduled for this date, "
        "but probable starting pitchers "
        "have not been announced yet."
    )
    st.stop()


pitcher_by_id = {
    pitcher["selection_id"]: pitcher
    for pitcher in pitchers
}


selected_id = st.selectbox(
    "Probable starting pitcher",
    options=list(pitcher_by_id),
    format_func=lambda sid: (
        f"{pitcher_by_id[sid]['pitcher_name']} — "
        f"{pitcher_by_id[sid]['team']} vs "
        f"{pitcher_by_id[sid]['opponent']}"
    ),
)


selected_pitcher = pitcher_by_id[selected_id]


st.subheader("Selected Matchup")

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


try:

    with st.spinner(
        "Loading pitcher and opponent data..."
    ):

        pitcher_stats = fetch_pitcher_season_stats(
            selected_pitcher["pitcher_id"],
            game_date.year,
        )

        recent_starts = fetch_pitcher_game_log(
            selected_pitcher["pitcher_id"],
            game_date.year,
        )

        opponent_stats = (
            fetch_team_hitting_stats(
                selected_pitcher["opponent_id"],
                game_date.year,
            )
            if selected_pitcher.get("opponent_id")
            else {}
        )

except MLBApiError as exc:

    st.warning(
        f"Some live data could not be loaded: {exc}"
    )

    pitcher_stats = {}
    recent_starts = []
    opponent_stats = {}


st.divider()


st.header("M1 · Capacidad real de K — 20%")


if pitcher_stats:

    col1, col2, col3 = st.columns(3)

    with col1:

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
                    "calculatedKPercentage"
                ),
                1,
                "%",
            ),
        )

    with col2:

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
                    "calculatedBBPercentage"
                ),
                1,
                "%",
            ),
        )

    with col3:

        st.metric(
            "K-BB%",
            fmt(
                pitcher_stats.get(
                    "calculatedKMinusBBPercentage"
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

else:

    st.info(
        "N/A — no season pitching data returned."
    )


st.header("M2 · Volumen / Leash — 20%")


if pitcher_stats:

    col1, col2, col3 = st.columns(3)

    with col1:

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

    with col2:

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
                    "calculatedBFPerStart"
                ),
                1,
            ),
        )

    with col3:

        st.metric(
            "IP/start",
            fmt(
                pitcher_stats.get(
                    "calculatedIPPerStart"
                ),
                2,
            ),
        )

else:

    st.info(
        "N/A — no volume data returned."
    )


st.header("M3 · Splits — 10%")

st.info(
    "N/A por ahora — splits confiables "
    "vs LHB/RHB y home/away "
    "todavía no están conectados."
)


st.header(
    "M4 · Propensión del rival a poncharse — 20%"
)


if opponent_stats:

    col1, col2, col3 = st.columns(3)

    with col1:

        st.metric(
            "Opponent K",
            opponent_stats.get(
                "strikeOuts",
                "N/A",
            ),
        )

    with col2:

        st.metric(
            "PA",
            opponent_stats.get(
                "plateAppearances",
                "N/A",
            ),
        )

    with col3:

        st.metric(
            "Opponent K%",
            fmt(
                opponent_stats.get(
                    "calculatedStrikeoutRate"
                ),
                1,
                "%",
            ),
        )

else:

    st.info(
        "N/A — opponent hitting data unavailable."
    )


st.header("M5 · Arsenal vs Matchup — 15%")

st.info(
    "N/A por ahora — necesita Statcast "
    "para pitch mix, velocity, whiff% "
    "y matchup por tipo de pitcheo."
)


st.header("M6 · Forma / Cambios recientes — 5%")


if recent_starts:

    for start in recent_starts:

        st.write(
            f"**{start['date']} "
            f"vs {start['opponent']}** — "
            f"{start['innings']} IP · "
            f"{start['strikeouts']} K · "
            f"{start['walks']} BB · "
            f"{start['earned_runs']} ER · "
            f"{start['home_runs']} HR · "
            f"{start['pitches']} pitches"
        )

else:

    st.info(
        "N/A — no recent start log returned."
    )


st.header("M7 · Contexto del juego — 5%")


col1, col2 = st.columns(2)


with col1:

    st.metric(
        "Venue",
        selected_pitcher.get(
            "venue",
            "N/A",
        ),
    )


with col2:

    st.metric(
        "Status",
        selected_pitcher.get(
            "status",
            "N/A",
        ),
    )


st.caption(
    "Weather, umpire and park K factors "
    "are not connected yet."
)


st.header("M8 · Lineup confirmado — 5%")


st.info(
    "N/A por ahora — confirmed batting order "
    "todavía no está conectado."
)


st.header("M9 · Mercado / Líneas / Edge")


st.info(
    "N/A por ahora — necesita una fuente "
    "de odds/props en vivo. "
    "No se inventará línea, edge ni EV."
)


st.divider()


st.subheader("Estado del modelo")


st.warning(
    "Predicción FINAL todavía desactivada. "
    "Primero vamos a automatizar y validar "
    "los datos de los 9 módulos. "
    "Después activaremos proyección central, "
    "rango, P(4+…9+), cuota justa, edge y EV."
)
