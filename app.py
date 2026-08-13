from datetime import date, datetime, timezone
from urllib.parse import urlencode

import requests
import streamlit as st


MLB_SCHEDULE_URL = "https://statsapi.mlb.com/api/v1/schedule"
MLB_PEOPLE_URL = "https://statsapi.mlb.com/api/v1/people"
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


@st.cache_data(ttl=300, show_spinner=False)
def fetch_pitcher_profile(player_id: int) -> dict:
    payload = get_json(f"{MLB_PEOPLE_URL}/{player_id}")
    people = payload.get("people", [])
    if not people or not isinstance(people[0], dict):
        raise MLBApiError("The MLB Stats API did not return this pitcher's profile.")
    return people[0]


def format_game_time(game_date: str | None) -> str:
    if not game_date:
        return "Time TBD"

    try:
        parsed = datetime.fromisoformat(game_date.replace("Z", "+00:00"))
    except ValueError:
        return "Time TBD"

    time_label = parsed.astimezone(timezone.utc).strftime("%I:%M %p").lstrip("0")
    return f"{parsed.astimezone(timezone.utc).strftime('%B')} {parsed.day}, {time_label} UTC"


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

    pitcher_options: list[dict] = []
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
            if not isinstance(game_teams, dict):
                continue

            for side, opponent_side in (("away", "home"), ("home", "away")):
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
                team = team_data.get("team", {}) if isinstance(team_data, dict) else {}
                opponent = (
                    opponent_data.get("team", {})
                    if isinstance(opponent_data, dict)
                    else {}
                )

                if not pitcher_id or not pitcher_name or not isinstance(team, dict):
                    continue

                throwing_hand = "Not listed by MLB"
                try:
                    profile = fetch_pitcher_profile(int(pitcher_id))
                    pitch_hand = profile.get("pitchHand", {})
                    if isinstance(pitch_hand, dict):
                        throwing_hand = (
                            pitch_hand.get("description") or "Not listed by MLB"
                        )
                except MLBApiError:
                    profile_errors += 1

                pitcher_options.append(
                    {
                        "selection_id": f"{game.get('gamePk')}-{side}-{pitcher_id}",
                        "pitcher_name": pitcher_name,
                        "team": team.get("name", "Unknown team"),
                        "opponent": (
                            opponent.get("name", "Unknown opponent")
                            if isinstance(opponent, dict)
                            else "Unknown opponent"
                        ),
                        "throwing_hand": throwing_hand,
                        "game_time": format_game_time(game.get("gameDate")),
                    }
                )

    return {
        "pitchers": pitcher_options,
        "scheduled_games": scheduled_games,
        "profile_errors": profile_errors,
    }


st.set_page_config(
    page_title="MLB Strikeout Predictor",
    page_icon="MLB",
    layout="centered",
)

st.title("MLB Starting Pitcher Strikeout Predictor")
st.write(
    "Select an MLB game date and probable starting pitcher. "
    "Pitcher and matchup details are loaded live from MLB."
)

game_date = st.date_input(
    "Game date",
    value=date.today(),
    min_value=date(2000, 1, 1),
    help="Select the date of the pitcher's scheduled start.",
)

source_url = (
    f"{MLB_SCHEDULE_URL}?{urlencode({'sportId': 1, 'date': game_date.isoformat()})}"
)
st.caption(f"Live data source: [MLB Stats API]({source_url})")

try:
    with st.spinner("Loading MLB schedule and probable pitchers..."):
        matchup_data = fetch_pitchers_for_date(game_date.isoformat())
except MLBApiError as exc:
    st.error(str(exc))
    st.stop()

pitchers = matchup_data["pitchers"]
scheduled_games = matchup_data["scheduled_games"]

if not scheduled_games:
    st.info("No MLB games are scheduled for this date.")
    st.stop()

if not pitchers:
    st.warning(
        "MLB games are scheduled for this date, but probable starting pitchers "
        "have not been announced yet."
    )
    st.stop()

if matchup_data["profile_errors"]:
    st.warning(
        "MLB listed the probable pitchers, but some throwing-hand details "
        "could not be loaded from their MLB profiles."
    )

pitcher_by_id = {pitcher["selection_id"]: pitcher for pitcher in pitchers}
selected_id = st.selectbox(
    "Probable starting pitcher",
    options=list(pitcher_by_id),
    format_func=lambda selection_id: (
        f"{pitcher_by_id[selection_id]['pitcher_name']} — "
        f"{pitcher_by_id[selection_id]['team']} vs "
        f"{pitcher_by_id[selection_id]['opponent']}"
    ),
    help="Only probable starting pitchers returned by MLB for the selected date appear here.",
)
selected_pitcher = pitcher_by_id[selected_id]

st.subheader("Selected matchup")
st.write(f"**Pitcher:** {selected_pitcher['pitcher_name']}")
st.write(f"**Team:** {selected_pitcher['team']}")
st.write(f"**Opponent:** {selected_pitcher['opponent']}")
st.write(f"**Throwing hand:** {selected_pitcher['throwing_hand']}")
st.write(f"**Game time:** {selected_pitcher['game_time']}")

st.info(
    "The strikeout prediction model is disabled for now. "
    "No prediction is being calculated."
)
