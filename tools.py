"""Tools the agent can call.

Each tool is a plain Python function that takes simple arguments and returns a
string. The model never runs these itself. It asks for a tool by name,
`execute_tool` runs it, and the result goes back to the model as text.
"""

from __future__ import annotations

import ast
import operator

import httpx

# ---------------------------------------------------------------------------
# Tool 1: a safe calculator
# ---------------------------------------------------------------------------
# `eval()` would let anyone on the internet run arbitrary Python on the server.
# Instead the expression is parsed into a syntax tree and only a whitelist of
# arithmetic nodes is allowed.

_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {ast.UAdd: operator.pos, ast.USub: operator.neg}

MAX_EXPRESSION_LENGTH = 200
MAX_EXPONENT = 100  # stops things like 9**9**9 from freezing the server


def _eval_node(node: ast.AST) -> float:
    if isinstance(node, ast.Constant) and type(node.value) in (int, float):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {MAX_EXPONENT})")
        return _BIN_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    raise ValueError("Only numbers and + - * / // % ** ( ) are allowed")


def calculator(expression: str) -> str:
    """Evaluate a basic arithmetic expression safely."""
    if len(expression) > MAX_EXPRESSION_LENGTH:
        raise ValueError(f"Expression too long (max {MAX_EXPRESSION_LENGTH} characters)")
    try:
        tree = ast.parse(expression, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid expression: {expression!r}") from exc
    try:
        result = _eval_node(tree.body)
    except ZeroDivisionError as exc:
        raise ValueError("Division by zero") from exc
    if isinstance(result, float):
        result = round(result, 10)
        if result.is_integer():
            result = int(result)
    return str(result)


# ---------------------------------------------------------------------------
# Tool 2: live weather from Open-Meteo (free, no API key needed)
# ---------------------------------------------------------------------------

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
HTTP_TIMEOUT = 10.0

# WMO weather interpretation codes used by Open-Meteo
WEATHER_CODES = {
    0: "clear sky",
    1: "mainly clear",
    2: "partly cloudy",
    3: "overcast",
    45: "fog",
    48: "depositing rime fog",
    51: "light drizzle",
    53: "moderate drizzle",
    55: "dense drizzle",
    56: "light freezing drizzle",
    57: "dense freezing drizzle",
    61: "slight rain",
    63: "moderate rain",
    65: "heavy rain",
    66: "light freezing rain",
    67: "heavy freezing rain",
    71: "slight snow",
    73: "moderate snow",
    75: "heavy snow",
    77: "snow grains",
    80: "slight rain showers",
    81: "moderate rain showers",
    82: "violent rain showers",
    85: "slight snow showers",
    86: "heavy snow showers",
    95: "thunderstorm",
    96: "thunderstorm with slight hail",
    99: "thunderstorm with heavy hail",
}


def get_weather(city: str) -> str:
    """Look up the current weather for a city using Open-Meteo."""
    city = city.strip()
    if not city:
        raise ValueError("City name is empty")

    with httpx.Client(timeout=HTTP_TIMEOUT) as http:
        geo = http.get(GEOCODING_URL, params={"name": city, "count": 1, "language": "en"})
        geo.raise_for_status()
        matches = geo.json().get("results") or []
        if not matches:
            return f"No location found matching '{city}'."
        place = matches[0]

        forecast = http.get(
            FORECAST_URL,
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,apparent_temperature,relative_humidity_2m,"
                "wind_speed_10m,weather_code",
                "timezone": "auto",
            },
        )
        forecast.raise_for_status()
        now = forecast.json()["current"]

    name = ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    conditions = WEATHER_CODES.get(now.get("weather_code"), "unknown conditions")
    return (
        f"{name} (local time {now['time']}): {conditions}, "
        f"{now['temperature_2m']}°C (feels like {now['apparent_temperature']}°C), "
        f"humidity {now['relative_humidity_2m']}%, wind {now['wind_speed_10m']} km/h."
    )


# ---------------------------------------------------------------------------
# Registry and schemas
# ---------------------------------------------------------------------------
# The schemas are what the model sees. Good descriptions matter: they are how
# the model decides *when* to use a tool, not just how.

TOOLS = {
    "calculator": calculator,
    "get_weather": get_weather,
}

TOOL_SPECS = [
    {
        "name": "calculator",
        "description": (
            "Evaluate an arithmetic expression and return the exact result. "
            "Supports + - * / // % ** and parentheses. Use this for any maths, "
            "including unit conversions such as Celsius to Fahrenheit (C * 9/5 + 32), "
            "rather than calculating in your head."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": "A plain arithmetic expression, e.g. '(17 * 23) + 2**10'.",
                }
            },
            "required": ["expression"],
        },
    },
    {
        "name": "get_weather",
        "description": (
            "Get the current, live weather for a city: conditions, temperature in "
            "Celsius, feels-like temperature, humidity and wind speed. Call it once "
            "per city."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "city": {
                    "type": "string",
                    "description": "City name, e.g. 'Paris' or 'Portland, US'.",
                }
            },
            "required": ["city"],
        },
    },
]

MAX_TOOL_OUTPUT_CHARS = 2000


def execute_tool(name: str, args: dict) -> tuple[str, bool]:
    """Run a tool by name. Returns (output, is_error).

    Errors are returned as text rather than raised, so the model can read what
    went wrong and try again (for example, fixing a malformed expression).
    """
    func = TOOLS.get(name)
    if func is None:
        return f"Unknown tool: {name}", True
    try:
        output = func(**args)
    except TypeError as exc:
        return f"Bad arguments for {name}: {exc}", True
    except httpx.HTTPError as exc:
        return f"Weather service unavailable: {exc.__class__.__name__}", True
    except Exception as exc:  # noqa: BLE001 - any tool failure goes back to the model
        return f"{exc}", True
    return str(output)[:MAX_TOOL_OUTPUT_CHARS], False
