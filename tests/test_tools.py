"""Unit tests for the tools. No network access or API key needed."""

import httpx
import pytest

import tools
from tools import TOOL_SPECS, TOOLS, calculator, execute_tool, get_weather

# --- calculator --------------------------------------------------------------


@pytest.mark.parametrize(
    "expr, expected",
    [
        ("2 + 2", "4"),
        ("(17 * 23) + 2**10", "1415"),
        ("14 * 9/5 + 32", "57.2"),
        ("-3 + 5", "2"),
        ("7 // 2", "3"),
        ("7 % 4", "3"),
        ("1284.50 / 7", "183.5"),
        ("0.1 + 0.2", "0.3"),
    ],
)
def test_calculator_arithmetic(expr, expected):
    assert calculator(expr) == expected


@pytest.mark.parametrize(
    "malicious",
    [
        "__import__('os').system('ls')",
        "open('/etc/passwd').read()",
        "().__class__.__bases__[0].__subclasses__()",
        "lambda: 1",
        "x + 1",
        "'a' * 10",
        "[1, 2, 3]",
    ],
)
def test_calculator_rejects_code(malicious):
    with pytest.raises(ValueError):
        calculator(malicious)


def test_calculator_blocks_huge_exponents():
    with pytest.raises(ValueError, match="Exponent too large"):
        calculator("9 ** 9 ** 9")


def test_calculator_rejects_long_input():
    with pytest.raises(ValueError, match="too long"):
        calculator("1+" * 200 + "1")


def test_calculator_division_by_zero():
    with pytest.raises(ValueError, match="Division by zero"):
        calculator("1 / 0")


# --- get_weather (Open-Meteo is mocked) --------------------------------------


def _mock_open_meteo(monkeypatch, geo_results, status=200):
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host.startswith("geocoding-api"):
            return httpx.Response(status, json={"results": geo_results} if geo_results else {})
        return httpx.Response(
            status,
            json={
                "current": {
                    "time": "2026-09-23T21:00",
                    "temperature_2m": 14.2,
                    "apparent_temperature": 12.9,
                    "relative_humidity_2m": 81,
                    "wind_speed_10m": 11.5,
                    "weather_code": 61,
                }
            },
        )

    real_client = httpx.Client
    monkeypatch.setattr(
        tools.httpx,
        "Client",
        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw),
    )


def test_get_weather_formats_live_data(monkeypatch):
    _mock_open_meteo(
        monkeypatch,
        [
            {
                "name": "London",
                "admin1": "England",
                "country": "United Kingdom",
                "latitude": 51.5,
                "longitude": -0.12,
            }
        ],
    )
    result = get_weather("London")
    assert "London, England, United Kingdom" in result
    assert "slight rain" in result
    assert "14.2°C" in result


def test_get_weather_unknown_city(monkeypatch):
    _mock_open_meteo(monkeypatch, [])
    assert "No location found" in get_weather("Qwxyzzz")


def test_get_weather_service_down_is_reported_not_raised(monkeypatch):
    _mock_open_meteo(monkeypatch, [], status=503)
    output, is_error = execute_tool("get_weather", {"city": "London"})
    assert is_error
    assert "unavailable" in output


# --- execute_tool and schemas -----------------------------------------------


def test_execute_tool_success():
    assert execute_tool("calculator", {"expression": "6 * 7"}) == ("42", False)


def test_execute_tool_unknown_tool():
    output, is_error = execute_tool("delete_everything", {})
    assert is_error and "Unknown tool" in output


def test_execute_tool_bad_arguments():
    output, is_error = execute_tool("calculator", {"wrong_arg": "1+1"})
    assert is_error and "Bad arguments" in output


def test_every_schema_has_a_matching_function():
    assert {spec["name"] for spec in TOOL_SPECS} == set(TOOLS)
    for spec in TOOL_SPECS:
        assert spec["description"]
        assert spec["input_schema"]["type"] == "object"
