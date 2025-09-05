import pytest
from yans_2025_hackathon.eval import (
    _normalize_number,
    extra_answer_from_response,
    evaluate_response,
)


@pytest.mark.parametrize(
    "input_value,expected",
    [
        (123, "123"),
        (0, "0"),
        (-123, "-123"),
        ("123", "123"),
        ("0123", "123"),
        ("1,234", "1234"),
        (" 123 ", "123"),
        ("0,001,", "1"),
    ],
)
def test_normalize_number(input_value, expected):
    assert _normalize_number(input_value) == expected


@pytest.mark.parametrize(
    "response,expected",
    [
        ("The answer is 42", "42"),
        ("Result: 123.45", "123.45"),
        ("Final value is -56", "-56"),
        ("First 10, then 20, finally 30", "30"),
        ("Values are 1.5, 2.7, and 3.9", "3.9"),
        ("No numbers here", ""),
        ("", ""),
        ("The total is 1,234", "1,234"),
        ("Amount: 5,678.90", "5,678.90"),
    ],
)
def test_extra_answer_from_response(response, expected):
    assert extra_answer_from_response(response) == expected


@pytest.mark.parametrize(
    "response,expected_answer,expected_result",
    [
        ("The answer is 42", 42, True),
        ("Result is 1,234", 1234, True),  # with comma
        ("Value: 0123", 123, True),
        ("The answer is 42", 43, False),
        ("Result is 100", 200, False),
        ("No answer provided", 42, False),
        ("", 42, False),
        ("The result is -25", -25, True),
        ("The result is -25", 25, False),
    ],
)
def test_evaluate_response(response, expected_answer, expected_result):
    assert evaluate_response(response, expected_answer) is expected_result
