import pytest

from backend.input_validator import (
    validate_input_text,
    MAX_INPUT_LENGTH
)


def test_valid_input():

    result = validate_input_text(
        "India gained independence in 1947."
    )

    assert result == "India gained independence in 1947."


def test_empty_input():

    with pytest.raises(ValueError):
        validate_input_text("")


def test_whitespace_input():

    with pytest.raises(ValueError):
        validate_input_text("     ")


def test_non_string_input():

    with pytest.raises(ValueError):
        validate_input_text(123)


def test_input_with_extra_spaces():

    result = validate_input_text(
        "   Hello World   "
    )

    assert result == "Hello World"


def test_input_exceeding_maximum_length():

    text = "a" * (MAX_INPUT_LENGTH + 1)

    with pytest.raises(ValueError):
        validate_input_text(text)


def test_input_at_maximum_length():

    text = "a" * MAX_INPUT_LENGTH

    result = validate_input_text(text)

    assert len(result) == MAX_INPUT_LENGTH