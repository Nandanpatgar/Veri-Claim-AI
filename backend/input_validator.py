# ==========================================
# INPUT VALIDATION CONFIGURATION
# ==========================================

MAX_INPUT_LENGTH = 20000


# ==========================================
# INPUT VALIDATION FUNCTION
# ==========================================

def validate_input_text(text):

    # Check whether input is a string
    if not isinstance(text, str):
        raise ValueError("Input must be a string.")

    # Remove unnecessary spaces
    text = text.strip()

    # Check for empty input
    if not text:
        raise ValueError("Input cannot be empty.")

    # Check maximum input length
    if len(text) > MAX_INPUT_LENGTH:
        raise ValueError(
            f"Input exceeds maximum length of {MAX_INPUT_LENGTH} characters."
        )

    return text