import re


def remove_unicode(input_string: str) -> str:
    # Regular expression to match Unicode encoded characters like \uXXXX
    cleaned_string = re.sub(r'\\u[0-9a-fA-F]{4}', '', input_string)
    return cleaned_string
