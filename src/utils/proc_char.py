import unicodedata


def remove_unicode(input_string: str) -> str:
    # Removes special categories of unicode characters like Symbols (So).
    cleaned_string = "".join(char for char in input_string if unicodedata.category(char) not in {"So", "Cn", "Co", "Cf", "Cs"})
    # Removes special white spaces.
    cleaned_string = cleaned_string.replace(u'\u2002', " ").replace(u'\u2003', " ")

    return cleaned_string
