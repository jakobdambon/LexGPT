from ..utils.proc_char import remove_unicode


def test_remove_unicode() -> None:
    text1 = 'Some text'
    text2 = ', more text'
    text3 = '; final text.'
    u1 = '\\u1234'
    u2 = '\\u2042'
    input1 = text1 + u1 + text2 + u2 + text3
    expected_out = text1 + text2 + text3
    input2 = expected_out
    assert remove_unicode(input1) == expected_out
    assert remove_unicode(input2) == expected_out