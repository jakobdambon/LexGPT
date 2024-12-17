from ..utils.proc_char import remove_unicode


def test_remove_unicode() -> None:
    text1 = 'Some text'
    text2 = ', more text.'
    u1 = ''
    input1 = text1 + u1 + text2
    expected_out = text1 + text2
    input2 = expected_out
    assert remove_unicode(input1) == expected_out
    assert remove_unicode(input2) == expected_out