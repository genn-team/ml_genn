from ml_genn.utils.unique_name import UniqueName

def test_varname():
    u = UniqueName()

    a = u(None, "Name")
    b = u(None, "Name")
    c = u(None, "Name")

    assert a == "a"
    assert b == "b"
    assert c == "c"

def test_manual():
    u = UniqueName()

    a = u("A", "Name")
    b = u("B", "Name")
    c = u("C", "Name")

    assert a == "A"
    assert b == "B"
    assert c == "C"


def test_auto():
    u = UniqueName()

    assert u(None, "Name") == "Name"
    assert u(None, "Name") == "Name_1"
    assert u(None, "Name") == "Name_2"