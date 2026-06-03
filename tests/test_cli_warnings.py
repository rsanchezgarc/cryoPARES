import warnings


def test_cli_suppresses_third_party_warnings():
    from cryoPARES.cli import _suppress_user_warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")  # baseline: capture everything
        _suppress_user_warnings()

        warnings.warn_explicit("torch internal", UserWarning, filename="torch/ops.py", lineno=1, module="torch.ops")
        warnings.warn_explicit("numpy future", FutureWarning, filename="numpy/core.py", lineno=1, module="numpy.core")
        warnings.warn_explicit("lib deprecation", DeprecationWarning, filename="lib.py", lineno=1, module="somelib")
        warnings.warn_explicit("pending dep", PendingDeprecationWarning, filename="lib.py", lineno=1, module="somelib")

        assert len(w) == 0, f"Expected no warnings, got: {[str(x.message) for x in w]}"


def test_cli_preserves_cryopares_warnings():
    from cryoPARES.cli import _suppress_user_warnings

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _suppress_user_warnings()

        warnings.warn_explicit("important message", UserWarning,
                               filename="cryoPARES/utils/x.py", lineno=1, module="cryoPARES.utils.x")

        assert len(w) == 1
        assert "important message" in str(w[0].message)
