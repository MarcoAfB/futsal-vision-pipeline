"""Check the installed command-line entry point."""

import pytest

from futsal.cli import main


def testVersion(capsys):
    with pytest.raises(SystemExit) as result:
        main(["--version"])
    assert result.value.code == 0
    assert capsys.readouterr().out.strip() == "futsal 0.1.0"
