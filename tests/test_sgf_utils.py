import pytest

from go_rl.sgf_utils import (
    _require_sgfmill,
    iter_sgf_samples,
    load_sgf,
    load_sgf_dir,
    sample_positions,
)


def test_python_sgf_parser_is_deprecated():
    with pytest.raises(RuntimeError, match="deprecated"):
        load_sgf("unused.sgf")

    with pytest.raises(RuntimeError, match="deprecated"):
        list(iter_sgf_samples("unused", max_positions=1))

    with pytest.raises(RuntimeError, match="deprecated"):
        load_sgf_dir("unused", max_positions=1)

    with pytest.raises(RuntimeError, match="deprecated"):
        sample_positions(None, max_positions=1)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="deprecated"):
        _require_sgfmill()
