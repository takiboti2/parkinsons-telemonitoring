"""Development row selection, authenticated requests, and safe command diagnostics."""

import json
import sys
from pathlib import Path
from unittest.mock import MagicMock

import httpx
import pandas as pd
import pytest

from parkinsons_ml import predict_rows
from parkinsons_ml.schema import CANDIDATE_FEATURES

KEY = "isolated-row-client-key-not-a-real-secret"


@pytest.fixture
def client_root(prepared_directory: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    root = tmp_path / "client"
    (root / "data/processed").mkdir(parents=True)
    (root / "data/processed/train.csv").write_bytes((prepared_directory / "train.csv").read_bytes())
    dotenv = root / ".env"
    dotenv.write_text(f"PARKINSONS_API_KEY={KEY}\nSOURCE_REVISION=ignored\n")
    monkeypatch.setitem(predict_rows.ClientSettings.model_config, "env_file", dotenv)
    monkeypatch.delenv("PARKINSONS_API_KEY", raising=False)
    monkeypatch.setattr(predict_rows, "ROOT", root)
    return root


@pytest.mark.parametrize("start,end", [(1, 1), (1, 3), (4, 6), (18, 18)])
def test_selected_rows_and_authentication(
    client_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: int,
    end: int,
) -> None:
    path = client_root / "data/processed/train.csv"
    first, last = start - 1, end - 1
    expected = pd.read_csv(path).loc[first:last, list(CANDIDATE_FEATURES)]
    values = pd.read_csv(path)
    values.loc[start - 1, "jitter"] = float("nan")
    values.to_csv(path, index=False)
    expected.loc[start - 1, "jitter"] = float("nan")
    response = httpx.Response(
        200, json={"predictions": [1.0] * (end - start + 1)}, request=httpx.Request("POST", predict_rows.URL)
    )
    post = MagicMock(return_value=response)
    monkeypatch.setattr(predict_rows.httpx, "post", post)
    monkeypatch.chdir(client_root.parent)
    result = predict_rows.predict_rows(start, end)
    assert result["predictions"] == [1.0] * (end - start + 1)
    post.assert_called_once()
    assert post.call_args.args == (predict_rows.URL,)
    assert post.call_args.kwargs["headers"]["X-API-Key"] == KEY
    assert post.call_args.kwargs["trust_env"] is False
    records = json.loads(post.call_args.kwargs["content"])["records"]
    assert records[0]["jitter"] is None
    actual = pd.DataFrame(records).reindex(columns=list(CANDIDATE_FEATURES))
    pd.testing.assert_frame_equal(actual.astype(float), expected.reset_index(drop=True).astype(float))
    assert all(set(row) == set(CANDIDATE_FEATURES) for row in records)


@pytest.mark.parametrize("start,end", [(0, 2), (3, 2), (1, 1001), (18, 19), (100, 150)])
def test_invalid_range_never_submits(
    client_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    start: int,
    end: int,
) -> None:
    post = MagicMock()
    monkeypatch.setattr(predict_rows.httpx, "post", post)
    with pytest.raises(predict_rows.RowSelectionError):
        predict_rows.predict_rows(start, end)
    post.assert_not_called()


def test_environment_key_override(client_root: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    override = "environment-row-client-key-not-a-real-secret"
    monkeypatch.setenv("PARKINSONS_API_KEY", override)
    post = MagicMock(return_value=httpx.Response(200, json={}, request=httpx.Request("POST", predict_rows.URL)))
    monkeypatch.setattr(predict_rows.httpx, "post", post)
    predict_rows.predict_rows(1, 1)
    assert post.call_args.kwargs["headers"]["X-API-Key"] == override


@pytest.mark.parametrize("problem", ["missing_key", "nonfinite", "unexpected_response"])
def test_invalid_inputs_or_response(client_root: Path, monkeypatch: pytest.MonkeyPatch, problem: str) -> None:
    post = MagicMock(return_value=httpx.Response(200, json=[], request=httpx.Request("POST", predict_rows.URL)))
    monkeypatch.setattr(predict_rows.httpx, "post", post)
    if problem == "missing_key":
        (client_root / ".env").write_text("")
    elif problem == "nonfinite":
        path = client_root / "data/processed/train.csv"
        frame = pd.read_csv(path)
        frame.loc[0, "age"] = float("inf")
        frame.to_csv(path, index=False)
    with pytest.raises(ValueError):
        predict_rows.predict_rows(1, 1)
    assert post.call_count == int(problem == "unexpected_response")


def test_cli_success(client_root: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["predict_rows", "--start-row", "1", "--end-row", "2"])
    monkeypatch.setattr(
        predict_rows.httpx,
        "post",
        MagicMock(
            return_value=httpx.Response(
                200,
                json={"predictions": [1.0, 2.0]},
                request=httpx.Request("POST", predict_rows.URL),
            )
        ),
    )
    predict_rows.main()
    assert json.loads(capsys.readouterr().out) == {"predictions": [1.0, 2.0]}


@pytest.mark.parametrize("failure", ["range", "http", "connection"])
def test_cli_failures_are_sanitized(
    client_root: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    failure: str,
) -> None:
    monkeypatch.setattr(sys, "argv", ["predict_rows", "--start-row", "1", "--end-row", "2"])
    if failure == "range":
        monkeypatch.setattr(sys, "argv", ["predict_rows", "--start-row", "0", "--end-row", "2"])
    elif failure == "http":
        response = httpx.Response(401, text=KEY, request=httpx.Request("POST", predict_rows.URL))
        monkeypatch.setattr(predict_rows.httpx, "post", MagicMock(return_value=response))
    else:
        monkeypatch.setattr(predict_rows.httpx, "post", MagicMock(side_effect=httpx.ConnectError(KEY)))
    with pytest.raises(SystemExit) as caught:
        predict_rows.main()
    assert caught.value.code == 1
    captured = capsys.readouterr()
    assert captured.err and not captured.out
    assert KEY not in captured.err
    if failure == "http":
        assert "HTTP 401" in captured.err
