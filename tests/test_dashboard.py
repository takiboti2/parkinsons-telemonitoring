"""Streamlit user actions remain read-only until explicit report generation."""

from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock

import pandas as pd
import numpy as np
import pytest
from mlflow.store.entities.paged_list import PagedList
from mlflow.tracking import MlflowClient
from streamlit.testing.v1 import AppTest

from parkinsons_ml import dashboard, monitoring

APP = str(Path(dashboard.__file__))


@pytest.fixture
def ui(monkeypatch: pytest.MonkeyPatch) -> tuple[AppTest, MagicMock, monitoring.MonitoringWindow]:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
    client = MagicMock(spec=MlflowClient)
    model = client.get_registered_model.return_value
    model.aliases, model.tags = {"champion": "1"}, {}
    version = MagicMock()
    version.version, version.status = "1", "READY"
    client.get_model_version.return_value = version
    client.search_model_versions.return_value = PagedList([version], None)
    client.search_runs.return_value = PagedList([], None)
    monkeypatch.setattr("mlflow.tracking.MlflowClient", MagicMock(return_value=client))
    cohort = monitoring.Cohort("motor-updrs", "1", "training", "experiment", ("age",), {})
    frame = pd.DataFrame({"age": np.tile(np.arange(50, 80, dtype=float), 2)})
    window = monitoring.MonitoringWindow(
        cohort,
        monitoring.WindowConfig(),
        1800000000000,
        "checksum",
        frame,
        frame,
        pd.DataFrame({"prediction": np.arange(60, dtype=float)}),
        (monitoring.Batch("inference", 1799999999000, 60, "inputs", "predictions"),),
        1,
        0,
        False,
    )
    monkeypatch.setattr(monitoring, "load_cohort", MagicMock(return_value=cohort))
    monkeypatch.setattr(monitoring, "load_window", MagicMock(return_value=window))
    return AppTest.from_file(APP, default_timeout=20), client, window


def click(app: AppTest, label: str) -> AppTest:
    return next(button for button in app.button if button.label == label).click().run()


def test_read_only_refresh_and_explicit_generation(ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow]) -> None:
    app, client, _ = ui
    client.create_run.return_value.info.run_id = "saved-monitoring"
    app.run()
    assert not app.exception and not app.error
    client.create_run.assert_not_called()
    click(app, "Refresh window")
    assert not app.exception and not app.error and len(app.dataframe) >= 3
    assert not app.checkbox[0].value
    client.create_run.assert_not_called()
    click(app, "Generate and save report")
    assert not app.exception and not app.error and "saved-monitoring" in app.success[0].value
    assert any("Report status: ready" in item.value for item in app.markdown)
    client.create_run.assert_called_once()
    app.run()
    assert not app.exception and not app.error and app.success
    client.create_run.assert_called_once()  # a normal rerun cannot generate another run
    before = len(app.dataframe)
    app.checkbox[0].check().run()
    assert not app.exception and not app.error and len(app.dataframe) == before + 2
    app.number_input[0].set_value(48).run()
    assert not app.exception and not app.error
    assert not app.success and app.session_state["window_key"][-1] == 48
    assert app.session_state["window_key"][2] == "1"
    assert "window" not in app.session_state and "report" not in app.session_state


def test_empty_and_truncated_window(
    ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, _, window = ui
    empty = replace(window, current=window.current.head(0), batches=(), truncated=True)
    monkeypatch.setattr(monitoring, "load_window", MagicMock(return_value=empty))
    app.run()
    click(app, "Refresh window")
    assert not app.exception and not app.error
    assert any("No compatible" in item.value for item in app.info)
    assert any("truncated" in item.value for item in app.warning)


@pytest.mark.parametrize("problem", ["registry", "configuration", "no_versions", "pending", "version_limit"])
def test_registry_and_configuration_states(
    ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow], problem: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, client, _ = ui
    if problem == "registry":
        client.get_registered_model.side_effect = ConnectionError("private credential")
    elif problem == "configuration":
        monkeypatch.setenv("MLFLOW_TRACKING_URI", "file:///private")
    elif problem == "no_versions":
        client.search_model_versions.return_value = PagedList([], None)
        client.get_registered_model.return_value.aliases = {}
    elif problem == "pending":
        client.get_registered_model.return_value.tags = {"pending_promotion_run_id": "pending"}
    else:
        client.search_model_versions.return_value.token = "next"
    app.run()
    assert not app.exception
    if problem in {"registry", "configuration"}:
        assert app.error and "private" not in app.error[0].value
    elif problem == "no_versions":
        assert not app.error
        assert any("No ready" in item.value for item in app.info)
    else:
        assert not app.error and app.warning
    client.create_run.assert_not_called()


@pytest.mark.parametrize("history_status", ["READY", "PENDING_REGISTRATION"])
def test_champion_is_resolved_outside_the_bounded_history(
    ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow], history_status: str
) -> None:
    app, client, _ = ui
    versions: list[MagicMock] = []
    for number in range(102, 202):
        version = MagicMock()
        version.version, version.status = str(number), history_status
        versions.append(version)
    client.search_model_versions.return_value = PagedList(versions, "more")
    app.run()
    assert not app.exception and not app.error
    assert app.selectbox[0].value == "1"
    assert "1" in app.selectbox[0].options
    assert len(app.selectbox[0].options) == (101 if history_status == "READY" else 1)
    client.get_model_version.assert_called_once_with("motor-updrs", "1")
    assert client.search_model_versions.call_args.kwargs["max_results"] == 100
    client.create_run.assert_not_called()


def test_version_change_clears_loaded_window_and_report(
    ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow],
) -> None:
    app, client, _ = ui
    second = MagicMock()
    second.version, second.status = "2", "READY"
    client.search_model_versions.return_value.append(second)
    client.create_run.return_value.info.run_id = "saved"
    app.run()
    click(app, "Refresh window")
    click(app, "Generate and save report")
    assert not app.exception and not app.error and app.success
    app.selectbox[0].select("2").run()
    assert not app.exception and not app.error and not app.success
    assert app.session_state["window_key"][2] == "2"
    assert "window" not in app.session_state and "report" not in app.session_state
    assert all(button.label != "Generate and save report" for button in app.button)
    client.create_run.assert_called_once()


def test_failed_upload_never_shows_saved_success(ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow]) -> None:
    app, client, _ = ui
    client.create_run.return_value.info.run_id = "failed-monitoring"
    client.log_dict.side_effect = ConnectionError("private outage")
    app.run()
    click(app, "Refresh window")
    click(app, "Generate and save report")
    assert not app.exception and not app.success
    assert app.error and "private" not in app.error[0].value
    client.set_terminated.assert_called_once_with("failed-monitoring", "FAILED")


def test_report_downloads_and_history(
    ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow], monkeypatch: pytest.MonkeyPatch
) -> None:
    app, client, window = ui
    report = monitoring.build_report(window)
    audit = MagicMock()
    audit.info.run_id, audit.info.status = "audit", "FINISHED"
    audit.data.tags = {"decision": "promoted", "state": "complete"}
    client.search_runs.return_value = PagedList([audit], None)
    monkeypatch.setattr(monitoring, "promotion_decision", MagicMock(return_value={"reason": "lower_oof_mae"}))
    monkeypatch.setattr(monitoring, "open_saved_report", MagicMock(return_value=(window.manifest(), report)))
    monkeypatch.setattr(monitoring, "build_report", MagicMock(return_value=report))
    client.create_run.return_value.info.run_id = "saved"
    app.run()
    click(app, "Open promotion decision")
    assert not app.exception and not app.error and any("lower_oof_mae" in item.value for item in app.json)
    click(app, "Open saved report")
    assert not app.exception and not app.error
    client.create_run.assert_not_called()
    click(app, "Refresh window")
    click(app, "Generate and save report")
    assert not app.exception and not app.error and app.success
    click(app, "Open saved report")
    assert not app.exception and not app.error


def test_invalid_parameters_are_sanitized(
    ui: tuple[AppTest, MagicMock, monitoring.MonitoringWindow], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    app, client, _ = ui
    (tmp_path / "params.yaml").write_text("[]")
    monkeypatch.setattr(dashboard, "ROOT", tmp_path)
    # Script imports ROOT from tracking, not from this already-imported module.
    monkeypatch.setattr("parkinsons_ml.tracking.ROOT", tmp_path)
    app.run()
    assert app.error and not app.exception
    client.create_run.assert_not_called()
