"""Thin, localhost Streamlit adapter for on-demand MLflow monitoring."""

import logging
from typing import cast

import pandas as pd
import streamlit as st
import yaml
from mlflow.tracking import MlflowClient
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from parkinsons_ml.experimentation import UniqueKeyLoader, training_config
from parkinsons_ml.monitoring import (
    MonitoringReport,
    MonitoringWindow,
    REPORT_POLICY,
    WindowConfig,
    build_report,
    load_cohort,
    load_window,
    open_saved_report,
    promotion_decision,
    save_report,
)
from parkinsons_ml.promotion import PENDING
from parkinsons_ml.tracking import ROOT, validate_tracking_uri

LOGGER = logging.getLogger(__name__)


class DashboardSettings(BaseSettings):
    model_config = SettingsConfigDict(populate_by_name=True, frozen=True, hide_input_in_errors=True)
    tracking_uri: str = Field(validation_alias="MLFLOW_TRACKING_URI", repr=False)

    @field_validator("tracking_uri")
    @classmethod
    def supported_uri(cls, value: str) -> str:
        validate_tracking_uri(value)
        return value


def show_window(window: MonitoringWindow) -> None:
    st.subheader("Recent inference")
    st.write(
        {
            "model": window.cohort.model_name,
            "version": window.cohort.model_version,
            "reference_run": window.cohort.training_run_id,
            "rows": len(window.current),
            "batches": len(window.batches),
            "incompatible_batches": window.incompatible,
        }
    )
    if window.truncated:
        st.warning("Window truncated by the scan/row limit; only whole batches are included.")
    with st.expander("Window provenance"):
        st.json(window.manifest())
    if window.current.empty:
        st.info("No compatible completed inference batches in this window.")
    else:
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run_id": batch.run_id,
                        "start_utc": pd.to_datetime(batch.start_ms, unit="ms", utc=True),
                        "rows": batch.rows,
                    }
                    for batch in window.batches
                ]
            )
        )
        st.subheader("Prediction distribution")
        st.dataframe(window.predictions[["prediction"]].describe())
        with st.expander("Feature distributions"):
            st.write("Reference")
            st.dataframe(window.reference.describe().transpose())
            st.write("Current")
            st.dataframe(window.current.describe().transpose())
        if st.checkbox("Show measurement and prediction preview"):
            st.dataframe(window.current.head(100))
            st.dataframe(window.predictions.head(100))


def show_report(report: MonitoringReport, *, key: str = "current") -> None:
    st.subheader("Data quality")
    st.dataframe(pd.DataFrame(cast(list[dict[str, object]], report.summary["quality"])))
    st.subheader("Feature drift")
    st.write(f"Report status: {report.summary['status']}")
    if report.summary["unavailable"]:
        st.warning(f"Drift unavailable for some features: {report.summary['unavailable']}")
    if "drift" in report.summary:
        st.dataframe(pd.DataFrame(cast(list[dict[str, object]], report.summary["drift"])))
        st.write(
            {
                "drifted_share_of_evaluated_features": report.summary["drifted_share"],
                "drift_signal": report.summary["dataset_drift"],
            }
        )
    st.info(str(report.summary["performance"]))
    if report.evidently_json is not None:
        st.download_button(
            "Download Evidently JSON", report.evidently_json, "evidently.json", "application/json", key=f"{key}-json"
        )
        assert report.html is not None
        st.download_button("Download Evidently HTML", report.html, "evidently.html", "text/html", key=f"{key}-html")


def main() -> None:
    st.set_page_config(page_title="Parkinson's monitoring", layout="wide")
    st.title("Parkinson's telemonitoring")
    st.caption("Research only, not a clinical diagnostic service. Drift is not proof of performance degradation.")
    try:
        settings = DashboardSettings.model_validate({})
        parameters = yaml.load((ROOT / "params.yaml").read_text(), Loader=UniqueKeyLoader)
        if not isinstance(parameters, dict):
            raise ValueError("Invalid server parameters")
        name = training_config(parameters.get("train")).promotion.registered_model_name
        client = MlflowClient(tracking_uri=settings.tracking_uri, registry_uri=settings.tracking_uri)
        model = client.get_registered_model(name)
        st.subheader("Model status")
        st.write({"registered_model": name, "champion": model.aliases.get("champion")})
        if model.tags.get(PENDING):
            st.warning(
                "Champion promotion is unconfirmed. Serving is blocked; historical monitoring remains available."
            )
        versions = client.search_model_versions(f"name = '{name}'", max_results=100, order_by=["version_number DESC"])
        options = [item.version for item in versions if item.status == "READY"]
        champion = model.aliases.get("champion")
        if champion is not None:
            current = client.get_model_version(name, champion)
            if current.status == "READY" and champion not in options:
                options.append(champion)
        options.sort(key=int, reverse=True)
        if not options:
            st.info("No ready model versions are available. Train through the API first.")
            return
        if versions.token:
            st.warning("Version history is limited to 100 versions, plus the champion.")
        selected = st.selectbox(
            "Immutable model version", options, index=options.index(champion) if champion in options else 0
        )
        hours = st.number_input("Lookback hours", min_value=1, max_value=168, value=24, step=1)
        config = WindowConfig(hours=int(hours))
        key = (settings.tracking_uri, name, selected, config.hours)
        if st.session_state.get("window_key") != key:
            for state in ("window", "report", "saved_run"):
                st.session_state.pop(state, None)
            st.session_state["window_key"] = key
        if st.button("Refresh window"):
            st.session_state.pop("report", None)
            st.session_state.pop("saved_run", None)
            st.session_state.pop("window", None)
            st.session_state["window"] = load_window(client, name, selected, config)
        window = cast(MonitoringWindow | None, st.session_state.get("window"))
        if window is not None:
            show_window(window)
            if st.button("Generate and save report"):
                st.session_state.pop("report", None)
                st.session_state.pop("saved_run", None)
                report = build_report(window)
                run_id = save_report(client, window, report)
                st.session_state["report"] = report
                st.session_state["saved_run"] = run_id
            report = cast(MonitoringReport | None, st.session_state.get("report"))
            if report is not None:
                st.success(f"Monitoring report saved: {st.session_state['saved_run']}")
                show_report(report)
        cohort = load_cohort(client, name, selected)
        decisions = client.search_runs(
            [cohort.experiment_id],
            filter_string=f"tags.purpose = 'champion-promotion' AND tags.registered_model_name = '{name}'",
            max_results=20,
            order_by=["attributes.start_time DESC"],
        )
        st.subheader("Recent promotion decisions")
        st.caption("Up to 20 recent assessments; unfinished audits may not yet have a decision artifact.")
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "run_id": run.info.run_id,
                        "status": run.info.status,
                        "decision": run.data.tags.get("decision"),
                        "state": run.data.tags.get("state"),
                    }
                    for run in decisions
                ]
            )
        )
        if decisions:
            audit = st.selectbox("Promotion audit", [run.info.run_id for run in decisions])
            if st.button("Open promotion decision"):
                st.json(promotion_decision(client, cohort, audit))
        saved = client.search_runs(
            [cohort.experiment_id],
            filter_string=(
                f"tags.purpose = 'monitoring' AND tags.registered_model_name = '{name}' "
                f"AND tags.model_version = '{selected}' AND tags.report_policy = '{REPORT_POLICY}' "
                "AND attributes.status = 'FINISHED'"
            ),
            max_results=10,
            order_by=["attributes.start_time DESC"],
        )
        st.subheader("Saved monitoring reports")
        st.caption("Up to 10 recent completed reports for the selected version.")
        if saved:
            saved_id = st.selectbox("Saved report", [run.info.run_id for run in saved])
            if st.button("Open saved report"):
                manifest, stored = open_saved_report(client, cohort, saved_id)
                st.json(manifest)
                show_report(stored, key="saved")
        else:
            st.info("No saved reports for this model version.")
    except Exception as error:
        LOGGER.error("Dashboard operation failed (%s)", type(error).__name__)
        st.error("Monitoring is unavailable: check server configuration, MLflow, and artifact integrity.")


if __name__ == "__main__":
    main()
