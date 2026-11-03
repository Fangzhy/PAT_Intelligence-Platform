"""Verify Phase 2 navigation, fault review, filtering, and reference availability."""

from pathlib import Path

from streamlit.testing.v1 import AppTest


ROOT = Path(__file__).resolve().parents[1]
PAGE = "pages/2_Data_Integration.py"


def page():
    return AppTest.from_file(str(ROOT / PAGE), default_timeout=60).run()


def test_second_page_is_reachable_from_overview_and_preserves_simulation_settings():
    app = AppTest.from_file(str(ROOT / "app.py"), default_timeout=60).run()
    app.slider(key="n_batches").set_value(3)
    app.selectbox(key="duration").select(72)
    app.number_input(key="seed").set_value(17)
    app.button[0].click().run()
    app.switch_page(PAGE).run()
    assert not app.exception
    assert app.metric[0].value == "219"
    assert app.metric[1].value == "219"
    assert app.metric[2].value == "12"
    app.switch_page("app.py").run()
    assert not app.exception
    assert app.number_input(key="seed").value == 17
    assert app.slider(key="n_batches").value == 3


def test_faulty_copy_reports_issues_and_time_tolerance_changes_match_count():
    app = page()
    assert not app.exception
    assert app.metric[0].value == "870"
    assert app.metric[1].value == "870"
    assert app.metric[2].value == "42"
    app.checkbox(key="quality_demo").check().run()
    assert not app.exception
    assert app.metric[1].value == "868"
    ledger = next(frame.value for frame in app.dataframe if "issue" in frame.value.columns)
    assert {"duplicate_identity", "invalid_timestamp", "missing_value", "outside_sensor_range"}.issubset(ledger["issue"])
    app.number_input(key="sensor_tolerance").set_value(5.0).run()
    assert not app.exception
    assert app.metric[1].value == "0"


def test_availability_view_does_not_show_future_assay_values():
    app = page()

    def known_table():
        return next(frame.value for frame in app.dataframe if "assay_sample_id" in frame.value and "spectrum_sample_id" not in frame.value)

    assert known_table().empty
    app.slider(key="availability_time").set_value(6.0).run()
    assert not app.exception
    assert len(known_table()) == 1
    assert known_table()["assay_sample_id"].iloc[0] == "B001-S00000"


def test_integration_batch_filter_and_empty_selection():
    app = page()
    app.multiselect(key="integration_batches").set_value(["B002"]).run()
    assert not app.exception
    assert app.metric[0].value == "145"
    assert app.metric[2].value == "7"
    aligned = next(frame.value for frame in app.dataframe if "sensor_matched" in frame.value)
    assert aligned["batch_id"].unique().tolist() == ["B002"]
    app.multiselect(key="integration_batches").set_value([]).run()
    assert not app.exception
    assert any("Select at least one batch" in warning.value for warning in app.warning)


def test_upload_mode_explains_missing_files_and_offers_schema():
    app = page()
    app.radio(key="integration_source").set_value("Upload CSV sources").run()
    assert not app.exception
    assert len(app.get("file_uploader")) == 4
    assert any("Upload sensors.csv" in warning.value for warning in app.warning)
    assert len(app.code) == 4
