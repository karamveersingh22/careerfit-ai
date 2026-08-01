"""A lightweight smoke test for the Streamlit entry point."""

from streamlit.testing.v1 import AppTest


def test_app_starts_without_exceptions() -> None:
    app = AppTest.from_file("app.py").run(timeout=15)

    assert not app.exception
    assert app.title[0].value == "CareerFit AI"
    assert len(app.file_uploader) == 2

