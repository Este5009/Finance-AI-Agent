"""Dedicated PyInstaller helper entry point for the owned Streamlit server."""

from finance_agent.desktop.runtime import streamlit_helper_main


if __name__ == "__main__":
    raise SystemExit(streamlit_helper_main())
