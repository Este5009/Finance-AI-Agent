"""Small native macOS controller window for the browser-backed desktop app."""

from __future__ import annotations

import queue
import signal
import threading
import webbrowser
from typing import Any

from finance_agent.desktop.runtime import DesktopRuntime, DesktopStartupError, StartupStatus


def run_macos_controller() -> int:
    """Run a Tk macOS lifecycle controller around the Streamlit session.

    Inputs: none; configuration comes from platform application data.
    Outputs: process exit code after the user closes/quits the controller.
    Assumptions: closing the controller means closing Finance AI Agent; closing
    only the browser leaves the controller available to reopen the interface.
    """

    import tkinter as tk
    from tkinter import messagebox

    events: queue.Queue[tuple[str, Any]] = queue.Queue()
    runtime = DesktopRuntime(status_callback=lambda status: events.put(("status", status)))
    root = tk.Tk()
    root.title("Finance AI Agent")
    root.geometry("470x220")
    root.resizable(False, False)
    status_text = tk.StringVar(value="Preparando Finance AI Agent…")
    tk.Label(root, text="Finance AI Agent", font=("Helvetica", 20, "bold")).pack(pady=(24, 10))
    tk.Label(root, textvariable=status_text, wraplength=420, justify="center").pack(pady=8)
    buttons = tk.Frame(root)
    buttons.pack(pady=18)
    open_button = tk.Button(
        buttons,
        text="Abrir interfaz",
        state="disabled",
        command=lambda: runtime.current_url and webbrowser.open(runtime.current_url, new=1),
    )
    open_button.pack(side="left", padx=8)

    closing = False

    def close_application(reason: str = "controller_close") -> None:
        """Close only this session's Streamlit helper and native controller."""

        nonlocal closing
        if closing:
            return
        closing = True
        status_text.set("Cerrando Finance AI Agent…")
        runtime.shutdown(reason=reason)
        root.after(0, root.destroy)

    tk.Button(buttons, text="Cerrar aplicación", command=close_application).pack(side="left", padx=8)
    root.protocol("WM_DELETE_WINDOW", close_application)

    def startup_worker() -> None:
        """Perform bounded service/model/server checks without freezing macOS UI."""

        try:
            runtime.startup()
            events.put(("ready", None))
        except DesktopStartupError as exc:
            events.put(("error", exc.status.message_es))
        except Exception:
            if runtime.logger:
                runtime.logger.exception(
                    "session=%s launcher_pid=%s unexpected_controller_startup_failure",
                    runtime.session_id,
                    runtime.launcher_pid,
                )
            events.put(("error", "Finance AI Agent no pudo iniciarse. Revise el registro técnico."))

    def poll_events() -> None:
        """Apply worker/lifecycle events on Tk's main thread."""

        if closing:
            return
        try:
            while True:
                kind, value = events.get_nowait()
                if kind == "status" and isinstance(value, StartupStatus):
                    status_text.set(value.message_es)
                elif kind == "ready":
                    status_text.set("Finance AI Agent está listo. Puede cerrar esta ventana para terminar la aplicación.")
                    open_button.configure(state="normal")
                elif kind == "error":
                    messagebox.showerror("Finance AI Agent", str(value))
                    close_application(reason="startup_error")
                    return
        except queue.Empty:
            pass
        if runtime.streamlit_process is not None and runtime.streamlit_process.poll() is not None:
            close_application(reason="streamlit_exit")
            return
        root.after(100, poll_events)

    def signal_handler(signum: int, _frame: object) -> None:
        root.after(0, lambda: close_application(reason=f"signal_{signum}"))

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    threading.Thread(target=startup_worker, daemon=True).start()
    root.after(100, poll_events)
    try:
        root.mainloop()
    finally:
        runtime.shutdown(reason="controller_exit")
    return 0
