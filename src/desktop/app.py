# app.py
# Purpose: the PhishLens Tkinter window. Layout and event wiring only.
# Author: Colile Sibanda

from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, ttk

from desktop import messages, theme
from desktop.api_client import ApiClient, ApiError, ExplainResult
from desktop.email_file import read_email_text
from desktop.header_bar import HeaderBar
from desktop.input_panel import InputPanel
from desktop.results_panel import ResultsPanel

# How often the Tk thread drains the workers' queues.
_POLL_MILLISECONDS = 100
# How often the header re-checks whether the API is up.
_HEALTH_MILLISECONDS = 8000


class PhishLensApp:
    """The desktop client window.

    Holds no analysis logic: it sends text to the API and renders what comes
    back (src/desktop/README.md). The model is never imported here.

    Inputs: api_base_url : where the FastAPI service is served; timeout :
    seconds to wait for a response; root : an existing Tk root to build into,
    which the tests supply so one root serves the whole session. A new one is
    created when it is omitted, which is what `run_app.py` does.
    Output: a constructed window; call `run()` to enter the Tk main loop.
    """

    def __init__(
        self, api_base_url: str, timeout: float, root: tk.Misc | None = None
    ) -> None:
        self.client = ApiClient(api_base_url, timeout)
        # Workers never touch a widget: Tk is not thread-safe. They post their
        # outcome to a queue and the Tk thread renders it. Health checks get
        # their own queue so a poll can never interleave with an analysis.
        self._results: queue.Queue = queue.Queue()
        self._health: queue.Queue = queue.Queue()
        if root is None:
            theme.enable_high_dpi()  # must precede the root, or Tk renders blurry
            root = tk.Tk()
        self.root = root
        self.root.title(messages.WINDOW_TITLE)
        scale = self.root.winfo_fpixels("1i") / 96
        self.root.geometry(f"{int(1240 * scale)}x{int(800 * scale)}")
        self.root.minsize(int(900 * scale), int(600 * scale))
        theme.apply_styles(self.root)
        self._build_widgets()
        self._bind_shortcuts()
        self.root.after(_POLL_MILLISECONDS, self._drain_results)
        self._check_health()

    def run(self) -> None:
        """Enter the Tk main loop.

        Inputs: none.
        Output: none; blocks until the window is closed.
        """
        self.input_text.focus_set()
        self.root.mainloop()

    def _build_widgets(self) -> None:
        """Lay out the header, the input and results panes, and the status bar.

        Inputs: none.
        Output: none; populates `self.root`.
        """
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.header = HeaderBar(self.root, lambda: self._check_health(False))
        self.header.frame.grid(row=0, column=0, sticky="ew")

        panes = ttk.PanedWindow(self.root, orient="horizontal")
        panes.grid(row=1, column=0, sticky="nsew", padx=theme.PADDING,
                   pady=theme.PADDING)
        self.input_panel = InputPanel(
            panes, on_analyse=self._on_analyse, on_clear=self._on_clear,
            on_paste=self._on_paste, on_open=self._on_open,
        )
        self.results_panel = ResultsPanel(panes)
        panes.add(self.input_panel.frame, weight=2)
        panes.add(self.results_panel.frame, weight=3)

        # Shortcuts to the widgets the event handlers and the tests read.
        self.input_text = self.input_panel.input_text
        self.analyse_button = self.input_panel.analyse_button
        self.verdict_label = self.results_panel.verdict_label
        self.model_label = self.results_panel.model_label
        self.explanation_text = self.results_panel.explanation_text

        self.status = ttk.Label(self.root, style="Status.TLabel", anchor="w")
        self.status.grid(row=2, column=0, sticky="ew")
        self.results_panel.reset()
        self._set_status(messages.STATUS_READY)

    def _bind_shortcuts(self) -> None:
        """Bind Ctrl+Enter to Analyse, Ctrl+L to Clear and Ctrl+O to Open.

        Inputs: none.
        Output: none.
        """
        def analyse(_event: tk.Event) -> str:
            self._on_analyse()
            return "break"  # stop the Text widget inserting a newline

        self.input_text.bind("<Control-Return>", analyse)
        self.root.bind("<Control-l>", lambda _event: self._on_clear())
        self.root.bind("<Control-o>", lambda _event: self._on_open())

    def _set_status(self, text: str, error: bool = False) -> None:
        """Show a line in the status bar, in red when it reports a failure.

        Inputs: text : the message; error : whether it reports a failure.
        Output: none.
        """
        style = "StatusError.TLabel" if error else "Status.TLabel"
        self.status.configure(text=text, style=style)

    def _on_clear(self) -> None:
        """Reset the window to its initial state.

        Inputs: none.
        Output: none.
        """
        self.input_panel.set_text("")
        self.results_panel.reset()
        self._set_status(messages.STATUS_READY)

    def _on_paste(self) -> None:
        """Replace the input with the clipboard's text.

        Inputs: none.
        Output: none.
        """
        try:
            text = self.root.clipboard_get()
        except tk.TclError:
            self._set_status(messages.CLIPBOARD_EMPTY, error=True)
            return
        self.input_panel.set_text(text)
        self._set_status(messages.STATUS_READY)

    def _on_open(self) -> None:
        """Load a .eml or .txt file into the input as subject and body.

        Inputs: none.
        Output: none.
        """
        chosen = filedialog.askopenfilename(
            parent=self.root, title=messages.OPEN_DIALOG_TITLE,
            filetypes=[("Email files", "*.eml *.txt"), ("All files", "*.*")],
        )
        if not chosen:
            return
        path = Path(chosen)
        try:
            text = read_email_text(path)
        except (OSError, ValueError, LookupError) as error:
            self._set_status(
                messages.FILE_UNREADABLE.format(name=path.name, detail=error), error=True)
            return
        self.input_panel.set_text(text)
        self._set_status(messages.FILE_LOADED.format(name=path.name))

    def _on_analyse(self) -> None:
        """Validate the input and start the analysis on a background thread.

        Blank input is refused here, before any request is sent. The call runs
        off the Tk thread because /explain runs LIME and would otherwise
        freeze the window; the result comes back through `self._results`.

        Inputs: none.
        Output: none.
        """
        if str(self.analyse_button.cget("state")) == tk.DISABLED:
            return  # already running; Ctrl+Enter does not go through the button
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            self._set_status(messages.EMPTY_INPUT, error=True)
            return

        self.analyse_button.configure(state=tk.DISABLED)
        self._set_status(messages.STATUS_ANALYSING)
        self.root.configure(cursor="watch")
        threading.Thread(target=self._analyse_worker, args=(text,), daemon=True).start()

    def _analyse_worker(self, text: str) -> None:
        """Call the API off the Tk thread and hand the outcome back to it.

        Inputs: text : the email text to analyse.
        Output: none; posts the outcome to `self._results` for the Tk thread
        to render. Every ApiError is caught here, so no traceback reaches the
        GUI and the Analyse button is never left disabled.
        """
        try:
            result = self.client.explain(text)
        except ApiError as error:
            self._results.put(("error", str(error)))
            return
        self._results.put(("result", result))

    def _drain_results(self) -> None:
        """Render whatever the workers have posted, then reschedule.

        Runs on the Tk thread, so it is the only place a worker's outcome
        reaches a widget.

        Inputs: none.
        Output: none; reschedules itself every `_POLL_MILLISECONDS`.
        """
        try:
            while True:
                outcome, payload = self._results.get_nowait()
                if outcome == "result":
                    self._show_result(payload)
                else:
                    self._show_error(payload)
        except queue.Empty:
            pass
        try:
            while True:
                self.header.show_health(*self._health.get_nowait())
        except queue.Empty:
            pass
        self.root.after(_POLL_MILLISECONDS, self._drain_results)

    def _show_result(self, result: ExplainResult) -> None:
        """Render a successful analysis and return to idle.

        Inputs: result : the parsed /explain response.
        Output: none.
        """
        self.results_panel.show(result)
        self.header.show_health("online", True)
        self._set_status(messages.STATUS_DONE)
        self._set_idle()

    def _show_error(self, message: str) -> None:
        """Show a readable failure message, return to idle, re-check the API.

        Inputs: message : the ApiError's message, already fit to show a user.
        Output: none.
        """
        self._set_status(message, error=True)
        self._set_idle()
        self._check_health(reschedule=False)

    def _set_idle(self) -> None:
        """Re-enable Analyse and restore the normal cursor.

        Inputs: none.
        Output: none.
        """
        self.analyse_button.configure(state=tk.NORMAL)
        self.root.configure(cursor="")

    def _check_health(self, reschedule: bool = True) -> None:
        """Poll /health on a worker thread; the result lands in `self._health`.

        Inputs: reschedule : whether to queue the next periodic check.
        Output: none.
        """
        def worker() -> None:
            try:
                loaded = self.client.health()
            except ApiError:
                self._health.put(("offline", False))
                return
            self._health.put(("online", loaded))

        threading.Thread(target=worker, daemon=True).start()
        if reschedule:
            self.root.after(_HEALTH_MILLISECONDS, self._check_health)
