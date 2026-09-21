# app.py
# Purpose: the PhishLens Tkinter window. Presentation and event wiring only.
# Author: Colile Sibanda

from __future__ import annotations

import queue
import threading
import tkinter as tk
from tkinter import ttk

from desktop import highlight, messages
from desktop.api_client import ApiClient, ApiError, ExplainResult

_PADDING = 10

# How often the Tk thread drains the worker's result queue.
_POLL_MILLISECONDS = 100


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
        # The worker thread never touches a widget: Tk is not thread-safe, and
        # calling into it off the main thread raises "main thread is not in
        # main loop". It posts its outcome here and the Tk thread renders it.
        self._results: queue.Queue = queue.Queue()
        self.root = tk.Tk() if root is None else root
        self.root.title(messages.WINDOW_TITLE)
        self.root.geometry("980x760")
        self._build_widgets()
        self.root.after(_POLL_MILLISECONDS, self._drain_results)

    def run(self) -> None:
        """Enter the Tk main loop.

        Inputs: none.
        Output: none; blocks until the window is closed.
        """
        self.root.mainloop()

    def _drain_results(self) -> None:
        """Render whatever the worker thread has posted, then reschedule.

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
        self.root.after(_POLL_MILLISECONDS, self._drain_results)

    def _build_widgets(self) -> None:
        """Lay out every widget in the window.

        Inputs: none.
        Output: none; populates `self.root`.
        """
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)
        self.root.rowconfigure(4, weight=2)

        ttk.Label(self.root, text=messages.INPUT_HEADING).grid(
            row=0, column=0, sticky="w", padx=_PADDING, pady=(_PADDING, 0)
        )
        self.input_text = tk.Text(self.root, height=10, wrap="word")
        self.input_text.grid(
            row=1, column=0, sticky="nsew", padx=_PADDING, pady=(0, _PADDING)
        )

        self._build_controls(row=2)
        self._build_verdict(row=3)
        self._build_explanation(row=4)

        self.status = ttk.Label(self.root, text=messages.STATUS_READY, anchor="w")
        self.status.grid(row=5, column=0, sticky="ew", padx=_PADDING, pady=(0, _PADDING))

    def _build_controls(self, row: int) -> None:
        """Build the Analyse and Clear buttons.

        Inputs: row : the grid row to place the control strip in.
        Output: none.
        """
        controls = ttk.Frame(self.root)
        controls.grid(row=row, column=0, sticky="ew", padx=_PADDING)

        self.analyse_button = ttk.Button(
            controls, text=messages.ANALYSE_BUTTON, command=self._on_analyse
        )
        self.analyse_button.pack(side="left")
        ttk.Button(
            controls, text=messages.CLEAR_BUTTON, command=self._on_clear
        ).pack(side="left", padx=(6, 0))

    def _build_verdict(self, row: int) -> None:
        """Build the verdict line and its provenance line.

        Inputs: row : the grid row to place the verdict panel in.
        Output: none.
        """
        panel = ttk.Frame(self.root)
        panel.grid(row=row, column=0, sticky="ew", padx=_PADDING, pady=_PADDING)

        self.verdict_label = ttk.Label(
            panel, text=messages.VERDICT_PLACEHOLDER, font=("Segoe UI", 13, "bold")
        )
        self.verdict_label.pack(anchor="w")

        self.model_label = ttk.Label(panel, text="")
        self.model_label.pack(anchor="w")

        ttk.Label(panel, text=messages.CAUTION, wraplength=900).pack(
            anchor="w", pady=(6, 0)
        )

    def _build_explanation(self, row: int) -> None:
        """Build the explanation heading, legend and highlighted text panel.

        Inputs: row : the grid row to place the explanation panel in.
        Output: none.
        """
        panel = ttk.Frame(self.root)
        panel.grid(row=row, column=0, sticky="nsew", padx=_PADDING)
        panel.columnconfigure(0, weight=1)
        panel.rowconfigure(2, weight=1)

        ttk.Label(panel, text=messages.EXPLANATION_HEADING).grid(
            row=0, column=0, sticky="w"
        )
        self._build_legend(panel, row=1)

        self.explanation_text = tk.Text(panel, height=14, wrap="word")
        self.explanation_text.grid(row=2, column=0, sticky="nsew", pady=(4, 0))
        highlight.configure_tags(self.explanation_text)
        self.explanation_text.insert("1.0", messages.EXPLANATION_PLACEHOLDER)
        self.explanation_text.configure(state=tk.DISABLED)

    def _build_legend(self, parent: ttk.Frame, row: int) -> None:
        """Build the two-swatch legend explaining the highlight colours.

        Inputs: parent : the frame to place the legend in; row : its grid row.
        Output: none.
        """
        legend = ttk.Frame(parent)
        legend.grid(row=row, column=0, sticky="w", pady=(4, 0))

        phishing_colour, legitimate_colour = highlight.legend_colours()
        for colour, text in (
            (phishing_colour, messages.LEGEND_PHISHING),
            (legitimate_colour, messages.LEGEND_LEGITIMATE),
        ):
            tk.Label(legend, text="    ", background=colour, relief="solid", bd=1).pack(
                side="left", padx=(0, 4)
            )
            ttk.Label(legend, text=text).pack(side="left", padx=(0, 16))

    def _on_clear(self) -> None:
        """Reset the window to its initial state.

        Inputs: none.
        Output: none.
        """
        self.input_text.delete("1.0", tk.END)
        self.verdict_label.configure(text=messages.VERDICT_PLACEHOLDER)
        self.model_label.configure(text="")
        self.explanation_text.configure(state=tk.NORMAL)
        self.explanation_text.delete("1.0", tk.END)
        self.explanation_text.insert("1.0", messages.EXPLANATION_PLACEHOLDER)
        self.explanation_text.configure(state=tk.DISABLED)
        self.status.configure(text=messages.STATUS_READY)

    def _on_analyse(self) -> None:
        """Validate the input and start the analysis on a background thread.

        Blank input is refused here, before any request is sent. The call runs
        off the Tk thread because /explain runs LIME and would otherwise
        freeze the window; the result is marshalled back with `root.after`.

        Inputs: none.
        Output: none.
        """
        text = self.input_text.get("1.0", tk.END).strip()
        if not text:
            self.status.configure(text=messages.EMPTY_INPUT)
            return

        self.analyse_button.configure(state=tk.DISABLED)
        self.status.configure(text=messages.STATUS_ANALYSING)
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

    def _show_result(self, result: ExplainResult) -> None:
        """Render a successful analysis into the verdict and explanation panels.

        Inputs: result : the parsed /explain response.
        Output: none.
        """
        self.verdict_label.configure(
            text=messages.format_verdict(
                result.label, result.probability, result.threshold
            )
        )
        self.model_label.configure(
            text=messages.format_model_line(result.model_id, result.method)
        )
        highlight.render_attributions(
            self.explanation_text, result.text, result.tokens
        )
        self.status.configure(text=messages.STATUS_DONE)
        self.analyse_button.configure(state=tk.NORMAL)

    def _show_error(self, message: str) -> None:
        """Show a readable failure message and re-enable the Analyse button.

        Inputs: message : the ApiError's message, already fit to show a user.
        Output: none.
        """
        self.status.configure(text=message)
        self.analyse_button.configure(state=tk.NORMAL)
