# -*- coding: utf-8 -*-

# index_inputs_gui.py
# GUI to collect inputs for the index-building pipeline using customtkinter.
# Fields:
#   - root_dir (directory picker)
#   - channels (comma separated, data channels only)
#   - stim channel (separate single name)
#   - keywords to drop (comma separated)
#
# Returns on Save (lowercased):
#   {
#     "root_dir": str,
#     "channel_names": list[str],        # data channels only
#     "stim_channel_name": str,          # single stim name
#     "drop_keywords": list[str],
#   }
#
# Usage:
#   cfg = run_index_inputs_gui()
#   if cfg is None: print("Canceled.")
#   else: print(cfg)

# =============================================================================
#                                 Imports
# =============================================================================

import os
import customtkinter as ctk
import tkinter as tk
from tkinter import filedialog, messagebox
# =============================================================================
# =============================================================================

def _split_csv_lower(s: str) -> list[str]:
    return [t.strip().lower() for t in (s or "").split(",") if t.strip()]

class IndexInputsGUI(ctk.CTk):
    def __init__(self, defaults: dict | None = None):
        super().__init__()
        self.title("Index Builder — Inputs")
        self.geometry("680x320")
        ctk.set_appearance_mode("System")
        ctk.set_default_color_theme("blue")
        self.resizable(False, False)

        self.defaults = defaults or {
            "root_dir": "",
            "channels_csv": "bla, fc",
            "stim_name": "stim",
            "drop_keywords_csv": 'drop, empty, m2, m3, m4',
        }
        self.result = None

        pad = {"padx": 12, "pady": 8}
        gridw = dict(sticky="w")

        # --- Data directory ---
        ctk.CTkLabel(self, text="Data directory (.adicht files):").grid(row=0, column=0, **pad, **gridw)
        self.dir_entry = ctk.CTkEntry(self, width=440)
        self.dir_entry.grid(row=0, column=1, **pad, sticky="we")
        self.dir_entry.insert(0, self.defaults["root_dir"])
        ctk.CTkButton(self, text="Browse…", width=90, command=self._browse_dir).grid(row=0, column=2, **pad)

        # --- Data channels (no stim here) ---
        ctk.CTkLabel(self, text="Channels (comma separated, data only):").grid(row=1, column=0, **pad, **gridw)
        self.chan_entry = ctk.CTkEntry(self, width=540)
        self.chan_entry.grid(row=1, column=1, columnspan=2, **pad, sticky="we")
        self.chan_entry.insert(0, self.defaults["channels_csv"])

        # --- Stim channel (separate) ---
        ctk.CTkLabel(self, text="Stim channel (single name):").grid(row=2, column=0, **pad, **gridw)
        self.stim_entry = ctk.CTkEntry(self, width=240)
        self.stim_entry.grid(row=2, column=1, **pad, sticky="w")
        self.stim_entry.insert(0, self.defaults["stim_name"])

        # --- Drop keywords ---
        ctk.CTkLabel(self, text="Keywords to drop channels (comma separated):").grid(row=3, column=0, **pad, **gridw)
        self.kw_entry = ctk.CTkEntry(self, width=540)
        self.kw_entry.grid(row=3, column=1, columnspan=2, **pad, sticky="we")
        self.kw_entry.insert(0, self.defaults["drop_keywords_csv"])

        # --- Hint ---
        hint = (
            "Example:\n"
            "  channels = 'bla, fc'    stim = 'stim'    drop = 'empty, bio, drop'\n"
            "All parsing is case-insensitive; values are stored lowercase."
        )
        ctk.CTkLabel(self, text=hint, font=ctk.CTkFont(size=12), text_color=("gray30","gray70")).grid(
            row=4, column=0, columnspan=3, **pad, sticky="w"
        )

        # --- Buttons ---
        btn_frame = ctk.CTkFrame(self)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=(10, 12))
        ctk.CTkButton(btn_frame, text="Save", width=120, command=self._save).pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Cancel", width=120, command=self._cancel).pack(side="left", padx=8)

        # grid stretch
        self.grid_columnconfigure(1, weight=1)

    def _browse_dir(self):
        d = filedialog.askdirectory(title="Select data directory")
        if d:
            self.dir_entry.delete(0, tk.END)
            self.dir_entry.insert(0, d)

    def _save(self):
        try:
            root_dir = self.dir_entry.get().strip()
            if not root_dir:
                raise ValueError("Please select a data directory.")
            if not os.path.isdir(root_dir):
                raise ValueError("Data directory does not exist.")

            chans = _split_csv_lower(self.chan_entry.get())
            stim = (self.stim_entry.get() or "").strip().lower()
            if not stim:
                raise ValueError("Please enter the stim channel name.")

            # Optional: prevent stim from also appearing in channels list
            chans = [c for c in chans if c != stim]

            drop_keywords = _split_csv_lower(self.kw_entry.get())

            self.result = {
                "root_dir": root_dir,
                "channel_names": chans,
                "stim_channel_name": stim,
                "drop_keywords": drop_keywords,
            }
            self.destroy()
        except Exception as e:
            messagebox.showerror("Invalid input", str(e))

    def _cancel(self):
        self.result = None
        self.destroy()

# --- Add this helper anywhere above run_index_inputs_gui ---
def _cleanup_tk(app):
    """Cancel pending .after() callbacks and destroy the Tk app safely."""
    try:
        # Cancel all scheduled 'after' callbacks
        try:
            # returns a space-separated list of ids
            after_ids = app.tk.call('after', 'info')
            if isinstance(after_ids, str):
                after_ids = after_ids.split()
            for aid in after_ids:
                try:
                    app.after_cancel(aid)
                except Exception:
                    pass
        except Exception:
            pass

        # Finish pending idle tasks, then destroy
        try:
            app.update_idletasks()
        except Exception:
            pass
        try:
            app.destroy()
        except Exception:
            pass
    except Exception:
        pass

def run_index_inputs_gui(defaults: dict | None = None):
    app = IndexInputsGUI(defaults=defaults)

    # Ensure window close (X) also sets a result and exits cleanly
    def _on_close():
        app.result = None
        app.quit()  # leave mainloop
    app.protocol("WM_DELETE_WINDOW", _on_close)

    try:
        app.mainloop()   # blocks until quit/destroy
        return app.result
    finally:
        # Always clean up CustomTkinter/Tk callbacks and destroy root
        _cleanup_tk(app)

# Example direct run:
if __name__ == "__main__":
    cfg = run_index_inputs_gui()
    if cfg is None:
        print("Canceled by user.")
    else:
        print(cfg)
