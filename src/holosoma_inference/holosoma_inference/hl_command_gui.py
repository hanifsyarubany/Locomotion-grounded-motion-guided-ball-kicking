"""Lightweight popup window (tkinter) showing the live HlCommand stream a
NetworkControlledUnifiedPolicy process is receiving over UDP from a separate HL controller process
(run_classical_hl_controller.py) -- vx, vy, wyaw, and the kick-trigger level, refreshed in real
time. Purely a read-only observer: launched from run_network_controlled_policy.py alongside the
real 50Hz RL loop, in its own daemon thread, polling HlCommandReceiver.get_latest() (already
thread-safe -- see hl_command_channel.py's internal lock) at a much lower, human-perceptible rate.
Never touches the RL loop's own data path or timing.
"""

from __future__ import annotations

import threading
from typing import Callable

from loguru import logger

from holosoma_inference.hl_command_channel import HlCommandReceiver

_POLL_INTERVAL_MS = 100  # 10Hz -- plenty for a human-readable display, deliberately decoupled
# from the RL loop's own 50Hz rate; this window never reads/writes anything the RL loop touches
# except the same already-locked HlCommandReceiver.get_latest()/was_recently_triggered() the RL
# loop and receive thread themselves use.

_TRIGGER_DISPLAY_S = 1.5  # how long "KICK: TRIGGER" stays lit after an actual trigger packet --
# long enough for a human to actually see it (the underlying signal itself is a single ~20ms
# pulse -- see was_recently_triggered()'s docstring in hl_command_channel.py for why this can't
# just read the instantaneous value).

_STALE_COLOR = "#c0392b"  # red -- no signal / stale
_OK_COLOR = "#27ae60"  # green -- receiving
_IDLE_BG = "#2c3e50"  # dark slate -- kick not triggered
_TRIGGER_BG = "#c0392b"  # red -- kick triggered this tick
_FILTER_ACTIVE_BG = "#b8860b"  # amber -- locomotion forced to zero (--filter-out-loco active)
_FILTER_OFF_BG = "#2c3e50"  # same dark slate as idle kick state -- filter not active right now


def launch_hl_command_gui(
    receiver: HlCommandReceiver,
    command_port: int,
    get_loco_filter_remaining: Callable[[], float] | None = None,
    get_applied_command: Callable[[], tuple[float, float, float]] | None = None,
) -> threading.Thread | None:
    """Starts the popup in a daemon thread and returns immediately -- the caller's own run loop is
    never blocked by this. Returns None (and logs a warning, not an exception) if a GUI can't be
    created at all -- e.g. no reachable display -- since this is a pure convenience/observability
    feature and must never be able to take down the actual LL control process.

    get_loco_filter_remaining : optional callable returning seconds left in a --filter-out-loco
        window (0.0 if not active) -- see NetworkControlledUnifiedPolicy.loco_filter_remaining_s.
        If omitted, the filter status row is hidden entirely (matches --filter-out-loco 0, the
        default, where the feature doesn't exist at all as far as the display is concerned).
    get_applied_command : optional callable returning the ACTUAL (post-filter) (vx, vy, wyaw)
        currently applied to the robot -- see NetworkControlledUnifiedPolicy.get_applied_command.
        Deliberately separate from the main vx/vy/wyaw rows above, which show the RAW value
        received over UDP (this window's original purpose -- "what does the LL receive"): with a
        loco filter active, received and applied can legitimately differ, and conflating them
        made it look (during real testing) like the filter wasn't working, when actually the GUI
        was just never displaying the filtered value in the first place -- it was showing the
        unaffected raw command the whole time. If omitted, the applied-command row is hidden."""
    try:
        import tkinter as tk  # noqa: PLC0415 -- optional, deferred so a missing/broken display
        # never breaks anything for headless-only use of this module.
    except ImportError:
        logger.warning("tkinter not available -- skipping HL command monitor popup")
        return None

    ready = threading.Event()
    failure: list[BaseException] = []

    def _run() -> None:
        try:
            root = tk.Tk()
        except tk.TclError as e:
            failure.append(e)
            ready.set()
            return

        extra_h = (40 if get_loco_filter_remaining else 0) + (34 if get_applied_command else 0)
        root.title(f"LL command monitor — UDP :{command_port}")
        root.geometry(f"340x{240 + extra_h}")
        root.resizable(False, False)
        try:
            root.attributes("-topmost", True)
        except tk.TclError:
            pass  # not fatal -- some window managers reject this, window still opens normally

        font_label = ("Courier New", 11)
        font_value = ("Courier New", 22, "bold")

        status_var = tk.StringVar(value="waiting for HL...")
        status_label = tk.Label(root, textvariable=status_var, font=("Courier New", 10), fg="gray")
        status_label.pack(pady=(10, 6))

        if get_applied_command is not None:
            tk.Label(root, text="HL sends (raw)", font=("Courier New", 9), fg="gray").pack(anchor="w", padx=18)

        value_vars: dict[str, tk.StringVar] = {}
        specs = [("vx", "m/s"), ("vy", "m/s"), ("wyaw", "rad/s")]
        for name, unit in specs:
            row = tk.Frame(root)
            row.pack(fill="x", padx=18, pady=3)
            tk.Label(row, text=f"{name} ({unit})", font=font_label, width=14, anchor="w").pack(side="left")
            var = tk.StringVar(value="--.---")
            tk.Label(row, textvariable=var, font=font_value, width=8, anchor="e").pack(side="right")
            value_vars[name] = var

        applied_var = applied_label = None
        if get_applied_command is not None:
            applied_var = tk.StringVar(value="LL applies: --.--- / --.--- / --.---")
            applied_label = tk.Label(root, textvariable=applied_var, font=("Courier New", 11, "bold"), fg=_OK_COLOR)
            applied_label.pack(anchor="w", padx=18, pady=(2, 8))

        trigger_frame = tk.Frame(root, bg=_IDLE_BG, height=54)
        trigger_frame.pack(fill="x", padx=18, pady=(14, 10))
        trigger_frame.pack_propagate(False)
        trigger_var = tk.StringVar(value="KICK: idle")
        trigger_label = tk.Label(
            trigger_frame, textvariable=trigger_var, font=("Courier New", 15, "bold"), bg=_IDLE_BG, fg="white"
        )
        trigger_label.pack(expand=True)

        filter_frame = filter_label = filter_var = None
        if get_loco_filter_remaining is not None:
            filter_frame = tk.Frame(root, bg=_FILTER_OFF_BG, height=40)
            filter_frame.pack(fill="x", padx=18, pady=(0, 10))
            filter_frame.pack_propagate(False)
            filter_var = tk.StringVar(value="LOCO: normal")
            filter_label = tk.Label(
                filter_frame, textvariable=filter_var, font=("Courier New", 12, "bold"), bg=_FILTER_OFF_BG, fg="white"
            )
            filter_label.pack(expand=True)

        def poll() -> None:
            cmd = receiver.get_latest()
            if cmd is None:
                status_var.set("NO SIGNAL (stale or no HL connected)")
                status_label.config(fg=_STALE_COLOR)
                for var in value_vars.values():
                    var.set("--.---")
                trigger_var.set("KICK: --")
                trigger_frame.config(bg=_IDLE_BG)
                trigger_label.config(bg=_IDLE_BG)
            else:
                status_var.set("receiving")
                status_label.config(fg=_OK_COLOR)
                value_vars["vx"].set(f"{cmd.vx:+.3f}")
                value_vars["vy"].set(f"{cmd.vy:+.3f}")
                value_vars["wyaw"].set(f"{cmd.wyaw:+.3f}")
                # was_recently_triggered(), not cmd.trigger_kick directly: trigger_kick is only
                # True for a single ~20ms tick (one UDP packet) when the HL controller fires it --
                # sampling that instantaneous value on this window's own 10Hz (100ms) timer would
                # miss it on the large majority of actual triggers (confirmed by direct visual
                # testing: the indicator stayed on "idle" through real kicks). was_recently_
                # triggered() is latched in the receiver's own recv loop, which sees every packet
                # as it arrives (no aliasing), so it reliably catches every trigger regardless of
                # this window's poll rate.
                if receiver.was_recently_triggered(_TRIGGER_DISPLAY_S):
                    trigger_var.set("KICK: TRIGGER")
                    trigger_frame.config(bg=_TRIGGER_BG)
                    trigger_label.config(bg=_TRIGGER_BG)
                else:
                    trigger_var.set("KICK: idle")
                    trigger_frame.config(bg=_IDLE_BG)
                    trigger_label.config(bg=_IDLE_BG)

            # Applied (post-filter) command -- separate from the raw vx/vy/wyaw rows above on
            # purpose, see get_applied_command's docstring for why. Also independent of whether an
            # HL signal is currently live: the policy still applies *something* (zero, if cmd is
            # None) every tick regardless.
            if get_applied_command is not None:
                avx, avy, awyaw = get_applied_command()
                applied_var.set(f"LL applies: {avx:+.3f} / {avy:+.3f} / {awyaw:+.3f}")
                is_filtered = get_loco_filter_remaining is not None and get_loco_filter_remaining() > 0.0
                applied_label.config(fg=_FILTER_ACTIVE_BG if is_filtered else _OK_COLOR)

            # Filter status is independent of whether an HL signal is currently live (it reflects
            # NetworkControlledUnifiedPolicy's own internal timer, not anything in the cmd stream
            # itself), so this runs unconditionally, outside the cmd is-None branch above.
            if get_loco_filter_remaining is not None:
                remaining = get_loco_filter_remaining()
                if remaining > 0.0:
                    filter_var.set(f"LOCO: FILTERED ({remaining:.1f}s)")
                    filter_frame.config(bg=_FILTER_ACTIVE_BG)
                    filter_label.config(bg=_FILTER_ACTIVE_BG)
                else:
                    filter_var.set("LOCO: normal")
                    filter_frame.config(bg=_FILTER_OFF_BG)
                    filter_label.config(bg=_FILTER_OFF_BG)
            # Rescheduling itself via root.after (rather than a plain while-loop + sleep) keeps
            # this on Tk's own event loop, which is the only thread-safe way to touch these
            # widgets -- do not call any of the *_var.set()/*.config() lines above from outside
            # this callback.
            root.after(_POLL_INTERVAL_MS, poll)

        root.after(_POLL_INTERVAL_MS, poll)
        ready.set()
        root.mainloop()

    thread = threading.Thread(target=_run, daemon=True, name="hl-command-gui")
    thread.start()

    # Block briefly for the window to either come up or fail fast, so a bad/missing display
    # produces one clear log line right at startup instead of a silent no-op discovered later.
    ready.wait(timeout=5.0)
    if failure:
        logger.warning(f"Could not open HL command monitor popup ({failure[0]}) -- continuing without it")
        return None
    if not ready.is_set():
        logger.warning("HL command monitor popup did not start within 5s -- continuing without it")
        return None
    return thread
