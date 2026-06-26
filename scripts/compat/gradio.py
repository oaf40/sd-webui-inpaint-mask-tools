# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025-2026 oaf40

# This module handles some differences between v3 and v4 of Gradio

import sys

import gradio as gr
from modules import shared
from PIL import Image

GRADIO_V4 = gr.__version__.startswith("4")
if GRADIO_V4:
    from gradio.context import LocalContext as ctx
else:
    from gradio import context as ctx


def get_canvas_uuid(params: dict) -> str:
    """
    Extract the UUID of the canvas by parsing its inner HTML
    :param params: component parameters
    :return: UUID string
    """
    ui_html: str = params.get("value")
    idx_start: int = ui_html.index('id="container_uuid_') + len('id="container_uuid_')
    idx_end: int = ui_html.index('">', idx_start)
    return ui_html[idx_start:idx_end]


def canvas_to_image(canvas) -> Image:
    """
    Extract the image of the mask from whatever Gradio gave us
    :param canvas: ["image", "mask"] for Gradio v3, Image for Gradio v4
    :return: the image of the mask
    """
    if GRADIO_V4:
        return canvas
    else:
        return canvas.get("mask") if canvas else None


def show_notification(level: str, message: str):
    """
    Try to show a notification in the UI. In vanilla A1111 it's done using
    standard Gradio API however Forge and its derivatives run `process` in
    a separate thread and Gradio API doesn't want to show notifications from
    non-UI threads without a workaround. Error messages are processed a bit
    differently so they would be downgraded to warning in the workaround.
    :param level: ["info", "warning", "error"]
    :param message: message to show
    """

    # Exception to stderr
    def __stderr(ex: Exception):
        print("Can't enqueue a notification", file=sys.stderr)
        print(ex, file=sys.stderr)

    # Internal function to enqueue a log message
    def __enqueue(ui_root, event_id: str, message: str, level: str):
        try:
            if not ui_root.enable_queue:
                raise ValueError("Event queue is disable on this UI")
            ui_root._queue.log_message(event_id=event_id, log=message, level=level)
        except Exception as ex:
            # We don't know what could future versions of `log_message` throw,
            # catch all just to be sure.
            __stderr(ex)

    # The accessibility of `block` and `event_id` fields is an indicator of
    # our code running on UI thread. This means we are either on vanilla A1111
    # or in the far future when Gradio has fixed their notification handling.
    blocks = ctx.blocks.get(None) if GRADIO_V4 else getattr(ctx.thread_data, "blocks", None)
    event_id = ctx.event_id.get(None) if GRADIO_V4 else getattr(ctx.thread_data, "event_id", None)
    if blocks and event_id:
        if level == "error":  # Try the native way
            gr.Error(message)
        else:
            __enqueue(blocks, event_id, message, level)
        return

    # Try a workaround: access the root UI element's event queue
    if level == "error":
        level = "warning"
    try:  # EAFP
        blocks = shared.demo
        # Get the id of the first valid event of the first valid active job
        for job in [j for j in blocks._queue.active_jobs if j]:
            for event in [e for e in job if e]:
                __enqueue(blocks, event._id, message, level)
                return
    except Exception as ex:
        __stderr(ex)
