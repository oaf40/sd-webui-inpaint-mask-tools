# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 oaf40

# This module handles some differences between v3 and v4 of Gradio

from PIL import Image

import gradio as gr

GRADIO_V4 = gr.__version__.startswith("4")


def get_canvas_uuid(params: dict) -> str:
    """
    Extract the UUID of the canvas by parsing its inner HTML
    :param params: component parameters
    :return: UUID string
    """
    ui_html: str = params.get("value")
    idx_start: int = ui_html.index("id=\"container_uuid_") + len("id=\"container_uuid_")
    idx_end: int = ui_html.index("\">", idx_start)
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
