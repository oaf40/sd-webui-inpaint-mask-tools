# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 oaf40

import logging
from math import sqrt
from textwrap import dedent

import cv2
import gradio as gr
import numpy as np
from PIL import Image, ImageOps

import modules.scripts as scripts
from modules import shared
from modules.masking import get_crop_region_v2
from modules.processing import create_binary_mask
from modules.script_callbacks import on_ui_settings
from modules.ui_components import ToolButton

SCRIPT_NAME = "Inpaint Mask Tools"
MEGA = 1024 * 1024
MULTIPLY_FACTOR = 1.1
WHOLEPICTURE_SAFEGUARD_TOLERANCE = 0.03  # 3%

logger = logging.getLogger(f"[{SCRIPT_NAME}]")
logger.setLevel(logging.INFO)


def round_by_8(val):
    return int(round(val / 8) * 8)


class MaskDimensionsScript(scripts.Script):
    ui_components = {
        "img2maskimg": None,
        "img2img_width": None,
        "img2img_height": None,
        "img2img_mask_blur": None,
        "img2img_inpaint_full_res_padding": None,
        "img2img_mask_mode": None,
    }

    def title(self):
        return SCRIPT_NAME

    def show(self, is_img2img):
        # The script will be active only in img2img mode, and we will  also hide
        # the UI controls on the client-side when the Inpaint tab is inactive.
        return scripts.AlwaysVisible if is_img2img else False

    def ui(self, is_img2img):
        if not is_img2img:
            return

        # Make the accordion invisible as it's going to be just a temporary
        # container for the quick controls.
        with gr.Accordion(SCRIPT_NAME, visible=False):
            # Generate the quick controls as the part of the Accordion first,
            # then move it up to the "Resize to" tab with client-size Javascript.
            with gr.Column(scale=1, elem_classes="imt_quickcontrols dimensions-tools"):
                calc_round_plus_blur = ToolButton(
                    value="🎭<sup>B</sup>",
                    elem_id="img2img_imt_calc_round_plus_blur",
                    tooltip="Calculate mask dimensions, round to the multiple of 8, add Mask Blur diameter",
                )
                calc_round = ToolButton(
                    value="🎭",
                    elem_id="img2img_imt_calc_round",
                    tooltip="Calculate mask dimensions, round to the multiple of 8",
                )
            with gr.Column(scale=1, elem_classes="imt_quickcontrols dimensions-tools"):
                calc_multiply = ToolButton(
                    value=f"x{MULTIPLY_FACTOR}",
                    elem_id="img2img_imt_calc_multiply",
                    tooltip=f"Multiply the current width and height by {MULTIPLY_FACTOR}, round to the multiple of 8",
                )
                calc_raw = ToolButton(
                    value="🎭<sup>RAW</sup>",
                    elem_id="img2img_imt_calc_raw",
                    tooltip="Calculate mask dimensions (raw unrounded value, no blur)",
                )

            # Some button don't need all the inputs, still keep them the same
            # across all buttons for better code consistency
            button_inputs = [
                self.ui_components[x]
                for x in [
                    "img2maskimg",
                    "img2img_mask_blur",
                    "img2img_inpaint_full_res_padding",
                    "img2img_mask_mode",
                    "img2img_width",
                    "img2img_height",
                ]
            ]
            button_outputs = [
                self.ui_components[x] for x in ["img2img_width", "img2img_height"]
            ]

            def set_on_click_listener(btn, fn):
                btn.click(
                    fn=fn,
                    inputs=button_inputs,
                    outputs=button_outputs,
                    show_progress=False,
                )

            set_on_click_listener(
                calc_round_plus_blur, self.imt_on_calc_round_plus_blur
            )
            set_on_click_listener(calc_round, self.imt_on_calc_round)
            set_on_click_listener(calc_multiply, self.imt_on_calc_multiply)
            set_on_click_listener(calc_raw, self.imt_on_calc_raw)

        return None

    def imt_on_calc_round_plus_blur(
        self, canvas, blur, padding, mode, fallback_width, fallback_height
    ):
        w, h = self.imt_calculate_bbox(
            canvas.get("mask") if canvas else None,
            blur,
            padding,
            mode,
            fallback_width,
            fallback_height,
        )

        if w == fallback_width and h == fallback_height:
            return w, h

        w = round_by_8(w)
        h = round_by_8(h)
        return w, h

    def imt_on_calc_round(
        self, canvas, blur, padding, mode, fallback_width, fallback_height
    ):
        width, height = self.imt_on_calc_raw(
            canvas, blur, padding, mode, fallback_width, fallback_height
        )
        return round_by_8(width), round_by_8(height)

    def imt_on_calc_multiply(
        self, canvas, blur, padding, mode, fallback_width, fallback_height
    ):
        return round_by_8(fallback_width * MULTIPLY_FACTOR), round_by_8(
            fallback_height * MULTIPLY_FACTOR
        )

    def imt_on_calc_raw(
        self, canvas, blur, padding, mode, fallback_width, fallback_height
    ):
        w, h = self.imt_calculate_bbox(
            canvas.get("mask") if canvas else None,
            0,
            0,
            mode,
            fallback_width,
            fallback_height,
        )
        return w, h

    def imt_calculate_bbox(
        self, image_mask, mask_blur, padding, invert, fallback_width, fallback_height
    ):
        # Check if mask actually exists
        if not (image_mask and (bbox := get_crop_region_v2(image_mask))):
            gr.Error("Cannot access the mask")
            logger.error("Cannot access the mask")
            return fallback_width, fallback_height

        # TODO: This will break once A1111 adds support for separate blur values
        mask_blur_x = mask_blur
        mask_blur_y = mask_blur

        # Unfortunately A1111 doesn't have a standalone function for blurring
        # the mask, so I had to copy-paste and adapt the code from there.
        # Reference: modules/processing.py , commit 1c0a0c4c (v1.9.3)
        # SPDX-SnippetBegin
        # SPDX-License-Identifier: AGPL-3.0-only
        # SPDX-SnippetCopyrightText: 2022 AUTOMATIC1111 and contributors
        image_mask = create_binary_mask(image_mask)
        if invert:
            image_mask = ImageOps.invert(image_mask)

        # Edge case: some users expand the inpainting area by drawing the dots
        # with the *smallest* brush size, however these dots get completely washed
        # away when the blur value is too high which results in smaller
        # calculated area. One way to solve this issue is to calculate mask
        # dimensions before and after blurring the mask then pick the biggest one.
        x1, y1, x2, y2 = bbox
        width_no_blur = x2 - x1 + 1
        height_no_blur = y2 - y1 + 1

        if mask_blur_x > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(2.5 * mask_blur_x + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur_x)
            image_mask = Image.fromarray(np_mask)

        if mask_blur_y > 0:
            np_mask = np.array(image_mask)
            kernel_size = 2 * int(2.5 * mask_blur_y + 0.5) + 1
            np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur_y)
            image_mask = Image.fromarray(np_mask)
        # SPDX-SnippetEnd

        bbox = get_crop_region_v2(image_mask, padding)
        width_blur = 0
        height_blur = 0
        if bbox:
            x1, y1, x2, y2 = bbox
            width_blur = x2 - x1 + 1
            height_blur = y2 - y1 + 1

        if width_no_blur > width_blur or height_no_blur > height_blur:
            logger.info("Blurred mask is smaller than non-blurred, fixing")
            return width_no_blur, height_no_blur
        else:
            return width_blur, height_blur

    # Do our thing and let the rest of the workflow run as usual, return nothing
    def process(self, p):
        if not p.image_mask:  # we need a mask to work
            return

        if shared.opts.imt_wholepicture_safeguard:
            self.imt_process_wholepicture_safeguard(p)
        if shared.opts.imt_autoadjust_onlymasked:
            self.imt_process_autoadjust_onlymasked(p)
        if shared.opts.imt_multipleof8_safeguard:
            self.imt_process_multipleof8_safeguard(p)

    # Interrupt generating when user forgets to switch from "Whole picture" to the
    # "Masked area" inpainting mode.
    def imt_process_wholepicture_safeguard(self, p):
        src_width, src_height = p.init_images[0].size

        # Allow some slight drift (defined by `WHOLEPICTURE_SAFEGUARD_TOLERANCE`)
        # in width and height in "Whole picture" mode because such usecases are
        # most likely legit use cases of resizing an image at high denoising strengths.
        tolerance_ok = np.allclose(
            p.init_images[0].size,
            (p.width, p.height),
            WHOLEPICTURE_SAFEGUARD_TOLERANCE,
            0,
        )
        if not p.inpaint_full_res and not tolerance_ok:
            msg = """Detected unusual dimensions set for the \"Whole
                picture\" mode. Did you mean to use \"Only masked\" instead?"""
            shared.state.interrupt()
            gr.Warning(msg)

    # Handle autoadjusting of width and height
    def imt_process_autoadjust_onlymasked(self, p):
        image_mask = p.image_mask
        if not p.inpaint_full_res:
            return  # Expecting "Only masked" mode with mask

        width = p.width
        height = p.height
        log_line = f"Requested {width}x{height}"

        # Compute bounding box without accounting for the blur and padding
        bbox_width, bbox_height = self.imt_calculate_bbox(
            image_mask, 0, 0, p.inpainting_mask_invert, width, height
        )
        log_line += f", measured {bbox_width}x{bbox_height}"
        # Upscale to target resolution if bounding box is too small
        if bbox_width * bbox_height / MEGA < shared.opts.imt_autoadjust_upscaleto:
            # Calculate upscaled width and height
            aspect_ratio = bbox_width / bbox_height
            height = int(
                sqrt(shared.opts.imt_autoadjust_upscaleto * MEGA / aspect_ratio)
            )
            width = int(height * aspect_ratio)
            scale_factor = width / bbox_width
            # Temporarily upscale the whole mask to account for blur and padding below
            image_mask = image_mask.resize(
                tuple([int(side * scale_factor) for side in image_mask.size])
            )
            log_line += f", upscaled {width}x{height}"

        width, height = self.imt_calculate_bbox(
            image_mask,
            p.mask_blur_x,
            p.inpaint_full_res_padding,
            p.inpainting_mask_invert,
            width,
            height,
        )
        log_line += f", blur&padding {width}x{height}"

        width = round_by_8(width)
        height = round_by_8(height)
        log_line += f", rounded {width}x{height}"
        logger.info(log_line)
        gr.Info(f"Adjusted size from {p.width}x{p.height} to {width}x{height}")
        p.width = width
        p.height = height

    # Store references to the core UI elements
    def after_component(self, component, **kwargs):
        for ui_cid in self.ui_components:
            if kwargs.get("elem_id") == ui_cid:
                self.ui_components[ui_cid] = component

    # Automatically round width and height to the closest multiple of 8
    def imt_process_multipleof8_safeguard(self, p):
        old_width = p.width
        old_height = p.height
        if old_width % 8:
            p.width = round_by_8(old_width)
        if old_height % 8:
            p.height = round_by_8(old_height)
        if p.width != old_width or p.height != old_height:
            log_line = (
                f"Adjusted size from {old_width}x{old_height} to {p.width}x{p.height}"
            )
            logger.info(log_line)
            gr.Info(log_line)


def imt_init_settings():
    section = ("imt", SCRIPT_NAME)
    shared.opts.add_option(
        "imt_autoadjust_onlymasked",
        shared.OptionInfo(
            False,
            'Autoadjust Width and Height in "Only masked" mode',
            gr.Checkbox,
            section=section,
        ).info(
            dedent("""Automatically override specified width and height when you click 
                "Generate". Provides faster UX but less flexible: doesn't work 
                well when the masked area is under 1 Mpx. <b>Check out the 
                "Upscale" setting for better results!</b>""")  # noqa: W291
        ),
    )
    shared.opts.add_option(
        "imt_autoadjust_upscaleto",
        shared.OptionInfo(
            0,
            "Upscale small areas to resolution (Mpx)",
            gr.Slider,
            {"minimum": 0, "maximum": 4, "step": 0.1},
            section=section,
        ).info(
            dedent("""Upscale the width and height if the masked area's resolution 
        is below the specified value. The upscaled values are rounded to the 
        nearest multiple of 8, causing minimal impact on the original aspect 
        ratio. Set to 0 to disable this option. <b>Recommended values: 1–1.5</b>""")  # noqa: W291
        ),
    )
    shared.opts.add_option(
        "imt_wholepicture_safeguard",
        shared.OptionInfo(
            True, '"Whole picture" inpainting safeguard', gr.Checkbox, section=section
        ).info(
            dedent("""Prevent image generation in "Whole Picture" mode if the 
        target dimensions exceed 3% of the original size. Helps avoid wasted 
        time on shrunken images when "Only Masked" mode is not selected.""")  # noqa: W291
        ),
    )
    shared.opts.add_option(
        "imt_multipleof8_safeguard",
        shared.OptionInfo(
            True, "Auto-round width & height to ×8", gr.Checkbox, section=section
        ).info(
            dedent("""Prevent nasty visual glitches on the edges of inpainted 
        areas. Keeping this option enabled is recommended.""")  # noqa: W291
        ),
    )


on_ui_settings(imt_init_settings)
