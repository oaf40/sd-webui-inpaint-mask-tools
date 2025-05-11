# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 oaf40

import logging
from enum import auto, Enum
from math import ceil, sqrt
from textwrap import dedent

import cv2
import gradio as gr
import modules.scripts as scripts
import numpy as np
from PIL import Image, ImageOps
from modules import shared
from modules.masking import get_crop_region_v2
from modules.processing import create_binary_mask, StableDiffusionProcessingImg2Img
from modules.script_callbacks import on_ui_settings
from modules.ui_components import ToolButton

SCRIPT_NAME = "Inpaint Mask Tools"
MEGA = 1000 * 1000
MULTIPLY_FACTOR = 1.1
WHOLEPICTURE_SAFEGUARD_TOLERANCE = 0.03  # 3%


class CalcMode(Enum):
    RAW = auto()
    RAW_ROUND = auto()
    BLUR_PAD_ROUND = auto()
    INTERNAL = auto()  # used in Autoadjusting algorithm


logger = logging.getLogger(f"[{SCRIPT_NAME}]")
logger.setLevel(logging.INFO)


def round_by_8(val: float) -> int:
    """
    Round up the value to the nearest multiple of 8
    :param val: source value
    :return: rounded value
    """
    return int(ceil((val if val > 0 else 1) / 8) * 8)


def measure_bbox(bbox: tuple[int, int, int, int]) -> tuple[int, int, float]:
    """
    Measure the rectangle given its top left and bottom right coordinates
    :param bbox: bounding box (x1, y1, x2, y2)
    :return: width and height in pixels, resolution in Megapixels
    """
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    resolution = width * height / MEGA
    return width, height, resolution


class MaskDimensionsScript(scripts.Script):
    ui_components = {
        "img2maskimg": None,
        "img2img_width": None,
        "img2img_height": None,
        "img2img_mask_blur": None,
        "img2img_inpaint_full_res_padding": None,
        "img2img_mask_mode": None,
    }

    def title(self) -> str:
        return SCRIPT_NAME

    def show(self, is_img2img: bool) -> bool:
        # The script will be active only in img2img mode, and we will  also hide
        # the UI controls on the client-side when the Inpaint tab is inactive.
        return scripts.AlwaysVisible if is_img2img else False

    # Store references to the core UI elements
    def after_component(self, component, **kwargs):
        for ui_cid in self.ui_components:
            if kwargs.get("elem_id") == ui_cid:
                self.ui_components[ui_cid] = component

    def ui(self, is_img2img: bool):
        if not is_img2img:
            return

        # Make the accordion invisible as it's going to be just a temporary
        # container for the quick controls.
        with gr.Accordion(SCRIPT_NAME, visible=False):
            # Generate the quick controls as the part of the Accordion first,
            # then move it up to the "Resize to" tab with client-size Javascript.
            with gr.Column(scale=1, elem_classes="imt_quickcontrols dimensions-tools"):
                calc_blur_pad_round = ToolButton(
                    value="🎭<sup>BP</sup>",
                    elem_id="img2img_imt_calc_blur_pad_round",
                    tooltip="Calculate mask dimensions accounting for blur and padding, round to the multiple of 8",
                )
                calc_raw_round = ToolButton(
                    value="🎭",
                    elem_id="img2img_imt_calc_round",
                    tooltip="Calculate mask dimensions, round to the multiple of 8 (do not account for blur and padding)",
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
                    tooltip="Calculate mask dimensions (raw unrounded value, do not account for blur and padding)",
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
                calc_blur_pad_round, self.imt_on_calc_blur_pad_round
            )
            set_on_click_listener(calc_raw_round, self.imt_on_calc_raw_round)
            set_on_click_listener(calc_multiply, self.imt_on_calc_multiply)
            set_on_click_listener(calc_raw, self.imt_on_calc_raw)

        return None

    # Do our thing and let the rest of the workflow run as usual
    def process(self, p: StableDiffusionProcessingImg2Img) -> StableDiffusionProcessingImg2Img:
        if not p.image_mask:  # we need a mask to work
            return

        if shared.opts.imt_wholepicture_safeguard:
            p = self.imt_process_wholepicture_safeguard(p)
        if shared.opts.imt_autoadjust_onlymasked:
            p = self.imt_process_autoadjust_onlymasked(p)
        if shared.opts.imt_multipleof8_safeguard:
            p = self.imt_process_multipleof8_safeguard(p)
        return p

    def imt_on_calc_raw(self, canvas: dict, blur: int, padding: int, inv: int, fallback_width: int,
                        fallback_height: int) -> tuple[int, int]:
        """
        Calculate the width and height of the bounding box surrounding the masked area
        :param canvas: ["image", "mask"]
        :param blur: not used
        :param padding: not used
        :param inv: not used
        :param fallback_width: fallback value if mask doesn't exist
        :param fallback_height: fallback value if mask doesn't exist
        :return: width and height in pixels
        """
        mask = canvas.get("mask") if canvas else None
        return self.imt_calculate_bbox(CalcMode.RAW, mask, blur, padding, inv, fallback_width, fallback_height)

    def imt_on_calc_raw_round(self, canvas: dict, blur: int, padding: int, inv: int, fallback_width: int,
                              fallback_height: int) -> tuple[int, int]:
        """
        Calculate the width and height of the bounding box surrounding the masked area,
        round up the dimensions to the nearest multiple of 8.
        :param canvas: ["image", "mask"]
        :param blur: not used
        :param padding: not used
        :param inv: not used
        :param fallback_width: fallback value if mask doesn't exist
        :param fallback_height: fallback value if mask doesn't exist
        :return: width and height in pixels
        """
        mask = canvas.get("mask") if canvas else None
        return self.imt_calculate_bbox(CalcMode.RAW_ROUND, mask, blur, padding, inv, fallback_width, fallback_height)

    def imt_on_calc_blur_pad_round(self, canvas: dict, blur: int, padding: int, inv: int, fallback_width: int,
                                   fallback_height: int) -> tuple[int, int]:
        """
        Calculate the width and height of the bounding box surrounding the masked area while
        accounting for blur and padding, round up the dimensions to the nearest multiple of 8.
        :param canvas: ["image", "mask"]
        :param blur: blur factor
        :param padding: pad N pixels on each side
        :param inv: mask inversion flag
        :param fallback_width: fallback value if mask doesn't exist
        :param fallback_height: fallback value if mask doesn't exist
        :return: width and height in pixels
        """
        mask = canvas.get("mask") if canvas else None
        return self.imt_calculate_bbox(CalcMode.BLUR_PAD_ROUND, mask, blur, padding, inv, fallback_width,
                                       fallback_height)

    def imt_on_calc_multiply(
            self, canvas: dict, blur: int, padding: int, inv: int, width: int, height: int
    ) -> tuple[int, int]:
        """
        Multiply width and height by MULTIPLY_FACTOR and round up each value to the nearest multiple of 8.
        :param canvas: not used
        :param blur: not used
        :param padding: not used
        :param inv: not used
        :param width: value to multiply
        :param height: value to multiply
        :return: width and height in pixels
        """
        return round_by_8(width * MULTIPLY_FACTOR), round_by_8(height * MULTIPLY_FACTOR)

    def imt_calculate_bbox(
            self,
            calc_mode: CalcMode,
            mask: Image,
            blur: int,
            padding: int,
            inv: int,
            fallback_width: int,
            fallback_height: int
    ) -> tuple[int, int] | tuple[float, int, int, float, tuple]:
        """
        Common function for calculating the bounding box around the masked area.
        Account for blur and padding if requested.
        Round up the values to the nearest multiple of 8 if requested.
        :param calc_mode: calculation mode
        :param mask: mask
        :param blur: blur factor
        :param padding: pad N pixels on each side
        :param inv: mask inversion flag
        :param fallback_width: fallback value if mask doesn't exist
        :param fallback_height: fallback value if mask doesn't exist
        :return: (raw aspect ratio, B&P width, B&P height, B&P resolution, blurred bbox) for CalcMode.INTERNAL
        :return: (width, height) for all other CalcModes.
        """

        # EAFP, too many edge-cases to check especially when the Inpaint UI
        # glitches out displaying small cropped preview of an image
        try:
            if not (mask and get_crop_region_v2(mask)):
                raise RuntimeError()
        except:  # noqa: E722
            gr.Error("Cannot access the mask")
            logger.error("Cannot access the mask")
            return fallback_width, fallback_height

        imt_mask = create_binary_mask(mask)
        if inv:
            imt_mask = ImageOps.invert(imt_mask)
        bbox = get_crop_region_v2(imt_mask)  # raw bbox

        imt_width, imt_height, _ = measure_bbox(bbox)
        imt_aspect_ratio = imt_width / imt_height
        if calc_mode == CalcMode.RAW:
            return imt_width, imt_height
        elif calc_mode == CalcMode.RAW_ROUND:
            return round_by_8(imt_width), round_by_8(imt_height)

        # Calculate accounting for blur and padding
        imt_mask = self.imt_apply_blur(imt_mask, blur, blur)  # same blur factor for X and Y axes
        imt_mask = imt_mask.convert("L")
        bbox = get_crop_region_v2(imt_mask, padding)
        if not bbox:
            logger.warning("The mask doesn't exist, check if the Mask Blur value is too big")
            return fallback_width, fallback_height

        imt_width, imt_height, imt_resolution = measure_bbox(bbox)
        if calc_mode == CalcMode.BLUR_PAD_ROUND:
            return round_by_8(imt_width), round_by_8(imt_height)
        elif calc_mode == CalcMode.INTERNAL:
            return imt_aspect_ratio, imt_width, imt_height, imt_resolution, get_crop_region_v2(imt_mask, 0)
        else:
            logger.error("Unhandled calc_mode!")
            return fallback_width, fallback_height

    def imt_apply_blur(self, image: Image, blur_x: int, blur_y: int) -> Image:
        """
        Apply Gaussian blur to the image, the code was taken from the original WebUI
        :param image: source image
        :param blur_x: horizontal blur factor
        :param blur_y: vertical blur factor
        :return: blurred image
        """

        # Reference: modules/processing.py , commit 1c0a0c4c (v1.9.3)
        # SPDX-SnippetBegin
        # SPDX-License-Identifier: AGPL-3.0-only
        # SPDX-SnippetCopyrightText: 2022 AUTOMATIC1111 and contributors
        if blur_x > 0:
            np_image = np.array(image)
            kernel_size = 2 * int(2.5 * blur_x + 0.5) + 1
            np_image = cv2.GaussianBlur(np_image, (kernel_size, 1), blur_x)
            image = Image.fromarray(np_image)
        if blur_y > 0:
            np_image = np.array(image)
            kernel_size = 2 * int(2.5 * blur_y + 0.5) + 1
            np_image = cv2.GaussianBlur(np_image, (1, kernel_size), blur_y)
            image = Image.fromarray(np_image)
        # SPDX-SnippetEnd

        return image

    def imt_process_wholepicture_safeguard(self,
                                           p: StableDiffusionProcessingImg2Img,
                                           force=False) -> StableDiffusionProcessingImg2Img:
        """
        Interrupt generating when user forgets to switch from "Whole picture" to the
        "Masked area" inpainting mode.
        :param p: img2img job data
        :param force: force the check regardless of the Inpaint mode
        """

        # Allow some slight drift (defined by `WHOLEPICTURE_SAFEGUARD_TOLERANCE`)
        # in width and height in "Whole picture" mode because most likely these
        # are legit use cases of resizing an image at high denoising strengths.
        tolerance_ok = np.allclose(
            p.init_images[0].size,
            (p.width, p.height),
            WHOLEPICTURE_SAFEGUARD_TOLERANCE,
            0,
        )
        if (force or not p.inpaint_full_res) and not tolerance_ok:
            msg = """Detected unusual dimensions set for the \"Whole
                picture\" mode. Did you mean to use \"Only masked\" instead?"""
            shared.state.interrupt()
            gr.Warning(msg)
        return p

    def imt_process_autoadjust_onlymasked(self,
                                          p: StableDiffusionProcessingImg2Img) -> StableDiffusionProcessingImg2Img:
        """
        Measure the bounding box around the blurred and padded masked area,
        update p.width and p.height with measured dimensions,
        upscale to target resolution if it's too small.
        :param p: img2img job data
        """
        original_mask: Image = p.image_mask.convert("L")
        original_width: int = p.width
        original_height: int = p.height
        log_line: str = f"Requested {original_width}x{original_height}"

        if not p.inpaint_full_res:
            logger.warning(f"{log_line}, but not in Only masked mode. Nothing for us to do.")
            return p

        values = self.imt_calculate_bbox(
            CalcMode.INTERNAL, original_mask, p.mask_blur_x, p.inpaint_full_res_padding, p.inpainting_mask_invert, -1,
            -1
        )
        if values[0] == -1:
            # Parent function failed for whatever reason, most likely because the
            # user hasn't drawn any mask *and* didn't toggle the Inverse option.
            logger.warning(f"{log_line}, but failed to measure the raw mask. Bail out.")
            # At this point A1111 might quietly start a full run of unmasked img2img, and it's
            # prone to shrunk image error. Force the check before leaving.
            return self.imt_process_wholepicture_safeguard(p, True)
        # imt_aspect_ratio: aspect ratio of the raw mask (no blur, no padding); used in autoupscaling
        # imt_width: measured width of blurred and padded mask
        # imt_height: measured height of blurred and padded mask
        # imt_resolution: resolution of blurred and padded mask
        # bbox_blurred: coordinates of the bounding box of the blurred mask (no padding); used in autoupscaling
        imt_aspect_ratio, imt_width, imt_height, imt_resolution, bbox_blurred = values
        log_line += f", measured {imt_width}x{imt_height} ({round(imt_resolution, 2)} Mp)"

        # Autoupscaling routines
        if imt_resolution < shared.opts.imt_autoadjust_upscaleto:
            bbox_original = original_mask.getbbox()

            # There were very weird individual reports of `bbox_original` starting in (0, 0) while the `bbox_blurred`
            # derived from it was fine. Commit 31da54b should solve it, keep the safety check just in case.
            if bbox_blurred[0] > bbox_original[0] or bbox_blurred[1] > bbox_original[1] \
                    or bbox_blurred[2] < bbox_original[2] or bbox_blurred[3] < bbox_original[3]:
                gr.Error(dedent("""\
                    Fatal error: blurred mask is smaller than the original one.
                    Inpaint Mask Tools will not work this time. Please report this issue and
                    attach the mask and generation parameters."""))
                logger.error(dedent("""\
                    Fatal error: blurred mask is smaller than the original one.
                    Please report this issue and attach the mask and generation parameters."""))
                logger.error(f"bbox_original: {bbox_original}; bbox_blurred: {bbox_blurred}")
                return p

            # Width and height added by gaussian blur (in pixels)
            blurW = (bbox_original[0] - bbox_blurred[0]) + (bbox_blurred[2] - bbox_original[2])  # (left) + (right)
            blurH = (bbox_original[1] - bbox_blurred[1]) + (bbox_blurred[3] - bbox_original[3])  # (top) + (bottom)

            # Width and height added by padding the blurred mask (in pixels)
            def calc_space(allowance: int, requested: int) -> int:
                return max(min(allowance, requested), 0)

            padW = calc_space(bbox_blurred[0], p.inpaint_full_res_padding) + \
                   calc_space(original_mask.size[0] - bbox_blurred[2], p.inpaint_full_res_padding)  # (left) + (right)
            padH = calc_space(bbox_blurred[1], p.inpaint_full_res_padding) + \
                   calc_space(original_mask.size[1] - bbox_blurred[3], p.inpaint_full_res_padding)  # (top) + (bottom)

            # Calculate new width and height based on the target resolution
            target_resolution = shared.opts.imt_autoadjust_upscaleto * MEGA
            # The calculations below were derived from the following equation:
            # (padW + blurW + imt_width) * (padH + blurH + imt_height) = target_resolution, where
            # imt_width = imt_aspect_ratio * imt_height
            b = (padW + blurW) + imt_aspect_ratio * (padH + blurH)
            imt_height = (-b + sqrt(
                b * b - 4 * imt_aspect_ratio * ((padW + blurW) * (padH + blurH) - target_resolution))) / (
                                 2 * imt_aspect_ratio)

            imt_width = int(imt_aspect_ratio * imt_height) + blurW + padW
            imt_height = int(imt_height) + blurH + padH
            log_line += f", upscaled {imt_width}x{imt_height}"

        if imt_width % 8 or imt_height % 8:
            imt_width = round_by_8(imt_width)
            imt_height = round_by_8(imt_height)
            log_line += f", rounded {imt_width}x{imt_height} ({round(imt_width * imt_height / MEGA, 2)} Mp)"

        logger.info(log_line)
        gr.Info(f"Adjusted dimensions from {original_width}x{original_height} to {imt_width}x{imt_height}")
        p.width = imt_width
        p.height = imt_height
        return p

    def imt_process_multipleof8_safeguard(self,
                                          p: StableDiffusionProcessingImg2Img) -> StableDiffusionProcessingImg2Img:
        """
        Automatically round up width and height to the nearest multiple of 8
        :param p: img2img job data
        """
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
        return p


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
        is below the specified value. The upscaled values are rounded up to the 
        nearest multiple of 8, causing minimal impact on the original aspect 
        ratio. Set to 0 to disable this option. <b>Recommended values: 1–1.5</b>.""")  # noqa: W291
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
