#!/usr/bin/env python3
# SPDX-License-Identifier: GPL-3.0-only
# SPDX-FileCopyrightText: 2025 oaf40

import cv2
import numpy as np
from PIL import Image, ImageDraw


class MeasureBlur:
    def measure_content(self, img: Image) -> tuple[int, int, float]:
        """
        Measure the bounding box surrounding non-zero pixels of an image
        :param img: source image
        :return: width and height in pixels, resolution in Megapixels
        """
        bbox = img.getbbox()
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        resolution = width * height / (1000 * 1000)
        return width, height, resolution

    def run_tests(self):
        """
        Calculate how many pixels does each Mask Blur value add to the width and height of a white rectangle.
        Print the list of width deltas (the width differences between non-blurred and blurred images).
        """
        values = []
        canvas_width = 2048
        canvas_height = 2048

        for blur in range(0, 64 + 1):
            # for (fig_width, fig_height) in [(512, 512), (768, 364), (1024, 1200)]:
            for (fig_width, fig_height) in [(512, 512)]:
                mask_blur_x = blur
                mask_blur_y = blur

                img = Image.new("L", (canvas_width, canvas_height))
                canvas = ImageDraw.Draw(img)

                x1 = int(canvas_width / 2 - fig_width / 2)
                y1 = int(canvas_height / 2 - fig_height / 2)
                x2 = x1 + fig_width - 1
                y2 = y1 + fig_height - 1
                canvas.rectangle((x1, y1, x2, y2), fill="white")

                # SPDX-SnippetBegin
                # SPDX-License-Identifier: AGPL-3.0-only
                # SPDX-SnippetCopyrightText: 2022 AUTOMATIC1111 and contributors
                if mask_blur_x > 0:
                    np_mask = np.array(img)
                    kernel_size = 2 * int(2.5 * mask_blur_x + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (kernel_size, 1), mask_blur_x)
                    img = Image.fromarray(np_mask)

                if mask_blur_y > 0:
                    np_mask = np.array(img)
                    kernel_size = 2 * int(2.5 * mask_blur_y + 0.5) + 1
                    np_mask = cv2.GaussianBlur(np_mask, (1, kernel_size), mask_blur_y)
                    img = Image.fromarray(np_mask)
                # SPDX-SnippetEnd

                w, h, _ = self.measure_content(img)
                dW = w - fig_width
                # dH = h - fig_height
                values.append(dW)  # the difference is equal for width and height
        print(values)


if __name__ == "__main__":
    MeasureBlur().run_tests()
