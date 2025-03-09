// SPDX-License-Identifier: GPL-3.0-only
// SPDX-FileCopyrightText: 2025 oaf40

function imt_isInInpaintMode() {
	const uiActiveTab = gradioApp().querySelector(
		"#tabs div.tab-nav button.selected",
	);
	const uiActiveImg2ImgTab = gradioApp().querySelector(
		"#mode_img2img div.tab-nav button.selected",
	);

	return (
		uiActiveTab?.textContent.trim() === "img2img" &&
		uiActiveImg2ImgTab?.textContent.trim() === "Inpaint"
	);
}

function imt_onImg2ImgTabChanged(mutationsList, observer) {
	for (const mut of mutationsList) {
		if (
			mut.type === "childList" &&
			mut.target.tagName.toLowerCase() === "button"
		) {
			const uiQuickControls = gradioApp().querySelectorAll(
				".imt_quickcontrols button",
			);
			const inpaintModeActive = imt_isInInpaintMode();
			for (const uiButton of uiQuickControls)
				uiButton.disabled = !inpaintModeActive;
			// ignore all the following mutations, run just once
			return;
		}
	}
}

onUiLoaded(() => {
	// Reapply quick controls labels because Gradio doesn't allow setting arbitrary
	// HTML tags as values from Python without creating a custom control.
	new Map([
		["img2img_imt_calc_round_plus_blur", "🎭<sup>B</sup>"],
		["img2img_imt_calc_raw", "🎭<sup>RAW</sup>"],
	]).forEach((v, k, m) => {
		gradioApp().getElementById(k).innerHTML = v;
	});
	// Move quick controls to the "Resize to" tab. There should be a way to do it
	// in Python tho..?
	const uiTargetParent = gradioApp().getElementById(
		"img2img_dimensions_row",
	).parentNode;
	for (const dom of gradioApp().querySelectorAll(".imt_quickcontrols")) {
		uiTargetParent.appendChild(dom);
	}
	// Quick controls are meant to be used only in the Inpaint mode, try to hide
	// them when the Inpaint tab becomes inactive.
	const uiTabsContainer = gradioApp().querySelector("#tabs");
	const uiImg2ImgTabsContainer = gradioApp().querySelector(
		"#mode_img2img div.tab-nav",
	);
	const tabObserver = new MutationObserver(imt_onImg2ImgTabChanged);
	const observerConf = {
		// I expected `attributes` but Svelte seems to work in a different way
		childList: true,
		subtree: true,
	};
	tabObserver.observe(uiTabsContainer, observerConf);
	tabObserver.observe(uiImg2ImgTabsContainer, observerConf);
});
