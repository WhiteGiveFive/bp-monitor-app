# Source and model provenance

The deployed models and prepared recording format are derived from the sibling
`bp-monitor` research project. The deployment tools import that project's
frozen definitions and load existing checkpoints; no retraining is performed.

The original FusionNet architecture and feature pipeline originate from
Ninja-FL/Hybrid-model-BP at commit
`f6b0c58c47bd7a958dab013472b2908128d65020`. The original project's notice states:

> The upstream snapshot did not contain a license file. The organiser-supplied loader and data remain external inputs.

The Mean Beats model is the `bp-monitor` adaptation documented in
`src/bp_monitor_att_fusion/NOTICE.md` and its frozen M6 final-v2 recipe.
The export wrapper preserves the trained weights, replacing unsupported tensor
operations with equivalent fixed-shape operations verified against the source.

LightGBM is developed by Microsoft and contributors under its MIT license.
The native tree evaluator reads exported numerical tree parameters and mirrors
LightGBM prediction semantics, including near-zero and missing-value handling.

Core ML is Apple's on-device inference framework. Model conversion uses Apple's
Core ML Tools and PyTorch. Their installed package license notices accompany
the conversion environment; those Python packages are not embedded in the app.

The locally bundled research recordings come from the user's existing competition
download. This project does not grant rights to redistribute data, models, or
upstream source. Generated recordings and model bundles are excluded from Git by
default. No upload or publication is part of this implementation.
