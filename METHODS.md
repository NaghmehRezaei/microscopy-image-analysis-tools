\# Methods Appendix — Microscopy Image Analysis



\## Overview



This document describes the image analysis workflow implemented in this

repository for microscopy-based cell and nuclear segmentation and

quantification.



The methods emphasize transparency, parameter interpretability, and

visual quality control rather than fully automated high-throughput

processing.



---



\## Image input and handling



Microscopy images are provided as TIFF, PNG, or JPG files and may contain

single or multiple channels. For multi-channel images, channels may be

selected manually or automatically based on mean intensity.



Higher-dimensional images are collapsed using maximum-intensity

projection when required for segmentation.



---



\## Image preprocessing



Optional preprocessing steps include:



\- White top-hat filtering for background correction

\- Contrast-limited adaptive histogram equalization (CLAHE)

\- Gaussian smoothing to reduce pixel-level noise

\- Percentile-based intensity normalization



All preprocessing steps are configurable through the user interface to

support dataset-specific tuning.



---



\## Segmentation model



Object segmentation is performed using \*\*StarDist2D\*\*, a deep-learning

model optimized for star-convex object detection in microscopy images.



A publicly available pre-trained fluorescence model is used, consistent

with exploratory and QC-driven analysis workflows.



---



\## Segmentation parameters



Segmentation behavior is controlled through:



\- Probability threshold

\- Non-maximum suppression (NMS) threshold

\- Adaptive tiling for large images



Both preview-scale and full-resolution segmentation modes are supported

to balance responsiveness and accuracy.



---



\## Object filtering and quantification



Post-segmentation filtering is applied at the object level based on:



\- Object area (pixel thresholds)

\- Mean raw intensity bounds



For retained objects, the following metrics are reported:



\- Object count

\- Mean object area

\- Mean object intensity

\- Optional conversion to physical units when pixel size is provided



---



\## Visualization and quality control



Segmentation results are visualized as overlays on the original image to

support rapid qualitative assessment. Quantitative summaries are

displayed alongside images to facilitate parameter tuning and result

interpretation.



