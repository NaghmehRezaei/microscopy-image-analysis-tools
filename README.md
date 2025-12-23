\# 🔬 Microscopy Image Analysis Tools



Interactive Python-based tools for microscopy image analysis, with a

focus on \*\*cell and nuclear segmentation\*\* using \*\*StarDist\*\* and

interactive visualization via \*\*Streamlit\*\*.



This repository documents a real, working analysis application used for

exploratory segmentation, parameter tuning, and quantitative quality

control of microscopy images.



---



\## Key features



\- Interactive image upload (TIFF / PNG / JPG)

\- Multi-channel image handling with automatic channel selection

\- Optional image preprocessing (background correction, contrast enhancement)

\- StarDist-based object segmentation

\- Adjustable probability and NMS thresholds

\- Object-level filtering by size and intensity

\- Quantitative summaries (counts, area, intensity)

\- Segmentation overlay visualization for QC



---



\## Workflow summary



Microscopy image  

→ optional preprocessing  

→ StarDist segmentation  

→ object filtering  

→ quantitative metrics  

→ visual overlay and quality control



---



\## Running the app locally



```bash

pip install -r requirements.txt

streamlit run app/stardist\_streamlit\_app.py



