# TrackOff
This repository provides Python scripts to track offsets between two input images based on a CNN model.

Presented workflow is provided to determine displacmenets amongst horizontal axes between image pairs to determine displacement field. Main aim of presented solution is to calcualte glacier velocity maps based on SAR intenisty images. The full descirption is provided by Łucka (2025).

Main workflow conists of 4 scripts:\
(1) to create master image tiles,\
(2) to create sceondary image tiles,\
(3) to train CNN model and generate output dispalcement field,\
(4) to filter outliers and low-quality points.

Three sample datasets are provided: two manually shifted image pairs (cat photo and amplitude image of Daugaard-Jensen glacier) and one real-world data from Daugaard-Jensen glacier. The real-world dataset consists of two coregistered SAR images from Sentinel-1 satellite from 2024/02/03 and 2024/02/15 dates.

### References
Łucka M., 2025; "Investigation of machine learning algorithms to determine glaciers displacements", Remote Sensing Applications: Society and Environment, ISSN 2352-9385, DOI: https://doi.org/10.1016/j.rsase.2025.101476.
