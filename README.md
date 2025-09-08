# TrackOff
This repository provides Python scripts to track offsets between two input images based on a CNN model.

Presented workflow is provided to determine displacements amongst horizontal axes between image pairs to determine displacement field. Main aim of presented solution is to calculate glacier velocity maps based on SAR intensity images. The full description is provided by Łucka (2025).

Main workflow consits of 4 scripts:\
(1) to create master image tiles,\
(2) to create secondary image tiles,\
(3) to train CNN model and generate output displacement field,\
(4) to filter outliers and low-quality points.

Three sample datasets are provided in the specified catalogue: two manually shifted image pairs (cat photo and amplitude image of Daugaard-Jensen glacier) and one real-world dataset from Daugaard-Jensen glacier. The real-world dataset consists of two coregistered SAR images from Sentinel-1 satellite from 2024/02/03 and 2024/02/15 dates. Moreover, for the real-world dataset, the Python script to download original Sentinel-1 scenes is provided. In order to start calculations, the user has to choose one of three sample datasets and thus apply prepared scripts step by step from 1 to 4. As a result, the .csv file is generated with filtered offset values amongst X, Y axes of the master images and for real-world application, the velocity of each grid point is calculated based on temporal baseline between SAR acquisitions.

Datasets gathered in this repository and developed Python scripts are part of a scientific project "Novel view on the study of glacier kinematics in the context of global climate change" funded by National Science Center of Poland (project ID: 2022/45/N/ST10/02382). More information about the project and its progress: https://magdalucka.github.io/en/menu/preludium/

<img width="3082" height="268" alt="image" src="https://github.com/user-attachments/assets/2f53a1d8-ad18-4c02-b489-2d2be411f448" />

### References
Łucka M., 2025; "Investigation of machine learning algorithms to determine glaciers displacements", Remote Sensing Applications: Society and Environment, ISSN 2352-9385, DOI: https://doi.org/10.1016/j.rsase.2025.101476.
