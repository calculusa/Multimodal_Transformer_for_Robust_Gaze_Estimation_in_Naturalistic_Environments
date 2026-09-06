# GazeUnconstrained Model

This repository accompanies our work on token level fusion of face and eye representations for three dimensional gaze estimation.

## Overview

The proposed model extracts spatial features from the full face and the left and right eye regions. These features are retained as tokens and fused through a Transformer encoder before predicting gaze yaw and pitch. The face provides global information about head orientation, while the eye regions preserve local ocular detail.

The main model uses ResNet50 backbones and a six layer Transformer encoder. Training uses L1 loss on yaw and pitch. Performance is evaluated using three dimensional angular error in degrees.

## Demo video

A demonstration video will be available here.

<!-- Replace this paragraph with the demonstration video URL after uploading it to GitHub. -->

## Dataset

The study uses GazeUnconstrained and MPIIFaceGaze. The extended GazeUnconstrained annotations include face and eye crops, bounding boxes, gaze labels, and associated metadata.

Data preparation and access: [GazeUnconstrained Dataset](https://github.com/calculusa/GazeUnconstrained-Dataset).

## Code structure

| Folder | Contents |
| :--- | :--- |
| `Config/` | Experiment settings and dataset paths |
| `Model/` | Model definitions |
| `Dataloader/` | Data loading and image transforms |
| `Train/` | Training scripts |
| `Test/` | Evaluation scripts |
| `Visualisation/` | Gaze visualisation |

Set the local dataset and output paths in the configuration before running an experiment. The current code release is incomplete; the original model definition and the remaining experimental workflows will be added when available.

## Results

Mean angular error under leave one subject out cross validation, as reported in the manuscript. Lower is better.

| Model | MPIIFaceGaze | GazeUnconstrained |
| :--- | ---: | ---: |
| Proposed face and eye fusion | 3.72° | 4.09° |

The held out participant is also used for model selection in this protocol.

## Citation and license

Citation details for the accompanying manuscript will be added when available. Please also cite the dataset paper when using GazeUnconstrained.

The code is released under the MIT License. See `LICENSE` for details.
