



# Multimodal Transformer for Robust Gaze Estimation in Naturalistic Environments

Token level fusion of full face and eye features through a Transformer for 3D gaze estimation. The main model uses ResNet50 backbones and L1 training loss.

[Dataset and annotations](https://github.com/calculusa/GazeUnconstrained-Dataset)

## Demo video



https://github.com/user-attachments/assets/6c381bae-1e19-4435-97f3-f54b3b9bea27






## Usage

Requires PyTorch, torchvision, NumPy, Pillow, PyYAML, tqdm, and OpenCV. The current release still requires the original `Model/model_multi_r50_face_eye_old.py`.

In `Config/config_freeze_r50_face_eye_old.yaml`, set `dataset.root`, update pretrained weight paths, and set `train.save_dir` to `outputs/checkpoints`. Run from the repository root:

```bash
export PYTHONPATH="$PWD/Model:$PWD/Dataloader${PYTHONPATH:+:$PYTHONPATH}"
```

### Train

```bash
python Train/train_loso_multi_r50_freeze_face_eye_old.py \
  --config Config/config_freeze_r50_face_eye_old.yaml \
  --gpu 0
```

Saves a checkpoint for each participant, such as `outputs/checkpoints/best_p06.pth`.

### Test

```bash
python Test/test_loso.py \
  --config Config/config_freeze_r50_face_eye_old.yaml \
  --pid p06 \
  --ckpt_dir outputs/checkpoints \
  --out_dir outputs/predictions \
  --gpu 0
```

Replace `p06` with your participant. Predictions are saved as `preds_p06.csv`.

### Visualize

```bash
python Visualisation/visual_from_gt_preds.py \
  --config Config/config_freeze_r50_face_eye_old.yaml \
  --pid p06 \
  --pred_csv outputs/predictions/preds_p06.csv \
  --out_dir outputs/visualization \
  --max_samples 200
```

Use the same dataset and participant as testing. Exports PNG pairs to `outputs/visualization/vis_p06/`: reference gaze in red, prediction in green.

## Results

Manuscript mean angular error (°) under LOSO cross validation; the held out participant also supports model selection.

| Model | MPIIFaceGaze | GazeUnconstrained |
| :--- | ---: | ---: |
| Face and eye fusion | 3.72 | 4.09 |

## Citation and license

Manuscript citation forthcoming. Please also cite the dataset paper when using GazeUnconstrained. Code: MIT License; see `LICENSE`.
