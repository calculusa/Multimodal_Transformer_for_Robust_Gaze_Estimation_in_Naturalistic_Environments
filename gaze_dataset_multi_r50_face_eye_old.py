# gaze_dataset_multi_face_eye.py  (L1 兼容 + 小增强)
# ------------------------------------------------------------
import json, torch, random
from pathlib import Path
from typing import List, Dict
from PIL import Image
import torchvision.transforms as T
import torchvision.transforms.functional as Fv

YAW_MAX   =  40.0
PITCH_MAX =  40.0

class GazeDatasetMulti(torch.utils.data.Dataset):
    def __init__(self, root_dir: str, pid_list: List[str],
                 face_size: int, eye_size: int,
                 augment: bool=True, use_frame: bool=False,
                 hflip_p: float = 0.0):
        self.use_frame = use_frame
        self.hflip_p = hflip_p if augment else 0.0
        self.samples: List[Dict] = []
        root = Path(root_dir)

        for pid in pid_list:
            ann = root/pid/"annotations.json"
            if not ann.is_file():
                continue
            with ann.open() as f:
                meta = json.load(f)

            for img, info in meta.items():
                ga = info.get("gaze_angles")
                if ga is None or len(ga) != 2:
                    continue
                yaw, pitch = float(ga[0]), float(ga[1])
                if abs(yaw) > YAW_MAX or abs(pitch) > PITCH_MAX:
                    continue

                paths = {
                    "face":  root/pid/"images/face"/img,
                    "eye_l": root/pid/"images/left_eye"/img,
                    "eye_r": root/pid/"images/right_eye"/img,
                }
                if self.use_frame:
                    paths["frame"] = root/pid/"images/frame"/img

                need_exist = ["face", "eye_l", "eye_r"] + (["frame"] if self.use_frame else [])
                if not all(paths[k].is_file() for k in need_exist):
                    continue

                self.samples.append({
                    "paths": paths,
                    "label": torch.tensor([yaw, pitch], dtype=torch.float32)
                })

        print(f"[GazeDatasetMulti] Loaded {len(self.samples)} clean samples "
              f"from {len(pid_list)} subjects (use_frame={self.use_frame})")

        # transforms
        self.face_tf = T.Compose([
            T.Resize((face_size, face_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        self.eye_tf = T.Compose([
            T.Resize((eye_size, eye_size)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225]),
        ])
        self.frame_tf = self.face_tf

        self.color_jit = T.ColorJitter(0.2, 0.2, 0.2, 0.05) if augment else None

    def __len__(self):
        return len(self.samples)

    def _same_color_jitter(self, imgs):
        if self.color_jit is None:
            return imgs
        fn_idx, b, c, s, h = T.ColorJitter.get_params(
            self.color_jit.brightness, self.color_jit.contrast,
            self.color_jit.saturation, self.color_jit.hue
        )
        def apply(im):
            for fn_id in fn_idx:
                if fn_id == 0 and b is not None: im = Fv.adjust_brightness(im, b)
                elif fn_id == 1 and c is not None: im = Fv.adjust_contrast(im, c)
                elif fn_id == 2 and s is not None: im = Fv.adjust_saturation(im, s)
                elif fn_id == 3 and h is not None: im = Fv.adjust_hue(im, h)
            return im
        return [apply(im) if im is not None else None for im in imgs]

    def __getitem__(self, idx):
        samp  = self.samples[idx]
        p = samp["paths"]

        face  = Image.open(p["face"]).convert("RGB")
        eye_l = Image.open(p["eye_l"]).convert("RGB")
        eye_r = Image.open(p["eye_r"]).convert("RGB")
        frame = Image.open(p["frame"]).convert("RGB") if self.use_frame else None

        # 同步颜色扰动
        face, eye_l, eye_r, frame = self._same_color_jitter([face, eye_l, eye_r, frame])

        label = samp["label"].clone()

        # 可选水平翻转（若启用，交换左右眼且 yaw 取负）
        if self.hflip_p > 0.0 and random.random() < self.hflip_p:
            face  = face.transpose(Image.FLIP_LEFT_RIGHT)
            eye_l, eye_r = eye_r, eye_l
            if frame is not None:
                frame = frame.transpose(Image.FLIP_LEFT_RIGHT)
            label[0] = -label[0]

        out = {
            "face" : self.face_tf(face),
            "eye_l": self.eye_tf(eye_l),
            "eye_r": self.eye_tf(eye_r),
            "label": label,
        }
        if self.use_frame:
            out["frame"] = self.frame_tf(frame)

        return out
