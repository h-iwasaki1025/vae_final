import os
import numpy as np
import tensorflow as tf
from PIL import Image
import logging

logger = logging.getLogger(__name__)


class OnTheFlyPDBDataLoader:
    def __init__(self, data_path, sn_levels, allowed_train_pdbs, allowed_test_pdbs, pixel_size=128, parser_config=None, angle_filter=None):
        self.data_path = data_path
        self.sn_levels = sn_levels
        self.allowed_train_pdbs = set(allowed_train_pdbs)
        self.allowed_test_pdbs = set(allowed_test_pdbs)
        self.pixel_size = pixel_size
        
        # 📄 デフォルトのパース規則（従来の規則）
        self.parser_config = parser_config or {
            "delimiter": "_",
            "xxx_idx": 1,
            "yyy_idx": 2,
            "zzz_idx": 3,
            "state_prefix_offset": 1
        }
        
        # 📐 デフォルトの角度フィルター（全通し）
        self.angle_filter = angle_filter or {
            "xxx": None,
            "yyy": None,
            "zzz": None
        }

    def parse_angles(self, fname):
        """
        config で指定された delimiter およびインデックス規則に基づいて、
        ファイル名から xxx, yyy, zzz の各角度をパースします。
        """
        # 拡張子を除去して指定の区切り文字で分割
        base_name = fname.split(".")[0]
        delim = self.parser_config.get("delimiter", "_")
        parts = base_name.split(delim)

        # ファイル名の先頭が State (0_, 1_, 2_) で始まっている場合のオフセット処理
        offset = 0
        if parts[0] in ("0", "1", "2"):
            offset = self.parser_config.get("state_prefix_offset", 1)

        xxx_idx = self.parser_config["xxx_idx"] + offset
        yyy_idx = self.parser_config["yyy_idx"] + offset
        zzz_idx = self.parser_config["zzz_idx"] + offset

        xxx = int(parts[xxx_idx])
        yyy = int(parts[yyy_idx])
        zzz = int(parts[zzz_idx])

        return xxx, yyy, zzz

    def scan_dataset(self):
        """
        ファイル名から情報をパースし、パスとメタデータのインデックスリストのみをメモリ上に作成します。
        """
        train_paths, train_y, train_angles, train_pdbs = [], [], [], []
        test_paths, test_y, test_angles, test_pdbs = [], [], [], []

        if not os.path.exists(self.data_path):
            logger.error(f"Data path not found: {self.data_path}")
            return None, None

        logger.info(f"Scanning directory: {self.data_path}...")

        # フォルダ構造のスキャン
        m_dirs = sorted(os.listdir(self.data_path))
        for m in m_dirs:
            m_path = os.path.join(self.data_path, m)
            if not os.path.isdir(m_path):
                continue
            
            # フォルダ名のパース
            state_p = m.split("_")
            if len(state_p) < 2:
                continue
            try:
                state = int(state_p[0])
            except ValueError:
                continue
            pdbid = state_p[1]

            noise_root = os.path.join(m_path, "noise")
            if not os.path.isdir(noise_root):
                continue

            for sn in self.sn_levels:
                sn_dir = os.path.join(noise_root, f"sn{sn}")
                if not os.path.isdir(sn_dir):
                    continue

                try:
                    for entry in os.scandir(sn_dir):
                        if entry.is_file() and entry.name.endswith(".tif"):
                            fname = entry.name
                            f_path = entry.path

                            # 💡 config のパーサー設定に基づき角度を動的パース
                            try:
                                xxx, yyy, zzz = self.parse_angles(fname)
                            except Exception:
                                continue

                            # 💡 config のアングルフィルター設定に基づき条件判定
                            # xxx フィルタ
                            if self.angle_filter.get("xxx") is not None:
                                if xxx != self.angle_filter["xxx"]:
                                    continue
                            # yyy フィルタ
                            if self.angle_filter.get("yyy") is not None:
                                if yyy != self.angle_filter["yyy"]:
                                    continue
                            # zzz フィルタ
                            if self.angle_filter.get("zzz") is not None:
                                if zzz != self.angle_filter["zzz"]:
                                    continue

                            # 割り振りの判定
                            if pdbid in self.allowed_test_pdbs:
                                test_paths.append(f_path)
                                test_y.append(state)
                                test_angles.append([xxx, yyy, zzz])
                                test_pdbs.append(pdbid)
                            elif pdbid in self.allowed_train_pdbs:
                                train_paths.append(f_path)
                                train_y.append(state)
                                train_angles.append([xxx, yyy, zzz])
                                train_pdbs.append(pdbid)
                except Exception as e:
                    logger.warning(f"Error scanning directory {sn_dir}: {e}")

        logger.info(f"Scan finished. Train: {len(train_paths)} paths, Test: {len(test_paths)} paths.")

        return (
            {"paths": train_paths, "y": np.array(train_y), "angles": np.array(train_angles), "pdbids": train_pdbs},
            {"paths": test_paths, "y": np.array(test_y), "angles": np.array(test_angles), "pdbids": test_pdbs}
        )


def tf_py_load_tif(path_tensor, pixel_size=128):
    path_str = path_tensor.numpy().decode("utf-8")
    try:
        with Image.open(path_str) as img:
            arr = np.array(img).astype(np.float32) / 255.0
            if len(arr.shape) == 2:
                arr = np.expand_dims(arr, axis=-1)
            # 必要な解像度にリサイズ
            if arr.shape[0] != pixel_size or arr.shape[1] != pixel_size:
                img_resized = img.resize((pixel_size, pixel_size), Image.Resampling.BILINEAR)
                arr = np.array(img_resized).astype(np.float32) / 255.0
                if len(arr.shape) == 2:
                    arr = np.expand_dims(arr, axis=-1)
            return arr
    except Exception:
        return np.zeros((pixel_size, pixel_size, 1), dtype=np.float32)


def load_and_preprocess_image(path_tensor, pixel_size=128):
    image = tf.py_function(lambda p: tf_py_load_tif(p, pixel_size), [path_tensor], tf.float32)
    image.set_shape([pixel_size, pixel_size, 1])
    return image
