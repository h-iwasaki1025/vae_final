#!/usr/bin/env python3
"""
2ec6 の特定角度帯での系統的誤分類を検証するスクリプト（①〜⑤）。

★ 正しい参照（この図と一致）:
  モデル:  04_Savemodel/20260603_z_noise10
  結果:    07_result/20260607_z_noise10_20260603_z_noise10/classifier_results/
  図:      error_angle_dodge_mlp_2class_without_1qvi.png
  角度軸:  Z（x_angle / y_angle は常に 0。z_angle_test.npy / r_test.npy を使う）
  分母:    各角度 60 枚（2ec6@80° で 58/60 誤答が再現する）

Classifier 参照:
  - 06_Analysis/classifier.py … parse_noise_level / MLP 基本形
  - 06_Analysis/classifier_error_analysis.py … 角度パース・dodge 集計の考え方
  - 実際の dodge 図を出した評価ロジックは
    06_Analysis_refactored/old_scripts/classifier.py
    （threshold=0.5 固定、axis_idx=2=Z、prefix=mlp_2class_without_1qvi）

使い方:
  cd /Users/hikaru/Code/2026_Project/06_Analysis
  /Users/hikaru/Code/miniforge3/envs/tf_vae/bin/python verify_2ec6_angle_errors.py
  .../tf_vae/bin/python verify_2ec6_angle_errors.py --skip-images
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.optimizers import Adam

# 同ディレクトリの Classifier ユーティリティを優先利用
sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    from classifier import parse_noise_level  # 06_Analysis/classifier.py
except Exception:  # pragma: no cover
    def parse_noise_level(path):
        if isinstance(path, bytes):
            path = path.decode("utf-8")
        m = re.search(r"sn([0-9.]+)", str(path))
        if m:
            return m.group(1)
        m = re.search(r"sn(inf)", str(path), flags=re.I)
        return m.group(1) if m else "Unknown"


# ---------------------------------------------------------------------------
# Defaults — 先輩指定の 20260603_z_noise10
# ---------------------------------------------------------------------------
DEFAULT_MODEL_DIR = (
    "/Users/hikaru/Code/2026_Project/04_Savemodel/20260603_z_noise10"
)
DEFAULT_RESULT_DIR = (
    "/Users/hikaru/Code/2026_Project/07_result/"
    "20260607_z_noise10_20260603_z_noise10"
)
DEFAULT_CLF_KERAS = os.path.join(
    DEFAULT_RESULT_DIR, "classifier_results", "mlp_confidence_model.keras"
)

# このデータでは y_test と一致: 1l2o=0, 2ec6=1
CLASS_NAME = {0: "state0(1l2o側)", 1: "state1(2ec6側)"}
# pdb_PSI_noise のディレクトリ prefix と一致
#   state0: 0_1kk8, 0_1kqm, 0_1kwo, 0_1l2o
#   state1: 1_1kk7, 1_1s5g, 1_1sr6, 1_2ec6, 1_1qvi
TRAIN_STATE0 = ("1kk8", "1kqm", "1kwo")
TRAIN_STATE1 = ("1kk7", "1s5g", "1sr6")
DEFAULT_DATA_ROOT = "/Users/hikaru/Code/2026_Project/05_Data/pdb_PSI_noise"
DATA_ROOT_ACTIVE: Optional[str] = None


# ---------------------------------------------------------------------------
# Parsing helpers（classifier_error_analysis.py / old_scripts と同じ命名規則）
# Side PSI: 1_2ec6_000_000_080_sn0.6_rep08.tif → X,Y,Z = 0,0,80
# ---------------------------------------------------------------------------
def parse_angles_xyz(path: str) -> Tuple[int, int, int]:
    """06_Analysis/classifier_error_analysis.py と同じファイル名パース。"""
    fname = os.path.basename(str(path))
    try:
        parts = fname.split(".")[0].split("_")
        if parts[0] in ("0", "1", "2"):
            return int(parts[2]), int(parts[3]), int(parts[4])
        return int(parts[1]), int(parts[2]), int(parts[3])
    except Exception:
        return 0, 0, 0


def extract_noise_level_sn(path: str) -> str:
    """old_scripts/classifier.py の extract_noise_level 互換。"""
    s = str(path)
    m = re.search(r"SN([\d.]+)", s)
    if m:
        return f"SN{m.group(1)}"
    m = re.search(r"sn([\d.]+)", s)
    if m:
        return f"SN{m.group(1)}"
    return "Unknown"


def resolve_path(path: str, data_root: Optional[str] = None) -> str:
    """
    kills 内パスを解決。/Volumes/... が無い場合は /Users/... や data_root にフォールバック。
    """
    p = str(path).split("#")[0]
    candidates = [p]
    if p.startswith("/Volumes/hikaru/Code/"):
        candidates.append(p.replace("/Volumes/hikaru/Code/", "/Users/hikaru/Code/", 1))
    if p.startswith("/Users/hikaru/Code/"):
        candidates.append(p.replace("/Users/hikaru/Code/", "/Volumes/hikaru/Code/", 1))
    if data_root:
        # .../pdb_PSI_noise/<rest>
        marker = "pdb_PSI_noise/"
        if marker in p:
            candidates.append(os.path.join(data_root, p.split(marker, 1)[1]))
    for c in candidates:
        if os.path.exists(c):
            return c
    return p


def load_projection(path: str, data_root: Optional[str] = None) -> np.ndarray:
    """単枚 .tif / .tiff を読む（このデータセットは #frame 無し）。"""
    if data_root is None:
        data_root = DATA_ROOT_ACTIVE
    p = resolve_path(path, data_root=data_root)
    if not os.path.exists(p):
        raise FileNotFoundError(p)
    try:
        import tifffile

        arr = tifffile.imread(p)
    except Exception:
        from PIL import Image

        arr = np.array(Image.open(p))
    if arr.ndim == 3:
        # まれに (C,H,W) or multi-page → 先頭
        arr = arr[0]
    return arr.astype(np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return float("nan")
    return float(np.dot(a, b) / (na * nb))


def mse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


# ---------------------------------------------------------------------------
# Data / model
# ---------------------------------------------------------------------------
def build_mlp(input_dim: int):
    """old_scripts/classifier.py::build_mlp と同じ構造（load 失敗時用）。"""
    model = Sequential(
        [
            Dense(128, activation="relu", input_shape=(input_dim,)),
            Dropout(0.3),
            Dense(64, activation="relu"),
            Dropout(0.3),
            Dense(32, activation="relu"),
            Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_mlp(model_path: str, input_dim: int = 10):
    try:
        return load_model(model_path)
    except Exception as e:
        print(f"load_model failed ({type(e).__name__}: {e})")
        print("Rebuilding architecture (old_scripts/classifier.build_mlp) and loading weights...")
        model = build_mlp(input_dim)
        model.load_weights(model_path)
        return model


def load_arrays(model_dir: str) -> Dict[str, np.ndarray]:
    def _np(name, allow_pickle=False):
        p = os.path.join(model_dir, name)
        if not os.path.exists(p):
            raise FileNotFoundError(p)
        return np.load(p, allow_pickle=allow_pickle)

    z_test = _np("z_mean_test.npy")
    pdbids_test = _np("pdbids_test.npy", True).astype(str)
    kills_test = _np("kills_test.npy", True).astype(str)
    y_test = _np("y_test.npy").astype(int)

    # ★ Z回転データ: z_angle_test / r_test を優先（x_angle は全部 0）
    if os.path.exists(os.path.join(model_dir, "z_angle_test.npy")):
        angles = _np("z_angle_test.npy").astype(int)
        angle_source = "z_angle_test.npy"
    elif os.path.exists(os.path.join(model_dir, "r_test.npy")):
        angles = _np("r_test.npy").astype(int)
        angle_source = "r_test.npy"
    else:
        angles = np.array([parse_angles_xyz(p)[2] for p in kills_test], dtype=int)
        angle_source = "filename_Z"

    # sanity: ファイル名の Z と一致するか
    ang_from_name = np.array([parse_angles_xyz(p)[2] for p in kills_test], dtype=int)
    n_mismatch = int(np.sum(ang_from_name != angles))
    print(f"angle source: {angle_source} | filename-Z mismatches: {n_mismatch}/{len(angles)}")

    noise = np.array([parse_noise_level(p) for p in kills_test])
    out = {
        "z_test": z_test,
        "pdbids_test": pdbids_test,
        "kills_test": kills_test,
        "y_test": y_test,
        "angles": angles,
        "noise": noise,
    }

    for key, fname, pickle in [
        ("z_train", "z_mean_train.npy", False),
        ("pdbids_train", "pdbids_train.npy", True),
        ("kills_train", "kills_train.npy", True),
        ("y_train", "y_train.npy", False),
        ("z_angle_train", "z_angle_train.npy", False),
        ("r_train", "r_train.npy", False),
    ]:
        p = os.path.join(model_dir, fname)
        if os.path.exists(p):
            arr = np.load(p, allow_pickle=pickle)
            if pickle:
                arr = np.array([str(x) for x in arr])
            out[key] = arr

    if "z_angle_train" in out:
        out["angles_train"] = out["z_angle_train"].astype(int)
    elif "r_train" in out:
        out["angles_train"] = out["r_train"].astype(int)
    elif "kills_train" in out:
        out["angles_train"] = np.array(
            [parse_angles_xyz(p)[2] for p in out["kills_train"]], dtype=int
        )
    return out


def build_pred_df(data: Dict[str, np.ndarray], proba: np.ndarray, threshold: float) -> pd.DataFrame:
    # old_scripts は `> 0.5`（厳密には >= ではない）。再現のため同じにする。
    y_true = data["y_test"]
    if threshold == 0.5:
        y_pred = (proba > 0.5).astype(int)
    else:
        y_pred = (proba >= threshold).astype(int)
    return pd.DataFrame(
        {
            "path": data["kills_test"],
            "pdb_id": data["pdbids_test"],
            "angle": data["angles"],
            "noise": data["noise"],
            "y_true": y_true,
            "y_pred": y_pred,
            "proba_state1": proba,
            "correct": y_true == y_pred,
            "true_name": [CLASS_NAME.get(int(v), str(v)) for v in y_true],
            "pred_name": [CLASS_NAME.get(int(v), str(v)) for v in y_pred],
        }
    )


# ---------------------------------------------------------------------------
# ①〜⑤
# ---------------------------------------------------------------------------
def step1_true_vs_pred(df, out_dir: Path, focus_pdb: str, focus_angle: int, angle_tol: int = 0):
    print("\n=== ① True vs Pred ===")
    sub = df[
        (df["pdb_id"] == focus_pdb)
        & (df["angle"] >= focus_angle - angle_tol)
        & (df["angle"] <= focus_angle + angle_tol)
    ].copy()
    n = len(sub)
    n_err = int((~sub["correct"]).sum())
    print(f"{focus_pdb} @ {focus_angle}±{angle_tol}°: N={n}, errors={n_err} ({n_err/max(n,1):.1%})")
    print("y_true:", Counter(sub["y_true"].tolist()))
    print("y_pred:", Counter(sub["y_pred"].tolist()))
    print("confusion (rows=true, cols=pred):\n", confusion_matrix(sub["y_true"], sub["y_pred"], labels=[0, 1]))

    if n and sub["y_true"].nunique() == 1:
        true_c = int(sub["y_true"].iloc[0])
        mean_p = float(sub["proba_state1"].mean())
        # 真が1なら P が低いほど逆、真が0なら P が高いほど逆
        if true_c == 1:
            frac_flip = float((sub["y_pred"] == 0).mean())
            high_conf_flip = float((sub["proba_state1"] <= 0.2).mean())
        else:
            frac_flip = float((sub["y_pred"] == 1).mean())
            high_conf_flip = float((sub["proba_state1"] >= 0.8).mean())
        print(
            f"one-way: true={true_c} ({CLASS_NAME[true_c]}), "
            f"flip_frac={frac_flip:.1%}, meanP(state1)={mean_p:.3f}, "
            f"high-conf-flip={high_conf_flip:.1%}"
        )
        if frac_flip >= 0.9:
            if high_conf_flip >= 0.5:
                print("→ 高confidenceで一方向に逆判定（見え方の逆転 or ラベル系統ミスを強く疑う）")
            else:
                print("→ 一方向誤分類。境界付近〜中程度の確信度で寄せている可能性")

    sub.sort_values(["noise"]).to_csv(
        out_dir / f"step1_{focus_pdb}_angle{focus_angle:03d}_predictions.csv", index=False
    )

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(sub["proba_state1"], bins=20, color="salmon", edgecolor="black")
    ax.axvline(0.5, color="gray", ls="--", label="thr=0.5")
    ax.set_title(f"{focus_pdb} @ {focus_angle}° — P(state1={CLASS_NAME[1]})")
    ax.set_xlabel("Predicted probability of state1")
    ax.set_ylabel("Count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"step1_{focus_pdb}_angle{focus_angle:03d}_proba_hist.png", dpi=200)
    plt.close(fig)
    print(f"saved step1_* under {out_dir}")
    return sub


def _montage(paths, titles, out_path: Path, ncols: int = 10):
    n = len(paths)
    if n == 0:
        print(f"skip empty montage: {out_path.name}")
        return
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 1.1, nrows * 1.2))
    axes = np.atleast_1d(axes).ravel()
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i >= n:
            continue
        try:
            ax.imshow(load_projection(paths[i]), cmap="gray")
            ax.set_title(titles[i], fontsize=6)
        except Exception as e:
            ax.set_title(f"ERR:{e}", fontsize=5)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"saved: {out_path}")


def step2_image_gallery(sub: pd.DataFrame, out_dir: Path, focus_pdb: str, focus_angle: int, max_images: int = 60):
    print("\n=== ② Image gallery ===")
    rows = sub.sort_values(["noise"]).head(max_images)
    titles = [
        f"{r.pdb_id} a{int(r.angle)}\nsn{r.noise}\nT{int(r.y_true)}→P{int(r.y_pred)} p={r.proba_state1:.2f}"
        for r in rows.itertuples()
    ]
    _montage(
        rows["path"].tolist(),
        titles,
        out_dir / f"step2_{focus_pdb}_angle{focus_angle:03d}_gallery.png",
    )


def _pick_same_angle(kills, pdbids, angles, pdb, angle, n=8, noise_prefer="0.6"):
    mask = (pdbids == pdb) & (angles == angle)
    if not np.any(mask):
        return []
    cand = kills[mask]
    noises = np.array([parse_noise_level(p) for p in cand])
    prefer = cand[noises == str(noise_prefer)]
    pool = prefer if len(prefer) else cand
    idx = np.linspace(0, len(pool) - 1, num=min(n, len(pool)), dtype=int)
    return [str(pool[i]) for i in idx]


def step3_compare_same_angle(data, out_dir: Path, focus_pdb: str, focus_angle: int,
                             train_pdbs=TRAIN_STATE0[:2] + TRAIN_STATE1[:2]):
    print("\n=== ③ Same-angle comparison ===")
    panels: List[Tuple[str, List[str]]] = []
    panels.append(
        (
            focus_pdb,
            _pick_same_angle(
                data["kills_test"], data["pdbids_test"], data["angles"], focus_pdb, focus_angle
            ),
        )
    )
    counterpart = "1l2o" if focus_pdb == "2ec6" else "2ec6"
    panels.append(
        (
            counterpart,
            _pick_same_angle(
                data["kills_test"], data["pdbids_test"], data["angles"], counterpart, focus_angle
            ),
        )
    )
    if "kills_train" in data:
        for tp in train_pdbs:
            panels.append(
                (
                    f"train:{tp}",
                    _pick_same_angle(
                        data["kills_train"],
                        data["pdbids_train"],
                        data["angles_train"],
                        tp,
                        focus_angle,
                    ),
                )
            )

    max_n = min(6, max((len(p) for _, p in panels), default=0))
    if max_n == 0:
        print("no images")
        return
    fig, axes = plt.subplots(len(panels), max_n, figsize=(max_n * 1.4, len(panels) * 1.4))
    axes = np.atleast_2d(axes)
    for r, (label, paths) in enumerate(panels):
        for c in range(max_n):
            ax = axes[r, c]
            ax.axis("off")
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, labelpad=45, va="center")
            if c >= len(paths):
                continue
            try:
                ax.imshow(load_projection(paths[c]), cmap="gray")
                ax.set_title(f"sn{parse_noise_level(paths[c])}", fontsize=7)
            except Exception as e:
                ax.set_title(str(e)[:30], fontsize=6)
    fig.suptitle(f"Same-angle projections @ Z={focus_angle}°")
    fig.tight_layout()
    out = out_dir / f"step3_compare_angle{focus_angle:03d}.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved: {out}")


def step4_proba_vs_angle(df, out_dir: Path, threshold: float, focus_pdbs=("2ec6", "1l2o")):
    print("\n=== ④ Proba vs angle (Z) ===")
    sub = df[df["pdb_id"].isin(focus_pdbs)].copy()
    g = (
        sub.groupby(["pdb_id", "angle"])["proba_state1"]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    g.to_csv(out_dir / "step4_proba_by_angle.csv", index=False)

    fig, ax = plt.subplots(figsize=(12, 4))
    for pdb, color in zip(focus_pdbs, ["#e07a5f", "#81b29a"]):
        gg = g[g["pdb_id"] == pdb].sort_values("angle")
        ax.plot(gg["angle"], gg["mean"], "-o", ms=3, color=color, label=pdb)
        ax.fill_between(
            gg["angle"],
            gg["mean"] - gg["std"].fillna(0),
            gg["mean"] + gg["std"].fillna(0),
            color=color,
            alpha=0.15,
        )
    ax.axhline(threshold, color="red", ls="--", label=f"threshold={threshold}")
    ax.set_xlabel("Angle Z (deg)")
    ax.set_ylabel("Mean P(state1=2ec6)")
    ax.set_title("Predicted probability vs Angle Z")
    ax.set_xlim(0, 350)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "step4_proba_vs_angle.png", dpi=200)
    plt.close(fig)

    err = (
        sub.assign(err=(~sub["correct"]).astype(int))
        .groupby(["pdb_id", "angle"])["err"]
        .mean()
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(12, 4))
    for pdb, color in zip(focus_pdbs, ["#e07a5f", "#81b29a"]):
        gg = err[err["pdb_id"] == pdb].sort_values("angle")
        ax.plot(gg["angle"], gg["err"], "-o", ms=3, color=color, label=pdb)
    ax.set_xlabel("Angle Z (deg)")
    ax.set_ylabel("Error rate")
    ax.set_ylim(0, 1)
    ax.set_title("Error rate vs Angle Z")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_dir / "step4_error_rate_vs_angle.png", dpi=200)
    plt.close(fig)

    for pdb in focus_pdbs:
        s = sub[sub["pdb_id"] == pdb]
        if s.empty:
            continue
        true_mode = int(s["y_true"].mode().iloc[0])
        if true_mode == 1:
            hc = s[(~s["correct"]) & (s["proba_state1"] <= 0.2)]
        else:
            hc = s[(~s["correct"]) & (s["proba_state1"] >= 0.8)]
        print(
            f"{pdb}: N={len(s)}, err={int((~s['correct']).sum())} "
            f"({(~s['correct']).mean():.1%}), high-conf-wrong={len(hc)} "
            f"({len(hc)/len(s):.1%}), meanP={s['proba_state1'].mean():.3f}"
        )


def _mean_image_band(kills, pdbids, angles, pdb, a0, a1, max_n=40):
    mask = (pdbids == pdb) & (angles >= a0) & (angles <= a1)
    if not np.any(mask):
        return None
    pool = kills[mask]
    if len(pool) > max_n:
        pool = pool[np.linspace(0, len(pool) - 1, max_n, dtype=int)]
    imgs = []
    for p in pool:
        try:
            imgs.append(load_projection(str(p)))
        except Exception:
            pass
    return np.mean(np.stack(imgs), 0) if imgs else None


def _mean_z_band(z, pdbids, angles, pdb, a0, a1):
    mask = (pdbids == pdb) & (angles >= a0) & (angles <= a1)
    return z[mask].mean(0) if np.any(mask) else None


def step5_band_similarity(data, out_dir: Path, focus_pdb="2ec6",
                          band_a=(70, 90), band_b=(250, 280)):
    print("\n=== ⑤ Band similarity (70–90 vs 250–280, +180°) ===")
    rows = []
    img_a = _mean_image_band(data["kills_test"], data["pdbids_test"], data["angles"], focus_pdb, *band_a)
    img_b = _mean_image_band(data["kills_test"], data["pdbids_test"], data["angles"], focus_pdb, *band_b)
    if img_a is not None and img_b is not None:
        rows += [
            {"metric": "image_cosine", "pair": f"{focus_pdb}{band_a}_vs_{band_b}", "value": cosine_sim(img_a, img_b)},
            {"metric": "image_mse", "pair": f"{focus_pdb}{band_a}_vs_{band_b}", "value": mse(img_a, img_b)},
        ]
        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(img_a, cmap="gray"); axes[0].set_title(f"{focus_pdb} {band_a[0]}-{band_a[1]}")
        axes[1].imshow(img_b, cmap="gray"); axes[1].set_title(f"{focus_pdb} {band_b[0]}-{band_b[1]}")
        axes[2].imshow(np.abs(img_a - img_b), cmap="magma"); axes[2].set_title("|diff|")
        for ax in axes:
            ax.axis("off")
        fig.tight_layout()
        fig.savefig(out_dir / "step5_mean_image_bands.png", dpi=150)
        plt.close(fig)

    z_a = _mean_z_band(data["z_test"], data["pdbids_test"], data["angles"], focus_pdb, *band_a)
    z_b = _mean_z_band(data["z_test"], data["pdbids_test"], data["angles"], focus_pdb, *band_b)
    if z_a is not None and z_b is not None:
        rows.append({"metric": "latent_cosine", "pair": f"{focus_pdb}{band_a}_vs_{band_b}", "value": cosine_sim(z_a, z_b)})

    # +180 pairs
    sims_img, sims_z = [], []
    for a in sorted({int(x) for x in data["angles"] if int(x) < 180}):
        b = (a + 180) % 360
        ia = _mean_image_band(data["kills_test"], data["pdbids_test"], data["angles"], focus_pdb, a, a, max_n=20)
        ib = _mean_image_band(data["kills_test"], data["pdbids_test"], data["angles"], focus_pdb, b, b, max_n=20)
        if ia is not None and ib is not None:
            sims_img.append(cosine_sim(ia, ib))
        za = _mean_z_band(data["z_test"], data["pdbids_test"], data["angles"], focus_pdb, a, a)
        zb = _mean_z_band(data["z_test"], data["pdbids_test"], data["angles"], focus_pdb, b, b)
        if za is not None and zb is not None:
            sims_z.append(cosine_sim(za, zb))
    if sims_img:
        rows.append({"metric": "image_cosine_mean_+180", "pair": focus_pdb, "value": float(np.nanmean(sims_img))})
    if sims_z:
        rows.append({"metric": "latent_cosine_mean_+180", "pair": focus_pdb, "value": float(np.nanmean(sims_z))})

    counterpart = "1l2o" if focus_pdb == "2ec6" else "2ec6"
    img_c = _mean_image_band(data["kills_test"], data["pdbids_test"], data["angles"], counterpart, *band_a)
    if img_a is not None and img_c is not None:
        rows.append({"metric": "image_cosine", "pair": f"{focus_pdb}{band_a}_vs_{counterpart}{band_a}", "value": cosine_sim(img_a, img_c)})
    z_c = _mean_z_band(data["z_test"], data["pdbids_test"], data["angles"], counterpart, *band_a)
    if z_a is not None and z_c is not None:
        rows.append({"metric": "latent_cosine", "pair": f"{focus_pdb}{band_a}_vs_{counterpart}{band_a}", "value": cosine_sim(z_a, z_c)})

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "step5_similarity_summary.csv", index=False)
    print(df.to_string(index=False))
    return df


def step0_meta_check(sub: pd.DataFrame, out_dir: Path, data_root: str):
    print("\n=== ⓪ Metadata integrity (focus subset) ===")
    rows = []
    n_missing = 0
    n_ang_mismatch = 0
    n_prefix_mismatch = 0
    for r in sub.itertuples():
        resolved = resolve_path(r.path, data_root=data_root)
        exists = os.path.exists(resolved)
        if not exists:
            n_missing += 1
        ax, ay, az = parse_angles_xyz(r.path)
        if int(az) != int(r.angle):
            n_ang_mismatch += 1
        # prefix state from dirname: 0_xxx / 1_xxx
        m = re.search(r"/([01])_([0-9a-z]+)_", str(r.path))
        prefix_state = int(m.group(1)) if m else None
        prefix_pdb = m.group(2) if m else None
        if prefix_pdb != r.pdb_id or (prefix_state is not None and prefix_state != int(r.y_true)):
            n_prefix_mismatch += 1
        rows.append(
            {
                "path": r.path,
                "resolved": resolved,
                "exists": exists,
                "angle_meta": int(r.angle),
                "angle_name_Z": az,
                "pdb_id": r.pdb_id,
                "y_true": int(r.y_true),
                "prefix_state": prefix_state,
                "prefix_pdb": prefix_pdb,
            }
        )
    meta = pd.DataFrame(rows)
    meta.to_csv(out_dir / "step0_meta_check.csv", index=False)
    print(f"files missing: {n_missing}/{len(sub)}")
    print(f"angle filename mismatch: {n_ang_mismatch}/{len(sub)}")
    print(f"prefix/label mismatch: {n_prefix_mismatch}/{len(sub)}")
    if n_missing == 0 and n_ang_mismatch == 0 and n_prefix_mismatch == 0:
        print("→ メタデータは整合。データ破損より見え方／表現側が本命寄り")
    return meta


def step6_latent_knn(
    data: Dict[str, np.ndarray],
    sub: pd.DataFrame,
    out_dir: Path,
    k: int = 10,
):
    """
    focus サンプル（例: 2ec6@80）の潜在最近傍が、train のどの state/PDB か。
    """
    print(f"\n=== ⑥ Latent kNN (k={k}, against TRAIN) ===")
    if "z_train" not in data:
        print("z_train missing; skip knn")
        return None

    from sklearn.neighbors import NearestNeighbors
    from sklearn.preprocessing import StandardScaler

    z_tr = data["z_train"]
    y_tr = data["y_train"].astype(int)
    pdb_tr = data["pdbids_train"]
    ang_tr = data["angles_train"]

    scaler = StandardScaler()
    z_tr_s = scaler.fit_transform(z_tr)
    nn = NearestNeighbors(n_neighbors=k, metric="euclidean")
    nn.fit(z_tr_s)

    # focus indices via path matching
    path_to_i = {p: i for i, p in enumerate(data["kills_test"])}
    focus_idx = [path_to_i[p] for p in sub["path"].tolist() if p in path_to_i]
    if not focus_idx:
        print("no focus indices")
        return None

    z_f = scaler.transform(data["z_test"][focus_idx])
    dists, inds = nn.kneighbors(z_f)

    rows = []
    state0_votes = 0
    state1_votes = 0
    for local_i, gi in enumerate(focus_idx):
        neigh = inds[local_i]
        y_n = y_tr[neigh]
        pdb_n = pdb_tr[neigh]
        ang_n = ang_tr[neigh]
        frac0 = float(np.mean(y_n == 0))
        frac1 = float(np.mean(y_n == 1))
        if frac0 >= frac1:
            state0_votes += 1
        else:
            state1_votes += 1
        rows.append(
            {
                "query_path": data["kills_test"][gi],
                "query_pdb": data["pdbids_test"][gi],
                "query_angle": int(data["angles"][gi]),
                "query_y": int(data["y_test"][gi]),
                "neighbor_frac_state0": frac0,
                "neighbor_frac_state1": frac1,
                "neighbor_pdb_mode": Counter(pdb_n.tolist()).most_common(1)[0][0],
                "neighbor_angle_mean": float(np.mean(ang_n)),
                "mean_dist": float(np.mean(dists[local_i])),
            }
        )

    knn_df = pd.DataFrame(rows)
    knn_df.to_csv(out_dir / "step6_latent_knn_focus.csv", index=False)
    n = len(knn_df)
    print(
        f"queries={n}: majority neighbors state0={state0_votes} ({state0_votes/n:.1%}), "
        f"state1={state1_votes} ({state1_votes/n:.1%})"
    )
    print("neighbor_pdb_mode top:", Counter(knn_df["neighbor_pdb_mode"]).most_common(6))
    print(
        "mean neighbor_frac_state0={:.3f}, state1={:.3f}".format(
            knn_df["neighbor_frac_state0"].mean(), knn_df["neighbor_frac_state1"].mean()
        )
    )
    true_mode = int(Counter(knn_df["query_y"]).most_common(1)[0][0])
    if true_mode == 1 and state0_votes / n >= 0.7:
        print("→ 潜在最近傍は主に state0。見え方／表現が逆クラス寄りを強く支持")
    elif true_mode == 0 and state1_votes / n >= 0.7:
        print("→ 潜在最近傍は主に state1。見え方／表現が逆クラス寄りを強く支持")
    else:
        print("→ 最近傍が混在。境界付近か別要因の可能性")
    return knn_df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    p.add_argument("--clf-keras", default=DEFAULT_CLF_KERAS)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data-root", default=DEFAULT_DATA_ROOT,
                   help="学習元画像ルート（既定: .../05_Data/pdb_PSI_noise）")
    p.add_argument("--focus-pdb", default="2ec6")
    p.add_argument("--angle", type=int, default=80)
    p.add_argument("--angle-tol", type=int, default=0)
    p.add_argument("--threshold", type=float, default=0.5,
                   help="元の dodge 図は 0.5 固定（old_scripts/classifier.py）")
    p.add_argument("--skip-images", action="store_true")
    p.add_argument("--max-gallery", type=int, default=60)
    p.add_argument("--knn-k", type=int, default=10)
    return p.parse_args()


def main():
    global DATA_ROOT_ACTIVE
    args = parse_args()
    DATA_ROOT_ACTIVE = args.data_root
    out_dir = Path(args.out_dir) if args.out_dir else Path(DEFAULT_RESULT_DIR) / "angle_error_audit"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"model_dir: {args.model_dir}")
    print(f"clf:       {args.clf_keras}")
    print(f"data_root: {args.data_root} (exists={os.path.isdir(args.data_root)})")
    print(f"out_dir:   {out_dir}")
    print("NOTE: このランは Z 回転（pdb_PSI_noise）。角度は z_angle_test を使用。")

    data = load_arrays(args.model_dir)
    print(
        f"test N={len(data['y_test'])}, pdbs={Counter(data['pdbids_test'])}, "
        f"unique Z angles={len(set(map(int, data['angles'])))}"
    )
    for pdb in ("2ec6", "1l2o"):
        ys = data["y_test"][data["pdbids_test"] == pdb]
        if len(ys):
            mode = int(Counter(ys.tolist()).most_common(1)[0][0])
            print(f"label: {pdb} -> y={mode} ({CLASS_NAME.get(mode)})")

    print("predicting...")
    proba = load_mlp(args.clf_keras, data["z_test"].shape[1]).predict(
        data["z_test"], batch_size=256, verbose=0
    ).flatten()

    mask = data["pdbids_test"] != "1qvi"
    if args.threshold == 0.5:
        pred = (proba[mask] > 0.5).astype(int)
    else:
        pred = (proba[mask] >= args.threshold).astype(int)
    acc = accuracy_score(data["y_test"][mask], pred)
    print(f"2-class acc @ thr={args.threshold}: {acc:.4f}")

    df = build_pred_df(data, proba, args.threshold)
    df.to_csv(out_dir / "all_test_predictions.csv", index=False)

    focus_n = int(((df.pdb_id == args.focus_pdb) & (df.angle == args.angle)).sum())
    print(f"NOTE: {args.focus_pdb}@{args.angle}° N={focus_n} （図の 55/60 はこの分母）")

    sub = step1_true_vs_pred(df, out_dir, args.focus_pdb, args.angle, args.angle_tol)
    step0_meta_check(sub, out_dir, args.data_root)
    step4_proba_vs_angle(df, out_dir, args.threshold)
    step6_latent_knn(data, sub, out_dir, k=args.knn_k)

    if not args.skip_images:
        step2_image_gallery(sub, out_dir, args.focus_pdb, args.angle, args.max_gallery)
        step3_compare_same_angle(data, out_dir, args.focus_pdb, args.angle)
        step5_band_similarity(data, out_dir, focus_pdb=args.focus_pdb)
    else:
        print("skip image steps")

    with open(out_dir / "README_NEXT.txt", "w") as f:
        f.write(
            "Refs\n"
            f"- model: {args.model_dir}\n"
            f"- clf:   {args.clf_keras}\n"
            f"- data:  {args.data_root}\n"
            "- classifier code: 06_Analysis/classifier.py , classifier_error_analysis.py\n"
            "- dodge plot logic: 06_Analysis_refactored/old_scripts/classifier.py\n"
            "\nOutputs\n"
            "- step0_meta_check.csv\n"
            "- step1_* predictions / proba hist\n"
            "- step2_* gallery\n"
            "- step3_* same-angle compare\n"
            "- step4_* proba/error vs angle\n"
            "- step5_* band similarity\n"
            "- step6_* latent kNN vs train\n"
        )
    print(f"\nDone. See {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
