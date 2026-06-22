"""
Parkinson Screening Backend — FastAPI
=====================================
วิธีรัน:
    pip install fastapi uvicorn xgboost numpy scipy
    python main.py

Endpoints:
    POST /predict/keyboard    — Mode 1 (Keyboard only)
    POST /predict/mobile      — Mode 2 (Mobile only)
    POST /predict/combined    — Mode 3 (Both devices)
    GET  /health              — Health check
"""

from __future__ import annotations
import json, math, statistics
from pathlib import Path
from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
import xgboost as xgb
from scipy import stats as scipy_stats
from pydantic import BaseModel
from typing import Optional, List


# ─────────────────────────────────────────────
# Paths  (วางไฟล์ JSON ไว้ใน models/ ข้างๆ main.py)
# ─────────────────────────────────────────────
BASE = Path(__file__).parent
KB_DIR   = BASE / "models" / "keyboard"
MOB_DIR  = BASE / "models" / "mobile"

# ─────────────────────────────────────────────
# App
# ─────────────────────────────────────────────
app = FastAPI(
    title="Parkinson Screening API",
    version="1.0.0",
    description="ระบบคัดกรองความเสี่ยงโรคพาร์กินสันจาก Keystroke / Tap data"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # ปรับเป็น domain จริงก่อน deploy production
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load assets at startup
# ─────────────────────────────────────────────
def load_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


class KeyboardAssets:
    """โหลดโมเดลและ params ทั้งหมดของ Keyboard"""
    def __init__(self):
        sp      = load_json(KB_DIR / "scaler_params_v5.json")
        bp      = load_json(KB_DIR / "best_params_v5.json")
        res     = load_json(KB_DIR / "results_v5.json")
        anchor  = load_json(KB_DIR / "calibration_anchor.json")
        ratio   = load_json(KB_DIR / "calibration_ratio.json")
        refstat = load_json(KB_DIR / "reference_stats.json")

        self.feature_names: list[str] = sp["feature_names"]
        self.scaler_mean   = np.array(sp["mean"])
        self.scaler_scale  = np.array(sp["scale"])
        self.threshold: float = res.get("threshold", 0.5)

        # calibration
        self.anchor = anchor   # ZScoreAnchor
        self.ratio  = ratio    # RatioNormalizer
        self.ref    = refstat  # population stats

        # XGBoost model
        self.model = xgb.Booster()
        # model ถูก save ด้วย best_params → ใช้ Booster.load_model
        self.model.load_model(str(KB_DIR / "xgboost_model_v5.json"))

    def scale(self, feat_dict: dict[str, float]) -> np.ndarray:
        """StandardScaler transform"""
        vec = np.array([feat_dict.get(n, 0.0) for n in self.feature_names])
        return (vec - self.scaler_mean) / self.scaler_scale

    def predict_proba(self, feat_dict: dict[str, float]) -> float:
        """คืน probability PD (0-1)"""
        x = self.scale(feat_dict).reshape(1, -1)
        dm = xgb.DMatrix(x, feature_names=self.feature_names)
        prob = float(self.model.predict(dm)[0])
        return prob


class MobileAssets:
    """โหลดโมเดล Mobile (parkinson_xgb_model.json)"""
    FEATURE_NAMES = [
        "rp_count","rp_mean","rp_std","rp_diff5","rp_diff7","rp_diff10",
        "lp_count","lp_mean","lp_std","lp_diff5","lp_diff7","lp_diff10",
        "tap_diff",
    ]

    def __init__(self):
        self.model = xgb.Booster()
        self.model.load_model(str(MOB_DIR / "parkinson_xgb_model.json"))
        cv = load_json(MOB_DIR / "cv_results.json")
        self.threshold: float = cv.get("threshold", 0.5)

    def predict_proba(self, feat_dict: dict[str, float]) -> float:
        vec = np.array([feat_dict.get(n, 0.0) for n in self.FEATURE_NAMES]).reshape(1, -1)
        dm  = xgb.DMatrix(vec, feature_names=self.FEATURE_NAMES)
        return float(self.model.predict(dm)[0])


kb_assets  = KeyboardAssets()
mob_assets = MobileAssets()

# ─────────────────────────────────────────────
# Feature Engineering — Keyboard
# ─────────────────────────────────────────────
def compute_keyboard_features(events: list[dict]) -> dict[str, float]:
    """
    คำนวณ features จาก raw keystroke events

    event schema:
        { "key": "f"|"k", "hand": "left"|"right",
          "hold_time": float (sec), "flight_time": float|null (sec) }
    """
    if len(events) < 10:
        raise HTTPException(422, "ต้องมีอย่างน้อย 10 keystrokes")

    hold_times   = [e["hold_time"]   for e in events]
    flight_times = [e["flight_time"] for e in events if e.get("flight_time") is not None]
    left_ht      = [e["hold_time"]   for e in events if e.get("hand") == "left"]
    right_ht     = [e["hold_time"]   for e in events if e.get("hand") == "right"]
    n            = len(events)

    def safe_stats(arr: list[float]) -> dict:
        if len(arr) < 2:
            return {k: 0.0 for k in ["mean","std","median","iqr","cv","p10","p90",
                                      "p90_p10_ratio","skew","kurt",
                                      "autocorr1","autocorr2","autocorr3"]}
        a   = np.array(arr)
        p10 = float(np.percentile(a, 10))
        p90 = float(np.percentile(a, 90))
        mn  = float(a.mean())
        sd  = float(a.std(ddof=1)) if len(a) > 1 else 0.0

        def autocorr(lag):
            if len(a) <= lag:
                return 0.0
            c = np.corrcoef(a[:-lag], a[lag:])
            return float(c[0, 1]) if not np.isnan(c[0, 1]) else 0.0

        return {
            "mean":            mn,
            "std":             sd,
            "median":          float(np.median(a)),
            "iqr":             float(np.percentile(a, 75) - np.percentile(a, 25)),
            "cv":              sd / mn if mn != 0 else 0.0,
            "p10":             p10,
            "p90":             p90,
            "p90_p10_ratio":   p90 / p10 if p10 != 0 else 0.0,
            "skew":            float(scipy_stats.skew(a)),
            "kurt":            float(scipy_stats.kurtosis(a)),
            "autocorr1":       autocorr(1),
            "autocorr2":       autocorr(2),
            "autocorr3":       autocorr(3),
        }

    # IKI (inter-keystroke interval) = hold + flight
    ikis = []
    for i in range(len(events) - 1):
        ht = events[i]["hold_time"]
        ft = events[i].get("flight_time") or 0.0
        ikis.append(ht + ft)

    def iki_entropy(ikis_: list[float]) -> float:
        if len(ikis_) < 2:
            return 0.0
        hist, _ = np.histogram(ikis_, bins=10)
        prob    = hist / hist.sum()
        prob    = prob[prob > 0]
        return float(-np.sum(prob * np.log(prob + 1e-9)))

    total_time = sum(hold_times) + sum(e.get("flight_time") or 0.0 for e in events)
    typing_speed = n / total_time if total_time > 0 else 0.0

    # bimanual
    bim_asym  = 0.0
    bim_ratio = 1.0
    if left_ht and right_ht:
        lm = statistics.mean(left_ht)
        rm = statistics.mean(right_ht)
        bim_asym  = abs(lm - rm)
        bim_ratio = lm / rm if rm != 0 else 1.0

    ht_s = safe_stats(hold_times)
    ft_s = safe_stats(flight_times)

    # pause ratios (based on IKI)
    total_iki = len(ikis) or 1
    pause_500  = sum(1 for x in ikis if x > 0.5)  / total_iki
    pause_1000 = sum(1 for x in ikis if x > 1.0)  / total_iki
    pause_2000 = sum(1 for x in ikis if x > 2.0)  / total_iki

    lht_s = safe_stats(left_ht)
    rht_s = safe_stats(right_ht)

    return {
        "HT_mean":           ht_s["mean"],
        "HT_std":            ht_s["std"],
        "HT_median":         ht_s["median"],
        "HT_iqr":            ht_s["iqr"],
        "HT_cv":             ht_s["cv"],
        "HT_p10":            ht_s["p10"],
        "HT_p90":            ht_s["p90"],
        "HT_p90_p10_ratio":  ht_s["p90_p10_ratio"],
        "HT_skew":           ht_s["skew"],
        "HT_kurt":           ht_s["kurt"],
        "HT_autocorr_lag1":  ht_s["autocorr1"],
        "HT_autocorr_lag2":  ht_s["autocorr2"],
        "HT_autocorr_lag3":  ht_s["autocorr3"],
        "FT_mean":           ft_s["mean"],
        "FT_std":            ft_s["std"],
        "FT_median":         ft_s["median"],
        "FT_iqr":            ft_s["iqr"],
        "FT_cv":             ft_s["cv"],
        "FT_skew":           ft_s["skew"],
        "FT_kurt":           ft_s["kurt"],
        "FT_p90":            ft_s["p90"],
        "FT_autocorr_lag1":  ft_s["autocorr1"],
        "pause_500ms_ratio":  pause_500,
        "pause_1000ms_ratio": pause_1000,
        "pause_2000ms_ratio": pause_2000,
        "IKI_entropy":        iki_entropy(ikis),
        "typing_speed_kps":   typing_speed,
        "n_keystrokes":       float(n),
        "bimanual_asymmetry": bim_asym,
        "bimanual_ratio":     bim_ratio,
        "left_HT_cv":         lht_s["cv"],
        "right_HT_cv":        rht_s["cv"],
    }


def compute_mobile_features(taps: list[dict]) -> dict[str, float]:
    """
    คำนวณ features จาก raw tap events (มือถือ)

    tap schema:
        { "hand": "left"|"right", "reaction_time": float (ms) }
    """
    left  = [t["reaction_time"] for t in taps if t.get("hand") == "left"]
    right = [t["reaction_time"] for t in taps if t.get("hand") == "right"]

    if len(left) < 3 or len(right) < 3:
        raise HTTPException(422, "ต้องมี tap ซ้าย/ขวา อย่างน้อย 3 ครั้ง")

    def diff_pct(arr: list[float], pct: int) -> float:
        if len(arr) < 2:
            return 0.0
        diffs = [abs(arr[i+1] - arr[i]) for i in range(len(arr)-1)]
        return float(np.percentile(diffs, pct))

    lp, rp = np.array(left), np.array(right)
    all_rt = sorted(left + right)

    return {
        "rp_count": float(len(right)),
        "rp_mean":  float(rp.mean()),
        "rp_std":   float(rp.std(ddof=1)) if len(right) > 1 else 0.0,
        "rp_diff5":  diff_pct(right, 5),
        "rp_diff7":  diff_pct(right, 7),
        "rp_diff10": diff_pct(right, 10),
        "lp_count": float(len(left)),
        "lp_mean":  float(lp.mean()),
        "lp_std":   float(lp.std(ddof=1)) if len(left) > 1 else 0.0,
        "lp_diff5":  diff_pct(left, 5),
        "lp_diff7":  diff_pct(left, 7),
        "lp_diff10": diff_pct(left, 10),
        "tap_diff": float(abs(lp.mean() - rp.mean())),
    }


# ─────────────────────────────────────────────
# Risk level helper
# ─────────────────────────────────────────────
def risk_label(prob: float, threshold: float) -> dict[str, Any]:
    if prob >= threshold + 0.15:
        level, color = "สูง", "red"
    elif prob >= threshold:
        level, color = "ปานกลาง", "orange"
    elif prob >= threshold - 0.15:
        level, color = "ต่ำ-ปานกลาง", "yellow"
    else:
        level, color = "ต่ำ", "green"
    return {
        "label": level,
        "color": color,
        "is_pd": prob >= threshold,
        "probability": round(prob, 4),
        "threshold_used": round(threshold, 4),
    }


# ─────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────
class KeyEvent(BaseModel):
    key:         str   = Field(..., description="'f' หรือ 'k'")
    hand:        str   = Field(..., description="'left' หรือ 'right'")
    hold_time:   float = Field(..., description="วินาที ที่กดค้าง")
    flight_time: float | None = Field(None, description="วินาที จนกว่าจะกดครั้งถัดไป")

class TapEvent(BaseModel):
    hand:          str   = Field(..., description="'left' หรือ 'right'")
    reaction_time: float = Field(..., description="milliseconds")

class KeyboardRequest(BaseModel):
    events: list[KeyEvent] = Field(..., min_length=10)
    session_id: str | None = None

class MobileRequest(BaseModel):
    taps:       list[TapEvent] = Field(..., min_length=6)
    session_id: str | None = None

class CombinedRequest(BaseModel):
    events:     list[KeyEvent]  = Field(..., min_length=10)
    taps:       list[TapEvent]  = Field(..., min_length=6)
    kb_weight:  float = Field(0.5, ge=0, le=1, description="น้ำหนัก keyboard score (0-1)")
    session_id: str | None = None


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "models_loaded": ["keyboard_v5", "mobile_xgb"]}


@app.post("/predict/keyboard")
def predict_keyboard(req: KeyboardRequest):
    """Mode 1 — Keyboard only"""
    events = [e.model_dump() for e in req.events]
    feats  = compute_keyboard_features(events)
    prob   = kb_assets.predict_proba(feats)
    risk   = risk_label(prob, kb_assets.threshold)

    return {
        "session_id": req.session_id,
        "mode":       "keyboard",
        "risk":       risk,
        "features":   {k: round(v, 6) for k, v in feats.items()},
        "n_keystrokes": len(events),
    }


@app.post("/predict/mobile")
def predict_mobile(req: MobileRequest):
    """Mode 2 — Mobile only"""
    taps  = [t.model_dump() for t in req.taps]
    feats = compute_mobile_features(taps)
    prob  = mob_assets.predict_proba(feats)
    risk  = risk_label(prob, mob_assets.threshold)

    return {
        "session_id": req.session_id,
        "mode":       "mobile",
        "risk":       risk,
        "features":   {k: round(v, 6) for k, v in feats.items()},
        "n_taps": len(taps),
    }


@app.post("/predict/combined")
def predict_combined(req: CombinedRequest):
    """Mode 3 — Keyboard + Mobile, weighted average"""
    events = [e.model_dump() for e in req.events]
    taps   = [t.model_dump() for t in req.taps]

    kb_feats  = compute_keyboard_features(events)
    mob_feats = compute_mobile_features(taps)

    kb_prob  = kb_assets.predict_proba(kb_feats)
    mob_prob = mob_assets.predict_proba(mob_feats)

    # Weighted combination
    w_kb  = req.kb_weight
    w_mob = 1.0 - w_kb
    combined_prob = w_kb * kb_prob + w_mob * mob_prob

    # ใช้ threshold เฉลี่ยของทั้งสอง
    combined_threshold = w_kb * kb_assets.threshold + w_mob * mob_assets.threshold
    risk = risk_label(combined_prob, combined_threshold)

    return {
        "session_id": req.session_id,
        "mode":       "combined",
        "risk":       risk,
        "breakdown": {
            "keyboard": {
                "probability": round(kb_prob, 4),
                "weight":      round(w_kb, 4),
                "threshold":   round(kb_assets.threshold, 4),
            },
            "mobile": {
                "probability": round(mob_prob, 4),
                "weight":      round(w_mob, 4),
                "threshold":   round(mob_assets.threshold, 4),
            },
        },
        "n_keystrokes": len(events),
        "n_taps":       len(taps),
    }

class FrontendAnalyzeRequest(BaseModel):
    test_type: str
    device_mode: Optional[str] = None   # รับแบบเดิม
    device: Optional[str] = None        # รับจาก frontend ที่ส่ง "device"
    event_log: Optional[list] = []
    ht_array_ms: Optional[list] = []    # ← เปลี่ยนเป็น Optional + default []
    ft_array_ms: Optional[list] = []    # ← เปลี่ยนเป็น Optional + default []
    calibration: Optional[dict] = None
    summary: Optional[dict] = None
    calib_baseline: Optional[dict] = None
    calibrated_features: Optional[dict] = None
    raw_features: Optional[dict] = None
    tap_count: Optional[int] = None

@app.post("/api/analyze")
def api_analyze(req: FrontendAnalyzeRequest):
    # รองรับ field "device" จาก frontend เก่า
    device_mode = req.device_mode or req.device or "pc"
    
    try:
        ht_sec = [v / 1000 for v in (req.ht_array_ms or []) if v > 0]
        ft_sec = [v / 1000 for v in (req.ft_array_ms or []) if v > 0]

        if len(ht_sec) < 5:
            # ถ้าข้อมูลน้อยไป ใช้ calibrated_features จาก frontend แทน
            if req.calibrated_features:
                feats = {k: v for k, v in req.calibrated_features.items()
                         if not k.startswith('_') and isinstance(v, (int, float))}
                prob = kb_assets.predict_proba(feats)
            else:
                prob = 0.15
        else:
            # สร้าง events จาก arrays
            events = []
            for i, ht in enumerate(ht_sec):
                ft = ft_sec[i] if i < len(ft_sec) else None
                hand = "left" if i % 2 == 0 else "right"
                events.append({
                    "key": "f" if hand == "left" else "k",
                    "hand": hand,
                    "hold_time": ht,
                    "flight_time": ft
                })
            feats = compute_keyboard_features(events)

            # ถ้ามี calibration จาก frontend → normalize
            if req.calibration:
                ht_baseline = req.calibration.get("baseline_ht_ms", 120) / 1000
                ft_baseline = req.calibration.get("baseline_ft_ms", 80) / 1000
                if ht_baseline > 0:
                    for k in ["HT_mean","HT_std","HT_median","HT_p10","HT_p90"]:
                        if k in feats: feats[k] /= ht_baseline
                if ft_baseline > 0:
                    for k in ["FT_mean","FT_std","FT_median","FT_p90"]:
                        if k in feats: feats[k] /= ft_baseline

            prob = kb_assets.predict_proba(feats)

        risk_pct = round(prob * 100, 1)

        if risk_pct < 20:
            level = "✅ ความเสี่ยงต่ำ — ผลปกติ"
            xai   = "รูปแบบการกดแป้นพิมพ์อยู่ในเกณฑ์ปกติ ไม่พบสัญญาณผิดปกติ"
        elif risk_pct < 50:
            level = "⚠️ ความเสี่ยงปานกลาง — แนะนำติดตาม"
            xai   = "พบความแปรปรวนสูงกว่าค่าเฉลี่ยเล็กน้อย แนะนำทดสอบซ้ำใน 2-4 สัปดาห์"
        else:
            level = "🔴 ความเสี่ยงสูง — ควรพบแพทย์"
            xai   = "พบรูปแบบผิดปกติของการควบคุมกล้ามเนื้อมัดเล็ก ควรพบแพทย์ประสาทวิทยา"

        return {
            "status": "success",
            "analysis": {
                "risk_percentage": risk_pct,
                "risk_level":      level,
                "xai_explanation": xai,
                "radar_metrics":   [
                    max(5, round(100 - prob * 80)),
                    max(5, round(90  - prob * 60)),
                    max(5, round(85  - prob * 55)),
                    max(5, round(92  - prob * 70)),
                    max(5, round(88  - prob * 65)),
                ]
            },
            "drift": None
        }

    except Exception as e:
        raise HTTPException(500, str(e))

# ─────────────────────────────────────────────
# Run
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)