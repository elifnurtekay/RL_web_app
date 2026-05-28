from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Any
import json
import time
import zipfile

import numpy as np
import torch
import torch.nn as nn

from rl_airfoil.geometry.cst import compute_cst_geometry_features
from .base import Evaluator, AeroOutput


class ResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.05):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(dropout),
        )
        self.activation = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.net(x))


class ResMLPSurrogate(nn.Module):
    """
    Paper'daki surrogate mimarisiyle uyumlu ResMLP:
    input_dim=10 -> hidden_dim=256 -> 4 residual block -> output_dim=3
    """
    def __init__(
        self,
        input_dim: int = 10,
        hidden_dim: int = 256,
        output_dim: int = 3,
        num_blocks: int = 4,
        dropout: float = 0.05,
    ):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.ModuleList(
            [ResidualBlock(hidden_dim=hidden_dim, dropout=dropout) for _ in range(num_blocks)]
        )
        self.output_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.input_layer(x)
        for block in self.blocks:
            h = block(h)
        return self.output_layer(h)


@dataclass
class JsonScaler:
    x_mean: np.ndarray
    x_scale: np.ndarray
    y_mean: np.ndarray
    y_std: np.ndarray
    use_log_re: bool = True

    @classmethod
    def from_json(cls, path: Path, model_name: Optional[str] = None) -> "JsonScaler":
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        if model_name is not None and isinstance(raw, dict) and model_name in raw:
            raw = raw[model_name]

        def pick(keys: list[str]) -> Any:
            for key in keys:
                if key in raw:
                    return raw[key]
            raise KeyError(f"None of these keys found in scaler json: {keys}")

        x_mean = pick(["x_mean", "input_mean", "mean_x", "X_mean"])
        x_scale = pick(["x_scale", "x_std", "input_std", "std_x", "X_std"])
        y_mean = pick(["y_mean", "target_mean", "mean_y", "Y_mean"])
        y_std = pick(["y_std", "y_scale", "target_std", "std_y", "Y_std"])

        return cls(
            x_mean=np.asarray(x_mean, dtype=np.float32),
            x_scale=np.asarray(x_scale, dtype=np.float32),
            y_mean=np.asarray(y_mean, dtype=np.float32),
            y_std=np.asarray(y_std, dtype=np.float32),
            use_log_re=bool(raw.get("use_log_re", True)),
        )

    def transform_x(self, x: np.ndarray) -> np.ndarray:
        denom = np.where(self.x_scale == 0.0, 1.0, self.x_scale)
        return (x - self.x_mean) / denom

    def inverse_y(self, y: np.ndarray) -> np.ndarray:
        return y * self.y_std + self.y_mean


@dataclass
class SurrogateArtifacts:
    model: torch.nn.Module
    scaler: JsonScaler | None
    torchscript_has_scaler: bool = False
    torchscript_expects_raw_re: bool = False


class SurrogateEvaluator(Evaluator):
    name = "surrogate"

    def __init__(
        self,
        checkpoint_path: str,
        model_name: str,
        scaler_json_path: str = "checkpoints/scalers.json",
        device: str = "cpu",
    ):
        self.checkpoint_path = Path(checkpoint_path)
        self.model_name = model_name
        self.scaler_json_path = Path(scaler_json_path)
        self.device = torch.device(device)

        self._torchscript_has_scaler = False
        self._torchscript_expects_raw_re = False

        self.artifacts = self._load_artifacts()

    def _torch_load(self, path: Path):
        try:
            return torch.load(path, map_location=self.device, weights_only=False)
        except TypeError:
            return torch.load(path, map_location=self.device)

    def _detect_torchscript_scaler(self, model: torch.nn.Module) -> bool:
        try:
            code = str(model.code).lower()
            scaler_terms = ["x_mean", "x_std", "x_scale", "y_mean", "y_std", "y_scale"]
            if any(term in code for term in scaler_terms):
                return True
        except Exception:
            pass

        try:
            buffer_names = [name.lower() for name, _ in model.named_buffers()]
            scaler_terms = ["x_mean", "x_std", "x_scale", "y_mean", "y_std", "y_scale"]
            if any(term in name for name in buffer_names for term in scaler_terms):
                return True
        except Exception:
            pass

        return False

    def _detect_torchscript_raw_re(self, model: torch.nn.Module) -> bool:
        """
        TorchScript wrapper forward içinde log10(Re) uyguluyorsa,
        son input kolonunda ham Reynolds bekliyor demektir.

        Eski kontrol sadece "log10" + "x_raw" arıyordu.
        Bazı TorchScript modellerde değişken adı x_raw olmayabilir.
        Bu nedenle log10 görülmesi ham Re beklentisi için yeterli sinyal kabul edilir.
        """
        try:
            code = str(model.code).lower()

            if "log10" in code:
                return True

        except Exception:
            pass

        return False

    def _extract_state_dict(self, ckpt: Any) -> Optional[Dict[str, torch.Tensor]]:
        if isinstance(ckpt, dict):
            for key in ["state_dict", "model_state_dict", "surrogate_state_dict", "net_state_dict"]:
                if key in ckpt and isinstance(ckpt[key], dict):
                    return ckpt[key]

            if "model" in ckpt:
                model_obj = ckpt["model"]
                if isinstance(model_obj, nn.Module):
                    return None
                if isinstance(model_obj, dict):
                    return model_obj

            if all(torch.is_tensor(v) for v in ckpt.values()):
                return ckpt

        return None

    def _strip_prefix_if_needed(
        self,
        state_dict: Dict[str, torch.Tensor],
        prefixes: list[str],
    ) -> Dict[str, torch.Tensor]:
        new_state = dict(state_dict)

        for prefix in prefixes:
            if all(k.startswith(prefix) for k in new_state.keys()):
                new_state = {k[len(prefix):]: v for k, v in new_state.items()}

        return new_state

    def _load_model(self) -> torch.nn.Module:
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(f"Surrogate checkpoint not found: {self.checkpoint_path}")

        self._last_jit_load_error = ""

        # TorchScript archive ise önce kesin olarak torch.jit.load denenir.
        if zipfile.is_zipfile(self.checkpoint_path):
            try:
                # Windows + OneDrive + Türkçe karakterli path sorunlarını önlemek için
                # TorchScript dosyasını doğrudan path ile değil, binary buffer ile yüklüyoruz.
                with open(self.checkpoint_path, "rb") as f:
                    buffer = io.BytesIO(f.read())

                model = torch.jit.load(buffer, map_location=self.device)
                model.eval()

                self._torchscript_has_scaler = self._detect_torchscript_scaler(model)
                self._torchscript_expects_raw_re = self._detect_torchscript_raw_re(model)

                print("[SURROGATE] Loaded with torch.jit.load from BytesIO")
                print(f"[SURROGATE] torchscript_has_scaler={self._torchscript_has_scaler}")
                print(f"[SURROGATE] torchscript_expects_raw_re={self._torchscript_expects_raw_re}")

                return model

            except Exception as exc:
                self._last_jit_load_error = str(exc)
                print("[SURROGATE] torch.jit.load from BytesIO failed:", self._last_jit_load_error)

        ckpt = self._torch_load(self.checkpoint_path)

        if isinstance(ckpt, nn.Module):
            model = ckpt.to(self.device)
            model.eval()

            # Önemli düzeltme:
            # torch.load bazı TorchScript archive dosyalarını içeride torch.jit.load'a yönlendirebilir.
            # Bu durumda model nn.Module gibi görünse bile TorchScript wrapper olabilir.
            # Bayrakları burada da tekrar set etmezsek external scaler yanlışlıkla uygulanabilir.
            self._torchscript_has_scaler = self._detect_torchscript_scaler(model)
            self._torchscript_expects_raw_re = self._detect_torchscript_raw_re(model)

            return model

        if isinstance(ckpt, dict) and "model" in ckpt and isinstance(ckpt["model"], nn.Module):
            model = ckpt["model"].to(self.device)
            model.eval()

            self._torchscript_has_scaler = self._detect_torchscript_scaler(model)
            self._torchscript_expects_raw_re = self._detect_torchscript_raw_re(model)

            return model

        state_dict = self._extract_state_dict(ckpt)

        if state_dict is None:
            raise TypeError(
                "Unsupported checkpoint format. Expected TorchScript, nn.Module, "
                "or a checkpoint containing state_dict/model_state_dict. "
                f"Last torch.jit.load error: {self._last_jit_load_error}"
            )

        model = ResMLPSurrogate(
            input_dim=10,
            hidden_dim=256,
            output_dim=3,
            num_blocks=4,
            dropout=0.05,
        ).to(self.device)

        candidate_states = [
            state_dict,
            self._strip_prefix_if_needed(state_dict, ["module."]),
            self._strip_prefix_if_needed(state_dict, ["model."]),
            self._strip_prefix_if_needed(state_dict, ["net."]),
            self._strip_prefix_if_needed(state_dict, ["surrogate."]),
        ]

        last_error = None

        for candidate in candidate_states:
            try:
                model.load_state_dict(candidate, strict=True)
                model.eval()
                return model
            except Exception as exc:
                last_error = exc

        raise RuntimeError(
            "Checkpoint is a state_dict, but its layer names do not match "
            f"ResMLPSurrogate. Last load error: {last_error}. "
            f"Last torch.jit.load error: {self._last_jit_load_error}"
        )

    def _load_artifacts(self) -> SurrogateArtifacts:
        model = self._load_model()

        # TorchScript SurrogateWrapper scaler'ı kendi içinde taşıyorsa
        # dışarıdaki scalers.json inference sırasında kullanılmaz.
        if self._torchscript_has_scaler:
            scaler = None
        else:
            if not self.scaler_json_path.exists():
                raise FileNotFoundError(f"Scaler json not found: {self.scaler_json_path}")

            scaler = JsonScaler.from_json(self.scaler_json_path, model_name=self.model_name)

            if scaler.x_mean.shape[-1] != 10:
                raise ValueError(f"Expected scaler x_mean dimension 10, got {scaler.x_mean.shape}.")
            if scaler.y_mean.shape[-1] != 3:
                raise ValueError(f"Expected scaler y_mean dimension 3, got {scaler.y_mean.shape}.")

        return SurrogateArtifacts(
            model=model,
            scaler=scaler,
            torchscript_has_scaler=self._torchscript_has_scaler,
            torchscript_expects_raw_re=self._torchscript_expects_raw_re,
        )

    def _raw_features_for_wrapper(self, cst: np.ndarray, aoa: float, re: float) -> np.ndarray:
        """
        TorchScript SurrogateWrapper için input:
            8 CST + AoA + ham Re

        Çünkü wrapper kendi içinde log10(Re) alıyor.
        """
        return np.concatenate(
            [
                np.asarray(cst, dtype=np.float32).reshape(8),
                np.array([aoa, re], dtype=np.float32),
            ],
            axis=0,
        ).reshape(1, -1).astype(np.float32)

    def _raw_features_for_external_scaler(
        self,
        cst: np.ndarray,
        aoa: float,
        re: float,
        use_log_re: bool = True,
    ) -> np.ndarray:
        """
        State_dict / normal PyTorch model için input:
            8 CST + AoA + Re özelliği

        scalers.json içinde use_log_re=true ise son kolon log10(Re),
        use_log_re=false ise son kolon ham Re olur.
        """
        re_value = np.log10(re) if use_log_re else re

        return np.concatenate(
            [
                np.asarray(cst, dtype=np.float32).reshape(8),
                np.array([aoa, re_value], dtype=np.float32),
            ],
            axis=0,
        ).reshape(1, -1).astype(np.float32)

    def _featurize(self, cst: np.ndarray, aoa: float, re: float) -> np.ndarray:
        if self.artifacts.torchscript_has_scaler:
            if self.artifacts.torchscript_expects_raw_re:
                return self._raw_features_for_wrapper(cst, aoa, re)

            # TorchScript scaler içeriyor ama içeride log10(Re) almıyorsa,
            # dışarıda log10(Re) hazırlanır.
            return self._raw_features_for_external_scaler(
                cst,
                aoa,
                re,
                use_log_re=True,
            )

        if self.artifacts.scaler is None:
            raise RuntimeError("Scaler is required for non-wrapper surrogate model.")

        x_raw = self._raw_features_for_external_scaler(
            cst,
            aoa,
            re,
            use_log_re=bool(self.artifacts.scaler.use_log_re),
        )

        return self.artifacts.scaler.transform_x(x_raw).astype(np.float32)

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        t0 = time.time()

        geometry = compute_cst_geometry_features(cst)
        geometry_dict = geometry.to_dict()

        if not geometry.is_valid:
            return AeroOutput(
                cl=0.0,
                cd=1.0,
                cm=0.0,
                tc=float(geometry.max_thickness),
                is_geometry_valid=False,
                solver_status="invalid_geometry",
                solver_error_message="CST geometry failed validity checks.",
                runtime_ms=(time.time() - t0) * 1000.0,
                geometry_features=geometry_dict,
            )

        try:
            x = self._featurize(cst, aoa, re)

            with torch.no_grad():
                xin = torch.from_numpy(x).to(self.device)
                y = self.artifacts.model(xin)

                if isinstance(y, tuple):
                    y = y[0]

                y_np = y.detach().cpu().numpy()

                # TorchScript wrapper zaten y_mean/y_std ile gerçek ölçeğe döndürüyor.
                # Sadece wrapper olmayan modellerde inverse_y uygulanır.
                if not self.artifacts.torchscript_has_scaler:
                    if self.artifacts.scaler is None:
                        raise RuntimeError("Scaler is required to inverse-transform surrogate outputs.")
                    y_np = self.artifacts.scaler.inverse_y(y_np)

            cl, cd, cm = [float(v) for v in y_np.reshape(-1)[:3]]

            finite_ok = bool(np.isfinite(cl) and np.isfinite(cd) and np.isfinite(cm))
            positive_cd = bool(cd > 0.0)

            if not finite_ok:
                return AeroOutput(
                    cl=0.0,
                    cd=1.0,
                    cm=0.0,
                    tc=float(geometry.max_thickness),
                    is_geometry_valid=False,
                    solver_status="nan_output",
                    solver_error_message="Surrogate produced non-finite output.",
                    runtime_ms=(time.time() - t0) * 1000.0,
                    geometry_features=geometry_dict,
                )

            if not positive_cd:
                # Negatif veya sıfır CD fiziksel değildir.
                # Environment içinde CD lower bound ayrıca kullanılıyor;
                # burada solver_status ile durumu işaretliyoruz.
                cd = max(float(cd), 1e-8)

            return AeroOutput(
                cl=float(cl),
                cd=float(cd),
                cm=float(cm),
                tc=float(geometry.max_thickness),
                is_geometry_valid=bool(geometry.is_valid and finite_ok),
                solver_status="ok" if positive_cd else "non_positive_cd_clipped",
                solver_error_message="" if positive_cd else "Surrogate returned non-positive CD; clipped to 1e-8.",
                runtime_ms=(time.time() - t0) * 1000.0,
                geometry_features=geometry_dict,
            )

        except Exception as exc:
            return AeroOutput(
                cl=0.0,
                cd=1.0,
                cm=0.0,
                tc=float(geometry.max_thickness),
                is_geometry_valid=False,
                solver_status="solver_error",
                solver_error_message=str(exc),
                runtime_ms=(time.time() - t0) * 1000.0,
                geometry_features=geometry_dict,
            )