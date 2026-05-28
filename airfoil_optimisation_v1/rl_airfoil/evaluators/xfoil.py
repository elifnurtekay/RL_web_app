from __future__ import annotations

from pathlib import Path
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass

import numpy as np

from rl_airfoil.evaluators.base import Evaluator, AeroOutput
from rl_airfoil.geometry.cst import (
    cst_surface,
    compute_cst_geometry_features,
)

def _safe_unlink(path: Path, retries: int = 3, sleep_sec: float = 0.05) -> None:
    """
    Windows'ta dosya kısa süre kilitli kalırsa birkaç kez deneyerek siler.
    Dosya yoksa hata vermez.
    """
    path = Path(path)

    for attempt in range(retries):
        try:
            if path.exists():
                path.unlink()
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(sleep_sec)
        except FileNotFoundError:
            return

@dataclass
class XFOILResult:
    converged: bool
    cl: float = 0.0
    cd: float = 1.0
    cm: float = 0.0
    alpha: float = 0.0
    error_message: str = ""


class XFOILEvaluator(Evaluator):
    name = "xfoil"

    def __init__(
        self,
        executable_path: str = "xfoil",
        timeout_sec: float = 30.0,
        max_iter: int = 100,
        ppar_n: int = 160,
        mach: float = 0.0,
        ncrit: float = 9.0,
        xtr_top: float = 1.0,
        xtr_bottom: float = 1.0,
        n_points: int = 201,
        min_local_thickness_required: float = 0.005,
        max_abs_surface_y: float = 0.75,
    ):
        self.executable_path = str(executable_path)
        self.timeout_sec = float(timeout_sec)
        self.max_iter = int(max_iter)
        self.ppar_n = int(ppar_n)
        self.mach = float(mach)
        self.ncrit = float(ncrit)
        self.xtr_top = float(xtr_top)
        self.xtr_bottom = float(xtr_bottom)
        self.n_points = int(n_points)
        self.min_local_thickness_required = float(min_local_thickness_required)
        self.max_abs_surface_y = float(max_abs_surface_y)

    def _resolve_executable(self) -> str:
        env_path = os.environ.get("XFOIL_PATH", "").strip()

        if env_path:
            p = Path(env_path)
            if p.exists():
                return str(p)

        configured = Path(self.executable_path)

        if configured.exists():
            return str(configured)

        found = shutil.which(self.executable_path)

        if found:
            return found

        raise FileNotFoundError(
            "XFOIL executable not found. "
            "Set XFOIL_PATH or pass --xfoil-executable-path."
        )

    def _write_airfoil_dat(self, cst: np.ndarray, airfoil_path: Path) -> None:
        """
        XFOIL coordinate ordering:
        upper TE -> LE, then lower LE -> TE.
        """
        cst = np.asarray(cst, dtype=np.float64).reshape(8)

        x = np.linspace(0.0, 1.0, self.n_points, dtype=np.float64)

        upper = cst_surface(cst[:4], x)
        lower = cst_surface(cst[4:], x)

        x_upper = x[::-1]
        y_upper = upper[::-1]

        x_lower = x[1:]
        y_lower = lower[1:]

        with open(airfoil_path, "w", encoding="utf-8") as f:
            f.write("CST8_RL_AIRFOIL\n")

            for xi, yi in zip(x_upper, y_upper):
                f.write(f"{xi:.8f} {yi:.8f}\n")

            for xi, yi in zip(x_lower, y_lower):
                f.write(f"{xi:.8f} {yi:.8f}\n")

    def _build_input_script(
        self,
        airfoil_filename: str,
        polar_filename: str,
        aoa: float,
        re: float,
    ) -> str:
        """
        XFOIL batch script.
        PACC writes polar output.
        ALFA solves one angle of attack.
        """
        lines = [
            "PLOP",
            "G",
            "",
            f"LOAD {airfoil_filename}",
            "PANE",
            "PPAR",
            f"N {self.ppar_n}",
            "",
            "",
            "OPER",
            f"VISC {float(re):.8g}",
            f"MACH {self.mach:.8g}",
            f"ITER {self.max_iter}",
            "VPAR",
            f"N {self.ncrit:.8g}",
            f"XTR {self.xtr_top:.8g} {self.xtr_bottom:.8g}",
            "",
            "PACC",
            polar_filename,
            "",
            f"ALFA {float(aoa):.8f}",
            "PACC",
            "",
            "QUIT",
            "",
        ]

        return "\n".join(lines)

    def _parse_polar(self, polar_path: Path, requested_aoa: float) -> XFOILResult:
        if not polar_path.exists():
            return XFOILResult(
                converged=False,
                alpha=float(requested_aoa),
                error_message="XFOIL did not create polar file.",
            )

        text = polar_path.read_text(errors="ignore")
        numeric_rows = []

        for line in text.splitlines():
            line = line.strip()

            if not line:
                continue

            parts = re.split(r"\s+", line)

            if len(parts) < 5:
                continue

            try:
                values = [float(x) for x in parts[:5]]
            except Exception:
                continue

            numeric_rows.append(values)

        if not numeric_rows:
            return XFOILResult(
                converged=False,
                alpha=float(requested_aoa),
                error_message="Polar file exists but contains no numeric aerodynamic row.",
            )

        alpha, cl, cd, _cdp, cm = numeric_rows[-1]

        if not np.isfinite(cl) or not np.isfinite(cd) or not np.isfinite(cm):
            return XFOILResult(
                converged=False,
                alpha=float(alpha),
                error_message="XFOIL returned non-finite CL/CD/CM.",
            )

        if cd <= 0.0:
            return XFOILResult(
                converged=False,
                alpha=float(alpha),
                error_message=f"XFOIL returned non-positive CD={cd}.",
            )

        return XFOILResult(
            converged=True,
            cl=float(cl),
            cd=float(cd),
            cm=float(cm),
            alpha=float(alpha),
            error_message="",
        )

    def evaluate(self, cst: np.ndarray, aoa: float, re: float) -> AeroOutput:
        t0 = time.time()

        geometry = compute_cst_geometry_features(
            cst,
            n_points=self.n_points,
            min_local_thickness_required=self.min_local_thickness_required,
            max_abs_surface_y=self.max_abs_surface_y,
        )

        geometry_dict = geometry.to_dict()

        if not bool(geometry.is_valid):
            return AeroOutput(
                cl=0.0,
                cd=1.0,
                cm=0.0,
                tc=float(geometry.max_thickness),
                is_geometry_valid=False,
                solver_status="invalid_geometry",
                solver_error_message="CST geometry failed validity checks before XFOIL.",
                runtime_ms=(time.time() - t0) * 1000.0,
                geometry_features={
                    **geometry_dict,
                    "xfoil_converged": False,
                    "xfoil_error_message": "invalid_geometry",
                },
            )

        try:
            xfoil_exe = self._resolve_executable()

            with tempfile.TemporaryDirectory(prefix="xfoil_eval_") as tmp:
                tmp_dir = Path(tmp)

                airfoil_path = tmp_dir / "airfoil.dat"
                input_path = tmp_dir / "xfoil.in"
                polar_path = tmp_dir / "polar.dat"
                stdout_path = tmp_dir / "xfoil_stdout.txt"
                stderr_path = tmp_dir / "xfoil_stderr.txt"

                # Her XFOIL çağrısı yeni temp klasörde çalışıyor.
                # Yine de aynı çağrı içinde oluşabilecek eski/yarım dosyaları güvenli şekilde temizliyoruz.
                for stale_file in [
                    polar_path,
                    input_path,
                    stdout_path,
                    stderr_path,
                    tmp_dir / "dump.dat",
                ]:
                    _safe_unlink(stale_file)

                # 1. Airfoil koordinat dosyasını yaz
                self._write_airfoil_dat(cst, airfoil_path)

                # 2. XFOIL batch input scriptini oluştur
                xfoil_input = self._build_input_script(
                    airfoil_filename=airfoil_path.name,
                    polar_filename=polar_path.name,
                    aoa=aoa,
                    re=re,
                )

                # 3. Komut dosyasını debug/tekrar üretilebilirlik için temp klasöre yaz
                input_path.write_text(xfoil_input, encoding="utf-8")

                # 4. XFOIL'i temp klasör içinde çalıştır
                xfoil_t0 = time.time()

                proc = subprocess.run(
                    [str(xfoil_exe)],
                    input=xfoil_input,
                    text=True,
                    cwd=str(tmp_dir),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    timeout=float(self.timeout_sec),
                )

                xfoil_runtime_ms = (time.time() - xfoil_t0) * 1000.0

                # 5. stdout/stderr temp klasöre yazılır.
                # TemporaryDirectory kapanınca silinir ama hata ayıklama sırasında istenirse breakpoint ile görülebilir.
                stdout_path.write_text(proc.stdout or "", encoding="utf-8", errors="ignore")
                stderr_path.write_text(proc.stderr or "", encoding="utf-8", errors="ignore")

                # 6. Polar dosyası gerçekten oluşmuş mu kontrol et
                if not polar_path.exists() or polar_path.stat().st_size == 0:
                    runtime_ms = (time.time() - t0) * 1000.0

                    message = "XFOIL did not create a non-empty polar file."

                    if proc.returncode not in (0, None):
                        message += f" | returncode={proc.returncode}"

                    if proc.stderr:
                        message += f" | stderr={proc.stderr.strip()}"

                    return AeroOutput(
                        cl=0.0,
                        cd=1.0,
                        cm=0.0,
                        tc=float(geometry.max_thickness),
                        is_geometry_valid=True,
                        solver_status="solver_error",
                        solver_error_message=message,
                        runtime_ms=runtime_ms,
                        geometry_features={
                            **geometry_dict,
                            "xfoil_converged": False,
                            "xfoil_error_message": message,
                            "xfoil_runtime_ms": float(xfoil_runtime_ms),
                        },
                    )

                # 7. Yeni polar.dat dosyasını parse et
                result = self._parse_polar(polar_path, requested_aoa=aoa)

                runtime_ms = (time.time() - t0) * 1000.0

                if not result.converged:
                    message = result.error_message

                    if proc.returncode not in (0, None):
                        message += f" | returncode={proc.returncode}"

                    if proc.stderr:
                        message += f" | stderr={proc.stderr.strip()}"

                    return AeroOutput(
                        cl=0.0,
                        cd=1.0,
                        cm=0.0,
                        tc=float(geometry.max_thickness),
                        is_geometry_valid=True,
                        solver_status="solver_error",
                        solver_error_message=message,
                        runtime_ms=runtime_ms,
                        geometry_features={
                            **geometry_dict,
                            "xfoil_converged": False,
                            "xfoil_error_message": message,
                            "xfoil_runtime_ms": float(xfoil_runtime_ms),
                        },
                    )

                return AeroOutput(
                    cl=float(result.cl),
                    cd=float(result.cd),
                    cm=float(result.cm),
                    tc=float(geometry.max_thickness),
                    is_geometry_valid=True,
                    solver_status="ok",
                    solver_error_message="",
                    runtime_ms=runtime_ms,
                    geometry_features={
                        **geometry_dict,
                        "xfoil_converged": True,
                        "xfoil_error_message": "",
                        "xfoil_runtime_ms": float(xfoil_runtime_ms),
                    },
                )

        except subprocess.TimeoutExpired:
            runtime_ms = (time.time() - t0) * 1000.0

            return AeroOutput(
                cl=0.0,
                cd=1.0,
                cm=0.0,
                tc=float(geometry.max_thickness),
                is_geometry_valid=True,
                solver_status="solver_error",
                solver_error_message=f"XFOIL timeout after {self.timeout_sec} seconds.",
                runtime_ms=runtime_ms,
                geometry_features={
                    **geometry_dict,
                    "xfoil_converged": False,
                    "xfoil_error_message": "timeout",
                },
            )

        except Exception as exc:
            runtime_ms = (time.time() - t0) * 1000.0

            return AeroOutput(
                cl=0.0,
                cd=1.0,
                cm=0.0,
                tc=float(geometry.max_thickness),
                is_geometry_valid=True,
                solver_status="solver_error",
                solver_error_message=str(exc),
                runtime_ms=runtime_ms,
                geometry_features={
                    **geometry_dict,
                    "xfoil_converged": False,
                    "xfoil_error_message": str(exc),
                },
            )