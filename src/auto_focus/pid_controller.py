"""PID controller for astigmatic autofocus.

Based on the CylLensGUI PID (Vliet lab) with additions:
  - Anti-windup with conditional integration
  - Derivative filtering (EMA on D term)
  - Command deadband to prevent dithering
  - Slew rate limiting
  - Excursion clamping around lock point
"""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass(slots=True)
class PidConfig:
    """PID tuning parameters."""

    kp: float = 0.6
    ki: float = 0.15
    kd: float = 0.0
    integral_limit_um: float = 2.0
    max_step_um: float = 0.25
    command_deadband_um: float = 0.005
    max_slew_rate_um_per_s: float | None = None
    derivative_alpha: float = 0.7
    stage_min_um: float | None = None
    stage_max_um: float | None = None
    max_excursion_um: float | None = 5.0


class PidController:
    """Discrete PID with anti-windup, derivative filtering, and safety clamps."""

    def __init__(self, config: PidConfig, initial_output: float = 0.0) -> None:
        self._config = config
        self._integral = 0.0
        self._last_error: float | None = None
        self._filtered_derivative = 0.0
        self._last_output = initial_output
        self._last_command_time: float | None = None
        self._lock_center: float | None = None

    def set_lock_center(self, z_um: float) -> None:
        self._lock_center = z_um

    def reset(self) -> None:
        self._integral = 0.0
        self._last_error = None
        self._filtered_derivative = 0.0
        self._last_command_time = None
        self._lock_center = None

    def update(self, error_um: float, dt_s: float, *, deadband_um: float | None = None) -> tuple[float, bool]:
        c = self._config
        dt_s = max(1e-6, dt_s)

        p_term = c.kp * error_um
        candidate_integral = self._integral + error_um * dt_s
        candidate_integral = max(-c.integral_limit_um, min(c.integral_limit_um, candidate_integral))
        i_term = c.ki * candidate_integral

        raw_derivative = 0.0 if self._last_error is None else (error_um - self._last_error) / dt_s
        self._last_error = error_um
        alpha = c.derivative_alpha
        self._filtered_derivative = alpha * self._filtered_derivative + (1.0 - alpha) * raw_derivative
        d_term = c.kd * self._filtered_derivative

        correction = -(p_term + i_term + d_term)
        correction = max(-c.max_step_um, min(c.max_step_um, correction))

        deadband = c.command_deadband_um if deadband_um is None else max(0.0, float(deadband_um))
        if abs(correction) <= deadband:
            return 0.0, False

        if c.max_slew_rate_um_per_s is not None and self._last_command_time is not None:
            age = max(1e-6, time.monotonic() - self._last_command_time)
            max_delta = c.max_slew_rate_um_per_s * age
            correction = max(-max_delta, min(max_delta, correction))

        proposed = self._last_output + correction

        if self._lock_center is not None and c.max_excursion_um is not None:
            lo = self._lock_center - c.max_excursion_um
            hi = self._lock_center + c.max_excursion_um
            proposed = max(lo, min(hi, proposed))

        if c.stage_min_um is not None:
            proposed = max(c.stage_min_um, proposed)
        if c.stage_max_um is not None:
            proposed = min(c.stage_max_um, proposed)

        if proposed == self._last_output + correction:
            self._integral = candidate_integral

        actual_correction = proposed - self._last_output
        self._last_output = proposed
        self._last_command_time = time.monotonic()
        return actual_correction, True

    @property
    def output(self) -> float:
        return self._last_output

    @output.setter
    def output(self, value: float) -> None:
        self._last_output = value
