"""YAML-configured Parquet loading for recorded IMU data."""

from dataclasses import dataclass
import os

import numpy as np
import pandas as pd
import yaml


@dataclass
class ParquetSensorRun:
    """Recorded IMU signals normalized to the parser's physical units."""

    run_name: str
    time_s: np.ndarray
    accel_mps2: np.ndarray
    gyro_radps: np.ndarray
    gyro_dps: np.ndarray
    baseline_pitch_deg: np.ndarray
    baseline_roll_deg: np.ndarray
    mahony_roll_offset_deg: float


def _load_mapping(mapping_file):
    with open(mapping_file, "r") as mapping_stream:
        cfg = yaml.safe_load(mapping_stream)
    if not isinstance(cfg, dict) or "parquet_column_map" not in cfg:
        raise ValueError("Mapping file must contain 'parquet_column_map'.")
    return cfg


def _mapping_angle_msg_name(angle_msg_map, angle_key):
    if angle_key in angle_msg_map:
        return angle_msg_map[angle_key]
    legacy = {"pitch": "y_pitch", "roll": "x_roll"}
    if legacy[angle_key] in angle_msg_map:
        return angle_msg_map[legacy[angle_key]]
    raise KeyError("Missing angle msg mapping for '%s'." % angle_key)


def _collect_required_msg_folders(column_map):
    required = set()
    for group in ("accel", "gyro"):
        required.update(column_map[group]["msg"].values())
    angle_msg = column_map["angle"]["msg"]
    required.add(_mapping_angle_msg_name(angle_msg, "pitch"))
    required.add(_mapping_angle_msg_name(angle_msg, "roll"))
    return sorted(required)


def _resolve_run_name(sensor_dir, required_msgs, run_name):
    def parquet_names(msg_folder):
        folder = os.path.join(sensor_dir, msg_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError("Missing message folder: %s" % folder)
        return {
            name for name in os.listdir(folder) if name.lower().endswith(".parquet")
        }

    if not run_name.lower().endswith(".parquet"):
        run_name = run_name + ".parquet"

    for msg, available_runs in zip(required_msgs, map(parquet_names, required_msgs)):
        if run_name not in available_runs:
            raise FileNotFoundError(
                "Run file '%s' is missing in folder '%s'." % (run_name, msg)
            )
    return run_name


def _resolve_mapping_file(sensor_dir, sensor_folder_name):
    default_mapping = os.path.join(sensor_dir, sensor_folder_name + ".yaml")
    if os.path.isfile(default_mapping):
        return default_mapping

    yaml_files = [
        os.path.join(sensor_dir, name)
        for name in os.listdir(sensor_dir)
        if name.lower().endswith((".yaml", ".yml"))
    ]
    if len(yaml_files) == 1:
        return yaml_files[0]
    if not yaml_files:
        raise FileNotFoundError(
            "No YAML mapping file found in sensor folder '%s'. Expected '%s.yaml' or one YAML file."
            % (sensor_dir, sensor_folder_name)
        )
    raise ValueError(
        "Multiple YAML mapping files found in sensor folder '%s'. Keep one YAML file in this folder."
        % sensor_dir
    )


def _read_msg_frames(sensor_dir, msg_folders, run_name):
    return {
        msg: pd.read_parquet(os.path.join(sensor_dir, msg, run_name))
        for msg in msg_folders
    }


def _interp_to_target(source_t, source_v, target_t):
    idx = np.argsort(source_t)
    sorted_t = source_t[idx]
    sorted_v = source_v[idx]
    unique_t, unique_idx = np.unique(sorted_t, return_index=True)
    return np.interp(target_t, unique_t, sorted_v[unique_idx])


def _datetime_int_scale_seconds(dtype_obj):
    dtype_str = str(dtype_obj)
    if "datetime64[" not in dtype_str:
        return 1.0
    unit = dtype_str.split("datetime64[", 1)[1].split("]", 1)[0].split(",", 1)[0].strip()
    return {"ns": 1e-9, "us": 1e-6, "ms": 1e-3, "s": 1.0}.get(unit, 1e-9)


def _time_series_to_seconds(time_series):
    if pd.api.types.is_datetime64_any_dtype(time_series):
        ticks = time_series.astype("int64").to_numpy()
        return ticks.astype(float) * _datetime_int_scale_seconds(time_series.dtype)

    if pd.api.types.is_object_dtype(time_series):
        datetime_series = pd.to_datetime(time_series, errors="coerce", utc=True)
        if datetime_series.notna().all():
            ticks = datetime_series.astype("int64").to_numpy()
            return ticks.astype(float) * _datetime_int_scale_seconds(datetime_series.dtype)
        numeric = pd.to_numeric(time_series, errors="coerce")
        if np.isnan(numeric.to_numpy()).any():
            raise ValueError("Time column contains values that are neither datetime nor numeric.")
        values = numeric.to_numpy(dtype=float)
    else:
        values = pd.to_numeric(time_series, errors="coerce").to_numpy(dtype=float)
        if np.isnan(values).any():
            raise ValueError("Time column contains non-numeric values.")

    intervals = np.diff(values)
    intervals = intervals[intervals > 0.0]
    if intervals.size == 0:
        return values
    median_interval = np.median(intervals)
    if median_interval > 1e6:
        return values * 1e-9
    if median_interval > 1e3:
        return values * 1e-6
    if median_interval > 1.0:
        return values * 1e-3
    return values


def _extract_series(frames, msg_name, signal_name, time_col, target_t):
    frame = frames[msg_name]
    if signal_name not in frame.columns:
        raise KeyError("Missing signal column '%s' in msg '%s'." % (signal_name, msg_name))

    values = frame[signal_name].to_numpy(dtype=float)
    if time_col in frame.columns:
        source_t = _time_series_to_seconds(frame[time_col])
        return _interp_to_target(source_t, values, target_t)
    if len(values) != len(target_t):
        raise ValueError(
            "Column '%s' in msg '%s' has no time column '%s' and length differs from target timeline."
            % (signal_name, msg_name, time_col)
        )
    return values


def _input_scale(cfg, name):
    scale = float(cfg.get("input_scale", {}).get(name, 1.0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("input_scale.%s must be a finite value greater than zero." % name)
    return scale


def _baseline_angle_sign(cfg, axis):
    sign = float(cfg.get("baseline_angle_sign", {}).get(axis, 1.0))
    if sign not in (-1.0, 1.0):
        raise ValueError("baseline_angle_sign.%s must be either -1 or 1." % axis)
    return sign


def _build_data_arrays(frames, column_map, accel_to_mps2, gyro_to_dps):
    time_col = column_map["time"]
    master_msg = column_map["accel"]["msg"]["x"]
    if time_col not in frames[master_msg].columns:
        raise KeyError("Time column '%s' not found in master msg '%s'." % (time_col, master_msg))

    absolute_time_s = _time_series_to_seconds(frames[master_msg][time_col])
    time_s = absolute_time_s - absolute_time_s[0]
    accel_mps2 = np.zeros((len(time_s), 3))
    gyro_dps = np.zeros((len(time_s), 3))

    for index, axis in enumerate(("x", "y", "z")):
        accel_mps2[:, index] = _extract_series(
            frames,
            column_map["accel"]["msg"][axis],
            column_map["accel"]["signal"][axis],
            time_col,
            absolute_time_s,
        )
        gyro_dps[:, index] = _extract_series(
            frames,
            column_map["gyro"]["msg"][axis],
            column_map["gyro"]["signal"][axis],
            time_col,
            absolute_time_s,
        )

    accel_mps2 *= accel_to_mps2
    gyro_dps *= gyro_to_dps
    angle_msg_map = column_map["angle"]["msg"]
    baseline_pitch_deg = _extract_series(
        frames,
        _mapping_angle_msg_name(angle_msg_map, "pitch"),
        column_map["angle"]["signal"]["pitch"],
        time_col,
        absolute_time_s,
    )
    baseline_roll_deg = _extract_series(
        frames,
        _mapping_angle_msg_name(angle_msg_map, "roll"),
        column_map["angle"]["signal"]["roll"],
        time_col,
        absolute_time_s,
    )
    return time_s, accel_mps2, gyro_dps, baseline_pitch_deg, baseline_roll_deg


def load_sensor_run(root_path, sensor_folder, test_file):
    """Load one sensor run using its YAML mapping and normalize its units."""
    sensor_dir = os.path.join(root_path, sensor_folder)
    if not os.path.isdir(sensor_dir):
        raise FileNotFoundError("Sensor folder not found: %s" % sensor_dir)

    cfg = _load_mapping(_resolve_mapping_file(sensor_dir, sensor_folder))
    column_map = cfg["parquet_column_map"]
    run_name = _resolve_run_name(
        sensor_dir, _collect_required_msg_folders(column_map), test_file
    )
    frames = _read_msg_frames(
        sensor_dir, _collect_required_msg_folders(column_map), run_name
    )
    time_s, accel_mps2, gyro_dps, baseline_pitch_deg, baseline_roll_deg = _build_data_arrays(
        frames,
        column_map,
        _input_scale(cfg, "accel_to_mps2"),
        _input_scale(cfg, "gyro_to_dps"),
    )
    baseline_pitch_deg *= _baseline_angle_sign(cfg, "pitch")
    baseline_roll_deg *= _baseline_angle_sign(cfg, "roll")
    roll_offset_deg = float(cfg.get("mahony_output", {}).get("roll_offset_deg", 0.0))

    return ParquetSensorRun(
        run_name=run_name,
        time_s=time_s,
        accel_mps2=accel_mps2,
        gyro_radps=np.deg2rad(gyro_dps),
        gyro_dps=gyro_dps,
        baseline_pitch_deg=baseline_pitch_deg,
        baseline_roll_deg=baseline_roll_deg,
        mahony_roll_offset_deg=roll_offset_deg,
    )
