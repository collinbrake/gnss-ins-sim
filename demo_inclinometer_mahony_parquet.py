# -*- coding: utf-8 -*-
# Filename: demo_inclinometer_mahony_parquet.py

"""
Run Mahony inclinometer using recorded parquet files described by a YAML mapping.

This parser assumes standardized source units:
- time in seconds
- accel in m/s^2
- gyro in deg/s
- pitch/roll in degrees
Only the filter-required gyro conversion (deg/s -> rad/s) is applied.
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml

from demo_algorithms import inclinometer_mahony
from gnss_ins_sim.attitude import attitude


def _load_mapping(mapping_file):
    with open(mapping_file, "r") as f:
        cfg = yaml.safe_load(f)
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


def _collect_required_msg_folders(col_map):
    required = set()
    for group in ("accel", "gyro"):
        required.update(col_map[group]["msg"].values())
    angle_msg = col_map["angle"]["msg"]
    required.add(_mapping_angle_msg_name(angle_msg, "pitch"))
    required.add(_mapping_angle_msg_name(angle_msg, "roll"))
    return sorted(required)


def _resolve_run_name(sensor_dir, required_msgs, run_name=None):
    def parquet_names(msg_folder):
        folder = os.path.join(sensor_dir, msg_folder)
        if not os.path.isdir(folder):
            raise FileNotFoundError("Missing message folder: %s" % folder)
        names = set()
        for name in os.listdir(folder):
            if name.lower().endswith(".parquet"):
                names.add(name)
        return names

    if run_name is not None and not run_name.lower().endswith(".parquet"):
        run_name = run_name + ".parquet"

    run_sets = [parquet_names(msg) for msg in required_msgs]

    if run_name is not None:
        for msg, run_set in zip(required_msgs, run_sets):
            if run_name not in run_set:
                raise FileNotFoundError(
                    "Run file '%s' is missing in folder '%s'." % (run_name, msg)
                )
        return run_name

    common = run_sets[0]
    for run_set in run_sets[1:]:
        common = common.intersection(run_set)

    if not common:
        raise FileNotFoundError("No common parquet run file across required message folders.")

    run_name = sorted(common)[0]
    if len(common) > 1:
        print(
            "Found multiple common run files. Using '%s'. Pass --run to override."
            % run_name
        )
    return run_name


def _resolve_mapping_file(sensor_dir, sensor_folder_name):
    default_mapping = os.path.join(sensor_dir, sensor_folder_name + ".yaml")
    if os.path.isfile(default_mapping):
        return default_mapping

    yaml_files = []
    for name in os.listdir(sensor_dir):
        low = name.lower()
        if low.endswith(".yaml") or low.endswith(".yml"):
            yaml_files.append(os.path.join(sensor_dir, name))

    if len(yaml_files) == 1:
        return yaml_files[0]

    if len(yaml_files) == 0:
        raise FileNotFoundError(
            "No YAML mapping file found in sensor folder '%s'. Expected '%s.yaml' or one YAML file."
            % (sensor_dir, sensor_folder_name)
        )

    raise ValueError(
        "Multiple YAML mapping files found in sensor folder '%s'. Keep one YAML file in this folder."
        % sensor_dir
    )


def _read_msg_frames(sensor_dir, msg_folders, run_name):
    frames = {}
    for msg in msg_folders:
        path = os.path.join(sensor_dir, msg, run_name)
        frames[msg] = pd.read_parquet(path)
    return frames


def _interp_to_target(source_t, source_v, target_t):
    idx = np.argsort(source_t)
    sorted_t = source_t[idx]
    sorted_v = source_v[idx]
    unique_t, unique_idx = np.unique(sorted_t, return_index=True)
    unique_v = sorted_v[unique_idx]
    return np.interp(target_t, unique_t, unique_v)


def _datetime_int_scale_seconds(dtype_obj):
    # dtype examples: datetime64[us, UTC], datetime64[ns], datetime64[ms, UTC]
    dtype_str = str(dtype_obj)
    if "datetime64[" not in dtype_str:
        return 1.0
    inside = dtype_str.split("datetime64[", 1)[1].split("]", 1)[0]
    unit = inside.split(",", 1)[0].strip()
    if unit == "ns":
        return 1e-9
    if unit == "us":
        return 1e-6
    if unit == "ms":
        return 1e-3
    if unit == "s":
        return 1.0
    # Fallback: nanoseconds convention
    return 1e-9


def _time_series_to_seconds(time_series):
    # Datetime path: convert to unix seconds.
    if pd.api.types.is_datetime64_any_dtype(time_series):
        ticks = time_series.astype("int64").to_numpy()
        scale = _datetime_int_scale_seconds(time_series.dtype)
        return ticks.astype(float) * scale

    # Object path: try datetime parsing first, then numeric.
    if pd.api.types.is_object_dtype(time_series):
        dt = pd.to_datetime(time_series, errors="coerce", utc=True)
        if dt.notna().all():
            ticks = dt.astype("int64").to_numpy()
            scale = _datetime_int_scale_seconds(dt.dtype)
            return ticks.astype(float) * scale
        numeric = pd.to_numeric(time_series, errors="coerce")
        if np.isnan(numeric.to_numpy()).any():
            raise ValueError("Time column contains values that are neither datetime nor numeric.")
        arr = numeric.to_numpy(dtype=float)
    else:
        arr = pd.to_numeric(time_series, errors="coerce").to_numpy(dtype=float)
        if np.isnan(arr).any():
            raise ValueError("Time column contains non-numeric values.")

    # Numeric path: infer scale if likely epoch-style units.
    dt = np.diff(arr)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        return arr
    dt_med = np.median(dt)
    if dt_med > 1e6:
        return arr * 1e-9   # ns -> s
    if dt_med > 1e3:
        return arr * 1e-6   # us -> s
    if dt_med > 1.0:
        return arr * 1e-3   # ms -> s
    return arr              # already seconds


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


def _build_data_arrays(frames, col_map):
    time_col = col_map["time"]
    master_msg = col_map["accel"]["msg"]["x"]
    if time_col not in frames[master_msg].columns:
        raise KeyError("Time column '%s' not found in master msg '%s'." % (time_col, master_msg))

    t_abs = _time_series_to_seconds(frames[master_msg][time_col])
    t = t_abs - t_abs[0]

    accel = np.zeros((len(t), 3))
    gyro_deg = np.zeros((len(t), 3))

    axis_order = ("x", "y", "z")
    for i, axis in enumerate(axis_order):
        a_msg = col_map["accel"]["msg"][axis]
        a_sig = col_map["accel"]["signal"][axis]
        g_msg = col_map["gyro"]["msg"][axis]
        g_sig = col_map["gyro"]["signal"][axis]
        accel[:, i] = _extract_series(frames, a_msg, a_sig, time_col, t_abs)
        gyro_deg[:, i] = _extract_series(frames, g_msg, g_sig, time_col, t_abs)

    # Source accel is already m/s^2. Convert source gyro deg/s to rad/s for Mahony.
    gyro_rad = np.deg2rad(gyro_deg)

    angle_msg_map = col_map["angle"]["msg"]
    pitch_msg = _mapping_angle_msg_name(angle_msg_map, "pitch")
    roll_msg = _mapping_angle_msg_name(angle_msg_map, "roll")
    pitch_sig = col_map["angle"]["signal"]["pitch"]
    roll_sig = col_map["angle"]["signal"]["roll"]

    ref_pitch = _extract_series(frames, pitch_msg, pitch_sig, time_col, t_abs)
    ref_roll = _extract_series(frames, roll_msg, roll_sig, time_col, t_abs)

    return t, accel, gyro_rad, gyro_deg, ref_pitch, ref_roll


def _estimate_fs(time_s):
    dt = np.diff(time_s)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Non-increasing time base. Cannot estimate sample frequency.")
    return 1.0 / np.median(dt)


def _run_mahony(fs, accel, gyro):
    algo = inclinometer_mahony.MahonyFilter()
    algo.run([fs, gyro, accel])
    quat = algo.get_results()[0]
    est_euler = np.zeros((quat.shape[0], 3))
    for i in range(quat.shape[0]):
        est_euler[i, :] = attitude.quat2euler(quat[i, :], rot_seq="zyx")
    est_pitch_deg = np.rad2deg(est_euler[:, 1])
    est_roll_deg = np.rad2deg(est_euler[:, 2])
    return est_pitch_deg, est_roll_deg


def _accel_tilt_deg(accel):
    ax = accel[:, 0]
    ay = accel[:, 1]
    az = accel[:, 2]
    pitch_deg = np.rad2deg(np.arctan2(-ax, np.sqrt(ay * ay + az * az)))
    roll_deg = np.rad2deg(np.arctan2(ay, az))
    return pitch_deg, roll_deg


def _plot_compare(time_s, ref_pitch, ref_roll, est_pitch, est_roll, run_name):
    pitch_err = est_pitch - ref_pitch
    roll_err = est_roll - ref_roll

    print("Run file: %s" % run_name)
    print("Pitch error RMS (deg): %.4f" % np.sqrt(np.mean(pitch_err * pitch_err)))
    print("Roll error RMS (deg): %.4f" % np.sqrt(np.mean(roll_err * roll_err)))

    fig, axes = plt.subplots(3, 1, sharex=True, num="Parquet Mahony Comparison")

    axes[0].plot(time_s, ref_pitch, label="ref pitch")
    axes[0].plot(time_s, est_pitch, "--", label="mahony pitch")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_s, ref_roll, label="ref roll")
    axes[1].plot(time_s, est_roll, "--", label="mahony roll")
    axes[1].set_ylabel("Roll (deg)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time_s, pitch_err, label="pitch error")
    axes[2].plot(time_s, roll_err, label="roll error")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Error (deg)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def _plot_baseline_with_gyro(time_s, ref_pitch, ref_roll, gyro_deg, run_name):
    # Common body-frame pairing for quick sign/alignment inspection.
    # pitch <-> gyro y, roll <-> gyro x
    gyro_x = gyro_deg[:, 0]
    gyro_y = gyro_deg[:, 1]

    print("Run file: %s" % run_name)

    fig, axes = plt.subplots(2, 1, sharex=True, num="Baseline Angle and Gyro")

    ax_pitch = axes[0]
    ax_pitch_gyro = ax_pitch.twinx()
    line_pitch, = ax_pitch.plot(time_s, ref_pitch, color="C0", label="ref pitch")
    line_pitch_gyro, = ax_pitch_gyro.plot(
        time_s, gyro_y, color="C1", linestyle="--", label="gyro y"
    )
    ax_pitch.set_ylabel("Pitch (deg)")
    ax_pitch_gyro.set_ylabel("Gyro Y (deg/s)")
    ax_pitch.grid(True)
    ax_pitch.legend([line_pitch, line_pitch_gyro], ["ref pitch", "gyro y"], loc="upper right")

    ax_roll = axes[1]
    ax_roll_gyro = ax_roll.twinx()
    line_roll, = ax_roll.plot(time_s, ref_roll, color="C0", label="ref roll")
    line_roll_gyro, = ax_roll_gyro.plot(
        time_s, gyro_x, color="C1", linestyle="--", label="gyro x"
    )
    ax_roll.set_xlabel("Time (s)")
    ax_roll.set_ylabel("Roll (deg)")
    ax_roll_gyro.set_ylabel("Gyro X (deg/s)")
    ax_roll.grid(True)
    ax_roll.legend([line_roll, line_roll_gyro], ["ref roll", "gyro x"], loc="upper right")

    plt.tight_layout()
    plt.show()


def _plot_baseline_with_accel_tilt(time_s, ref_pitch, ref_roll, accel, run_name):
    pitch_accel, roll_accel = _accel_tilt_deg(accel)

    print("Run file: %s" % run_name)

    fig, axes = plt.subplots(2, 1, sharex=True, num="Baseline vs Accel-Only Tilt")

    axes[0].plot(time_s, pitch_accel, color="gold", label="pitch from accel")
    axes[0].plot(time_s, ref_pitch, color="C0", label="baseline pitch")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_s, roll_accel, color="gold", label="roll from accel")
    axes[1].plot(time_s, ref_roll, color="C0", label="baseline roll")
    axes[1].set_xlabel("Time (s)")
    axes[1].set_ylabel("Roll (deg)")
    axes[1].grid(True)
    axes[1].legend()

    plt.tight_layout()
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Run Mahony inclinometer from parquet message folders using YAML mapping."
    )
    parser.add_argument(
        "--path",
        required=True,
        help="Root path containing sensor folders.",
    )
    parser.add_argument(
        "--sensor-folder",
        required=True,
        help="Sensor folder name under --path (e.g., sensor1).",
    )
    parser.add_argument(
        "--test-file",
        required=True,
        help="Parquet run filename (with or without .parquet).",
    )
    parser.add_argument(
        "--plot-mode",
        choices=["compare", "baseline-gyro", "baseline-accel"],
        default="compare",
        help="compare: baseline vs Mahony + error. baseline-gyro: baseline angle with direct gyro overlay. baseline-accel: baseline vs accelerometer-only tilt.",
    )
    args = parser.parse_args()

    sensor_dir = os.path.join(args.path, args.sensor_folder)
    if not os.path.isdir(sensor_dir):
        raise FileNotFoundError("Sensor folder not found: %s" % sensor_dir)

    mapping_file = _resolve_mapping_file(sensor_dir, args.sensor_folder)

    cfg = _load_mapping(mapping_file)
    col_map = cfg["parquet_column_map"]

    required_msgs = _collect_required_msg_folders(col_map)
    run_name = _resolve_run_name(sensor_dir, required_msgs, args.test_file)
    frames = _read_msg_frames(sensor_dir, required_msgs, run_name)

    time_s, accel, gyro_rad, gyro_deg, ref_pitch, ref_roll = _build_data_arrays(frames, col_map)
    if args.plot_mode == "baseline-gyro":
        _plot_baseline_with_gyro(time_s, ref_pitch, ref_roll, gyro_deg, run_name)
    elif args.plot_mode == "baseline-accel":
        _plot_baseline_with_accel_tilt(time_s, ref_pitch, ref_roll, accel, run_name)
    else:
        fs = _estimate_fs(time_s)
        est_pitch, est_roll = _run_mahony(fs, accel, gyro_rad)
        _plot_compare(time_s, ref_pitch, ref_roll, est_pitch, est_roll, run_name)


if __name__ == "__main__":
    main()
