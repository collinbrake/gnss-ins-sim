# -*- coding: utf-8 -*-
# Filename: demo_inclinometer_mahony_parquet.py

"""Run the Mahony inclinometer against recorded Parquet IMU data."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from demo_algorithms import inclinometer_mahony
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.parquet import load_sensor_run


def _estimate_fs(time_s):
    dt = np.diff(time_s)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Non-increasing time base. Cannot estimate sample frequency.")
    return 1.0 / np.median(dt)


def _run_mahony(fs, accel, gyro, roll_offset_deg=0.0):
    algo = inclinometer_mahony.MahonyFilter()
    algo.run([fs, gyro, accel])
    quat = algo.get_results()[0]
    est_euler = np.zeros((quat.shape[0], 3))
    for i in range(quat.shape[0]):
        est_euler[i, :] = attitude.quat2euler(quat[i, :], rot_seq="zyx")
    est_pitch_deg = np.rad2deg(est_euler[:, 1])
    est_roll_deg = np.rad2deg(est_euler[:, 2]) + roll_offset_deg
    est_roll_deg = np.rad2deg(
        np.arctan2(np.sin(np.deg2rad(est_roll_deg)), np.cos(np.deg2rad(est_roll_deg)))
    )
    return est_pitch_deg, est_roll_deg


def _accel_tilt_deg(accel):
    ax = accel[:, 0]
    ay = accel[:, 1]
    az = accel[:, 2]
    pitch_deg = np.rad2deg(np.arctan2(ax, np.sqrt(ay * ay + az * az)))
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
    print("Run file: %s" % run_name)
    fig, axes = plt.subplots(3, 1, sharex=True, num="Baseline Angle and Gyro")

    ax_pitch = axes[0]
    ax_pitch_gyro = ax_pitch.twinx()
    line_pitch, = ax_pitch.plot(time_s, ref_pitch, color="C0", label="ref pitch")
    line_pitch_gyro, = ax_pitch_gyro.plot(
        time_s, gyro_deg[:, 1], color="C1", linestyle="--", label="gyro y"
    )
    ax_pitch.set_ylabel("Pitch (deg)")
    ax_pitch_gyro.set_ylabel("Gyro Y (deg/s)")
    ax_pitch.grid(True)
    ax_pitch.legend([line_pitch, line_pitch_gyro], ["ref pitch", "gyro y"], loc="upper right")

    ax_roll = axes[1]
    ax_roll_gyro = ax_roll.twinx()
    line_roll, = ax_roll.plot(time_s, ref_roll, color="C0", label="ref roll")
    line_roll_gyro, = ax_roll_gyro.plot(
        time_s, gyro_deg[:, 0], color="C1", linestyle="--", label="gyro x"
    )
    ax_roll.set_ylabel("Roll (deg)")
    ax_roll_gyro.set_ylabel("Gyro X (deg/s)")
    ax_roll.grid(True)
    ax_roll.legend([line_roll, line_roll_gyro], ["ref roll", "gyro x"], loc="upper right")

    axes[2].plot(time_s, gyro_deg[:, 0], label="gyro x")
    axes[2].plot(time_s, gyro_deg[:, 1], label="gyro y")
    axes[2].plot(time_s, gyro_deg[:, 2], label="gyro z")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Gyro (deg/s)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def _plot_baseline_with_accel_tilt(time_s, ref_pitch, ref_roll, accel, run_name):
    pitch_accel, roll_accel = _accel_tilt_deg(accel)

    print("Run file: %s" % run_name)
    fig, axes = plt.subplots(3, 1, sharex=True, num="Baseline vs Accel-Only Tilt")
    axes[0].plot(time_s, pitch_accel, color="gold", label="pitch from accel")
    axes[0].plot(time_s, ref_pitch, color="C0", label="baseline pitch")
    axes[0].set_ylabel("Pitch (deg)")
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time_s, roll_accel, color="gold", label="roll from accel")
    axes[1].plot(time_s, ref_roll, color="C0", label="baseline roll")
    axes[1].set_ylabel("Roll (deg)")
    axes[1].grid(True)
    axes[1].legend()

    axes[2].plot(time_s, accel[:, 0], label="acc x")
    axes[2].plot(time_s, accel[:, 1], label="acc y")
    axes[2].plot(time_s, accel[:, 2], label="acc z")
    axes[2].set_xlabel("Time (s)")
    axes[2].set_ylabel("Accel (m/s^2)")
    axes[2].grid(True)
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def _apply_time_window(time_s, accel, gyro_rad, gyro_deg, ref_pitch, ref_roll, start_s=None, end_s=None):
    if start_s is None and end_s is None:
        return time_s, accel, gyro_rad, gyro_deg, ref_pitch, ref_roll
    if start_s is None:
        start_s = time_s[0]
    if end_s is None:
        end_s = time_s[-1]
    if end_s < start_s:
        raise ValueError("--end-s must be greater than or equal to --start-s.")

    mask = (time_s >= start_s) & (time_s <= end_s)
    if not np.any(mask):
        raise ValueError("No samples found in the requested time window.")
    selected_time_s = time_s[mask]
    selected_time_s = selected_time_s - selected_time_s[0]
    return (
        selected_time_s,
        accel[mask, :],
        gyro_rad[mask, :],
        gyro_deg[mask, :],
        ref_pitch[mask],
        ref_roll[mask],
    )


def main():
    parser = argparse.ArgumentParser(
        description="Run Mahony inclinometer from parquet message folders using YAML mapping."
    )
    parser.add_argument("--path", required=True, help="Root path containing sensor folders.")
    parser.add_argument(
        "--sensor-folder", required=True, help="Sensor folder name under --path (e.g., sensor1)."
    )
    parser.add_argument(
        "--test-file", required=True, help="Parquet run filename (with or without .parquet)."
    )
    parser.add_argument("--start-s", type=float, default=None)
    parser.add_argument("--end-s", type=float, default=None)
    parser.add_argument(
        "--plot-mode",
        choices=["compare", "baseline-gyro", "baseline-accel"],
        default="compare",
    )
    args = parser.parse_args()

    run = load_sensor_run(args.path, args.sensor_folder, args.test_file)
    time_s, accel, gyro_rad, gyro_deg, ref_pitch, ref_roll = _apply_time_window(
        run.time_s,
        run.accel_mps2,
        run.gyro_radps,
        run.gyro_dps,
        run.baseline_pitch_deg,
        run.baseline_roll_deg,
        start_s=args.start_s,
        end_s=args.end_s,
    )
    if args.plot_mode == "baseline-gyro":
        _plot_baseline_with_gyro(time_s, ref_pitch, ref_roll, gyro_deg, run.run_name)
    elif args.plot_mode == "baseline-accel":
        _plot_baseline_with_accel_tilt(time_s, ref_pitch, ref_roll, accel, run.run_name)
    else:
        fs = _estimate_fs(time_s)
        est_pitch, est_roll = _run_mahony(
            fs, accel, gyro_rad, run.mahony_roll_offset_deg
        )
        _plot_compare(time_s, ref_pitch, ref_roll, est_pitch, est_roll, run.run_name)


if __name__ == "__main__":
    main()
