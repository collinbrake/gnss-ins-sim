# -*- coding: utf-8 -*-
# Filename: demo_inclinometer_mahony_parquet.py

"""Run the Mahony inclinometer against recorded Parquet IMU data."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from demo_algorithms import inclinometer_mahony
from demo_algorithms import inclinometer_mahony_approx
from gnss_ins_sim.attitude import attitude
from gnss_ins_sim.parquet import load_sensor_run


def _estimate_fs(time_s):
    dt = np.diff(time_s)
    dt = dt[dt > 0.0]
    if dt.size == 0:
        raise ValueError("Non-increasing time base. Cannot estimate sample frequency.")
    return 1.0 / np.median(dt)


def _run_mahony(fs, accel, gyro, roll_offset_deg=0.0, approx_gains=None, gyro_only=False):
    if approx_gains is None:
        algo = inclinometer_mahony.MahonyFilter()
        if gyro_only:
            algo.kp_acc_high = 0.0
            algo.ki_acc_high = 0.0
            algo.kp_acc_low = 0.0
            algo.ki_acc_low = 0.0
    else:
        ki, kp = approx_gains
        if gyro_only:
            ki = 0.0
            kp = 0.0
        algo = inclinometer_mahony_approx.MahonyFilter(kp_acc=kp, ki_acc=ki)
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


def _accel_orientation(accel):
    ax = accel[:, 0]
    ay = accel[:, 1]
    az = accel[:, 2]
    pitch_deg = np.rad2deg(np.arctan2(ax, np.sqrt(ay * ay + az * az)))
    roll_deg = np.rad2deg(np.arctan2(ay, az))
    return pitch_deg, roll_deg


def _plot_bode(magnitude_axis, phase_axis, frequency_hz, response, title):
    magnitude_axis.semilogx(frequency_hz, 20.0 * np.log10(np.maximum(np.abs(response), 1e-15)))
    magnitude_axis.set_title(title)
    magnitude_axis.set_ylabel("Magnitude (dB)")
    magnitude_axis.grid(True, which="both")

    phase_axis.semilogx(frequency_hz, np.rad2deg(np.unwrap(np.angle(response))))
    phase_axis.set_xlabel("Frequency (Hz)")
    phase_axis.set_ylabel("Phase (deg)")
    phase_axis.grid(True, which="both")


def _add_locus_arrows(axis, poles, loop_gains, color):
    for target_gain in (0.1, 2.0):
        index = int(np.argmin(np.abs(loop_gains - target_gain)))
        for branch in range(2):
            start = poles[max(index - 12, 0), branch]
            end = poles[index, branch]
            axis.annotate(
                "",
                xy=(end.real, end.imag),
                xytext=(start.real, start.imag),
                arrowprops={"arrowstyle": "->", "color": color, "linewidth": 1.5},
            )


def _plot_approx_control_analysis(ki, kp):
    """Plot the fixed-PI small-angle model used by the approximate filter."""
    frequency_scale = max(np.sqrt(ki), kp, 1.0)
    frequency_radps = np.logspace(
        np.log10(frequency_scale) - 3.0,
        np.log10(frequency_scale) + 3.0,
        600,
    )
    frequency_hz = frequency_radps / (2.0 * np.pi)
    s = 1j * frequency_radps
    denominator = s * s + kp * s + ki
    accel_response = (kp * s + ki) / denominator
    gyro_angle_response = s * s / denominator

    fig = plt.figure(num="Approximate Mahony Control Analysis", figsize=(11, 9))
    grid = fig.add_gridspec(3, 2, height_ratios=[1.15, 1.0, 1.0])
    root_axis = fig.add_subplot(grid[0, :])
    accel_magnitude_axis = fig.add_subplot(grid[1, 0])
    accel_phase_axis = fig.add_subplot(grid[2, 0], sharex=accel_magnitude_axis)
    gyro_magnitude_axis = fig.add_subplot(grid[1, 1])
    gyro_phase_axis = fig.add_subplot(grid[2, 1], sharex=gyro_magnitude_axis)

    loop_gains = np.concatenate(([0.0], np.logspace(-4, 3, 600)))
    discriminant = (loop_gains * kp) ** 2 - 4.0 * loop_gains * ki
    discriminant_root = np.lib.scimath.sqrt(discriminant)
    poles = np.column_stack((
        (-loop_gains * kp + discriminant_root) / 2.0,
        (-loop_gains * kp - discriminant_root) / 2.0,
    ))
    root_axis.plot(poles[:, 0].real, poles[:, 0].imag, color="C0", label="root locus")
    root_axis.plot(poles[:, 1].real, poles[:, 1].imag, color="C0")
    _add_locus_arrows(root_axis, poles, loop_gains, "C0")
    selected_poles = np.roots([1.0, kp, ki])
    root_axis.plot(
        selected_poles.real,
        selected_poles.imag,
        "x",
        color="C3",
        markersize=8,
        markeredgewidth=2,
        label="selected gains (K=1)",
    )
    root_axis.plot(0.0, 0.0, "x", color="black", markersize=8, markeredgewidth=2, label="2 poles")
    root_axis.annotate("two open-loop poles", xy=(0.0, 0.0), xytext=(8, 8), textcoords="offset points")
    zero_location = None
    if kp > 0.0:
        zero_location = -ki / kp
        root_axis.plot(zero_location, 0.0, "o", color="black", fillstyle="none", markersize=8, label="PI zero")
        root_axis.annotate("PI zero", xy=(zero_location, 0.0), xytext=(8, -14), textcoords="offset points")
    root_axis.axhline(0.0, color="black", linewidth=0.8)
    root_axis.axvline(0.0, color="black", linewidth=0.8)
    root_axis.set_title("Root Locus: $s^2 + K(K_p s + K_i) = 0$")
    root_axis.set_xlabel("Real axis (rad/s)")
    root_axis.set_ylabel("Imaginary axis (rad/s)")
    view_scale = max(np.max(np.abs(selected_poles)), abs(zero_location or 0.0), 0.1)
    root_axis.set_xlim(-3.0 * view_scale, 0.5 * view_scale)
    root_axis.set_ylim(-2.0 * view_scale, 2.0 * view_scale)
    root_axis.grid(True)
    root_axis.legend()
    root_axis.text(
        0.02,
        0.97,
        "$K=1:\\quad s^2 + %.4g s + %.4g = 0$" % (kp, ki),
        transform=root_axis.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "square", "facecolor": "white", "alpha": 0.9},
    )

    _plot_bode(
        accel_magnitude_axis,
        accel_phase_axis,
        frequency_hz,
        accel_response,
        "Accel Path: $(K_p s + K_i)/(s^2 + K_p s + K_i)$",
    )
    _plot_bode(
        gyro_magnitude_axis,
        gyro_phase_axis,
        frequency_hz,
        gyro_angle_response,
        "Integrated Gyro Path: $s^2/(s^2 + K_p s + K_i)$",
    )
    fig.tight_layout()


def _plot_compare(
    time_s, ref_pitch, ref_roll, est_pitch, est_roll, run_name,
    mode="mahony", accel=None, gyro_rad=None):
    pitch_err = est_pitch - ref_pitch
    roll_err = est_roll - ref_roll

    print("Run file: %s" % run_name)
    print("Pitch error RMS (deg): %.4f" % np.sqrt(np.mean(pitch_err * pitch_err)))
    print("Roll error RMS (deg): %.4f" % np.sqrt(np.mean(roll_err * roll_err)))

    diagnostic_signal = None
    diagnostic_name = None
    diagnostic_unit = None
    if mode == "gyro":
        if gyro_rad is None:
            raise ValueError("gyro_rad is required when mode is 'gyro'.")
        diagnostic_signal = gyro_rad
        diagnostic_name = "Gyro"
        diagnostic_unit = "rad/s"
    elif mode == "accel":
        if accel is None:
            raise ValueError("accel is required when mode is 'accel'.")
        diagnostic_signal = accel
        diagnostic_name = "Accel"
        diagnostic_unit = "m/s^2"

    plot_count = 5 if diagnostic_signal is not None else 3
    fig, axes = plt.subplots(plot_count, 1, sharex=True, num="Parquet Mahony Comparison")
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

    if diagnostic_signal is not None:
        for axis_index, axis_name in enumerate(("x", "y", "z")):
            diagnostic_axis = axes[axis_index + 2]
            diagnostic_axis.plot(time_s, diagnostic_signal[:, axis_index])
            diagnostic_axis.set_ylabel(
                "%s %s\n(%s)" % (diagnostic_name, axis_name, diagnostic_unit)
            )
            diagnostic_axis.grid(True)
    else:
        axes[2].plot(time_s, pitch_err, label="pitch error")
        axes[2].plot(time_s, roll_err, label="roll error")
        axes[2].set_ylabel("Error (deg)")
        axes[2].grid(True)
        axes[2].legend()
    axes[-1].set_xlabel("Time (s)")

    plt.tight_layout()
    plt.show()


def _apply_time_window(time_s, accel, gyro_rad, ref_pitch, ref_roll, start_s=None, end_s=None):
    if start_s is None and end_s is None:
        return time_s, accel, gyro_rad, ref_pitch, ref_roll
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
        "--approx",
        nargs=2,
        type=float,
        metavar=("KI", "KP"),
        default=None,
        help="Use fixed-gain PI tuning with integral and proportional gains, in KI KP order.",
    )
    parser.add_argument("--mode", choices=["mahony", "gyro", "accel"], default="mahony")
    args = parser.parse_args()

    if args.approx is not None:
        ki, kp = args.approx
        if kp < 0.0 or ki < 0.0:
            parser.error("--approx KI KP values must be greater than or equal to zero.")
        natural_frequency = np.sqrt(ki)
        damping_ratio = kp / (2.0 * natural_frequency) if natural_frequency > 0.0 else np.inf
        print("Approximate PI tuning: Ki=%.6g, Kp=%.6g" % (ki, kp))
        print("Approximate wn=%.6g rad/s, zeta=%.6g" % (natural_frequency, damping_ratio))

    run = load_sensor_run(args.path, args.sensor_folder, args.test_file)
    time_s, accel, gyro_rad, ref_pitch, ref_roll = _apply_time_window(
        run.time_s,
        run.accel_mps2,
        run.gyro_radps,
        run.baseline_pitch_deg,
        run.baseline_roll_deg,
        start_s=args.start_s,
        end_s=args.end_s,
    )
    fs = _estimate_fs(time_s)
    if args.approx is not None and args.mode != "gyro":
        _plot_approx_control_analysis(*args.approx)
    if args.mode == "accel" and args.approx is None:
        est_pitch, est_roll = _accel_orientation(accel)
    else:
        gyro_input = np.zeros_like(gyro_rad) if args.mode == "accel" else gyro_rad
        est_pitch, est_roll = _run_mahony(
            fs,
            accel,
            gyro_input,
            run.mahony_roll_offset_deg,
            approx_gains=args.approx,
            gyro_only=args.mode == "gyro",
        )
    _plot_compare(
        time_s,
        ref_pitch,
        ref_roll,
        est_pitch,
        est_roll,
        run.run_name,
        mode=args.mode,
        accel=accel,
        gyro_rad=gyro_rad,
    )


if __name__ == "__main__":
    main()
