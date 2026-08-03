# -*- coding: utf-8 -*-
"""Fixed-gain Mahony inclinometer for classical PI-loop tuning experiments."""

import math

import numpy as np

from gnss_ins_sim.attitude import attitude


class MahonyFilter(object):
    """Mahony IMU observer with direct, fixed proportional-integral correction.

    Near level, each tilt axis is approximated by the characteristic equation:
    s^2 + kp_acc * s + ki_acc = 0.
    """

    def __init__(self, kp_acc=1.0, ki_acc=0):
        if not np.isfinite(kp_acc) or kp_acc < 0.0:
            raise ValueError("kp_acc must be a finite value greater than or equal to zero.")
        if not np.isfinite(ki_acc) or ki_acc < 0.0:
            raise ValueError("ki_acc must be a finite value greater than or equal to zero.")

        self.input = ["fs", "gyro", "accel"]
        self.output = ["att_quat", "wb", "ab"]
        self.batch = True
        self.kp_acc = float(kp_acc)
        self.ki_acc = float(ki_acc)
        self.ini = False
        self.dt = 1.0
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.err_int = np.array([0.0, 0.0, 0.0])
        self.gyro_bias = np.array([0.0, 0.0, 0.0])
        self.tmp = np.array([0.0, 0.0, 0.0])
        self.quat = None
        self.wb = None
        self.ab = None

    def run(self, set_of_input):
        self.dt = 1.0 / set_of_input[0]
        gyro = set_of_input[1]
        accel = set_of_input[2]
        sample_count = accel.shape[0]
        self.quat = np.zeros((sample_count, 4))
        self.wb = np.zeros((sample_count, 3))
        self.ab = np.zeros((sample_count, 3))
        for index in range(sample_count):
            self.update(gyro[index, :], accel[index, :])
            self.quat[index, :] = self.q
            self.wb[index, :] = self.gyro_bias
            self.ab[index, :] = self.tmp

    def update(self, gyro, accel):
        accel_norm = math.sqrt(np.dot(accel, accel))
        if accel_norm == 0.0:
            self.q = attitude.quat_update(self.q, gyro, self.dt)
            return
        accel = accel / accel_norm

        if not self.ini:
            self._initialize_from_accel(accel)

        gravity_body = self._gravity_body()
        accel_error = np.cross(accel, gravity_body)
        self.err_int += self.ki_acc * accel_error * self.dt
        self.gyro_bias = self.kp_acc * accel_error + self.err_int
        self.tmp = accel_error
        self.q = attitude.quat_update(self.q, gyro + self.gyro_bias, self.dt)

    def _initialize_from_accel(self, accel):
        self.ini = True
        self.err_int[:] = 0.0
        if accel[0] >= 1.0:
            pseudo_mag = np.array([0.0, 0.0, 1.0])
        elif accel[1] <= -1.0:
            pseudo_mag = np.array([0.0, 0.0, -1.0])
        else:
            pseudo_mag = np.array([
                math.sqrt(1.0 - accel[0] * accel[0]),
                -accel[1] * accel[0] / math.sqrt(1.0 - accel[0] * accel[0]),
                -accel[0] * accel[2] / math.sqrt(1.0 - accel[0] * accel[0]),
            ])
        self.q = attitude.dcm2quat(attitude.get_cn2b_acc_mag_ned(accel, pseudo_mag))

    def _gravity_body(self):
        q0, q1, q2, q3 = self.q
        return np.array([
            -2.0 * (q1 * q3 - q0 * q2),
            -2.0 * (q0 * q1 + q2 * q3),
            -q0 * q0 + q1 * q1 + q2 * q2 - q3 * q3,
        ])

    def get_results(self):
        return [self.quat, self.wb, self.ab]

    def reset(self):
        self.ini = False
        self.q = np.array([1.0, 0.0, 0.0, 0.0])
        self.err_int[:] = 0.0
        self.gyro_bias[:] = 0.0
        self.tmp[:] = 0.0
