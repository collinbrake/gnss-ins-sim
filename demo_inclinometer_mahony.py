# -*- coding: utf-8 -*-
# Filename: demo_inclinometer_mahony.py

"""
An inclinometer demo.
Created on 2018-01-23
@author: dongxiaoguang
"""

import os
import math
import numpy as np
import matplotlib.pyplot as plt
from gnss_ins_sim.sim import imu_model
from gnss_ins_sim.sim import ins_sim

# globals
D2R = math.pi/180

motion_def_path = os.path.abspath('.//demo_motion_def_files//')
fs = 100.0          # IMU sample frequency

def test_inclinometer_mahony():
    '''
    test Sim with a inclinometer algorithm based on Mahony's theory.
    '''
    #### choose a built-in IMU model, typical for IMU380
    imu_err = {'gyro_b': np.array([5.0, -5.0, 2.5]) * 3600.0,
               'gyro_arw': np.array([0.25, 0.25, 0.25]) * 1.0,
               'gyro_b_stability': np.array([3.5, 3.5, 3.5]) * 1.0,
               'gyro_b_corr': np.array([100.0, 100.0, 100.0]),
               'accel_b': np.array([50.0e-3, 50.0e-3, 50.0e-3]) * 0.0,
               'accel_vrw': np.array([0.03119, 0.03009, 0.04779]) * 1.0,
               'accel_b_stability': np.array([4.29e-5, 5.72e-5, 8.02e-5]) * 1.0,
               'accel_b_corr': np.array([200.0, 200.0, 200.0]),
               'mag_std': np.array([0.2, 0.2, 0.2]) * 1.0
              }
    imu_err = 'high-accuracy'
    # do not generate GPS and magnetometer data
    imu = imu_model.IMU(accuracy=imu_err, axis=6, gps=False)

    #### Algorithm
    # ECF based inclinometer
    from demo_algorithms import inclinometer_mahony
    algo = inclinometer_mahony.MahonyFilter()

    #### start simulation
    sim = ins_sim.Sim([fs, 0.0, 0.0],
                      motion_def_path+"//motion_def.csv",
                      ref_frame=1,
                      imu=imu,
                      mode=None,
                      env=None,#'[0.01 0.01 0.11]g-random',
                      algorithm=algo)
    sim.run()
    # generate simulation results, summary, and save data to files
    sim.results()  # do not save data

    # Plot true (simulated) vs estimated pitch/roll for the first algo/run.
    time, ref_att_euler, att_euler = sim.get_data(['time', 'ref_att_euler', 'att_euler'])
    first_key = next(iter(att_euler))
    est_att = att_euler[first_key]

    fig, axes = plt.subplots(2, 1, sharex=True, num='Pitch/Roll: Truth vs Mahony')
    axes[0].plot(time, np.rad2deg(ref_att_euler[:, 1]), label='true pitch')
    axes[0].plot(time, np.rad2deg(est_att[:, 1]), '--', label='filter pitch')
    axes[0].set_ylabel('Pitch (deg)')
    axes[0].grid(True)
    axes[0].legend()

    axes[1].plot(time, np.rad2deg(ref_att_euler[:, 2]), label='true roll')
    axes[1].plot(time, np.rad2deg(est_att[:, 2]), '--', label='filter roll')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Roll (deg)')
    axes[1].grid(True)
    axes[1].legend()

    # plot data
    sim.plot(['att_euler', 'wb', 'ab'], opt={'att_euler': 'error'})

if __name__ == '__main__':
    test_inclinometer_mahony()
