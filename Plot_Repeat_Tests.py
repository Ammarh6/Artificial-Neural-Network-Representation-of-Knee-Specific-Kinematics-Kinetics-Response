# Import Packages
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import Create_ANN_Models as ANN
from sklearn.metrics import r2_score, mean_squared_error


def main():
    # Font size for legend
    plt.rc('legend', fontsize=14)
    # Select knee to include in the figure
    knee = 'oks001'
    # Read raw data
    kinetics_raw = pd.read_hdf(
        'dat/joint_mechanics-' + str(knee) + '/joint_mechanics-' + str(knee) + '_x_hierarchical_data.h5')
    kinematics_raw = pd.read_hdf(
        'dat/joint_mechanics-' + str(knee) + '/joint_mechanics-' + str(knee) + '_y_hierarchical_data.h5')
    # Simulate real experiment by moving Flexion to Kinetics and Extension Torque to Kinematics
    kinematics = kinematics_raw.join(kinetics_raw['JCS Load Extension Torque(Nm)'])
    kinetics = kinetics_raw.join(kinematics_raw['Knee JCS Flexion(deg)'])
    del kinematics['Knee JCS Flexion(deg)']
    del kinetics['JCS Load Extension Torque(Nm)']
    # Partition the data to separate test included in raw data
    kinetics_keys, kinematics_keys = ANN.partition_data(kinetics, kinematics)
    # List of kinematics
    var_names = list(kinematics)
    ############################################################################
    ############################################################################
    # Select test to plot
    if knee == 'oks001':
        tests = '017_AP laxity_30degrees_main_processed.tdms'
    if knee == 'oks002':
        tests = '005_A-P laxity_30deg_main_processed.tdms'
    if knee == 'oks003':
        tests = '005_A-P Laxity 30deg_main_processed.tdms'
    if knee == 'oks004':
        tests = '039_AP Laxity_30deg_main_processed.tdms'
    if knee == 'oks006':
        tests = '006_A-P Laxity 30deg_main_processed.tdms'
    if knee == 'oks007':
        tests = '016_A-P Laxity_main_processed.tdms'
    if knee == 'oks008':
        tests = '221_A-P Laxity_main_processed.tdms'
    if knee == 'oks009':
        tests = '026_AP_Laxity_main_processed.tdms'
    kinetics = np.asarray(kinetics_keys[tests])
    kinematics = np.asarray(kinematics_keys[tests])
    # Initiate a figure
    fig, ax = plt.subplots(nrows=2, ncols=3)
    # Loop through kinematics
    for var in range(len(kinematics[0])):
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        PredKinmatics = model.predict(kinetics)
        # Compute R-Square value for validation set
        R2Value = r2_score(kinematics[:, var], PredKinmatics)
        mseValue = mean_squared_error(kinematics[:, var], PredKinmatics)
        # Print R^2 and RMSE
        print("R-Square=", R2Value)
        print("RMSE=", np.sqrt(mseValue))
        # Create Plots
        if var in range(0, 3):
            ax[0, var].plot(kinematics[:, var], label='E')
            ax[0, var].plot(PredKinmatics, label='NN')
            ax[0, var].legend()
            ax[0, var].set_ylabel(str(var_names[int(var)]), size=26)

        if var in range(3, 6):
            ii = var - 3
            ax[1, ii].plot(kinematics[:, var], label='E')
            ax[1, ii].plot(PredKinmatics, label='NN')
            ax[1, ii].legend()
            ax[1, ii].set_ylabel(str(var_names[int(var)]), size=26)
            ax[1, ii].set_xlabel('Data Index', size=26)

    ax[0, 0].set_ylabel('Medial(mm)', size=26)
    ax[0, 1].set_ylabel('Posterior(mm)', size=26)
    ax[0, 2].set_ylabel('Superior(mm)', size=26)
    ax[1, 0].set_ylabel('Valgus(deg)', size=26)
    ax[1, 1].set_ylabel('Internal Rotation(deg)', size=26)
    ax[1, 2].set_ylabel('Extension Torque(Nm)', size=26)
    fig.suptitle(str(tests), fontsize=28)
    print(datetime.now() - startTime)
    plt.show()
    plt.figure()


if __name__ == '__main__':
    from datetime import datetime

    startTime = datetime.now()
    main()
