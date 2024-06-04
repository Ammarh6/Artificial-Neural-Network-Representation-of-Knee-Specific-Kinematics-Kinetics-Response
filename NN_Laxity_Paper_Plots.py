import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import NN


def main():
    plt.rc('legend', fontsize=14)
    # Choose Knee. i.e., oks001, oks006
    knee = ('oks006')
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
    # List of kinematics
    var_names = list(kinematics)
    # Partition the data to separate test included in raw data
    kinetics_keys, kinematics_keys = NN.partition_data(kinetics, kinematics)
    ############################################################################
    ############################################################################
    # Select indices for oks006 laxity data
    if knee == 'oks006':
        # Select data for 30 degree
        test = '009_All Laxity 30deg_main_processed.tdms'
        # List to Numpy array
        kinetics = np.asarray(kinetics_keys[test])
        kinematics = np.asarray(kinematics_keys[test])
        # Create empty lists to add experimental laxity data
        AP_kinetics = []
        VV_kinetics = []
        IE_kinetics = []
        AP_kinematics = []
        VV_kinematics = []
        IE_kinematics = []
        for i in range(len(kinetics)):
            # A-P data
            if 17579 < i < len(kinetics):
                AP_kinetics.append(kinetics[i,])
                AP_kinematics.append(kinematics[i,])
            if 9913 < i < 17580:
                VV_kinetics.append(kinetics[i,])
                VV_kinematics.append(kinematics[i,])
            if 505 < i < 9913:
                IE_kinetics.append(kinetics[i,])
                IE_kinematics.append(kinematics[i,])
        # List to Numpy array
        AP_kinetics = np.asarray(AP_kinetics)
        VV_kinetics = np.asarray(VV_kinetics)
        IE_kinetics = np.asarray(IE_kinetics)
        AP_kinematics = np.asarray(AP_kinematics)
        VV_kinematics = np.asarray(VV_kinematics)
        IE_kinematics = np.asarray(IE_kinematics)
        # Initiate a figure
        fig, ax = plt.subplots(ncols=3, nrows=3)
        ##################################################################################
        # A-P plots
        var = 1  # index for A-P data
        # 0 for off-axis data, -20 for superior load
        AP_kinetics[:, 0] = 0
        AP_kinetics[:, 2] = -20
        AP_kinetics[:, 3] = 0
        AP_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(AP_kinetics)
        # Create Plots
        ax[0, 0].plot(AP_kinetics[:, var], -1 * AP_kinematics[:, var], 'r--', label='E (30' + u'\N{DEGREE SIGN})')
        ax[0, 0].legend()
        ax[0, 0].plot(AP_kinetics[:, var], -1 * PredKinmatics, 'k', label='NN (30' + u'\N{DEGREE SIGN})')
        ax[0, 0].legend()
        #########################################################################################################
        # V-V plots
        var = 3  # index for V-V data
        # 0 for off-axis data, -20 for superior load
        VV_kinetics[:, 0] = 0
        VV_kinetics[:, 1] = 0
        VV_kinetics[:, 2] = -20
        VV_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(VV_kinetics)
        # Create Plots
        ax[0, 1].plot(VV_kinetics[:, var], -1 * VV_kinematics[:, var], 'r--', label='E (30' + u'\N{DEGREE SIGN})')
        ax[0, 1].legend()
        ax[0, 1].plot(VV_kinetics[:, var], -1 * PredKinmatics, 'k', label='NN (30' + u'\N{DEGREE SIGN})')
        ax[0, 1].legend()
        ####################################################################################
        # I-E plots
        var = 4  # index for I-E data
        # 0 for off-axis data, -20 for superior load
        IE_kinetics[:, 0] = 0
        IE_kinetics[:, 1] = 0
        IE_kinetics[:, 2] = -20
        IE_kinetics[:, 3] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(IE_kinetics)
        # Create Plots
        ax[0, 2].plot(IE_kinetics[:, var], -1 * IE_kinematics[:, var], 'r--', label='E (30' + u'\N{DEGREE SIGN})')
        ax[0, 2].legend()
        ax[0, 2].plot(IE_kinetics[:, var], -1 * PredKinmatics, 'k', label='NN (30' + u'\N{DEGREE SIGN})')
        ax[0, 2].legend()
        #####################################################################
        #####################################################################
        # Select data for 60 degree
        test = '014_All Laxity 60deg_main_processed.tdms'
        # List to Numpy array
        kinetics = np.asarray(kinetics_keys[test])
        kinematics = np.asarray(kinematics_keys[test])
        # Create empty lists to add experimental laxity data
        AP_kinetics = []
        VV_kinetics = []
        IE_kinetics = []
        AP_kinematics = []
        VV_kinematics = []
        IE_kinematics = []
        # Select indices for oks006 laxity data
        for i in range(len(kinetics)):
            # A-P data
            if 32966 < i < len(kinetics):
                AP_kinetics.append(kinetics[i,])
                AP_kinematics.append(kinematics[i,])
            if 19773 < i < 32967:
                VV_kinetics.append(kinetics[i,])
                VV_kinematics.append(kinematics[i,])
            if 0 < i < 19774:
                IE_kinetics.append(kinetics[i,])
                IE_kinematics.append(kinematics[i,])
        # List to Numpy array
        AP_kinetics = np.asarray(AP_kinetics)
        VV_kinetics = np.asarray(VV_kinetics)
        IE_kinetics = np.asarray(IE_kinetics)
        AP_kinematics = np.asarray(AP_kinematics)
        VV_kinematics = np.asarray(VV_kinematics)
        IE_kinematics = np.asarray(IE_kinematics)

        ##################################################################################
        # A-P plots
        # 0 for off-axis data, -20 for superior load
        var = 1
        AP_kinetics[:, 0] = 0
        AP_kinetics[:, 2] = -20
        AP_kinetics[:, 3] = 0
        AP_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(AP_kinetics)
        # Create Plots
        ax[1, 0].plot(AP_kinetics[:, var], -1 * AP_kinematics[:, var], 'b--', label='E (60' + u'\N{DEGREE SIGN})')
        ax[1, 0].legend()
        ax[1, 0].plot(AP_kinetics[:, var], -1 * PredKinmatics, 'm', label='NN (60' + u'\N{DEGREE SIGN})')
        ax[1, 0].legend()

        #####################################################################################################
        # V-V plots
        var = 3  # index for V-V data
        # 0 for off-axis data, -20 for superior load
        VV_kinetics[:, 0] = 0
        VV_kinetics[:, 1] = 0
        VV_kinetics[:, 2] = -20
        VV_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(VV_kinetics)
        # Create Plots
        ax[1, 1].plot(VV_kinetics[:, var], -1 * VV_kinematics[:, var], 'b--', label='E (60' + u'\N{DEGREE SIGN})')
        ax[1, 1].legend()
        ax[1, 1].plot(VV_kinetics[:, var], -1 * PredKinmatics, 'm', label='NN (60' + u'\N{DEGREE SIGN})')
        ax[1, 1].legend()

        ############################################################################################
        # IE plots
        var = 4  # index for I-E data
        # 0 for off-axis data, -20 for superior load
        IE_kinetics[:, 0] = 0
        IE_kinetics[:, 1] = 0
        IE_kinetics[:, 2] = -20
        IE_kinetics[:, 3] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(IE_kinetics)
        # Create Plots
        ax[1, 2].plot(IE_kinetics[:, var], -1 * IE_kinematics[:, var], 'b--', label='E (60' + u'\N{DEGREE SIGN})')
        ax[1, 2].legend()
        ax[1, 2].plot(IE_kinetics[:, var], -1 * PredKinmatics, 'm', label='NN (60' + u'\N{DEGREE SIGN})')
        ax[1, 2].legend()
        #####################################################################
        #####################################################################
        # Select data for 90 degree
        test = '016_All Laxity 90deg_main_processed.tdms'
        kinetics = np.asarray(kinetics_keys[test])
        kinematics = np.asarray(kinematics_keys[test])
        AP_kinetics = []
        VV_kinetics = []
        IE_kinetics = []
        AP_kinematics = []
        VV_kinematics = []
        IE_kinematics = []
        # Select indices for oks006 laxity data
        for i in range(len(kinetics)):
            # A-P data
            if 16510 < i < len(kinetics):
                AP_kinetics.append(kinetics[i,])
                AP_kinematics.append(kinematics[i,])
            if 9934 < i < 16511:
                VV_kinetics.append(kinetics[i,])
                VV_kinematics.append(kinematics[i,])
            if 501 < i < 9935:
                IE_kinetics.append(kinetics[i,])
                IE_kinematics.append(kinematics[i,])
        # List to Numpy array
        AP_kinetics = np.asarray(AP_kinetics)
        VV_kinetics = np.asarray(VV_kinetics)
        IE_kinetics = np.asarray(IE_kinetics)
        AP_kinematics = np.asarray(AP_kinematics)
        VV_kinematics = np.asarray(VV_kinematics)
        IE_kinematics = np.asarray(IE_kinematics)

        ##################################################################################
        # A-P plots
        # 0 for off-axis data, -20 for superior load
        var = 1
        AP_kinetics[:, 0] = 0
        AP_kinetics[:, 2] = -20
        AP_kinetics[:, 3] = 0
        AP_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(AP_kinetics)
        # Create Plots
        ax[2, 0].plot(AP_kinetics[:, var], -1 * AP_kinematics[:, var], 'g--', label='E (90' + u'\N{DEGREE SIGN})')
        ax[2, 0].legend()
        ax[2, 0].plot(AP_kinetics[:, var], -1 * PredKinmatics, 'y', label='NN (90' + u'\N{DEGREE SIGN})')
        ax[2, 0].legend()
        ax[1, 0].set_ylabel('Anterior (mm)', size=26)
        ax[2, 0].set_xlabel('Anterior Drawer(N)', size=26)
        #####################################################################################################
        # V-V plots
        var = 3  # index for V-V data
        VV_kinetics[:, 0] = 0
        VV_kinetics[:, 1] = 0
        VV_kinetics[:, 2] = -20
        VV_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(VV_kinetics)
        # Create Plots
        ax[2, 1].plot(VV_kinetics[:, var], -1 * VV_kinematics[:, var], 'g--', label='E (90' + u'\N{DEGREE SIGN})')
        ax[2, 1].legend()
        ax[2, 1].plot(VV_kinetics[:, var], -1 * PredKinmatics, 'y', label='NN (90' + u'\N{DEGREE SIGN})')
        ax[2, 1].legend()
        ax[1, 1].set_ylabel('Varus (deg)', size=26)
        ax[2, 1].set_xlabel('Varus Torque(Nm)', size=26)
        ############################################################################################
        # IE plots
        var = 4  # index for I-E data
        # 0 for off-axis data, -20 for superior load
        IE_kinetics[:, 0] = 0
        IE_kinetics[:, 1] = 0
        IE_kinetics[:, 2] = -20
        IE_kinetics[:, 3] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(IE_kinetics)
        # Create Plots
        ax[2, 2].plot(IE_kinetics[:, var], -1 * IE_kinematics[:, var], 'g--', label='E (90' + u'\N{DEGREE SIGN})')
        ax[2, 2].legend()
        ax[2, 2].plot(IE_kinetics[:, var], -1 * PredKinmatics, 'y', label='NN (90' + u'\N{DEGREE SIGN})')
        ax[2, 2].legend()
        ax[1, 2].set_ylabel('External Rotation(deg)', size=26)
        ax[2, 2].set_xlabel('External Rotation Torque(Nm)', size=26)
        ax[0, 0].legend(fontsize=18)
        ax[0, 1].legend(fontsize=18)
        ax[0, 2].legend(fontsize=18)
        ax[1, 0].legend(fontsize=18)
        ax[1, 1].legend(fontsize=18)
        ax[1, 2].legend(fontsize=18)
        ax[2, 0].legend(fontsize=18)
        ax[2, 1].legend(fontsize=18)
        ax[2, 2].legend(fontsize=18)
        ax[0, 0].tick_params(labelsize=12)
        ax[0, 1].tick_params(labelsize=12)
        ax[0, 2].tick_params(labelsize=12)
        ax[1, 0].tick_params(labelsize=12)
        ax[1, 1].tick_params(labelsize=12)
        ax[1, 2].tick_params(labelsize=12)
        ax[2, 0].tick_params(labelsize=12)
        ax[2, 1].tick_params(labelsize=12)
        ax[2, 2].tick_params(labelsize=12)
        fig.subplots_adjust(hspace=0)
        plt.show()
        plt.figure()

    ############################################################################################
    ############################################################################################
    ############################################################################################
    # Select indices for oks001 laxity data
    if knee == 'oks001':
        # Select data for 30 degree
        test = '013_AllLaxity_30degrees_main_processed.tdms'
        # List to Numpy array
        kinetics = np.asarray(kinetics_keys[test])
        kinematics = np.asarray(kinematics_keys[test])
        # Create empty lists to add experimental laxity data
        AP_kinetics = []
        VV_kinetics = []
        IE_kinetics = []
        AP_kinematics = []
        VV_kinematics = []
        IE_kinematics = []
        for i in range(len(kinetics)):
            # A-P data
            if 16587 < i < len(kinetics):
                AP_kinetics.append(kinetics[i,])
                AP_kinematics.append(kinematics[i,])
            if 9941 < i < 16587:  # 9914:17580
                VV_kinetics.append(kinetics[i,])
                VV_kinematics.append(kinematics[i,])
            if 520 < i < 9941:
                IE_kinetics.append(kinetics[i,])
                IE_kinematics.append(kinematics[i,])
        # List to Numpy array
        AP_kinetics = np.asarray(AP_kinetics)
        VV_kinetics = np.asarray(VV_kinetics)
        IE_kinetics = np.asarray(IE_kinetics)
        AP_kinematics = np.asarray(AP_kinematics)
        VV_kinematics = np.asarray(VV_kinematics)
        IE_kinematics = np.asarray(IE_kinematics)

        fig, ax = plt.subplots(ncols=3, nrows=3)
        ##################################################################################
        # A-P plots
        var = 1  # index for A-P data
        # 0 for off-axis data, -20 for superior load
        AP_kinetics[:, 0] = 0
        AP_kinetics[:, 2] = -20
        AP_kinetics[:, 3] = 0
        AP_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(AP_kinetics)
        # Create Plots
        ax[0, 0].plot(AP_kinetics[:, var], -1 * AP_kinematics[:, var], 'r--',
                      label='E (30' + u'\N{DEGREE SIGN})')
        ax[0, 0].legend()
        ax[0, 0].plot(AP_kinetics[:, var], -1 * PredKinmatics, 'k', label='NN (30' + u'\N{DEGREE SIGN})')
        ax[0, 0].legend()
        ############################################################################################
        # V-V plots
        var = 3  # index for V-V data
        # 0 for off-axis data, -20 for superior load
        VV_kinetics[:, 0] = 0
        VV_kinetics[:, 1] = 0
        VV_kinetics[:, 2] = -20
        VV_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(VV_kinetics)
        # Create Plots
        ax[0, 1].plot(VV_kinetics[:, var], -1 * VV_kinematics[:, var], 'r--',
                      label='E (30' + u'\N{DEGREE SIGN})')
        ax[0, 1].legend()
        ax[0, 1].plot(VV_kinetics[:, var], -1 * PredKinmatics, 'k', label='NN (30' + u'\N{DEGREE SIGN})')
        ax[0, 1].legend()

        ############################################################################################
        # IE plots
        var = 4  # index for I-E data
        # 0 for off-axis data, -20 for superior load
        IE_kinetics[:, 0] = 0
        IE_kinetics[:, 1] = 0
        IE_kinetics[:, 2] = -20
        IE_kinetics[:, 3] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(IE_kinetics)
        # Create Plots
        ax[0, 2].plot(IE_kinetics[:, var], -1 * IE_kinematics[:, var], 'r--',
                      label='E (30' + u'\N{DEGREE SIGN})')
        ax[0, 2].legend()
        ax[0, 2].plot(IE_kinetics[:, var], -1 * PredKinmatics, 'k', label='NN (30' + u'\N{DEGREE SIGN})')
        ax[0, 2].legend()

        ############################################################################################
        ############################################################################################
        # Select data for 60 degree
        test = '018_All Laxity_60degrees_main_processed.tdms'
        # List to Numpy array
        kinetics = np.asarray(kinetics_keys[test])
        kinematics = np.asarray(kinematics_keys[test])
        # Create empty lists to add experimental laxity data
        AP_kinetics = []
        VV_kinetics = []
        IE_kinetics = []
        AP_kinematics = []
        VV_kinematics = []
        IE_kinematics = []
        # Select indices for oks006 laxity data
        for i in range(len(kinetics)):
            # A-P data
            if 17533 < i < len(kinetics):
                AP_kinetics.append(kinetics[i,])
                AP_kinematics.append(kinematics[i,])
            if 9925 < i < 17533:
                VV_kinetics.append(kinetics[i,])
                VV_kinematics.append(kinematics[i,])
            if 0 < i < 9925:
                IE_kinetics.append(kinetics[i,])
                IE_kinematics.append(kinematics[i,])
        # List to Numpy array
        AP_kinetics = np.asarray(AP_kinetics)
        VV_kinetics = np.asarray(VV_kinetics)
        IE_kinetics = np.asarray(IE_kinetics)
        AP_kinematics = np.asarray(AP_kinematics)
        VV_kinematics = np.asarray(VV_kinematics)
        IE_kinematics = np.asarray(IE_kinematics)

        ############################################################################################
        # A-P plots
        # 0 for off-axis data, -20 for superior load
        var = 1
        AP_kinetics[:, 0] = 0
        AP_kinetics[:, 2] = -20
        AP_kinetics[:, 3] = 0
        AP_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(AP_kinetics)
        # Create Plots
        ax[1, 0].plot(AP_kinetics[:, var], -1 * AP_kinematics[:, var], 'b--',
                      label='E (60' + u'\N{DEGREE SIGN})')
        ax[1, 0].legend()
        ax[1, 0].plot(AP_kinetics[:, var], -1 * PredKinmatics, 'm', label='NN (60' + u'\N{DEGREE SIGN})')
        ax[1, 0].legend()

        ############################################################################################
        # V-V plots
        var = 3  # index for V-V data
        # 0 for off-axis data, -20 for superior load
        VV_kinetics[:, 0] = 0
        VV_kinetics[:, 1] = 0
        VV_kinetics[:, 2] = -20
        VV_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(VV_kinetics)
        # Create Plots
        ax[1, 1].plot(VV_kinetics[:, var], -1 * VV_kinematics[:, var], 'b--',
                      label='E (60' + u'\N{DEGREE SIGN})')
        ax[1, 1].legend()
        ax[1, 1].plot(VV_kinetics[:, var], -1 * PredKinmatics, 'm', label='NN (60' + u'\N{DEGREE SIGN})')
        ax[1, 1].legend()

        ############################################################################################
        # IE plots
        var = 4  # index for I-E data
        # 0 for off-axis data, -20 for superior load
        IE_kinetics[:, 0] = 0
        IE_kinetics[:, 1] = 0
        IE_kinetics[:, 2] = -20
        IE_kinetics[:, 3] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(IE_kinetics)
        # Create Plots
        ax[1, 2].plot(IE_kinetics[:, var], -1 * IE_kinematics[:, var], 'b--',
                      label='E (60' + u'\N{DEGREE SIGN})')
        ax[1, 2].legend()
        ax[1, 2].plot(IE_kinetics[:, var], -1 * PredKinmatics, 'm', label='NN (60' + u'\N{DEGREE SIGN})')
        ax[1, 2].legend()

        ############################################################################################
        ############################################################################################
        # Select data for 90 degree
        test = '020_All Laxity_90degrees_main_processed.tdms'
        kinetics = np.asarray(kinetics_keys[test])
        kinematics = np.asarray(kinematics_keys[test])
        AP_kinetics = []
        VV_kinetics = []
        IE_kinetics = []
        AP_kinematics = []
        VV_kinematics = []
        IE_kinematics = []
        # Select indices for oks006 laxity data
        for i in range(len(kinetics)):
            # A-P data
            if 34537 < i < len(kinetics):
                AP_kinetics.append(kinetics[i,])
                AP_kinematics.append(kinematics[i,])
            if 19621 < i < 34537:
                VV_kinetics.append(kinetics[i,])
                VV_kinematics.append(kinematics[i,])
            if 501 < i < 19621:
                IE_kinetics.append(kinetics[i,])
                IE_kinematics.append(kinematics[i,])
        # List to Numpy array
        AP_kinetics = np.asarray(AP_kinetics)
        VV_kinetics = np.asarray(VV_kinetics)
        IE_kinetics = np.asarray(IE_kinetics)
        AP_kinematics = np.asarray(AP_kinematics)
        VV_kinematics = np.asarray(VV_kinematics)
        IE_kinematics = np.asarray(IE_kinematics)

        ############################################################################################
        # A-P plots
        # 0 for off-axis data, -20 for superior load
        var = 1
        AP_kinetics[:, 0] = 0
        AP_kinetics[:, 2] = -20
        AP_kinetics[:, 3] = 0
        AP_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(AP_kinetics)
        # Create Plots
        ax[2, 0].plot(AP_kinetics[:, var], -1 * AP_kinematics[:, var], 'g--',
                      label='E (90' + u'\N{DEGREE SIGN})')
        ax[2, 0].legend()
        ax[2, 0].plot(AP_kinetics[:, var], -1 * PredKinmatics, 'y', label='NN (90' + u'\N{DEGREE SIGN})')
        ax[2, 0].legend()
        ax[1, 0].set_ylabel('Anterior (mm)', size=26)
        ax[2, 0].set_xlabel('Anterior Drawer(N)', size=26)
        ############################################################################################
        # V-V plots
        var = 3
        VV_kinetics[:, 0] = 0
        VV_kinetics[:, 1] = 0
        VV_kinetics[:, 2] = -20
        VV_kinetics[:, 4] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(VV_kinetics)
        # Create Plots
        ax[2, 1].plot(VV_kinetics[:, var], -1 * VV_kinematics[:, var], 'g--',
                      label='E (90' + u'\N{DEGREE SIGN})')
        ax[2, 1].legend()
        ax[2, 1].plot(VV_kinetics[:, var], -1 * PredKinmatics, 'y', label='NN (90' + u'\N{DEGREE SIGN})')
        ax[2, 1].legend()
        ax[1, 1].set_ylabel('Varus (deg)', size=26)
        ax[2, 1].set_xlabel('Varus Torque(Nm)', size=26)
        ############################################################################################
        # IE plots
        var = 4  # index for I-E data
        # 0 for off-axis data, -20 for superior load
        IE_kinetics[:, 0] = 0
        IE_kinetics[:, 1] = 0
        IE_kinetics[:, 2] = -20
        IE_kinetics[:, 3] = 0
        # Print name of selected kinematic
        print(str(var_names[int(var)]))
        # Load pretrained model
        model = keras.models.load_model(
            r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Predict Kinematics
        PredKinmatics = model.predict(IE_kinetics)
        # Create Plots
        ax[2, 2].plot(IE_kinetics[:, var], -1 * IE_kinematics[:, var], 'g--',
                      label='E (90' + u'\N{DEGREE SIGN})')
        ax[2, 2].legend()
        ax[2, 2].plot(IE_kinetics[:, var], -1 * PredKinmatics, 'y', label='NN (90' + u'\N{DEGREE SIGN})')
        ax[2, 2].legend()
        ax[1, 2].set_ylabel('External Rotation(deg)', size=26)
        ax[2, 2].set_xlabel('External Rotation Torque(Nm)', size=26)
        ax[0, 0].legend(fontsize=18)
        ax[0, 1].legend(fontsize=18)
        ax[0, 2].legend(fontsize=18)
        ax[1, 0].legend(fontsize=18)
        ax[1, 1].legend(fontsize=18)
        ax[1, 2].legend(fontsize=18)
        ax[2, 0].legend(fontsize=18)
        ax[2, 1].legend(fontsize=18)
        ax[2, 2].legend(fontsize=18)
        ax[0, 0].tick_params(labelsize=12)
        ax[0, 1].tick_params(labelsize=12)
        ax[0, 2].tick_params(labelsize=12)
        ax[1, 0].tick_params(labelsize=12)
        ax[1, 1].tick_params(labelsize=12)
        ax[1, 2].tick_params(labelsize=12)
        ax[2, 0].tick_params(labelsize=12)
        ax[2, 1].tick_params(labelsize=12)
        ax[2, 2].tick_params(labelsize=12)
        fig.subplots_adjust(hspace=0)
        plt.show()
        plt.figure()


if __name__ == '__main__':
    from datetime import datetime

    startTime = datetime.now()
    main()
    print(datetime.now() - startTime)
