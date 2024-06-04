# Import Packages
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras import callbacks


def main():
    # Set seed to reproduce results
    # np.random.seed(6)
    # Batch size
    batch_size = 1000
    # Choose Knee to create ANN for
    knee = 'oks009'
    # Read raw data
    kinetics_raw = pd.read_hdf('dat/joint_mechanics-' + str(knee) + '/joint_mechanics-' + str(knee) + '_x_hierarchical_data.h5')
    kinematics_raw = pd.read_hdf('dat/joint_mechanics-' + str(knee) + '/joint_mechanics-' + str(knee) + '_y_hierarchical_data.h5')
    # Simulate real experiment by moving Flexion to Kinetics and Extension Torque to Kinematics
    kinematics = kinematics_raw.join(kinetics_raw['JCS Load Extension Torque(Nm)'])
    kinetics = kinetics_raw.join(kinematics_raw['Knee JCS Flexion(deg)'])
    del kinematics['Knee JCS Flexion(deg)']
    del kinetics['JCS Load Extension Torque(Nm)']
    # List of kinematics
    var_names = list(kinematics)
    # Partition the data to separate tests included in raw data
    kinetics_keys, kinematics_keys = partition_data(kinetics, kinematics)
    # Tests to be included to create the ANN model
    if knee == 'oks001':  #, 017_AP laxity_30degrees_main_processed.tdms
        tests = ['004_passive flexion_main_processed.tdms', '005_passive flexion_optimized_main_processed.tdms',
                 '006_preconditioning_30degrees_main_processed.tdms', '007_AP_30degrees_main_processed.tdms',
                 '008_AllLaxity_0degrees_main_processed.tdms', '009_CombinedLoads_0degrees_main_processed.tdms',
                 '013_AllLaxity_30degrees_main_processed.tdms', '014_CombinedLoads_30degrees_main_processed.tdms',
                 '015_CombinedLoads_30degrees_longer_main_processed.tdms',
                 '022_AP Laxity_30degrees_main_processed.tdms',
                 '018_All Laxity_60degrees_main_processed.tdms',
                 '019_CombineLoads_60degrees_main_processed.tdms', '020_All Laxity_90degrees_main_processed.tdms',
                 '021_CombinedLoads_90degrees_main_processed.tdms']
    elif knee == 'oks002':  #, '005_A-P laxity_30deg_main_processed.tdms'
        tests = ['002_Passive Flexion 0-90_main_processed.tdms',
                 '003_Optimized Passive Flexion 0-90_main_processed.tdms',
                 '004_Preconditioning_30deg_main_processed.tdms',
                 '006_All Laxity 0_deg_main_processed.tdms', '007_Combined Loading 0_deg_main_processed.tdms',
                 '008_All Laxity 30_deg_main_processed.tdms', '009_Combined Loading 30_deg_main_processed.tdms',
                 '010_A-P Laxity 30_deg_main_processed.tdms', '011_All Laxity 60_deg_main_processed.tdms',
                 '012_Combined Loading 60_deg_main_processed.tdms', '013_All laxity 90_deg_main_processed.tdms',
                 '014_Combined Loading 90_deg_main_processed.tdms', '015_A-P laxity 30_deg_main_processed.tdms']
    elif knee == 'oks003':  #'005_A-P Laxity 30deg_main_processed.tdms',
        tests = ['001_native passive flexion 0 - 60_main_processed.tdms',
                 '002_native optimized passive flexion 0 - 60_main_processed.tdms', '003_0flex_20N_main_processed.tdms',
                 '004_0flex_100N_main_processed.tdms', '005_0flex_200N_main_processed.tdms',
                 '006_0flex_300N_main_processed.tdms', '007_0flex_400N_main_processed.tdms',
                 '008_0flex_500N_main_processed.tdms', '009_0flex_600N_main_processed.tdms',
                 '010_0flex_20N_main_processed.tdms', '011_0flex_100N_main_processed.tdms',
                 '012_0flex_200N_main_processed.tdms', '013_0flex_300N_main_processed.tdms',
                 '014_0flex_400N_main_processed.tdms', '015_0flex_500N_main_processed.tdms',
                 '016_0flex_20N_main_processed.tdms', '017_0flex_100N_main_processed.tdms',
                 '018_0flex_200N_main_processed.tdms', '019_0flex_300N_main_processed.tdms',
                 '020_0flex_400N_main_processed.tdms', '021_0flex_500N_main_processed.tdms',
                 '022_0flex_600N_main_processed.tdms', '023_15flex_20N_main_processed.tdms',
                 '024_15flex_100N_main_processed.tdms', '025_15flex_200N_main_processed.tdms',
                 '026_15flex_300N_main_processed.tdms', '027_15flex_400N_main_processed.tdms',
                 '028_15flex_500N_main_processed.tdms', '029_15flex_600N_main_processed.tdms',
                 '030_30flex_20N_main_processed.tdms', '031_30flex_100N_main_processed.tdms',
                 '032_30flex_200N_main_processed.tdms', '033_30flex_300N_main_processed.tdms',
                 '034_30flex_400N_main_processed.tdms', '035_30flex_500N_main_processed.tdms',
                 '036_30flex_600N_main_processed.tdms', '037_45flex_20N_main_processed.tdms',
                 '038_45flex_100N_main_processed.tdms', '039_45flex_200N_main_processed.tdms',
                 '040_45flex_300N_main_processed.tdms', '041_45flex_400N_main_processed.tdms',
                 '042_45flex_500N_main_processed.tdms', '043_45flex_600N_main_processed.tdms',
                 '044_60flex_20N_main_processed.tdms', '045_60flex_100N_main_processed.tdms',
                 '046_60flex_200N_main_processed.tdms', '047_60flex_300N_main_processed.tdms',
                 '048_60flex_400N_main_processed.tdms', '049_60flex_500N_main_processed.tdms',
                 '050_60flex_600N_main_processed.tdms', '002_passive flexion_0-90_main_processed.tdms',
                 '003_optimized passive flexion_0-90_main_processed.tdms', '004_preconditioning_main_processed.tdms',
                 '006_All Laxity 0deg_main_processed.tdms',
                 '007_CombinedLoads 0deg_main_processed.tdms', '009_AllLaxity 30deg_main_processed.tdms',
                 '010_CombinedLoads 30deg_main_processed.tdms', '011_A-P Laxity 30deg_main_processed.tdms',
                 '012_AllLaxity 60deg_main_processed.tdms', '013_CombinedLoads 60deg_main_processed.tdms',
                 '014_AllLaxity 90deg_main_processed.tdms', '017_CombinedLoads 90deg_continued_main_processed.tdms',
                 '019_A-P Laxity 30deg_main_processed.tdms']
    elif knee == 'oks004':  # '039_AP Laxity_30deg_main_processed.tdms'
        tests = ['010_Passive Flexion_main_processed.tdms', '013_Passive_Flexion_Optimized_main_processed.tdms',
                 '016_AP_Laxity_30deg_main_processed.tdms', '018_All_Laxity_0deg_main_processed.tdms',
                 '021_Combined Loads_0deg_main_processed.tdms', '023_All Laxity_30deg_main_processed.tdms',
                 '025_Combined Loads_30deg_main_processed.tdms', '027_AP Laxity_30deg_main_processed.tdms',
                 '028_All Laxity_60deg_main_processed.tdms', '029_Combined Loads_60deg_main_processed.tdms',
                 '034_Valgus_90deg_main_processed.tdms', '035_AP_90deg_main_processed.tdms',
                 '038_Combine Loads_FinishLastProfile_90deg_main_processed.tdms']
    elif knee == 'oks006':  # , '006_A-P Laxity 30deg_main_processed.tdms'
        tests = ['003_Passive Flexion 0-90_main_processed.tdms',
                 '004_Optimized Passive Flexion 0-90_main_processed.tdms',
                 '005_preconditioning_main_processed.tdms',
                 '007_All Laxity 0deg_main_processed.tdms', '008_CombinedLoads 0deg_main_processed.tdms',
                 '009_All Laxity 30deg_main_processed.tdms', '012_CombinedLoads 30deg_main_processed.tdms',
                 '013_A-P Laxity 30deg_main_processed.tdms', '014_All Laxity 60deg_main_processed.tdms',
                 '015_CombinedLoads 60deg_main_processed.tdms', '016_All Laxity 90deg_main_processed.tdms',
                 '018_CombinedLoads 90deg_main_processed.tdms', '019_A-P Laxity 30deg_main_processed.tdms']
    elif knee == 'oks007':  # '016_A-P Laxity_main_processed.tdms'
        tests = ['002_passive flexion_main_processed.tdms', '003_optimized passive flexion_main_processed.tdms',
                 '004_preconditioning_main_processed.tdms', '005_A-P Laxity_main_processed.tdms',
                 '006_All Laxity 0deg_main_processed.tdms', '007_Combined Loads 0deg_main_processed.tdms',
                 '008_All Laxity 30deg_main_processed.tdms', '009_Combined Loads 30deg_main_processed.tdms',
                 '010_A-P Laxity_main_processed.tdms', '011_All Laxity 60deg_main_processed.tdms',
                 '012_Combined Loads 60deg_main_processed.tdms', '013_All Laxity 90deg_main_processed.tdms',
                 '014_Combine Loads 90deg_main_processed.tdms', '015_1Nm IR Torque All Angles_main_processed.tdms']
    elif knee == 'oks009':  #, '026_AP_Laxity_main_processed.tdms'
        tests = ['002_passive flexion 0-90 degrees_main_processed.tdms',
                 '003_passive flexion 0-90 degrees_main_processed.tdms',
                 '007_passive flexion remount_main_processed.tdms',
                 '008_passive flexion remount_optimized_main_processed.tdms', '009_preconditioning_main_processed.tdms',
                 '010_A-P Laxity_main_processed.tdms', '011_All_Laxity 0 degrees_main_processed.tdms',
                 '012_Combined Loading 0_degrees_main_processed.tdms', '013_All_Laxity 30_degrees_main_processed.tdms',
                 '014_Combined loads 30_degrees_main_processed.tdms', '015_AP Laxity_main_processed.tdms',
                 '016_All Laxity 60_main_processed.tdms', '017_Combined Loads 60_main_processed.tdms',
                 '018_All Laxity 90_main_processed.tdms', '020_All Laxity_Just AP_ 90_main_processed.tdms',
                 '021_All Laxity_Just AP_ 90_main_processed.tdms', '022_combined loads 90_main_processed.tdms',
                 '024_All_Laxity Posterior Only 90_main_processed.tdms',
                 '025_All_Laxity Posterior Only 60_main_processed.tdms',
                 '027_all_laxity posterior_only 30_main_processed.tdms']
    elif knee == 'oks008':  #, '221_A-P Laxity_main_processed.tdms'
        tests = ['199_Passive Flexion_main_processed.tdms', '200_Passive Flexion_optimized_main_processed.tdms',
                 '201_Passive Flexion_optimized repeat_main_processed.tdms',
                 '204_Passive Flexion_slower_re-optimized_main_processed.tdms',
                 '205_preconditioning_main_processed.tdms', '206_A-P Laxity_main_processed.tdms',
                 '207_A-P Laxity_repeat_main_processed.tdms', '209_All Laxity_0 deg_main_processed.tdms',
                 '210_CombinedLoads_0 deg_main_processed.tdms', '211_All Laxity_30 deg_main_processed.tdms',
                 '212_Combined Loads_30 deg_main_processed.tdms', '213_A-P Laxity_main_processed.tdms',
                 '214_All Laxity_60 deg_main_processed.tdms', '216_Combined Loads_60 deg_main_processed.tdms',
                 '217_All Laxity_90 deg_main_processed.tdms', '219_Combined Loads_90 deg_main_processed.tdms',
                 '220_A-P Laxity_main_processed.tdms', '221_A-P Laxity_main_processed.tdms']

    # Create an empty list to append selected test to it
    kinetics_1 = []
    kinematics_1 = []
    for i in range(0, len(tests)):
        kinetics_1.append(kinetics_keys[tests[i]])
        kinematics_1.append(kinematics_keys[tests[i]])
    # List to Numpy array
    kinetics = np.vstack(kinetics_1)
    kinematics = np.vstack(kinematics_1)
    # Create variable to store all kinematics data
    kinematics_all = kinematics
    # Initiate empty figure
    fig, ax = plt.subplots(nrows=2, ncols=3)
    # Loop thought Kinematics to create ANN model
    for var in range(len(var_names)):
        print(var_names[var])
        print(var)
        kinematics = kinematics_all[:, var]
        # Crate Training and Validation test
        x_train, x_test, y_train, y_test = train_test_split(kinetics, kinematics, test_size=0.25)

        # Setup ANN sensational model
        model = Sequential()
        model.add(Dense(100, activation="tanh", input_dim=6))
        model.add(Dense(50, activation="tanh"))
        model.add(Dense(25, activation="tanh"))
        model.add(Dense(1, activation="linear"))
        # Compile model
        model.compile(loss='mse', optimizer='adam')
        # Setup early stopping to avoid overfitting
        earlystopping = callbacks.EarlyStopping(monitor="val_loss",
                                                mode="min", patience=5,
                                                restore_best_weights=True)
        # Fit the model
        model.fit(x_train, y_train, epochs=100, batch_size=batch_size, verbose=1, validation_data=(x_test, y_test),
                  callbacks=earlystopping)
        # Calculate predictions
        PredTestSet = model.predict(x_train)
        PredValSet = model.predict(x_test)

        # Save predictions
        np.savetxt("trainresults.csv", PredTestSet, delimiter=",")
        np.savetxt("valresults.csv", PredValSet, delimiter=",")

        # Plot actual vs prediction for training set
        TestResults = np.genfromtxt("trainresults.csv", delimiter=",")

        #Compute R-Square value for training set
        TestR2Value = r2_score(y_train, TestResults)
        TestMSEValue = mean_squared_error(y_train, TestResults)
        print("Training Set R-Square=", TestR2Value)
        print("Training Set MSE=", TestMSEValue)
        print("Training Set RMSE=", np.sqrt(TestMSEValue))


        # Plot actual vs predition for validation set
        ValResults = np.genfromtxt("valresults.csv", delimiter=",")

        #Compute R-Square value for validation set
        ValR2Value = r2_score(y_test, ValResults)
        ValmseValue = mean_squared_error(y_test, ValResults)
        # Print R^2 and MSE
        print("Validation Set R-Square=", ValR2Value)
        print("Validation Set MSE=", ValmseValue)
        print("Validation Set RMSE=", np.sqrt(ValmseValue))
        model.save(r'dat/ANN_Models/' + str(knee) + str(var_names[int(var)]))
        # Plot the data
        if var in range(0, 3):
            ax[0, var].plot(y_test, ValResults, 'ro')
            ax[0, var].set_title(
                str(var_names[int(var)]) + '\n$\mathregular{R^{2}}$= ' + str(round(ValR2Value, 4)) + ' ,RMSE= ' + str(
                    round(np.sqrt(ValmseValue), 4)), size=18)

        if var in range(3, 6):
            ii = var - 3
            ax[1, ii].plot(y_test, ValResults, 'ro')
            ax[1, ii].set_title(
                str(var_names[int(var)]) + '\n$\mathregular{R^{2}}$= ' + str(round(ValR2Value, 4)) + ' ,RMSE= ' + str(
                    round(np.sqrt(ValmseValue), 4)), size=18)
    plt.subplots_adjust(hspace=.3)
    ax[1, 0].set_xlabel('Data Point Index', size=18)
    ax[1, 1].set_xlabel('Data Point Index', size=18)
    ax[1, 2].set_xlabel('Data Point Index', size=18)
    print(datetime.now() - startTime)
    # plt.show()
    plt.setp(ax[-1, :], xlabel='Actual')
    plt.setp(ax[:, 0], ylabel='Predicted')
    # plt.show()


def partition_data(kinetics, kinematics):
    # Partition the data to separate tests included in raw data
    tests_list = []
    tests = {}
    tests[kinetics.index[1][1]] = None
    for row in kinetics.itertuples():
        test_name = row[0][1]
        if test_name in tests:
            tests_list.append(row[1:7])
            oldest = test_name
        else:
            tests[oldest] = tests_list
            tests[row[0][1]] = None
            tests_list = []
            tests_list.append(row[1:7])
    tests[oldest] = tests_list
    kinetics = tests
    tests_list = []
    tests = {}
    tests[kinematics.index[1][1]] = None
    # test_name =kinitics.index[1][1]
    for row in kinematics.itertuples():
        test_name = row[0][1]
        if test_name in tests:
            tests_list.append(row[1:7])
            oldest = test_name
        else:
            tests[oldest] = tests_list
            tests[row[0][1]] = None
            tests_list = []
            tests_list.append(row[1:7])

    tests[oldest] = tests_list
    kinematics = tests
    return kinetics, kinematics


if __name__ == '__main__':
    from datetime import datetime

    startTime = datetime.now()
    main()
