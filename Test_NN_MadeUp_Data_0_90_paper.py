# Import Packages
import pandas as pd
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np

# Select two knees to include in the figure
knees = ['oks001', 'oks006']
# Flexion angles to plot
flexion = np.linspace(0, 90, 10)
# Initiate the figure
fig, ax = plt.subplots(ncols=2, nrows=3)
# Loop through the selected knees
for knee in range(len(knees)):
    # Read raw data
    kinetics_raw = pd.read_hdf(
        'dat/joint_mechanics-' + str(knees[knee]) + '/joint_mechanics-' + str(knees[knee]) + '_x_hierarchical_data.h5')
    kinematics_raw = pd.read_hdf(
        'dat/joint_mechanics-' + str(knees[knee]) + '/joint_mechanics-' + str(knees[knee]) + '_y_hierarchical_data.h5')
    # Loop through flexion angles
    for i in range(len(flexion)):
        # Select load to apply
        applied_load = 'JCS Load External Rotation Torque(Nm)'
        # 0 off axis loads with -20 distraction load
        kinetics = {'JCS Load Lateral Drawer(N)': np.linspace(0, 0, 21),
                    'JCS Load Anterior Drawer(N)': np.linspace(0, 0, 21),
                    'JCS Load Distraction(N)': np.linspace(-20, -20, 21),
                    'JCS Load Varus Torque(Nm)': np.linspace(0, 0, 21),
                    'JCS Load External Rotation Torque(Nm)': np.linspace(-5, +5, 21),
                    'Knee JCS Flexion(deg)': np.linspace(flexion[i], flexion[i], 21)}
        # List of kinetics
        var_names = list(kinetics.keys())
        kinetics = pd.DataFrame(kinetics, columns=var_names)
        # List of kinematics
        var_names = ['Knee JCS Medial(mm)', 'Knee JCS Posterior(mm)', 'Knee JCS Superior(mm)', 'Knee JCS Valgus(deg)',
                     'Knee JCS Internal Rotation(deg)', 'JCS Load Extension Torque(Nm)']
        # Select kinematics to include in plots
        for var in range(len(var_names)):
            if var == 4:
                print(str(var_names[int(var)]))
                # Load pretrained model
                model = keras.models.load_model(
                    r'dat/POLS_GPU/NN_Models/Models/' + str(knees[knee]) + str(var_names[int(var)]))
                # Predict Kinematics
                PredKinmatics = model.predict(kinetics)
                # Plot data
                if flexion[i] == 0:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='blue', linestyle='dashed')
                elif flexion[i] == 10:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$', color='darkorange')
                elif flexion[i] == 20:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$', color='green')
                elif flexion[i] == 30:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='maroon', linestyle='dashed')
                elif flexion[i] == 40:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$', color='purple')
                elif flexion[i] == 50:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$', color='brown')
                elif flexion[i] == 60:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='pink', linestyle='dashed')
                elif flexion[i] == 70:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$', color='grey')
                elif flexion[i] == 80:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$', color='olive')
                elif flexion[i] == 90:
                    ax[1, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='darkturquoise', linestyle='dashed')
        ############################################################################
        # Select load to apply
        applied_load = 'JCS Load Anterior Drawer(N)'
        # 0 off axis loads with -20 distraction load
        kinetics = {'JCS Load Lateral Drawer(N)': np.linspace(0, 0, 21),
                    'JCS Load Anterior Drawer(N)': np.linspace(-100, +100, 21),
                    'JCS Load Distraction(N)': np.linspace(-20, -20, 21),
                    'JCS Load Varus Torque(Nm)': np.linspace(0, 0, 21),
                    'JCS Load External Rotation Torque(Nm)': np.linspace(0, 0, 21),
                    'Knee JCS Flexion(deg)': np.linspace(flexion[i], flexion[i], 21)}
        # List of kinetics
        var_names = list(kinetics.keys())
        kinetics = pd.DataFrame(kinetics, columns=var_names)
        # List of kinematics
        var_names = ['Knee JCS Medial(mm)', 'Knee JCS Posterior(mm)', 'Knee JCS Superior(mm)', 'Knee JCS Valgus(deg)',
                     'Knee JCS Internal Rotation(deg)', 'JCS Load Extension Torque(Nm)']
        # Select kinematics to include in plots
        for var in range(len(var_names)):
            if var == 1:
                print(str(var_names[int(var)]))
                # Load pretrained model
                model = keras.models.load_model(
                    r'dat/POLS_GPU/NN_Models/Models/' + str(knees[knee]) + str(var_names[int(var)]))
                # Predict Kinematics
                PredKinmatics = model.predict(kinetics[0:])
                # Plot data
                if flexion[i] == 0:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='blue', linestyle='dashed')
                elif flexion[i] == 10:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='darkorange')
                elif flexion[i] == 20:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='green')
                elif flexion[i] == 30:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='maroon', linestyle='dashed')
                elif flexion[i] == 40:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='purple')
                elif flexion[i] == 50:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='brown')
                elif flexion[i] == 60:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='pink', linestyle='dashed')
                elif flexion[i] == 70:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='grey')
                elif flexion[i] == 80:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='olive')
                elif flexion[i] == 90:
                    ax[0, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='darkturquoise', linestyle='dashed')
                ax[0, knee].tick_params(labelsize=24)
        ############################################################################
        # Select load to apply
        applied_load = 'JCS Load Varus Torque(Nm)'
        # 0 off axis loads with -20 distraction load
        kinetics = {'JCS Load Lateral Drawer(N)': np.linspace(0, 0, 21),
                    'JCS Load Anterior Drawer(N)': np.linspace(0, 0, 21),
                    'JCS Load Distraction(N)': np.linspace(-20, -20, 21),
                    'JCS Load Varus Torque(Nm)': np.linspace(-10, +10, 21),
                    'JCS Load External Rotation Torque(Nm)': np.linspace(0, 0, 21),
                    'Knee JCS Flexion(deg)': np.linspace(flexion[i], flexion[i], 21)}
        # List of kinetics
        var_names = list(kinetics.keys())
        kinetics = pd.DataFrame(kinetics, columns=var_names)
        # List of kinematics
        var_names = ['Knee JCS Medial(mm)', 'Knee JCS Posterior(mm)', 'Knee JCS Superior(mm)', 'Knee JCS Valgus(deg)',
                     'Knee JCS Internal Rotation(deg)', 'JCS Load Extension Torque(Nm)']
        # Select kinematics to include in plots
        for var in range(len(var_names)):
            if var == 3:
                print(str(var_names[int(var)]))
                # Load pretrained model
                model = keras.models.load_model(
                    r'dat/POLS_GPU/NN_Models/Models/' + str(knees[knee]) + str(var_names[int(var)]))
                # Predict Kinematics
                PredKinmatics = model.predict(kinetics)
                # Plot data
                if flexion[i] == 0:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='blue', linestyle='dashed')
                elif flexion[i] == 10:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='darkorange')
                elif flexion[i] == 20:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='green')
                elif flexion[i] == 30:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='maroon', linestyle='dashed')
                elif flexion[i] == 40:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='purple')
                elif flexion[i] == 50:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='brown')
                elif flexion[i] == 60:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='pink', linestyle='dashed')
                elif flexion[i] == 70:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='grey')
                elif flexion[i] == 80:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics,
                                     label=str(flexion[i]) + '$^\circ$',
                                     color='olive')
                elif flexion[i] == 90:
                    ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, linewidth=3,
                                     label=str(flexion[i]) + '$^\circ$', color='darkturquoise', linestyle='dashed')
                ax[2, knee].plot(kinetics[applied_load].values, -1 * PredKinmatics, label=str(flexion[i]) + '$^\circ$')
                ax[2, knee].tick_params(labelsize=24)
# Set labels and legends
ax[0, 0].set_ylabel('Anterior(mm)', size=24)
ax[0, 0].set_xlabel('Anterior Drawer(N)', size=24)
ax[0, 1].set_xlabel('Anterior Drawer(N)', size=24)

ax[2, 0].set_ylabel('Varus(deg)', size=24)
ax[2, 0].set_xlabel('Varus Torque(N)', size=24)
ax[2, 1].set_xlabel('Varus Torque(N)', size=24)

ax[1, 0].set_ylabel('External\nRotation(deg)', size=24)
ax[1, 0].set_xlabel('External Rotation Torque(Nm)', size=24)
ax[1, 1].set_xlabel('External Rotation Torque(Nm)', size=24)

fig.subplots_adjust(wspace=0.2)
fig.subplots_adjust(hspace=0.4)
ax[0, 0].set_title(knees[0], size=24)
ax[0, 1].set_title(knees[1], size=24)
box = ax[0, 0].get_position()
ax[0, 0].set_position([box.x0, box.y0, box.width, box.height])
ax[0, 0].legend(loc='center left', bbox_to_anchor=(0.55, 1.4),
                ncol=5, fancybox=True, shadow=True, fontsize=14)
ax[0, 0].tick_params(axis='both', which='major', labelsize=14)
ax[1, 0].tick_params(axis='both', which='major', labelsize=14)
ax[2, 0].tick_params(axis='both', which='major', labelsize=14)

ax[0, 1].tick_params(axis='both', which='major', labelsize=14)
ax[1, 1].tick_params(axis='both', which='major', labelsize=14)
ax[2, 1].tick_params(axis='both', which='major', labelsize=14)
fig.subplots_adjust()
plt.show()
