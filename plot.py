import numpy as np
import matplotlib.pyplot as plt
import glob
from matplotlib.collections import LineCollection

cma_data_list = []
dqn_data_list = []

cma = []
dqn = []

# Load the data
for file in glob.glob(r"First iteration\\*.npz"):
  with np.load(file) as f:
    if "CMA" in file:
        cma.append(f["arr_0"])
    else:
        dqn.append(f["arr_0"])
cma_data_list.append(cma)
dqn_data_list.append(dqn)
cma = []
dqn = []

# Load the data
for file in glob.glob(r"Second iteration\\*.npz"):
  with np.load(file) as f:
    if "CMA" in file:
        cma.append(f["arr_0"])
    else:
        dqn.append(f["arr_0"])
cma_data_list.append(cma)
dqn_data_list.append(dqn)
cma = []
dqn = []

# Load the data
for file in glob.glob(r"Third iteration\\*.npz"):
  with np.load(file) as f:
    if "CMA" in file:
        cma.append(f["arr_0"])
    else:
        dqn.append(f["arr_0"])
cma_data_list.append(cma)
dqn_data_list.append(dqn)
cma = []
dqn = []

# Load the data
for file in glob.glob(r"Fourth iteration\\*.npz"):
  with np.load(file) as f:
    if "CMA" in file:
        cma.append(f["arr_0"])
    else:
        dqn.append(f["arr_0"])
cma_data_list.append(cma)
dqn_data_list.append(dqn)
cma = []
dqn = []

# Load the data
for file in glob.glob(r"Fifth iteration\\*.npz"):
  with np.load(file) as f:
    if "CMA" in file:
        cma.append(f["arr_0"])
    else:
        dqn.append(f["arr_0"])
cma_data_list.append(cma)
dqn_data_list.append(dqn)

cma_data_list = np.array(cma_data_list)
dqn_data_list = np.array(dqn_data_list)

# Reading in all the files in a folder is sorted alphabetically. These lists contain the order
cma_labels = ["CMA 16 12 8", "CMA 16 8", "CMA 32 16", "CMA 32 24 12", "CMA 48 24 12", "CMA 4 3", "CMA 8 4"]
dqn_labels = ["DQN 16 12 8", "DQN 16 8", "DQN 32 16", "DQN 32 24 12", "DQN 48 24 12", "DQN 4 3", "DQN 8 4"]


def get_amount_of_epochs():
    # Print the amount of epochs (min, average, max) for each of the agents and the different network sizes
    label_index =  [5, 6, 1, 0, 2, 3, 4]

    dqn_sizes = []
    for i in range(0, len(label_index)):
        temp = []
        for j in range(0, 5):
            temp.append(dqn_data_list[j,label_index[i]].shape[0])
        dqn_sizes.append(temp)

    for model_epochs in dqn_sizes:
        print("min: ", min(model_epochs))
        print("average: ", np.mean(model_epochs))
        print("max: ", max(model_epochs))

    # Order of the network sizes
    label_index =  [5, 6, 1, 0, 2, 3, 4]

    cma_sizes = []
    for i in range(0, len(label_index)):
        temp = []
        for j in range(0, 5):
            temp.append(cma_data_list[j,label_index[i]].shape[0])
        cma_sizes.append(temp)

    for model_epochs in cma_sizes:
        print("min: ", min(model_epochs))
        print("average: ", np.mean(model_epochs))
        print("max: ", max(model_epochs))


def plot_moving_avg_all(data, labels):
    # Plot the moving average for all agents and the different network sizes in separate plots based on network size
    def mean(data):
        mean_data = []
        for series in data:
            mean_temp = []
            mean_temp.append(series[0])
            for i in range(1,len(series)):
                mean_temp.append((series[i] * 0.01 ) + (mean_temp[i-1] * 0.99 ))
            mean_data.append(mean_temp)
        return mean_data
    
    label_index =  [5, 6, 1, 0, 2, 3, 4]
    cma_labels = labels[0]
    dqn_labels = labels[1]

    cma_lines_all = []
    dqn_lines_all = []

    # For each network size
    for i in range(0,7):
        cma_data = data[0,:,i]
        dqn_data = data[1,:,i]
        
        mean_data_cma = mean(cma_data)
        mean_data_dqn = mean(dqn_data)

        cma_ys = mean_data_cma
        
        dqn_ys = mean_data_dqn

        # Create lines for each iteration
        cma_lines = LineCollection([list(zip(np.arange(len(y)),y)) for y in cma_ys], label= cma_labels[i], color = '#1f77b4')
        dqn_lines = LineCollection([list(zip(np.arange(len(y)),y)) for y in dqn_ys], label= dqn_labels[i], color = '#ff7f0e')
 
        cma_lines_all.append(cma_lines)
        dqn_lines_all.append(dqn_lines)

    # Plot the mean rewards
    fig, axs = plt.subplots(3, 3)
    fig.suptitle("Moving average reward over 100 episodes")
    axs[0,0].add_collection(cma_lines_all[label_index[0]])
    axs[0,0].add_collection(dqn_lines_all[label_index[0]])
    axs[0,0].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[0,0].legend(loc="lower right")
    axs[0,0].set_ylim([-525, 0])
    axs[0,0].set_xlim([-300, 7500])

    axs[0,1].add_collection(cma_lines_all[label_index[1]])
    axs[0,1].add_collection(dqn_lines_all[label_index[1]])
    axs[0,1].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[0,1].legend(loc="lower right")
    axs[0,1].set_ylim([-525, 0])
    axs[0,1].set_xlim([-400, 11000])

    axs[0,2].add_collection(cma_lines_all[label_index[2]])
    axs[0,2].add_collection(dqn_lines_all[label_index[2]])
    axs[0,2].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[0,2].legend(loc="lower right")
    axs[0,2].set_ylim([-525, 0])
    axs[0,2].set_xlim([-500, 13500])

    axs[1,0].add_collection(cma_lines_all[label_index[3]])
    axs[1,0].add_collection(dqn_lines_all[label_index[3]])
    axs[1,0].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[1,0].legend(loc="lower right")
    axs[1,0].set_ylim([-525, 0])
    axs[1,0].set_xlim([-450, 12000])

    axs[1,1].add_collection(cma_lines_all[label_index[4]])
    axs[1,1].add_collection(dqn_lines_all[label_index[4]])
    axs[1,1].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[1,1].legend(loc="lower right")
    axs[1,1].set_ylim([-525, 0])
    axs[1,1].set_xlim([-500, 14000])

    axs[1,2].add_collection(cma_lines_all[label_index[5]])
    axs[1,2].add_collection(dqn_lines_all[label_index[5]])
    axs[1,2].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[1,2].legend(loc="lower right")
    axs[1,2].set_ylim([-525, 0])
    axs[1,2].set_xlim([-500, 13000])

    axs[2,1].add_collection(cma_lines_all[label_index[6]])
    axs[2,1].add_collection(dqn_lines_all[label_index[6]])
    axs[2,1].grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    axs[2,1].legend(loc="lower right")
    axs[2,1].set_ylim([-525, 0])
    axs[2,1].set_xlim([-500, 13000])

    axs[1,0].set(xlabel="Episode")
    axs[1,2].set(xlabel="Episode")
    axs[2,2].set_axis_off()
    axs[2,0].set_axis_off()

    plt.setp(axs[-1, :], xlabel='Episode')
    plt.setp(axs[:, 0], ylabel='Average reward')
    plt.legend()
    plt.show()


def plot_moving_avg(data):
    # Plot the moving average for an agents and the different network sizes
    def mean(data):
        mean_data = []
        for series in data:
            mean_temp = []
            mean_temp.append(series[0])
            for i in range(1,len(series)):
                mean_temp.append((series[i] * 0.01 ) + (mean_temp[i-1] * 0.99 ))
            mean_data.append(mean_temp)
        return mean_data
    
    # Order of the network sizes
    label_index =  [5, 6, 1, 0, 2, 3, 4]
    lines_all = []
    colors = ["#35C9FF", "#A23FFF", "#FF5EDC", "#FF3D57", "#FFD644", "#FF7632", "#65FF51"]

    # For each network size
    for i in range(0,7):

        loop_data = np.array(data)[:,i]

        print(loop_data.shape)
        
        mean_data= mean(loop_data)

        cma_ys = mean_data
        
        # Create lines for each iteration
        cma_lines = LineCollection([list(zip(np.arange(len(y)),y)) for y in cma_ys], label= cma_labels[i], color = colors[i], alpha = 0.6)
 
        lines_all.append(cma_lines)

    # Plot the mean rewards
    fig, axs = plt.subplots(1,1)
    for i in range(0, len(lines_all)):
        axs.add_collection(lines_all[label_index[i]])
    
    axs.set_ylim([-525, 0])
    axs.set_xlim([-500, 14500])
    plt.setp(axs, xlabel='Episode', ylabel = "Average reward")
    plt.title("Mean average reward for the different CMA networks")
    axs.grid(True, linewidth = 0.1, color = 'black', linestyle = '-', alpha = 1)
    plt.legend(loc = 'lower right')
    plt.show()

plot_moving_avg(cma_data_list, labels = cma_labels)
plot_moving_avg(dqn_data_list, labels = dqn_labels)

plot_moving_avg_all(np.array([cma_data_list, dqn_data_list]), labels = [cma_labels, dqn_labels])