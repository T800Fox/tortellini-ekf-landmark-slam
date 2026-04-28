import matplotlib.pyplot as plt
import numpy as np
import os



class UncertaintyPlotter(object):
    def __init__(self, log_file_path=None):

        self.fig, self.axs = plt.subplots(2,2,
                                            figsize=(10, 10),
                                            facecolor='black',
                                            gridspec_kw={'width_ratios': [3, 1]})

        self.COLORS = [
                "tab:red",
                "tab:green",
                "tab:blue",
                "tab:purple",
                "tab:orange",
                "tab:cyan",
                "tab:brown",
                "tab:pink",
                "tab:olive",
                "tab:gray",
                ]
        
        self.timesteps = []
        self.running_mean = {}
        self.running_deviation = {}

        if log_file_path is not None:
            self.log_file_path = log_file_path

            if os.path.exists(self.log_file_path):
                print("Uncertainty Log File Exists, deleting...")
                os.remove(self.log_file_path)
        else:
            self.log_file_path = None


    def extract_state_deviations(self, covariance_matrix):
        state_covariances = list(np.diag(covariance_matrix))

        return [np.sqrt(s_c) for s_c in state_covariances]

    def decompose_and_label_covariance_matrices(self, pose, pose_covar, landmarks):
        labeled = []

        state_entry = {}
        state_entry['name'] = 'p'
        state_entry['labels'] = ['x', 'y', 'theta']
        state_entry['state'] = [float(v) for v in list(pose)]
        state_entry['covariance'] = pose_covar
        labeled.append(state_entry)

        
        for l in landmarks:
            # state_mat = np.zeros((2,2))
            # np.fill_diagonal(state_mat, l.mean)

            landmark_entry = {}
            landmark_entry['name'] = str(l.lm_id)# + " @ " + str(l.mean)
            landmark_entry['labels'] = ['x', 'y']
            landmark_entry['state'] = [float(v) for v in l.mean]
            landmark_entry['covariance'] = l.covariance
            labeled.append(landmark_entry)

        return labeled

    def plot_system(self, pose, pose_covar, landmarks, t):
        # for l in labeled_data:
        #     print("Name ->")
        #     print('\t', l['name'])
        #     print("Labels ->")
        #     print('\t', l['labels'])
        #     print("State ->")
        #     print('\t', l['state'])
        #     print("Covariance ->")
        #     print('\t', l['covariance'])
        #     print("Deviations ->")
        #     print('\t', self.extract_state_deviations(l['covariance']))

        labeled_data = self.decompose_and_label_covariance_matrices(pose, pose_covar, landmarks)

        self.timesteps.append(t)

        current_labels = []
        current_colours = []
        current_state_values = []
        current_deviation_values = []

        # if self.log_file_path is not None:
        line_parts = [f"time : {t}"]

        # gaussian matrix?
        for i, gm in enumerate(labeled_data):

            assert len(gm['state']) == len(gm['labels'])

            labeled_data[i]['colour'] = self.COLORS[i % len(self.COLORS)]

            deviations = self.extract_state_deviations(gm['covariance'])
            for j in range(len(deviations)):
                key = f"{gm['name']}_{gm['labels'][j]}"


                self.update_running(runningValueSet=self.running_mean, 
                                    label=f"{gm['name']}_{gm['labels'][j]}", 
                                    value=gm['state'][j], 
                                    time=t,
                                    colour=gm['colour'])
               
                self.update_running(runningValueSet=self.running_deviation, 
                                    label=f"{gm['name']}_{gm['labels'][j]}", 
                                    value=deviations[j], 
                                    time=t,
                                    colour=gm['colour'])
                
                current_labels.append(f"{gm['name']}_{gm['labels'][j]}")
                current_colours.append(gm['colour'])
                
                current_state_values.append(gm['state'][j])
                current_deviation_values.append(deviations[j])

                
                line_parts.append(f"{key}_mean : {gm['state'][j]}")
                line_parts.append(f"{key}_std : {deviations[j]}")

        if self.log_file_path is not None:
            with open(self.log_file_path, 'a') as f:
                f.write(", ".join(line_parts) + "\n")


        landmarks_x = [gm['state'][0] for gm in labeled_data]
        landmarks_y = [gm['state'][1] for gm in labeled_data]
        l_colours = [gm['colour'] for gm in labeled_data]

        self.plot_enviroment(ax=self.axs[0,1],
                             title="Enviroment Map",
                             x_label="X Value (m)",
                             y_label="Y Value (m)",
                             x_vals=landmarks_x,
                             y_vals=landmarks_y,
                             colours=l_colours)

        # self.plot_bar(ax=self.axs[0, 1],
        #                title="Mean",
        #                x_label='',
        #                y_label="Value (m)",
        #                colours=current_colours,
        #                labels=current_labels,
        #                values=current_state_values
        #               )
        
        self.plot_bar(ax=self.axs[1,1],
                      title="Standard Deviation",
                      x_label='',
                      y_label="Value (m)",
                      colours=current_colours,
                      labels=current_labels,
                      values=current_deviation_values
                      )

        self.setup_line_plots(self.axs[0,0], 
                              title="Mean Over Time", 
                              y_label="Value (m)", 
                              x_label="Time (s)")
        for k in self.running_mean.keys():
            self.plot_line(self.axs[0,0], 
                           self.running_mean[k]['time'],
                           self.running_mean[k]['data'], 
                           colour=self.running_mean[k]['colour'])
            
        self.setup_line_plots(self.axs[1,0], 
                              title="Standard Deviation Over Time", 
                              y_label="Value (m)", 
                              x_label="Time (s)")
        for k in self.running_deviation.keys():
            self.plot_line(self.axs[1,0], 
                           self.running_deviation[k]['time'], 
                           self.running_deviation[k]['data'], 
                           colour=self.running_deviation[k]['colour'])

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

    def plot_enviroment(self, ax, title, x_label, y_label, x_vals, y_vals, colours):
        ax.clear()
        ax.set_facecolor('darkgrey')
        ax.set_xlabel(x_label, fontweight='bold', color='white')
        ax.set_ylabel(y_label, fontweight='bold', color='white')
        ax.invert_xaxis()
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(title, fontweight='bold', color='white')

        ax.scatter(y_vals, x_vals, c=colours, s=300)

        ax.tick_params(axis='x', colors='white')
        for label in ax.get_xticklabels():
            label.set_fontweight('bold')

        ax.tick_params(axis='y', colors='white')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')

    def plot_bar(self, ax, title, x_label, y_label, labels, colours, values):
        # print(values)
        # print(colours)
        # print(labels)

        ax.clear()
        ax.set_facecolor('darkgrey')
        # ax.set_aspect("equal")
        ax.set_xlabel(x_label, fontweight='bold', color='white')
        ax.set_ylabel(y_label, fontweight='bold', color='white')
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(title, fontweight='bold', color='white')

        ax.bar(labels, values, color=colours)

        ax.set_xticks(labels)
        ax.set_xticklabels(labels, fontweight='bold', color='white',
                           rotation=45, ha='right')

        ticks_loc = ax.get_yticks()
        ax.set_yticks(ticks_loc)
        ax.set_yticklabels(ticks_loc, fontweight='bold', color='white')
        # ax.set_yticklabels(ax.get_yticks(), fontweight='bold', color='white')
        # ax.set_yticks(values)
        # ax.set_yticklabels(values, fontweight='bold')
    
    def setup_line_plots(self, ax, title, x_label, y_label):
        ax.clear()
        ax.set_facecolor('darkgrey')
        # ax.set_aspect("equal")
        ax.set_xlabel(x_label, fontweight='bold', color='white')
        ax.set_ylabel(y_label, fontweight='bold', color='white')
        ax.grid(True, linestyle=":", alpha=0.6)
        ax.set_title(title, fontweight='bold', color='white')

        ax.tick_params(axis='y', colors='white')
        for label in ax.get_yticklabels():
            label.set_fontweight('bold')
        # ax.set_yticklabels(ax.get_yticks(), fontweight='bold', color='white')


    def plot_line(self, ax, x_data , y_data, colour):
        ax.plot(x_data, y_data, color=colour)

    def update_running(self, runningValueSet:dict, label, value, time, colour):
        if label not in runningValueSet:
            runningValueSet[label] = {'colour': colour,
                                      'data': [value],
                                      'time': [time]}
        else:
            runningValueSet[label]['time'].append(time)
            runningValueSet[label]['data'].append(value)
