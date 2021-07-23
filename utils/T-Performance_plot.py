import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns; sns.set()
from plotsAnalysis import MeanAvgnMinMSEvsTry_all,MeanAvgnMinMSEvsTry,get_mse_mat_from_folder,reshape_xpred_list_to_mat,get_xpred_ytruth_xtruth_from_folder

def DrawAggregateMeanAvgnMSEPlot(datas, names, save_name='aggregate_plot', plot_points=200,resolution=None, dash_group='nobody',
                                dash_label='', solid_label='',worse_model_mode=False): # Depth=2 now based on current directory structure

    fig, (ax0_top, ax0_low, ax1, ax2) = plt.subplots(4, 1, gridspec_kw={'height_ratios': [1, 6, 7, 7]}, sharex=True)
    fig.set_figheight(12)
    ax_dict = {'Chen': (ax0_low, ax0_top), 'Peurifoy': [ax1], 'Yang_sim': [ax2]}

    print("AM: ",datas,names)
    for data_dir,data_name in zip(datas,names):
        print("DATA_NAME: ",data_name)
        """
        The function to draw the aggregate plot for Mean Average and Min MSEs
        :param data_dir: The mother directory to call
        :param data_name: The data set name
        :param plot_points: Number of points to be plot
        :param resolution: The resolution of points
        :return:
        """
        # Predefined name of the avg lists
        min_list_name = 'mse_min_list.txt'
        avg_list_name = 'mse_avg_list.txt'
        std_list_name = 'mse_std_list.txt'
        quan2575_list_name = 'mse_quan2575_list.txt'

        # Loop through the directories
        avg_dict, min_dict, std_dict, quan2575_dict = {}, {}, {}, {}
        for dirs in os.listdir(data_dir):
            # Dont include NA for now and check if it is a directory
            print("entering :", dirs)
            print("this is a folder?:", os.path.isdir(os.path.join(data_dir, dirs)))
            print("this is a file?:", os.path.isfile(os.path.join(data_dir, dirs)))
            if not os.path.isdir(os.path.join(data_dir, dirs)):# or dirs == 'NA':# or 'boundary' in dirs::
                print("skipping due to it is not a directory")
                continue
            for subdirs in os.listdir((os.path.join(data_dir, dirs))):
                if subdirs == data_name:
                    # Read the lists
                    mse_avg_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, avg_list_name),
                                               header=None, delimiter=' ').values
                    mse_min_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, min_list_name),
                                               header=None, delimiter=' ').values
                    if not (dirs == "TD") and not (dirs == "NN"):
                        mse_std_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, std_list_name),
                                               header=None, delimiter=' ').values
                        std_dict[dirs] = mse_std_list
                    mse_quan2575_list = pd.read_csv(os.path.join(data_dir, dirs, subdirs, quan2575_list_name),
                                               header=None, delimiter=' ').values
                    print("The quan2575 error range shape is ", np.shape(mse_quan2575_list))
                    print("dirs =", dirs)
                    print("shape of mse_min_list is:", np.shape(mse_min_list))
                    # Put them into dictionary
                    avg_dict[dirs] = mse_avg_list
                    min_dict[dirs] = mse_min_list
                    quan2575_dict[dirs] = mse_quan2575_list

        def plotDict(ax,dict, name, data_name=None, logy=False, logx=False, time_in_s_table=None, avg_dict=None,
                        plot_points=50,  resolution=None, err_dict=None, color_assign=False, dash_group='nobody',
                        dash_label='', solid_label='', plot_xlabel=False, worse_model_mode=False):
            """
            :param name: the name to save the plot
            :param dict: the dictionary to plot
            :param logy: use log y scale
            :param time_in_s_table: a dictionary of dictionary which stores the averaged evaluation time
                    in seconds to convert the graph
            :param plot_points: Number of points to be plot
            :param resolution: The resolution of points
            :param err_dict: The error bar dictionary which takes the error bar input
            :param avg_dict: The average dict for plotting the starting point
            :param dash_group: The group of plots to use dash line
            :param dash_label: The legend to write for dash line
            :param solid_label: The legend to write for solid line
            :param plot_xlabel: The True or False flag for plotting the x axis label or not
            :param worse_model_mode: The True or False flag for plotting worse model mode (1X, 10X, 50X, 100X worse model)
            """
            import matplotlib.colors as mcolors
            if worse_model_mode:
                color_dict = {"(1X": "limegreen", "(10X": "blueviolet", "(50X":"cornflowerblue", "(100X": "darkorange"}
            else:
                # # manual color setting
                # color_dict = {"VAE": "blueviolet","cINN":"crimson", "INN":"cornflowerblue", "Random": "limegreen",
                #                 "MDN": "darkorange", "NA_init_lr_0.1_decay_0.5_batch_2048":"limegreen"}
                # Automatic color setting
                color_dict = {}

                for ind, key in enumerate(dict.keys()):
                    color_dict[key] = list(mcolors.TABLEAU_COLORS.keys())[ind]

                print(color_dict)


            for a in ax:
                a.spines['left'].set_color('black')
                a.spines['right'].set_color('black')

                a.tick_params(which='major', right=True, left=True, direction='out')
                a.tick_params(which='minor', right=True, left=True, direction='out')
                a.tick_params(which='major', bottom=True, top=True, direction='inout')

            if len(ax) == 2:
                bot_ax,top_ax = ax
                bot_ax.spines['bottom'].set_color('black')
                top_ax.spines['top'].set_color('black')
            else:
                a0 = ax[0]
                a0.spines['bottom'].set_color('black')
                a0.spines['top'].set_color('black')

            text_pos = 0.01
            # List for legend
            legend_list = []
            print("All the keys=", dict.keys())
            print("All color keys=", color_dict.keys())
            z_point1 = 3.5+np.linspace(0.01,1,num=len(dict))
            z_point0 = z_point1 - 1
            z_line = int(z_point0[-1]) - 1
            z_ebar = z_line - 1

            strt_ord = sorted(dict.keys(), key= lambda x: dict[x][0], reverse=True)
            end_ord = sorted(dict.keys(),key= lambda x: dict[x][-1], reverse=True)

            for i,key in enumerate(sorted(dict)):
                ######################################################
                # This is for 02.02 getting the T=1, 50, 1000 result #
                ######################################################
                #text = key.replace('_',' ')+"\n" + ': t1={:.2e},t50={:.2e},t1000={:.2e}'.format(dict[key][0][0], dict[key][49][0], dict[key][999][0])
                #print("printing message on the plot now")
                #plt.text(1, text_pos, text, wrap=True)
                #text_pos /= 5

                # Linestyle
                if dash_group is not None and dash_group in key:
                    linestyle = 'dashed'
                else:
                    linestyle = 'solid'


                x_axis = np.arange(len(dict[key])).astype('float')
                x_axis += 1
                if time_in_s_table is not None:
                    x_axis *= time_in_s_table[data_name][key]

                print('key = ', key)
                if err_dict is None:
                    if color_assign:
                        line_axis, = plt.plot(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],c=color_dict[key.split('_')[0]],label=key, linestyle=linestyle)
                    else:
                        line_axis, = plt.plot(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],label=key, linestyle=linestyle)
                else:
                    if resolution is None:
                        label = key
                        linestyle = 'solid'

                        if data_name == 'Chen':
                            a0,a1 = ax
                        else:
                            a0 = ax[0]

                        if (dict[key][:plot_points].max() < 10e-4 and data_name=='Chen') or data_name!='Chen':
                            color_key = key
                            line_axis, = a0.plot(x_axis[:plot_points], dict[key][:plot_points], color=color_dict[color_key], linestyle=linestyle, label=label,zorder=z_line)
                            a0.plot(x_axis[0],dict[key][0],color=color_dict[color_key],marker='o',zorder=z_point0[strt_ord.index(key)],clip_on=False)
                            a0.plot(x_axis[-1], dict[key][-1], color=color_dict[color_key], marker='D', zorder=z_point1[end_ord.index(key)],clip_on=False)
                            lower = - err_dict[key][0, :plot_points] + np.ravel(dict[key][:plot_points])
                            higher = err_dict[key][1, :plot_points] + np.ravel(dict[key][:plot_points])
                            a0.fill_between(x_axis[:plot_points], lower, higher, color=color_dict[color_key], alpha=0.075,zorder=z_ebar)
                        else:
                            color_key = key
                            line_axis, = a1.plot(x_axis[:plot_points], dict[key][:plot_points], color=color_dict[color_key], linestyle=linestyle, label=label,zorder=z_line)
                            a1.plot(x_axis[0],dict[key][0],color=color_dict[color_key],marker='o',zorder=z_point0[strt_ord.index(key)],clip_on=False)
                            a1.plot(x_axis[-1], dict[key][-1], color=color_dict[color_key], marker='D', zorder=z_point1[end_ord.index(key)],clip_on=False)
                            lower = - err_dict[key][0, :plot_points] + np.ravel(dict[key][:plot_points])
                            higher = err_dict[key][1, :plot_points] + np.ravel(dict[key][:plot_points])
                            a1.fill_between(x_axis[:plot_points], lower, higher, color=color_dict[color_key], alpha=0.075,zorder=z_ebar)
                    else:
                        print('Resolution is not None')
                        if color_assign:
                            print("color_assign")
                            line_axis = plt.errorbar(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution],c=color_dict[key.split('_')[0]], yerr=err_dict[key][:, :plot_points:resolution], label=key.replace('_',' '), capsize=5, linestyle=linestyle)
                        else:
                            print("not color_assign")
                            line_axis = plt.errorbar(x_axis[:plot_points:resolution], dict[key][:plot_points:resolution], yerr=err_dict[key][:, :plot_points:resolution], label=key.replace('_',' '), capsize=5, linestyle=linestyle)

                legend_list.append(line_axis)

            if logy:
                for a in ax:
                    a.set_yscale('log')
                    a.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
                    a.yaxis.set_minor_locator(matplotlib.ticker.LogLocator(subs='all'))

            if logx:
                for a in ax:
                    a.set_xscale('log')
                a0 = ax[0]
                a0.set_xticks([1,10,50,100,200])
                a0.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
                a0.set_ylabel('MSE')

            # Setup markers to be fully visible
            plt.legend(handles=legend_list, loc='lower center',ncol=int(len(legend_list)/2)+1,bbox_to_anchor=(0.5,-.4))

            for a in ax:
                a.grid(True, axis='both', which='major', color='b', alpha=0.2)

            print(legend_list)


        plotDict(ax_dict[data_name],min_dict,'_oD.png', plot_points=plot_points, logy=True,logx=True, avg_dict=avg_dict, err_dict=quan2575_dict, data_name=data_name,
                dash_group=dash_group, dash_label=dash_label, solid_label=solid_label, resolution=resolution, worse_model_mode=worse_model_mode, plot_xlabel=True)

    plt.savefig(os.path.join(datas[0], 'Final_fig.png'), bbox_inches='tight', dpi=300, transparent=True)


if __name__ == '__main__':
    work_dir = 3*['../mm_bench_multi_eval_backup_to_Ashwin']
    MeanAvgnMinMSEvsTry_all(work_dir[0])
    datasets = ['Yang_sim','Chen','Peurifoy']
    DrawAggregateMeanAvgnMSEPlot(work_dir, datasets)


