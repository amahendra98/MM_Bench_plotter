import numpy as np
import matplotlib.pyplot as plt
import os

dirx = '../multi-eval-dat/my-plot-data'

DL = ('cINN','INN','MDN','NA','VAE','NN','GA1','GA2')
dataset = ('Chen','Peurifoy','Yang')

ids = dict(zip(dataset,(2,1,4))) # Data Sample from Set
sec_lim_dict = dict(zip(dataset,((-0.5,0.5),(-0.5,0.5),(-0.5,0.5)))) # Ylimits of 100% line

# Dataset definitions
x_dat_dict = dict(zip(dataset,((240,2000),(400,800),(100,500))))
x_ax_dict = dict(zip(dataset,('Wavelength (nm)','Wavelength (nm)','Wavelength (THz)')))
y_ax_dict = dict(zip(dataset,('Absorption',r'$\frac{\sigma}{\pi r^2}$','Transmission')))
color_dict = dict(zip(DL,('tab:red','tab:pink','tab:brown','tab:green','tab:olive','tab:orange','tab:blue','tab:gray')))

f, ax = plt.subplots(nrows=2,ncols=3,gridspec_kw={'height_ratios':[1,4]},sharex='col')
f.set_figwidth(15)
ax_dict = dict(zip(dataset,zip(*ax)))
legend_list = []

for s,set in enumerate(dataset):
    ax_sec,ax_main = ax_dict[set]
    x0, xf = x_dat_dict[set]
    ax_main.set_xlabel(x_ax_dict[set])
    ax_main.set_ylabel(y_ax_dict[set])
    ax_main.set_xlim(x0,xf)
    ax_sec.set_ylabel('Error')
    ax_sec.set_ylim(sec_lim_dict[set])

    yt = np.genfromtxt(os.path.join(dirx,set,'Ytruth.csv'))[ids[set],:]
    line_axis, = ax_main.plot(np.linspace(x0,xf,len(yt)),yt,color='black',linestyle='solid',linewidth=4,
                              label='True Spectrum')
    if s == 1:
        legend_list.append(line_axis)

    y_sec = np.empty((len(DL),len(yt)))
    y_main = np.empty_like(y_sec)

    for i,invs in enumerate(DL):
        y_main[i] = np.genfromtxt(os.path.join(dirx,set,'Ypred-'+invs+'-'+set+'-'+str(ids[set])+'.csv'))[0]
        y_sec[i] = y_main[i] - yt

    ymax = np.max(np.abs(y_sec))

    for (Ym,Ys,invs) in (zip(y_main,y_sec,DL)):
        line_axis, = ax_sec.plot(np.linspace(x0, xf, len(Ys)), Ys/ymax, color=color_dict[invs],
                                 linestyle='solid',linewidth=1.5, label=invs)

        ax_main.plot(np.linspace(x0, xf, len(Ym)), Ym, color=color_dict[invs],
                                  linestyle='dashed', linewidth=2)
        if s == 1:
            legend_list.append(line_axis)

plt.legend(handles=legend_list,loc='lower center',bbox_to_anchor=(-.9,-.5),ncol=int((len(DL)+2)/2))
plt.tight_layout()
plt.subplots_adjust(wspace=0.4)
plt.savefig(os.path.join(dirx,'plots200.png'),transparent=True,dpi=300)