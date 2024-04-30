import pandas as pd
import numpy as np 

import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib import rcParams

from utils import *


fig = plt.figure(layout="constrained",figsize=(18.3*cm*0.5, 10*cm))

w_space = 0.09
column_w = (1-w_space*3)/2
positions = np.cumsum([w_space,column_w]*2)

gs1 = GridSpec(2, 1, figure=fig,left=positions[0],right=positions[1],wspace=0,hspace=0)
ax2 = fig.add_subplot(gs1[0, 0])
ax3 = fig.add_subplot(gs1[1,0])

gs2= GridSpec(2, 1, figure=fig,left=positions[2],right=positions[3],wspace=0,hspace=0)
ax5 = fig.add_subplot(gs2[0,0])
ax6 = fig.add_subplot(gs2[1, 0])

gs3 = GridSpec(2, 1, figure=fig,left=positions[0],right=positions[1],wspace=0,hspace=0)
ax8 = fig.add_subplot(gs3[0, 0])
ax9 = fig.add_subplot(gs3[1,0])

gs4 = GridSpec(2, 1, figure=fig,left=positions[2],right=positions[3],wspace=0,hspace=0)
ax11 = fig.add_subplot(gs4[0, 0])
ax12 = fig.add_subplot(gs4[1,0])


# load fr data
x_m,y_m,x_p,y_p,x_r,y_r = load_data('fr')

#fr plot
ax2.plot(x_p,y_p,'-',color=colors[2],label='pair distribution function')
ax2.plot(x_m,y_m,'-',color=colors[1],label='moving distance distribution')
ax2.set_ylim([0.8e-9,1.8e-2])

format_axes_log(ax3)
ax3.plot(x_r,y_r,'k.',ms=1,label=f'intrinsic distance cost')
slope = 1.07
intercept = compute_intercept(x_r,y_r,slope)
ax3.plot(x_p,x_p**(-slope)*10**(intercept), 'k--',label=f'slope: {slope:.2f}',lw=0.5)
ax3.text(0.95,0.65,
        r' s$={:.4g}$'.format(float(slope)),
        fontsize=6,ha='right',transform=ax3.transAxes)

format_axes_log(ax2)
ax2.set_title('france',loc='left',y=0.93)

ax3.set_xlim(7,2e6)
ax2.set_xlim(7,2e6)
ax2.set_xticks(np.logspace(1,6,6))
ax3.set_xticks(np.logspace(1,6,6))

ax2.tick_params(which='both', direction='in')
ax3.tick_params(which='both', direction='in')
ax3.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
ax2.set_ylabel('pdf', rotation=0, labelpad=0,va='top', ha='right')
ax3.set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='right')
ax2.yaxis.set_label_coords(-0.05, 1)
ax3.yaxis.set_label_coords(-0.05, 1)
ax3.xaxis.set_label_coords(1, -0.3)

handles, labels = ax2.get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1),frameon=False, ncol=3)



#########    DAY-TO-DAY MOBILITY   ###############

def plot_city(axs,x_m,y_m,x_p,y_p,x_r,y_r,colors,name,slope):
    axs[1].plot(x_p,y_p ,'-',color=colors[2],label='pairwise distance')
    axs[1].plot(x_m,y_m,'-',color=colors[1],label='moving distance ')
  
    axs[2].plot(x_r,y_r,'k.',ms=1,label=f'intrinsic moving law')
    intercept = compute_intercept(x_r,y_r,slope)
    axs[2].plot(x_r,x_r**(-slope)*10**(intercept), 'k--',label=f'slope: {slope:.2f}',lw=0.5)
    
    format_axes_log(axs[1])
    format_axes_log(axs[2])

    axs[1].set_title(name,loc='left',y=0.93)
    axs[1].set_xticks(np.logspace(1,6,6))
    axs[2].set_xticks(np.logspace(1,6,6))
    axs[2].set_yticks([0.1,10])
    axs[1].set_yticks([1e-7,1e-5])

    axs[2].text(0.95,0.65,
        r' s$={:.2g}$'.format(float(slope)),
        fontsize=6,ha='right',transform=axs[2].transAxes)
    axs[1].tick_params(which='both', direction='in')
    axs[2].tick_params(which='both', direction='in')
    axs[2].set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
    axs[1].set_ylabel('pdf', rotation=0, labelpad=0,va='top', ha='right')
    axs[2].set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='right')
    axs[1].yaxis.set_label_coords(-0.05, 1)
    axs[2].yaxis.set_label_coords(-0.05, 1)
    axs[2].xaxis.set_label_coords(1, -0.3)



#load and plot data
x_m,y_m,x_p,y_p,x_r,y_r = load_data('singapore')
plot_city([1,ax11,ax12],x_m,y_m,x_p,y_p,x_r,y_r,colors,'singapore',slope=0.89)
x_m,y_m,x_p,y_p,x_r,y_r = load_data('san francisco')
plot_city([0,ax5,ax6],x_m,y_m,x_p,y_p,x_r,y_r,colors,'san francisco',slope=.94)
x_m,y_m,x_p,y_p,x_r,y_r = load_data('houston')
plot_city([1,ax8,ax9],x_m,y_m,x_p,y_p,x_r,y_r,colors,'houston',slope=0.95)


# layout
#gs1.tight_layout(fig,pad=0.1)
gs1.update(left=positions[0],right=positions[1],wspace=0.2, hspace=0.3,top=0.9,bottom=0.55)
#gs2.tight_layout(fig,pad=0.1)
gs2.update(left=positions[2],right=positions[3],wspace=0.2, hspace=0.3,top=0.9,bottom=0.55)
#gs3.tight_layout(fig,pad=0.1)
gs3.update(left=positions[0],right=positions[1],wspace=0.2, hspace=0.3,top=0.44,bottom=0.08)
#gs4.tight_layout(fig,pad=0.1)
gs4.update(left=positions[2],right=positions[3],wspace=0.2, hspace=0.3,top=0.44,bottom=0.08)
plt.tight_layout()

#save figure
plt.savefig(f'figures/figure-4.pdf')
plt.savefig(f'figures/figure-4.png',dpi=300)
plt.close('all')