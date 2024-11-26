from shapely.ops import unary_union

from matplotlib.path import Path

import matplotlib.patches as patches

import matplotlib.pyplot as pl
import pandas as pd
import numpy as np 
import matplotlib.ticker as ticker

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import colorsys
import matplotlib.colors as mc

from matplotlib import rcParams
from scipy.spatial.distance import pdist

import powerlaw
from scipy.stats import pareto, norm, gaussian_kde, lognorm, kstest
from fincoretails import distributions, unipareto, lognormal, general_expareto,unipareto,general_powpareto,general_algpareto, loglikelihood_ratio, aic

from utils import *

#declare figure layout
fig = plt.figure(layout="constrained",figsize=(18.3*cm*0.5, 10*cm))



gs2 = GridSpec(2, 2, figure=fig,hspace=0.2,bottom=0.6)

ax3 = fig.add_subplot(gs2[0, 0])
ax4 = fig.add_subplot(gs2[1, 0])
ax5 = fig.add_subplot(gs2[0, 1])
ax6 = fig.add_subplot(gs2[1, 1])

gs3 = GridSpec(1, 2, figure=fig,top=0.45)
ax7 = fig.add_subplot(gs3[0, 0])
ax8 = fig.add_subplot(gs3[0, 1])

gs5 = GridSpec(1, 1, figure=fig,bottom=0.1,hspace=0.2)
ax_sim = fig.add_subplot(gs5[0, 0])



font_size = 11 

x_m,y_m,x_p,y_p,x_r,y_r = load_data('dk')


def city_plot(ax3,ax4,city_code= 84910847):
    #load data 
    df_piecewise = pd.read_csv(f'data/{city_code}_piecewise.csv')
    df_piecewise_ratio = pd.read_csv(f'data/{city_code}_piecewise_ratio.csv')   
    df_piecewise_fit = pd.read_csv(f'data/{city_code}_piecewise_fit.csv')
    p = np.load(f'data/{city_code}_piecewise_params.npy')

    xx = df_piecewise['x_m']
    hist_mv = df_piecewise['y_m']
    hist_pd = df_piecewise['y_p']
    
    x_r = df_piecewise_ratio['x']
    ratio = df_piecewise_ratio['y']
    
    x_pw = df_piecewise_fit['x']
    y_pw = df_piecewise_fit['y']
    
    ms=1
    
    ax3.plot(xx,hist_mv,'.',color=colors[1],label='movement distance',markersize=ms)
    ax3.plot(xx,hist_pd,'.',color=colors[2],label='pair distribution function',markersize=ms)
    
    ax4.axvline(x=10**p[0], color=colors[-1], linestyle=':',lw=2)
    ax4.plot(x_pw, y_pw,'k--',alpha=0.5)
    ax4.plot(x_r,ratio,'.',color='k',markersize=ms)

    format_axes_log(ax3)
    format_axes_log(ax4)
    ax4.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right')
    
    ax4.axvspan(0,10**p[0], color=colors[3],edgecolor=None, alpha=0.2,lw=0)
    ax4.axvspan(10**p[0],1e6, color=colors[4],edgecolor=None, alpha=0.2,lw=0)

    ax3.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='right')
    ax3.yaxis.set_label_coords(-0.05, 0.96)
    ax4.set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='right')
    ax4.yaxis.set_label_coords(-0.05, 0.96)
    ax4.set_yticks([1e2,1e0])

    ax4.xaxis.set_label_coords(1, -0.25)
    ax4.set_xticks(np.logspace(1,6,6))
    ax3.set_xticks(ax4.get_xticks())
    ax3.set_xlim([10,1e6])
    ax4.set_xlim([10,1e6])

    ax3.tick_params(which='both', direction='in')  # 'both' applies the settings to both major and minor ticks
    ax4.tick_params(which='both', direction='in')  # 'both' applies the settings to both major and minor ticks

    
city_plot(ax3,ax4,city_code= 84910847)
city_plot(ax5,ax6,city_code= 51018355)
ax3.set_yticks([1e-5,1e-7])
ax5.set_yticks([1e-6,1e-10])


### exponents distribution 

data1 = np.load('data/pw-exponents_1.npy')
data2 = np.load('data/pw-exponents-2.npy')
print(f'n_city = {len(data1)}')

kde1 = gaussian_kde(data1)
kde2= gaussian_kde(data2)

x1 = np.linspace(min(data1)-0.1, max(data1)+0.1, 100)
x2 = np.linspace(min(data2), max(data2), 100)

ax7.plot(-x1, kde1(x1), color=colors[3])
ax7.fill_between(-x1, kde1(x1), color=colors[3], alpha=0.2)
ax7.plot(-x2, kde2(x2), color=colors[4])
ax7.fill_between(-x2, kde2(x2), color=colors[4], alpha=0.2)

ax7.axvline(-np.mean(data1), color=colors[3], linestyle='--',lw=1)
ax7.axvline(-np.mean(data2), color=colors[4], linestyle='--',lw=1)

ax7.set_xticks([2,0.5])
ax7.set_xticklabels(['2','0.5'])

ax7.set_xlabel('exponent value',rotation=0, labelpad=0, va='top', ha='right')
ax7.xaxis.set_label_coords(1, -0.2)
ax7.spines['right'].set_visible(False)
ax7.spines['top'].set_visible(False)
ax7.set_ylim([0,2.2])
ax7.set_xlim([-0.5,3])
ax7.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='right')
ax7.yaxis.set_label_coords(-0.05, 0.98)
ax7.set_yticklabels(['0','1',''])

#ax8

data = np.load('data/xmins.npy')
print(f'n_city = {len(data)}')

s = 0.954  # shape parameter

bins_ln = np.logspace(2,5,20)
shape, loc, scale = lognorm.fit(data, floc=0)
hist_ln,bins_ln = np.histogram(data,bins_ln,density=True)
xx = (bins_ln[1:]+bins_ln[:-1])/2
pdf = lognorm.pdf(xx, shape, loc=loc, scale=scale)
ax8.set_ylabel('pdf',rotation=0, labelpad=0, va='top', ha='right')
ax8.yaxis.set_label_coords(-0.05, 0.98)

ax8.set_xlabel('mobility city radius (m)',rotation=0, labelpad=0, va='top', ha='right')
ax8.xaxis.set_label_coords(1,-0.2)
ax8.tick_params(which='both', direction='in')

fit = powerlaw.Fit(data)
x = xx
alpha = fit.power_law.alpha
x_min = fit.power_law.xmin
pdf_pl = x**(-alpha) / (alpha - 1) / x_min if alpha > 1 else np.zeros_like(x)


ng = np.random.default_rng()

# KS test vs power law
ks, p = kstest(data, 'powerlaw', args=(alpha, x_min))
print(f"KS test vs power law: {ks}, p-value: {p}")
#test lognormal
ks, p = kstest(data, 'lognorm', args=(shape, loc, scale))
print(f"KS test vs lognormal: {ks}, p-value: {p}")

mu,sigma = lognormal.fit_params(data)
ax8.plot(xx,lognormal.pdf(xx, mu,sigma),'k--',lw=1,label='log-normal')
alpha, xmin, beta = general_powpareto.fit_params(data)
ax8.plot(xx,general_powpareto.pdf(xx, alpha, xmin, beta),'k:',lw=1,label='pow-Pareto')


ax8.plot(xx,hist_ln,'.',color=colors[5],ms=5.5)

ax8.legend(loc='lower left',frameon=False,fontsize=6)

ax8.set_ylim([1e-7,1e-3])
ax8.set_xlim([1e2,5e4])
ax8.set_yticks([1e-6,1e-4])

print(f'x_min {x_min}, alpha {alpha}, beta {beta}, mu {mu}, sigma {sigma}')
format_axes_log(ax8)

R, p = fit.distribution_compare('power_law', 'lognormal', normalized_ratio=True)


print(f"\nComparison with Lognormal Distribution:")
print(f"Likelihood ratio (R): {R}")
print(f"p-value: {p}")

#exp compare

R, p = fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)
print(f"\nComparison with Exponential Distribution:")
print(f"Likelihood ratio (R): {R}")
print(f"p-value: {p}")

#streched exp compare
R, p = fit.distribution_compare('power_law', 'stretched_exponential', normalized_ratio=True)
print(f"\nComparison with stretched Exponential Distribution:")
print(f"Likelihood ratio (R): {R}")
print(f"p-value: {p}")

# Compare the fitted power-law distribution with a truncated power law (Pareto) distribution
R, p = fit.distribution_compare('power_law', 'truncated_power_law')

print(f"\nComparison with Truncated Power-law (Pareto) Distribution:")
print(f"Likelihood ratio (R): {R}")
print(f"p-value: {p}")

# Compare the fitted power-law distribution with a truncated power law (Pareto) distribution
R, p = fit.distribution_compare('lognormal', 'truncated_power_law')

print(f"\nComparison with lognormal and Truncated Power-law (Pareto) Distribution:")
print(f"Likelihood ratio (R): {R}")
print(f"p-value: {p}")

mu, sigma = lognormal.fit_params(data)

Nsample = len(data)
ll_gpp = general_powpareto.loglikelihoods(data, alpha, xmin, beta)
ll_logn = lognormal.loglikelihoods(data, mu, sigma)
logL_gpp = ll_gpp.sum()
logL_logn = ll_logn.sum()

R, p = loglikelihood_ratio(ll_gpp, ll_logn)
AIC_gpp= aic(logL_gpp, number_of_free_parameters=3, nsamples=Nsample)
AIC_logn = aic(logL_logn, number_of_free_parameters=2, nsamples=Nsample)



print(f"gp pareto: {alpha=:4.2f}, {xmin=:4.2f}, {beta=:4.2f} ")
print(f"lognormal: {mu=:4.2f}, {sigma=:4.2f}")
print(f"logL gp pareto = {logL_gpp:4.2f}")
print(f"logL lognormal = {logL_logn:4.2f}")
print(f"log-likelihood ratio R={R:4.2f} with significance level p={p:4.2e}")
print(f"AIC gp pareto = {AIC_gpp:4.2f}")
print(f"AIC lognormal = {AIC_logn:4.2f}")
    



# List of colors for elements and their fills

darker_colors = [darken_color(color, 0.99) for color in colors]
lighter_colors = [darken_color(color, 1.5) for color in colors]
alpha_colors = [adjust_alpha(color, 1.5) for color in colors]

# Parameters
r1, r2, d = 1.2, 1.45, 6
c1 = np.array([0, d/5])
c2 = np.array([d, 0])

def format_axes_geo(ax):
    ax.axis('off')
    ax.set_aspect('equal')
    ax.set_xticks([])
    ax.set_yticks([])

def format_axes(ax):
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)

# Ellipse parameters (a, b) computed from circle radii and centers distance
a = 0.55 * d / 2 + max(r1, r2) + max(r1, r2)+0.3
b = 0.6  * (np.sqrt(a**2 - (d/2)**2) + max(r1, r2)) +0.2

# Center of the oval
center = (c1 + c2) / 2

# 4 control points for upper and lower half of the oval, respectively.
upper_cp = np.array([(center[0] - a, center[1]), (center[0] - a, center[1] + b), 
            (center[0] + a, center[1] + b), (center[0] + a, center[1])])
lower_cp = np.array([(center[0] + a, center[1]), (center[0] + a, center[1] - b), 
            (center[0] - a, center[1] - b), (center[0] - a, center[1])])

# Angle of the line connecting the two cities
angle = np.arctan2(c2[1] - c1[1], c2[0] - c1[0])

# Rotation matrix
rot = np.array([[np.cos(angle), -np.sin(angle)], 
                [np.sin(angle),  np.cos(angle)]])

# Rotate control points
upper_cp = np.dot(upper_cp - center, rot.T) + center
lower_cp = np.dot(lower_cp - center, rot.T) + center

# Introduce smooth noise
np.random.seed(0)  
for cp in [upper_cp, lower_cp]:
    for i in range(len(cp)):
        cp[i][0] += np.sin(cp[i][0]) * 0.25
        cp[i][1] += np.sin(cp[i][1]) * 0.25

codes = [Path.MOVETO, Path.CURVE4, Path.CURVE4, Path.CURVE4]
upper_path = Path(upper_cp, codes)
lower_path = Path(lower_cp, codes)


# Add the paths
patch_upper = patches.PathPatch(upper_path, facecolor= adjust_alpha(colors[4],0.2), lw=1.3, edgecolor=colors[4],zorder =1)
patch_lower = patches.PathPatch(lower_path, facecolor=adjust_alpha(colors[4],0.2), lw=1.3, edgecolor=colors[4],zorder =1)

ax7.tick_params(which='both', direction='in')  

#simulation
bins = np.logspace(1,5.7,100)
x = (bins[:-1]+bins[1:])/2


intrinsic_cost_sim = np.load('data/intrinsic_cost.npy')
x_city = np.load('data/xmins.npy')


ax8.set_yticks([1e-6,1e-5,1e-4])

#sim 

ax_sim.axvspan(10,np.min(x_city), alpha=0.2, color=colors[3],lw=0) 
ax_sim.axvspan(np.max(x_city),int(1e6), alpha=0.2, color=colors[4],lw=0)
ax_sim.axvspan(np.min(x_city),np.max(x_city), alpha=0.1, color=colors[5],lw=0)
ax_sim.vlines(np.min(x_city),0,200,color=colors[5],linestyle=':', lw = 2)
ax_sim.vlines(np.max(x_city),0,200,color=colors[5],linestyle=':', lw = 2 )


ax_sim.plot(x,intrinsic_cost_sim,'.',color='k',label='intrinsic moving distance',markersize=1)
x_all = np.logspace(1,6.1,100)
ax_sim.plot(x_all[:-1],x_all[:-1]**-1*intrinsic_cost_sim[0]*x_all[0]*2 ,'--',color='k',lw=0.5,label='$y=x^{-1}$')

ax_sim.set_ylim([1e-4,2e2])
ax_sim.set_xlim([10,1e6])
format_axes_log(ax_sim)
ax_sim.tick_params(which='both', direction='in')  
ax_sim.set_xlabel('distance (m)',rotation=0, labelpad=0, va='top', ha='right',x=1,y=0.1)
ax_sim.set_ylabel('ratio',rotation=0, labelpad=0, va='top', ha='left',x=0.1,y=1)
ax_sim.set_yticks([1e-4,1e-2,1e0])

# figure layout 


gs2.tight_layout(fig,pad=0.1)
gs2.update(left=0.1,right=0.95,top=0.96,bottom=0.564,wspace=0.3, hspace=0.35)

gs3.tight_layout(fig,pad=0.1)
gs3.update(left=0.1,right=0.95,top=0.475,bottom=0.29,wspace=0.3, hspace=0.2)

gs5.tight_layout(fig,pad=0.1)
gs5.update(left=0.1,right=0.95,top=0.22,bottom=0.07,wspace=0.3, hspace=0.25)

# save
plt.tight_layout()
plt.savefig(f'figures/figure_3b.png',dpi=300)
plt.savefig(f'figures/figure_3b.pdf')

plt.close('all')