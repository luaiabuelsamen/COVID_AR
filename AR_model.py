"""
MECH 309
Data from
https://www.donneesquebec.ca/recherche/dataset/covid-19-portrait-quotidien-des-cas-confirmes
Sample code for students
"""
from doctest import ELLIPSIS
import numpy as np
from scipy import linalg
from matplotlib import pyplot as plt

# %% Plotting parameters

# plt.rc('text', usetex=True)
plt.rc('font', family='serif', size=16)
plt.rc('lines', linewidth=2)
plt.rc('axes', grid=True)
plt.rc('grid', linestyle='--')

# %% Load data

data = np.loadtxt('./COVID19QC.csv', dtype=float,delimiter=',', skiprows=2, usecols=(2,))

y = data.copy()

# Plot the raw COVID-19 data, new cases per day versus time
N = data.size

t = np.linspace(0, N, N, endpoint=True)

fig, ax = plt.subplots()

fig.set_size_inches(18.5, 10.5, forward=True)
ax.set_title(r'New COVID-19 Cases per Day versus Time')
ax.set_xlabel(r'$t$ (days)')

ax.set_ylabel(r'$y_k$ (cases)')

plt.plot(t, y, label='New Cases per Day')

ax. legend(loc='center left', bbox_to_anchor=(1, 10.5))
fig.savefig('COVID19_cases_vs_time.pdf')

# %% AR model functions

#print (data)

def form_A_b(y, ell):
    N = y.size
    A = np.ones((N - ell, ell))
    for n in range(N - ell):
        A[n] = y[n : ell +n]
    b = y[ell : N]
    return A, b

def fit(A, b):
    # This function solves Ax = b where x are the AR model parameters
    return linalg.solve(A.T @ A, A.T @ b)

def predict(y, beta):
    N = y.size
    ell = beta.size
    y_pred =y * 0
    for n in range(ell, N):
        y_pred[n] = beta.reshape(1, -1) @ y[n - ell : n].reshape(-1, 1)
        # Set the first ell predictions to the average of the first ell measurments
        y_pred_mean = np.mean(y[:ell]) # Don't change this
        y_pred[:ell] = np.ones(ell) * y_pred_mean # Don't change this
        return y_pred

# %% Fit (train) the AR model

N_start = 49 # start day, don't change
N_end = 399 # end day, don't change

ell = 16 # memory of AR model, change to 14, 15, 16
t = np.arange(N_start, N_end) # time, don't change
y_scale = np.max(y[N_start:N_end])

ly=y/y_scale # non-dimensionalize the data


A, b = form_A_b(y[N_start:N_end], ell) # form A, b matrices
beta = fit(A, b) # find the beta parameters

y=y*y_scale # dimensionalize the data again

y_pred = predict(y[N_start:N_end], beta)

e = np.abs(y[N_start:N_end] - y_pred)


# Plotting
fig, ax = plt.subplots(2, 1)

fig.set_size_inches(18.5, 10.5, forward=True)

ax[0].set_title(r'AR Model Train')

for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (days)')
    ax[0].set_ylabel(r'$y_k¢ (cases)')
    ax[1].set_ylabel(r'$e_k$ (cases)')

    ax[0].plot(t[ell:], y_pred[ell:], label=r'sy_{k, pred, \ell=%s}$' % ell)
    ax[0].plot(t[ell:], y[N_start + ell:N_end], '--', label=r'$y {k, true}$')
    ax[1].plot(t[ell:], e[ell:], label=r'$e_{k, \ell=%s}$' % ell)

    ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))

fig.tight_layout()
plt.show()

fig. savefig('AR_response_train.pdf')

#%% Teast

y = data.copy()

N_start = 400 # start day, don't change
N_end = 625 # end day, don't change

t = np.arange(N_start, N_end)

y_pred = predict(y[N_start:N_end], beta)

# predictions
# Compute various metrics associated with predictino error

e = np.abs(y[N_start:N_end] - y_pred)

e_rel = np.abs(y[N_start:N_end] - y_pred)/y[N_start:N_end]

mu = np.mean(e)

sigma = np.std(e)

mu_e_rel = np.mean(e_rel)
sigma_e_rel = np.std(e_rel)

fig, ax = plt.subplots(3, 1)
# Format axes
fig.set_size_inches(18.5, 10.5, forward=True)
ax[0].set_title(r'AR Model Test')
for a in np.ravel(ax):
    a.set_xlabel(r'$t$ (days)')
ax[0].set_ylabel(r'$y_k$ (cases)')
ax[1].set_ylabel(r'$e_k$ (cases)')
ax[1].hlines(y=mu, xmin=N_start, xmax=N_end, linewidth=2, color='r')
ax[1].hlines(y=mu + 3*sigma, xmin=N_start, xmax=N_end, linewidth=2, color= 'r')
ax[1].hlines(y=mu - 3*sigma, xmin=N_start, xmax=N_end, linewidth=2, color= 'r')
ax[2].set_ylabel(r'$e_{k, rel}$ (%)')
ax[2].hlines(y=mu_e_rel, xmin=N_start, xmax=N_end, linewidth=2, color='r')

 

  

ax[2].hlines(y-mu_e_rel + 3*sigma_e_rel, xmin=N_start, xmax=N_end, linewidth=2,

color='r')

ax[2].hlines(y=mu_e_rel - 3*sigma_e_rel, xmin=N_start, xmax=N_end, linewidth=2,

color='r')

ax[0].plot(t[ell:], y_pred[ell:], label=r'$y_{k, pred, \ell=%s}$' % ell)
ax[1].plot(t[ell:], e[ell:], label=r'$e_{k, \ell=%s}$' % ell)
ax[0].plot(t[ell:], y[N_start + ell:N_end], '--', label=r'$y {k, true}$')
ax[2].plot(t[ell:], e_rel[ell:], label=r'$e_{k, rel, \ell-%s}$' % ell)
ax[0].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[1].legend(loc='center left', bbox_to_anchor=(1, 0.5))
ax[2].legend(loc='center left', bbox_to_anchor=(1, 0.5))
fig.tight_layout()

plt.show()

fig.savefig('AR_response_test_ell_%s.pdf' % ell)

print('Mean absolute error is ', mu, '\n')

print('Absolute error standard deviation is', sigma, '\n')
print('Mean relative error is ', mu_e_rel, '\n')
print('Relative error standard deviation is', sigma_e_rel, '\n')