# %% [markdown]
# # Modelling commuting in Denmark: gravity vs radiation
#
# This notebook reproduces everything in the **Commuting Playground**:
#
# 1. load the data (commuting flows, municipalities, distances)
# 2. the **gravity model**, with Zipf's parameters and with a Poisson regression fit
# 3. the **radiation model**, which has no parameters
# 4. the score (**CPC**), the observed-vs-predicted plot and the biggest misses
#
# Data: Statistics Denmark, table PEND100 (2024). You need `pandas`, `numpy`, `matplotlib` and `statsmodels`.

# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

flows = pd.read_csv('flows.csv')                  # origin, destination, commuters (98 x 98 pairs)
mun = pd.read_csv('municipalities.csv')           # workers_living, jobs, lon, lat
dist = pd.read_csv('distances.csv')               # distance_km between centres
flows.head()

# %% [markdown]
# ## 1. One table with everything we need
#
# We keep only flows **between different municipalities** (people working in their own municipality are not modelled), and attach to each pair:
# - `m_i`: employed people living in the origin
# - `m_j`: jobs in the destination
# - `d`: distance in km

# %%
df = flows[flows.origin != flows.destination].merge(
    dist[['origin', 'destination', 'distance_km']], on=['origin', 'destination'])
df = df.merge(mun[['municipality', 'workers_living']].rename(columns={'municipality': 'origin', 'workers_living': 'm_i'}), on='origin')
df = df.merge(mun[['municipality', 'jobs']].rename(columns={'municipality': 'destination', 'jobs': 'm_j'}), on='destination')
df = df.rename(columns={'distance_km': 'd', 'commuters': 'T'})
print(len(df), 'pairs,', (df['T'] > 0).sum(), 'with at least one commuter,', df['T'].sum(), 'commuters')
df[['origin', 'destination', 'T', 'm_i', 'm_j', 'd']].head()

# %% [markdown]
# ## 2. The score: common part of commuters (CPC)
#
# $$\text{CPC} = \frac{2 \sum \min(T^{obs}, T^{pred})}{\sum T^{obs} + \sum T^{pred}}$$
#
# The share of commuters the model puts in the right place: 1 is perfect, 0 is nothing right.

# %%
def cpc(obs, pred):
    return 2 * np.minimum(obs, pred).sum() / (obs.sum() + pred.sum())

# %% [markdown]
# ## 3. Gravity model
#
# $$T_{ij} = K \, m_i^{\alpha} \, m_j^{\beta} \, d_{ij}^{-\gamma}$$
#
# `K` is set so that the model predicts the right total number of commuters.

# %%
def gravity(df, alpha, beta, gamma):
    raw = df.m_i**alpha * df.m_j**beta * df.d**(-gamma)
    return raw * df['T'].sum() / raw.sum()        # K: match the total

df['zipf'] = gravity(df, 1, 1, 1)
print('Zipf (alpha = beta = gamma = 1): CPC =', round(cpc(df['T'], df.zipf), 3))

# %% [markdown]
# ### Fitting α, β, γ with a Poisson regression
#
# Take logs and the gravity model is linear in its parameters:
#
# $$\log T_{ij} = \log K + \alpha \log m_i + \beta \log m_j - \gamma \log d_{ij}$$
#
# Two ways to fit it:
#
# - **Linear regression on log T** (ordinary least squares). Simple, but it cannot use pairs with zero commuters (log 0 is −∞), and it treats a miss of 5 vs 50 commuters as just as bad as 5,000 vs 50,000.
# - **Poisson regression**. It assumes each flow is a count, $T_{ij} \sim \text{Poisson}(\mu_{ij})$ with $\log \mu_{ij}$ given by the formula above, and picks the parameters that make the observed counts most likely. Zeros are fine, and because the variance of a Poisson count equals its mean, big flows carry more weight. As a bonus, the predictions add up exactly to the observed total.
#
# Let's compare them.

# %%
X = sm.add_constant(pd.DataFrame({'log_m_i': np.log(df.m_i), 'log_m_j': np.log(df.m_j), 'minus_log_d': -np.log(df.d)}))

# (a) linear regression on log T, only pairs with T > 0
pos = df['T'] > 0
ols = sm.OLS(np.log(df.loc[pos, 'T']), X[pos]).fit()
print('Log-linear OLS: alpha = %.2f, beta = %.2f, gamma = %.2f' % tuple(ols.params[1:]))
df['ols'] = gravity(df, *ols.params[1:])
print('   CPC =', round(cpc(df['T'], df.ols), 3))

# (b) Poisson regression, all pairs including zeros
poi = sm.GLM(df['T'], X, family=sm.families.Poisson()).fit()
print('Poisson:        alpha = %.2f, beta = %.2f, gamma = %.2f' % tuple(poi.params[1:]))
df['gravity'] = poi.predict(X)
print('   CPC =', round(cpc(df['T'], df.gravity), 3))

# %% [markdown]
# ### Exponential distance decay
#
# With $d^{-\gamma}$ and $\gamma \approx 1$, a place 180 km away is only about 6 times less attractive than one 30 km away, so the model sends too many people on very long daily commutes. An exponential falls much faster:
#
# $$T_{ij} = K \, m_i^{\alpha} \, m_j^{\beta} \, e^{-d_{ij}/d_0}$$
#
# Same Poisson regression, with $-d$ instead of $-\log d$ as the last variable; then $d_0 = 1/\text{coefficient}$.

# %%
Xe = X.copy()
Xe['minus_d'] = -df.d
Xe = Xe.drop(columns='minus_log_d')
poi_exp = sm.GLM(df['T'], Xe, family=sm.families.Poisson()).fit()
a, b, k = poi_exp.params[1:]
print('Exponential: alpha = %.2f, beta = %.2f, d0 = %.0f km' % (a, b, 1 / k))
df['gravity_exp'] = poi_exp.predict(Xe)
print('   CPC =', round(cpc(df['T'], df.gravity_exp), 3))

# %% [markdown]
# ## 4. Radiation model (Simini et al., 2012)
#
# $$T_{ij} = T_i \, \frac{m_i \, n_j}{(m_i + s_{ij})(m_i + n_j + s_{ij})}$$
#
# - $T_i$: observed commuters leaving $i$
# - $m_i$: workers living in $i$; $n_j$: jobs in $j$
# - $s_{ij}$: jobs in all other municipalities closer to $i$ than $j$ is
#
# No parameters. Predictions are rescaled so they add up to $T_i$ for each origin.

# %%
names = mun.municipality.tolist()
W = mun.set_index('municipality').workers_living
J = mun.set_index('municipality').jobs
D = dist.pivot(index='origin', columns='destination', values='distance_km')
T_out = df.groupby('origin')['T'].sum()

rad = {}
for i in names:
    order = D.loc[i].drop(i).sort_values()          # other municipalities, nearest first
    s, row = 0.0, {}
    for j in order.index:
        row[j] = W[i] * J[j] / ((W[i] + s) * (W[i] + J[j] + s))
        s += J[j]                                    # jobs passed on the way
    total = sum(row.values())
    for j, v in row.items():
        rad[(i, j)] = v / total * T_out[i]
df['radiation'] = [rad[(o, d)] for o, d in zip(df.origin, df.destination)]
print('Radiation: CPC =', round(cpc(df['T'], df.radiation), 3))

# %% [markdown]
# ## 5. Observed vs predicted
#
# Each dot is a pair with at least one commuter. The solid line is a perfect prediction, the dashed lines are 10 times too many or too few. The orange line is the **median prediction** for flows of a given size: where it leaves the solid line, the model is systematically off.

# %%
def obs_vs_pred(df, col, ax):
    d = df[df['T'] > 0]
    ax.scatter(d['T'], d[col].clip(lower=0.1), s=3, alpha=0.2, color='#58355E')
    bins = np.logspace(0, 4.6, 16)
    mid, med = [], []
    for a, b in zip(bins[:-1], bins[1:]):
        sel = d[(d['T'] >= a) & (d['T'] < b)]
        if len(sel) >= 5:
            mid.append(np.sqrt(a * b)); med.append(sel[col].median())
    ax.plot(mid, med, '-o', color='#EC9A29', lw=2.5, ms=4, label='median prediction')
    x = np.array([1, 1e5])
    ax.plot(x, x, color='#58355E', lw=1.2, label='perfect')
    ax.plot(x, 10 * x, '--', color='#58355E', lw=0.8, label='10× off')
    ax.plot(x, x / 10, '--', color='#58355E', lw=0.8)
    ax.set(xscale='log', yscale='log', xlim=(1, 1e5), ylim=(0.1, 1e5),
           xlabel='Observed commuters', ylabel='Predicted commuters',
           title=f'{col}  (CPC {cpc(df["T"], df[col]):.2f})')
    ax.legend(loc='upper left', fontsize=8)

fig, axes = plt.subplots(2, 2, figsize=(11, 10))
for col, ax in zip(['zipf', 'gravity', 'gravity_exp', 'radiation'], axes.ravel()):
    obs_vs_pred(df, col, ax)
plt.tight_layout()

# %% [markdown]
# ## 6. Biggest misses, in number of people
#
# Ranked by the difference between observed and predicted commuters. These are the errors that cost the most in the CPC score.

# %%
def misses(df, col, k=8):
    d = df.assign(predicted=df[col].round(), difference=(df[col] - df['T']).round())
    cols = ['origin', 'destination', 'd', 'T', 'predicted', 'difference']
    too_few = d.nsmallest(k, 'difference')[cols]
    too_many = d.nlargest(k, 'difference')[cols]
    return too_few, too_many

too_few, too_many = misses(df, 'gravity')
print('Gravity predicts far too few:'); display(too_few)
print('Gravity predicts far too many:'); display(too_many)

# %% [markdown]
# ## Your turn
#
# - Repeat the misses table for `radiation`. Which pairs does each model get wrong?
# - Repeat it for `gravity_exp`. Which misses disappear with the exponential, and which remain?
# - Use only the men or only the women (columns `men`, `women` in `flows.csv`). Is γ different?
