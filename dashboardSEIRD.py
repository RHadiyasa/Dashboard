import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import streamlit as st
import pandas as pd
from scipy.integrate import odeint
import seaborn as sns

st.title("DASHBOARD DECISION SUPPORT MODEL USING SEIRD METHOD")
st.subheader("Data Default adalah data lapangan Provinsi DKI Jakarta")

col1, col2, col3 = st.columns(3)

effective_contact_rate = (0.1 * 10)
recovery_rate = 1/10


#@markdown Waktu Penerapan Protokol Kesehatan
Protokol_Kesehatan = st.sidebar.slider("Waktu Dimulai Protokol Kesehatan Hari ke:",min_value=0, max_value=90, value=15, step=1)
t_social_distancing = Protokol_Kesehatan
#@markdown Protokol Kesehatan (0 to 100%)
Protokol_Kesehatan_Level = st.sidebar.slider("Tingkat Protokol Kesehatan",min_value=0, max_value=90, value=60, step=1)
u_social_distancing = Protokol_Kesehatan_Level

#@markdown Masa Inkubasi Virus (Maks 14 Hari)
t_incubation = 7 #@param {type:"slider", min:1, max:14, step: 1}
#@markdown Masa Infeksius (Maks 10 Hari)
t_infective = 3 #@param {type:"slider", min:1, max:10, step:1}
# Population Size
N = st.sidebar.number_input("Enter Total Population: ",min_value=1, max_value=270000000, value=10610000)
# Jumlah Infeksi Aktif
n = st.sidebar.number_input("Enter Active Cases: ", min_value=1, max_value=270000000, value=19195)

R0 = st.sidebar.slider(
    "Tentukan Tingkat Transmission Rate (R0)",min_value=0.1, max_value=50.0, value=2.5, step=0.1
)

# initial number of infected and recovered individuals
e_initial = n/N
i_initial = 0.00
r_initial = 0
d_initial = 0
s_initial = 1 - e_initial - i_initial - r_initial - d_initial

alpha = 1/t_incubation
gamma = 1/t_infective
beta = R0*gamma
xi = 1 - 0.1



def step(t):
    return 1 if t >= 1*t_social_distancing else 0
  
# SEIR model differential equations.
def deriv(x, t, u, alpha, beta, gamma):
    s, e, i, r, d = x
    dsdt = -(1-u*step(t)/100)*beta * s * i - ( gamma * i - (gamma * i * xi)) 
    dedt =  (1-u*step(t)/100)*beta * s * i - alpha * e
    didt = alpha * e - gamma * i
    drdt = (gamma * i * xi)
    dddt = gamma * i - (gamma * i * xi)
    return [dsdt, dedt, didt, drdt, dddt]

days = range(0,3*30)

t = np.linspace(0, 3*30,3*30)
x_initial = s_initial, e_initial, i_initial, r_initial, d_initial
s, e, i, r, d = odeint(deriv, x_initial, t, args=(u_social_distancing, alpha, beta, gamma)).T
s0, e0, i0, r0, d0 = odeint(deriv, x_initial, t, args=(0, alpha, beta, gamma)).T

# plot the data
fig1 = plt.figure(figsize=(15, 18))
ax = [fig1.add_subplot(311, axisbelow=True), 
      fig1.add_subplot(312),fig1.add_subplot(313, axisbelow=True)]

pal = sns.color_palette()

ax[2].stackplot(t/1, N*s0, N*e0, N*i0, N*r0, N*d0, colors=pal, alpha=0.6)
ax[2].set_title('Populasi Retan dan Sembuh TANPA Protokol Kesehatan'.format(u_social_distancing))
ax[2].set_xlabel('Weeks following Initial Area Exposure')
ax[2].set_xlim(0, t[-1]/1)
ax[2].set_ylim(0, N)
ax[2].legend([
    'Susceptible', 
    'Exposed/no symptoms', 
    'Infectious/symptomatic',
    'Recovered',
    'Death'], 
    loc='best')
ax[2].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)
ax[2].plot(np.array([0, t[-1]])/1, [N/R0, N/R0], lw=3, label='herd immunity')
ax[2].annotate("New Normal",
    (t[-1]/1, N/R0), (t[-1]/1 - 8, N/R0 - N/5),
    arrowprops=dict(arrowstyle='->'))

ax[0].stackplot(t/1, N*s, N*e, N*i, N*r, N*d, colors=pal, alpha=0.6)
ax[0].set_title('Populasi Retan dan Sembuh Dengan{0:3.0f}% Protokol Kesehatan'.format(u_social_distancing))
ax[0].set_xlabel('Weeks following Initial Area Exposure')
ax[0].set_xlim(0, t[-1]/1)
ax[0].set_ylim(0, N)
ax[0].legend([
    'Susceptible', 
    'Exposed/no symptoms', 
    'Infectious/symptomatic',
    'Recovered',
    'Death'], 
    loc='best')
ax[0].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)
ax[0].plot(np.array([0, t[-1]])/1, [N/R0, N/R0], lw=3, label='herd immunity')
ax[0].annotate("Penerapan Protokol Kesehatan",
    (t_social_distancing, 0), (t_social_distancing + 1.5, N/10),
    arrowprops=dict(arrowstyle='->'))
ax[0].annotate("New Normal",
    (t[-1]/1, N/R0), (t[-1]/1 - 8, N/R0 - N/5),
    arrowprops=dict(arrowstyle='->'))

ax[1].stackplot(t/1, N*i0,N*e0, colors=pal[2:0:-1], alpha=0.5)
ax[1].stackplot(t/1, N*i, N*e, colors=pal[2:0:-1], alpha=0.5)
ax[1].set_title('Populasi Terinfeksi Tanpa Protokol Kesehatan dan Dengan{0:3.0f}% Protokol Kesehatan'.format(u_social_distancing))
ax[1].set_xlim(0, t[-1]/1)
ax[1].set_ylim(0, max(N, 1.05*max(N*(e + i))))
ax[1].set_xlabel('Weeks following Initial Area Exposure')
ax[1].legend([
    'Infective/Symptomatic', 
    'Exposed/Not Sympotomatic'],
    loc='upper right')
ax[1].plot(np.array([t_social_distancing, t_social_distancing]), ax[0].get_ylim(), 'r', lw=3)
ax[1].annotate("Penerapan Protokol Kesehatan",
    (t_social_distancing, 0), (t_social_distancing + 1.5, N/10),
    arrowprops=dict(arrowstyle='->'))
y0 = N*(e0 + i0)
k0 = np.argmax(y0)
ax[1].annotate("Tanpa Protokol Kesehatan", (t[k0]/1, y0[k0] + 100))

y = N*(e + i)
k = np.argmax(y)
ax[1].annotate("Penerapan{0:3.0f}% Protokol Kesehatan ".format(u_social_distancing), (t[k]/1, y[k] + 100))

for a in ax:
    a.xaxis.set_major_locator(plt.MultipleLocator(5))
    a.xaxis.set_minor_locator(plt.MultipleLocator(1))
    a.xaxis.set_major_formatter(plt.FormatStrFormatter('%d'))
    a.grid(True)

df = pd.DataFrame({
    'susceptible tanpa prokes' : s0,
    'exposed tanpa prokes' : e0,
    'infected tanpa prokes' : i0,
    'recovered tanpa prokes' : r0,
    'death tanpa prokes' : d0,
    'susceptible prokes' : s,
    'exposed prokes' : e,
    'infected rokes' : i,
    'recovered prokes' : r,
    'death prokes' : d,
    'days' : days
})

column = df["recovered tanpa prokes"]
last_value_dsm = column.max()*100
st._transparent_write("Prediksi Total Infeksi :",last_value_dsm,"% Populasi")

st.sidebar.write("Tingkat Transmission Rate (R0) adalah ", R0)

with col1:
    st.subheader("Forecasting")
    col1.pyplot(fig1)

st.pyplot(fig1)

# Generate universe variables
hospital = np.arange(0, 101, 1)
vaccine = np.arange(0, 101, 1)
seird = np.arange(0, 101, 1)
DSM = np.arange(0, 61, 1)

# Generate fuzzy membership functions
hospital_lo = fuzz.dsigmf(hospital, -20, 0.25, 40, 0.25)
hospital_md = fuzz.dsigmf(hospital, 40, 0.25, 80, 0.25)
hospital_hi = fuzz.dsigmf(hospital, 80, 0.25, 140, 0.25)

vaccine_lo = fuzz.dsigmf(vaccine, -20, 0.25, 20, 0.25)
vaccine_md = fuzz.dsigmf(vaccine, 30, 0.25, 60, 0.25)
vaccine_hi = fuzz.dsigmf(vaccine, 60, 0.25, 200, 0.25)

seird_lo = fuzz.dsigmf(seird, -20, 0.25, 20, 0.25)
seird_md = fuzz.dsigmf(seird, 20, 0.25, 60, 0.25)
seird_hi = fuzz.dsigmf(seird, 60, 0.25, 200, 0.25)

DSM_lo = fuzz.dsigmf(DSM, -20, 0.5, 10, 0.5)
DSM_md = fuzz.dsigmf(DSM, 10, 0.25, 50, 0.25)
DSM_hi = fuzz.dsigmf(DSM, 50, 0.25, 120, 0.25)

# -------------------------- Hospital --------------------------

#@markdown Presentase Keterisian Rumah Sakit
BOR = st.sidebar.slider("Tentukan Bed Occupation Rate (BOR)",min_value=1.0, max_value=100.0, value=14.0, step=0.1)
bedOccupationRate = BOR
st.sidebar.write("Tingkat Ketersian RS :",BOR,"%")

hospital_level_lo = fuzz.interp_membership(hospital, hospital_lo, bedOccupationRate)
hospital_level_md = fuzz.interp_membership(hospital, hospital_md, bedOccupationRate)
hospital_level_hi = fuzz.interp_membership(hospital, hospital_hi, bedOccupationRate)

# -------------------------- Vaksinasi --------------------------

st.write("Jumlah Populasi :", N)
st.write("Jumlah Kasus Aktif :",n)
target_vaksin = N

sudah_vaksin = st.sidebar.slider(
    "Total Vaksinasi :",min_value=0.0, max_value=50.0, value=95.0, step=0.1
)
print("Vaccinated :",sudah_vaksin,"%")
belum_vaksin = 100-sudah_vaksin

vaccine_level_lo = fuzz.interp_membership(vaccine, vaccine_lo, belum_vaksin)
vaccine_level_md = fuzz.interp_membership(vaccine, vaccine_md, belum_vaksin)
vaccine_level_hi = fuzz.interp_membership(vaccine, vaccine_hi, belum_vaksin)

st.sidebar.write("Belum Vaksin :", belum_vaksin,"%")

# -------------------------- SEIRD Model --------------------------

SEIRD_value = last_value_dsm

seird_level_lo = fuzz.interp_membership(seird, seird_lo, SEIRD_value)
seird_level_md = fuzz.interp_membership(seird, seird_md, SEIRD_value)
seird_level_hi = fuzz.interp_membership(seird, seird_hi, SEIRD_value)

#RULE 1
# Now we take our rules and apply them. Rule 1 concerns bad hospital OR seird.
# The OR operator means we take the maximum of these two.
active_rule1 = np.fmax(hospital_level_lo, vaccine_level_lo)
active_rule_lo = np.fmin(active_rule1, DSM_lo)
active_rule_lo_seird = np.fmax(active_rule_lo, seird_level_lo)
active_dsm_lo = np.fmin(active_rule_lo_seird, DSM_lo)

# RULE 2
active_rule2 = np.fmax(hospital_level_md, vaccine_level_md)
active_rule_md = np.fmin(active_rule2, DSM_md)
active_rule_md_seird = np.fmax(active_rule_md, seird_level_md)
active_dsm_md = np.fmin(active_rule_md_seird, DSM_md)

# For rule 3 we connect high service OR high food with high tipping
active_rule3 = np.fmax(hospital_level_hi, vaccine_level_hi)
active_rule_hospital_md = np.fmin(active_rule3, DSM_hi)
active_rule_hi_seird = np.fmax(active_rule_hospital_md, seird_level_hi)
active_dsm_hi = np.fmin(active_rule_hi_seird, DSM_hi)

DSM_activation_lo = active_dsm_lo
DSM_activation_md = active_dsm_md
DSM_activation_hi = active_dsm_hi
DSM0 = np.zeros_like(DSM)

# Visualize this
fig2, ax0 = plt.subplots(figsize=(12, 3))

ax0.fill_between(DSM, DSM0, DSM_activation_lo, facecolor='b', alpha=0.7)
ax0.plot(DSM, DSM_lo, 'b', linewidth=0.5, linestyle='--', )
ax0.fill_between(DSM, DSM0, DSM_activation_md, facecolor='g', alpha=0.7)
ax0.plot(DSM, DSM_md, 'g', linewidth=0.5, linestyle='--')
ax0.fill_between(DSM, DSM0, DSM_activation_hi, facecolor='r', alpha=0.7)
ax0.plot(DSM, DSM_hi, 'r', linewidth=0.5, linestyle='--')
ax0.set_title('Output membership activity')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

with col2:
    st.subheader("Membership Value")
    col2.pyplot(fig2)

st.pyplot(fig2)

# Aggregate all three output membership functions together
aggregated = np.fmax(DSM_activation_lo,np.fmax(DSM_activation_md, DSM_activation_hi))

# Calculate defuzzified result
decision = fuzz.defuzz(DSM, aggregated, 'centroid')
DSM_activation = fuzz.interp_membership(DSM, aggregated, decision)  # for plot

# Visualize this
fig3, ax0 = plt.subplots(figsize=(12, 3))

ax0.plot(DSM, DSM_lo, 'b', linewidth=0.5, linestyle='--')
ax0.plot(DSM, DSM_md, 'g', linewidth=0.5, linestyle='--')
ax0.plot(DSM, DSM_hi, 'r', linewidth=0.5, linestyle='--')
ax0.fill_between(DSM, DSM0, aggregated, facecolor='Red', alpha=0.5)
ax0.plot([decision, decision], [0, DSM_activation], 'k', linewidth=2, alpha=0.5)
ax0.set_title('Aggregated membership and result (line)')

# Turn off top/right axes
for ax in (ax0,):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

with col3:
    st.subheader("DSM Value")
    col3.pyplot(fig3)
    st.write("Decision Value is : ", decision)
    if decision > 0 and decision <10:
        st.header("Normal")
    elif decision >= 10 and decision <20:
        st.header("PPKM Level 1")
    elif decision >= 20 and decision <30:
        st.header("PPKM Level 2")
    elif decision >= 30 and decision <40:
        st.header("PPKM Level 3")
    elif decision >= 40 and decision <50:
        st.header("PPKM Level 4")
    else:
        st.header("LOCKDOWN")
st.pyplot(fig3)

st.subheader("Decision Value is : ")
st.subheader(decision)

if decision > 0 and decision <10:
    st.header("Normal")
elif decision >= 10 and decision <20:
    st.header("PPKM Level 1")
elif decision >= 20 and decision <30:
    st.header("PPKM Level 2")
elif decision >= 30 and decision <40:
    st.header("PPKM Level 3")
elif decision >= 40 and decision <50:
    st.header("PPKM Level 4")
else:
    st.header("LOCKDOWN")