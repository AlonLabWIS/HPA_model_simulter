# 1. Imports
import streamlit as st
import numpy as np
from scipy.optimize import fsolve
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import labellines
import io

# 2. Function definitions
def one_d_model_drift(x, t, u_val_treat=0, db_val_treat=0, start_time=10, end_time=100):
    x_cur, b_cur, u_cur = x
    if t  > start_time and t < end_time:
        dx = x_cur**2 * (1 - x_cur) - b_cur * x_cur + u_val_treat
        du = 0
        db = db_val_treat
    else:
        dx = x_cur**2 * (1 - x_cur) - b_cur * x_cur
        du = 0
        db = 0
    return np.array([dx, db, du])

def sde_solver_system(drift, x0, t, sigma, params=None):
    if params is None:
        params = ()
    n = len(t)
    d = len(x0)
    x = np.zeros((n, d))
    x[0] = x0
    dt = t[1] - t[0]
    for i in range(1, n):
        dw = np.random.normal(scale=np.sqrt(dt))  # Single noise term for x1
        x[i] = x[i - 1] + drift(x[i - 1], t[i - 1], *params) * dt
        x[i][0] += sigma * dw
    return x

def fig_to_png(fig):
    img = io.BytesIO()
    fig.savefig(img, format='png')
    img.seek(0)
    return img

# 3. Streamlit UI and parameter setup
st.sidebar.title("Simulation Parameters")

# db_1 = st.sidebar.slider("db", min_value=-0.1, max_value=3.0, value=0.0, step=1e-1) / 100 
du_1 = st.sidebar.slider("du_1", min_value=-7.0, max_value=0.0, value=-4.0, step=1e-2, format="%.4f") / 100
du_2 = st.sidebar.slider("du_2", min_value=-7.0, max_value=0.0, value=-6.5, step=1e-2, format="%.4f") / 100
b = st.sidebar.slider("b", min_value=0.0, max_value=1.0, value=0.2, step=1e-2)
u = st.sidebar.slider("u", min_value=0.0, max_value=5.0, value=0.0, step=0.1)
end_time = st.sidebar.slider("Until time", min_value=0, max_value=300, value=100, step=1)/8
start_time = st.sidebar.slider("From time", min_value=0, max_value=100, value=10, step=1) /8 

sigma = st.sidebar.slider("Noise Level (sigma)", min_value=0.0, max_value=0.1, value=0.00, step=0.001)
T_in_hours = st.sidebar.slider("Simulation Time (hours)", min_value=1, max_value=500, value=50, step=10) 
n_points = st.sidebar.slider("Time Steps", min_value=100, max_value=1000, value=400, step=50)

# Compute steady-state initial conditions
def f_to_solve(x):
    return one_d_model_drift(x, 0)

guess = [1, b, u]
steady_state = fsolve(f_to_solve, guess)
x0 = steady_state[0]
x0 = [x0, b, u]
variables = ["x", "b", "u"]

# Simulation time
T = T_in_hours 
t = np.linspace(0, T, n_points)

# 4. Solve the system and plot
parmas_1 = (du_1, 0, start_time, end_time)
parmas_2 = (du_2, 0, start_time, end_time)
sol_1 = sde_solver_system(one_d_model_drift, x0, t, sigma, parmas_1)
sol_2 = sde_solver_system(one_d_model_drift, x0, t, sigma, parmas_2)
start_time*=8
end_time*=8
t= t*8
# Plotting
fig, ax = plt.subplots()

ax.plot(t, sol_1[:, 0], label="relapse")
ax.plot(t, sol_2[:, 0], label="remission")
# ax.plot(t, sol_1[:, 1], label="b")
# plot a step function for u low where t is between start_time and end_time
u1_low = np.zeros_like(t)
u1_low[(t > start_time) & (t < end_time)] = du_1
# ax.plot(t, u1_low, label="u relapse value")
u2_low = np.zeros_like(t)
u2_low[(t > start_time) & (t < end_time)] = du_2
# ax.plot(t, u2_low, label="u remission value")
# 
ax.set_xlabel("Time [days]")
ax.set_ylabel("symptoms [x]")
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add grey area for treatment
ax.axvspan(start_time, end_time, facecolor='grey', alpha=0.1)

# Add text for treatment
ax.text(start_time + (end_time - start_time) / 2, 0.1, "treatment", ha='center', va='center', fontsize=12)

b_placed = st.number_input("Place b", 0, 300, 200)
u_placed = st.number_input("Place u", 0, 300, 200)

# ax.legend(loc='upper right')
ax.set_ylim(-0.1, 1)
# lableines
labellines.labelLines(ax.get_lines(), xvals=[300,300], zorder=2.5, )

st.pyplot(fig)

# SVG Download
filename = st.text_input("Filename", "hpa_simulation")

if "png_data" not in st.session_state:
    st.session_state.png_data = None

if st.button("Convert Plot to PNG"):
    st.session_state.png_data = fig_to_png(fig)

if st.session_state.png_data is not None:
    st.download_button(
        label="Download Plot as PNG",
        data=st.session_state.png_data,
        file_name=f"{filename}.png",
        mime="image/png",
    )

st.markdown(
    """
### Summary of the HPA Axis Model

1. **Equation for $ \\frac{dx_1}{dt} $**:
   $$\\frac{dx_1}{dt} = b_1 \\frac{1}{1 + \\left(\\frac{x_{3b}}{k_{gr}}\\right)^3} \\frac{1}{x_{3b}} u - a_1 x_1$$
   - $ x_1 $: CRH (Corticotropin-Releasing Hormone)
   - $ b_1 $: Production rate constant for CRH
   - $ \\frac{1}{1 + \\left(\\frac{x_{3b}}{k_{gr}}\\right)^3} $: GR (Glucocorticoid Receptor) response inside the BBB (Blood-Brain Barrier)
   - $ \\frac{1}{x_{3b}} $: MR (Mineralocorticoid Receptor) response
   - $ u $: External stimulus
   - $ a_1 $: Degradation rate of CRH
   - $ k_{gr} $: Concentration of cortisol at which the GR response is at half of its maximum effectiveness (EC50).

2. **Equation for $ \\frac{dx_2}{dt} $**:
   $$\\frac{dx_2}{dt} = b_2 x_1 \\frac{1}{1 + \\left(\\frac{x_3}{k_{gr}}\\right)^3} - a_2 x_2$$
   - $ x_2 $: Another form of CRH
   - $ b_2 $: Production rate constant for this form of CRH
   - $ x_1 $: Precursor CRH
   - $ \\frac{1}{1 + \\left(\\frac{x_3}{k_{gr}}\\right)^3} $: GR response (outside the BBB)
   - $ a_2 $: Degradation rate of this form of CRH

3. **Equation for $ \\frac{dx_3}{dt} $**:
   $$\\frac{dx_3}{dt} = b_3 x_2 - a_3 x_3$$
   - $ x_3 $: Cortisol
   - $ b_3 $: Production rate constant for cortisol
   - $ x_2 $: Precursor CRH
   - $ a_3 $: Degradation rate of cortisol

4. **Equation for $ \\frac{dx_{3b}}{dt} $**:
   $$\\frac{dx_{3b}}{dt} = k (x_3 - x_{3b}) - a_3 x_{3b}$$
   - $ x_{3b} $: Cortisol in the BBB
   - $ k $: Transfer rate constant between blood and brain
   - $ x_3 $: Cortisol
   - $ a_3 x_{3b} $: Degradation rate of cortisol in the BBB
    """
)
