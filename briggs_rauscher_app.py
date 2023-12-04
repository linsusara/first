# -*- coding: utf-8 -*-
"""
Created on Sat Dec  2 21:16:08 2023

@author: ABRAHAM JOSEPH
"""


import streamlit as st
import numpy as np
import scipy.integrate as spi
import scipy.interpolate as interp
from plotly.subplots import make_subplots
import plotly.io as pio

def BriggsRauscherODE(t, y, k):
    dydt = np.zeros_like(y)
    # Replace with the Briggs-Rauscher ODEs
    dydt[0] = -k[0] * y[0] + k[2] * y[2]
    dydt[1] = k[0] * y[0] - k[1] * y[1]
    dydt[2] = k[1] * y[1] - k[2] * y[2]
    dydt[3] = k[1] * y[1] - k[2] * y[2]
    return dydt

st.header('Briggs-Rauscher Solver')
A0 = st.sidebar.slider('[A]0', min_value=0.1, max_value=100.0, value=10.0, step=0.1)
B0 = st.sidebar.text_input('[B]0', value='1')
C0 = st.sidebar.text_input('[C]0', value='1')
D0 = st.sidebar.text_input('[D]0', value='0')
k1 = st.sidebar.text_input('k1', value='0.005')
k2 = st.sidebar.text_input('k2', value='0.1')
k3 = st.sidebar.text_input('k3', value='0.01')
tmax = st.sidebar.text_input('Simulation Time (s)', value='10000')
xscale = st.sidebar.selectbox('X Axis Scale', ['Linear', 'Logarithmic'], index=0)
plot = st.empty()

# Set up the simulation conditions, rates, initial concentrations, simulation time
k = [float(k1), float(k2), float(k3)]
y0 = [float(A0), float(B0), float(C0), float(D0)]
time = float(tmax)

# Solve the ODEs
solution = spi.solve_ivp(BriggsRauscherODE, [0, time], y0, method='Radau', max_step=time / 30, rtol=1e-8, args=[k])

# Generate a smooth interpolated line to connect the points from the ODE solution
smootht = np.linspace(0, time, 1000)
smoothy = []
for component in solution.y:
    cs = interp.CubicSpline(solution.t, component)
    smoothy.append(cs(smootht))

# Define our colors so that the ODE solver markers and the interpolated line are the same color
colorway = pio.templates['simple_white'].layout.colorway

# Plot the points and the interpolated lines
fig = make_subplots()
name = ['[A]', '[B]', '[C]', '[D]']
for i in range(4):
    fig.add_scatter(x=solution.t,
                    y=solution.y[i],
                    mode='markers',
                    name=name[i],
                    line={'color': colorway[i % len(colorway)]})

for i in range(4):
    fig.add_scatter(x=smootht,
                    y=smoothy[i],
                    mode='lines',
                    name='',
                    line={'color': colorway[i % len(colorway)]},
                    showlegend=False)
loglinear = {'Linear': 'linear', 'Logarithmic': 'log'}
fig.update_xaxes(type=loglinear[xscale])
fig.update_layout(height=600, margin={'t': 20})

# Here instead of fig.show() we assign the fig to a streamlit plotly_chart
# object and it handles the rest
st.plotly_chart(fig, height=600, use_container_width=True)