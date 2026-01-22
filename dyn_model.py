# -*- coding: utf-8 -*-
#
# Open-BLDC pysim - Open BrushLess DC Motor Controller python simulator
# Copyright (C) 2011 by Antoine Drouin <poinix@gmail.com>
# Copyright (C) 2011 by Piotr Esden-Tempski <piotr@esden.net>
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#

import numpy as np
import math
import misc_utils as mu

# parameters
pset = 2

if pset == 0:
    Inertia = 0.0022  # aka. 'J' in kg/(m^2)
    Damping = 0.001   # aka. 'B' in Nm/(rad/s)
    Kv = 1700.        # aka. motor constant in RPM/V
    L = 0.00312       # aka. Coil inductance in H
    M = 0.0           # aka. Mutual inductance in H
    R = 0.8           # aka. Phase resistence in Ohm
    VDC = 100.        # aka. Supply voltage
    NbPoles = 14.     # NbPoles / 2 = Number of pole pairs
    dvf = .7          # aka. freewheeling diode forward voltage
elif pset == 1:
    Inertia = 0.0022
    Damping = 0.001
    Kv = 70.
    L = 0.00521
    M = 0.0
    R = 0.7
    VDC = 100.
    NbPoles = 4.
    dvf = .7
elif pset == 2:  # psim
    Inertia = 0.000007
    tau_shaft = 0.006
    Damping = Inertia / tau_shaft
    Kv = 1. / 32.3 * 1000
    L = 0.00207
    M = -0.00069
    R = 11.9
    VDC = 100.
    NbPoles = 4.
    dvf = .0
elif pset == 3:  # modified psim
    Inertia = 0.000059
    tau_shaft = 0.006
    Damping = Inertia / tau_shaft
    Kv = 1. / 32.3 * 1000
    L = 0.00207
    M = -0.00069
    R = 11.9
    VDC = 300.
    NbPoles = 4.
    dvf = .0
else:
    print("Unknown pset {}".format(pset))

# Components of the state vector
sv_theta = 0      # angle of the rotor
sv_omega = 1      # angular speed of the rotor
sv_iu = 2         # phase u current
sv_iv = 3         # phase v current
sv_iw = 4         # phase w current
sv_size = 5

# Components of the command vector
iv_lu = 0
iv_hu = 1
iv_lv = 2
iv_hv = 3
iv_lw = 4
iv_hw = 5
iv_size = 6

# Components of the perturbation vector
pv_torque = 0
pv_friction = 1
pv_size = 2

# Components of the output vector
ov_iu = 0
ov_iv = 1
ov_iw = 2
ov_vu = 3
ov_vv = 4
ov_vw = 5
ov_theta = 6
ov_omega = 7
ov_size = 8

# Phases and star vector designators
ph_U = 0
ph_V = 1
ph_W = 2
ph_star = 3
ph_size = 4

# Debug vector components
dv_eu = 0
dv_ev = 1
dv_ew = 2
dv_ph_U = 3
dv_ph_V = 4
dv_ph_W = 5
dv_ph_star = 6
dv_size = 7


#
# Calculate backemf at a given omega offset from the current rotor position
#
def backemf(X, thetae_offset):
    phase_thetae = mu.norm_angle((X[sv_theta] * (NbPoles / 2.0)) + thetae_offset)

    bemf_constant = mu.vpradps_of_rpmpv(Kv)  # aka. ke in V/rad/s
    max_bemf = bemf_constant * X[sv_omega]

    bemf = 0.0
    if 0.0 <= phase_thetae <= (math.pi * (1.0 / 6.0)):
        bemf = (max_bemf / (math.pi * (1.0 / 6.0))) * phase_thetae
    elif (math.pi / 6.0) < phase_thetae <= (math.pi * (5.0 / 6.0)):
        bemf = max_bemf
    elif (math.pi * (5.0 / 6.0)) < phase_thetae <= (math.pi * (7.0 / 6.0)):
        bemf = -((max_bemf / (math.pi / 6.0)) * (phase_thetae - math.pi))
    elif (math.pi * (7.0 / 6.0)) < phase_thetae <= (math.pi * (11.0 / 6.0)):
        bemf = -max_bemf
    elif (math.pi * (11.0 / 6.0)) < phase_thetae <= (2.0 * math.pi):
        bemf = (max_bemf / (math.pi / 6.0)) * (phase_thetae - (2.0 * math.pi))
    else:
        print("ERROR: angle out of bounds can not calculate bemf {}".format(phase_thetae))

    return bemf


#
# Calculate phase voltages
# Returns a vector of phase voltages in reference to the star point
#
def voltages(X, U):

    eu = backemf(X, 0.0)
    ev = backemf(X, math.pi * (2.0 / 3.0))
    ew = backemf(X, math.pi * (4.0 / 3.0))

    # Check which phases are excited
    pux = (U[iv_hu] == 1) or (U[iv_lu] == 1)
    pvx = (U[iv_hv] == 1) or (U[iv_lv] == 1)
    pwx = (U[iv_hw] == 1) or (U[iv_lw] == 1)

    vu = 0.0
    vv = 0.0
    vw = 0.0
    vm = 0.0

    if pux and pvx and pwx:
        vu = VDC / 2.0 if (U[iv_hu] == 1) else -VDC / 2.0
        vv = VDC / 2.0 if (U[iv_hv] == 1) else -VDC / 2.0
        vw = VDC / 2.0 if (U[iv_hw] == 1) else -VDC / 2.0
        vm = (vu + vv + vw - eu - ev - ew) / 3.0

    elif pux and pvx:
        vu = VDC / 2.0 if (U[iv_hu] == 1) else -VDC / 2.0
        vv = VDC / 2.0 if (U[iv_hv] == 1) else -VDC / 2.0
        vm = (vu + vv - eu - ev) / 2.0
        vw = ew + vm

    elif pux and pwx:
        vu = VDC / 2.0 if (U[iv_hu] == 1) else -VDC / 2.0
        vw = VDC / 2.0 if (U[iv_hw] == 1) else -VDC / 2.0
        vm = (vu + vw - eu - ew) / 2.0
        vv = ev + vm

    elif pvx and pwx:
        vv = VDC / 2.0 if (U[iv_hv] == 1) else -VDC / 2.0
        vw = VDC / 2.0 if (U[iv_hw] == 1) else -VDC / 2.0
        vm = (vv + vw - ev - ew) / 2.0
        vu = eu + vm

    elif pux:
        vu = VDC / 2.0 if (U[iv_hu] == 1) else -VDC / 2.0
        vm = (vu - eu)
        vv = ev + vm
        vw = ew + vm

    elif pvx:
        vv = VDC / 2.0 if (U[iv_hv] == 1) else -VDC / 2.0
        vm = (vv - ev)
        vu = eu + vm
        vw = ew + vm

    elif pwx:
        vw = VDC / 2.0 if (U[iv_hw] == 1) else -VDC / 2.0
        vm = (vw - ew)
        vu = eu + vm
        vv = ev + vm

    else:
        vm = eu
        vv = ev
        vw = ew

    V = [vu, vv, vw, vm]
    return V


#
# Dynamic model
#
def dyn(X, t, U, W):
    Xd, _Xdebug = dyn_debug(X, t, U, W)
    return Xd


# Dynamic model with debug vector
def dyn_debug(X, t, U, W):

    eu = backemf(X, 0.0)
    ev = backemf(X, math.pi * (2.0 / 3.0))
    ew = backemf(X, math.pi * (4.0 / 3.0))

    # Electromagnetic torque
    # Защита от деления на ноль при omega ~ 0
    omega = X[sv_omega]
    if abs(omega) < 1e-12:
        etorque = 0.0
    else:
        etorque = (eu * X[sv_iu] + ev * X[sv_iv] + ew * X[sv_iw]) / omega

    # Mechanical torque
    mtorque = (etorque * (NbPoles / 2.0)) - (Damping * omega) - W[pv_torque]

    if (mtorque > 0) and (mtorque <= W[pv_friction]):
        mtorque = 0.0
    elif mtorque >= W[pv_friction]:
        mtorque = mtorque - W[pv_friction]
    elif (mtorque < 0) and (mtorque >= (-W[pv_friction])):
        mtorque = 0.0
    elif mtorque <= (-W[pv_friction]):
        mtorque = mtorque + W[pv_friction]

    # Acceleration of the rotor
    omega_dot = mtorque / Inertia

    V = voltages(X, U)

    iu_dot = (V[ph_U] - (R * X[sv_iu]) - eu - V[ph_star]) / (L - M)
    iv_dot = (V[ph_V] - (R * X[sv_iv]) - ev - V[ph_star]) / (L - M)
    iw_dot = (V[ph_W] - (R * X[sv_iw]) - ew - V[ph_star]) / (L - M)

    Xd = [
        X[sv_omega],
        omega_dot,
        iu_dot,
        iv_dot,
        iw_dot
    ]

    Xdebug = [
        eu,
        ev,
        ew,
        V[ph_U],
        V[ph_V],
        V[ph_W],
        V[ph_star]
    ]

    return Xd, Xdebug


def output(X, U):

    V = voltages(X, U)

    Y = [
        X[sv_iu], X[sv_iv], X[sv_iw],
        V[ph_U], V[ph_V], V[ph_W],
        X[sv_theta], X[sv_omega]
    ]

    return Y
