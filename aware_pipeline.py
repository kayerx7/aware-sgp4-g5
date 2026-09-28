#!/usr/bin/env python3
"""
================================================================================
SPACE WEATHER ASTRODYNAMICS RESEARCH ENGINE (AWARE)
G5 Superstorm (May 2024) Upper-LEO SGP4 Ephemeris Sensitivity Benchmarker
================================================================================
Author: Farhan Mahmood Saad Sarker
Revised for Rigorous Automated Catalog-Scale Execution
"""

import getpass
import os
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sgp4.api import Satrec, WGS72

# ------------------------------------------------------------------------------
# 1. AUTHENTICATION & CONFIGURATION
# ------------------------------------------------------------------------------
print("=" * 75)
print("   AWARE: AUTOMATED LEO SGP4 EMPIRICAL SENSITIVITY ENGINE")
print("=" * 75)

st_user = input("Enter Space-Track Email: ").strip()
st_pass = getpass.getpass("Enter Space-Track Password: ").strip()

session = requests.Session()
login_url = "https://www.space-track.org/ajaxauth/login"
resp = session.post(login_url, data={"identity": st_user, "password": st_pass})
if resp.status_code != 200 or "Failed" in resp.text:
    raise RuntimeError(f"Authentication failed: {resp.text}")
print("[+] Space-Track Authentication Successful.\n")

# TARGET SELECTION: Verified Upper-LEO Rocket Bodies (Circular, e < 0.005)
# In production, query directly via API: /OBJECT_TYPE/ROCKET BODY/ECCENTRICITY/<0.005
TARGET_CATALOG = {
    39160: "CZ-2C R/B",
    25942: "CZ-4B R/B",
    25400: "DELTA 2 R/B",
    22803: "SL-16 R/B",
    28353: "DELTA 2 R/B",
    14129: "SL-8 R/B",
    20511: "SL-8 R/B",
    40058: "VEGA R/B",
    23279: "COSMOS 2292 (Eccentric Baseline)",
}

cat_ids = ",".join(str(k) for k in TARGET_CATALOG.keys())

# ANCHORED TIME WINDOWS:
# Storm window: May 10, 2024 shock arrival -> May 13 (72h peak expansion)
# Quiet window: April 10, 2024 -> April 13 (Quiet baseline)
WINDOWS = {
    "storm": ("2024-05-09", "2024-05-14"),
    "quiet": ("2024-04-09", "2024-04-14")
}

# ------------------------------------------------------------------------------
# 2. CORE ASTRODYNAMICS ROUTINES
# ------------------------------------------------------------------------------
def get_sat_epoch_jd(sat):
    """Returns absolute Julian Date (integral + fraction)."""
    jd = getattr(sat, 'jdsatepoch', 0.0)
    fr = getattr(sat, 'jdsatepochF', getattr(sat, 'jdsatepochf', 0.0))
    return jd + fr

def parse_tle_stream(raw_text):
    """Splits raw TLE text into verified (Line 1, Line 2) tuples."""
    lines = [l.strip() for l in raw_text.strip().split('\n') if l.strip()]
    pairs = []
    i = 0
    while i < len(lines):
        if lines[i].startswith('1 ') and i + 1 < len(lines) and lines[i+1].startswith('2 '):
            pairs.append((lines[i], lines[i+1]))
            i += 2
        else:
            i += 1
    return pairs

def evaluate_arc(t0_lines, t1_lines):
    """
    Propagates initial TLE to reference TLE epoch.
    Resolves Radial-In-Track-Cross-Track (RIC) error and delta-a.
    Normalizes in-track error to exact 72.0-hour horizon via quadratic scaling.
    """
    sat0 = Satrec.twoline2rv(t0_lines[0], t0_lines[1])
    sat1 = Satrec.twoline2rv(t1_lines[0], t1_lines[1])
    
    # Gravitational parameter mu (km^3/s^2)
    mu = 398600.4418
    
    # Mean motion conversion (rad/min to rad/s)
    n0 = sat0.no_kozai / 60.0
    a0 = (mu / (n0**2))**(1.0 / 3.0)
    alt0 = a0 - 6378.137
    
    n1 = sat1.no_kozai / 60.0
    a1 = (mu / (n1**2))**(1.0 / 3.0)
    delta_a_m = (a1 - a0) * 1000.0
    
    # Extract propagation delta-t (hours)
    jd0 = get_sat_epoch_jd(sat0)
    jd1 = get_sat_epoch_jd(sat1)
    dt_hours = (jd1 - jd0) * 24.0
    if dt_hours <= 0:
        return None
        
    # Forward propagate sat0 to epoch of sat1
    fr1 = getattr(sat1, 'jdsatepochF', getattr(sat1, 'jdsatepochf', 0.0))
    jd1_int = getattr(sat1, 'jdsatepoch', 0.0)
    
    err0, r0, v0 = sat0.sgp4(jd1_int, fr1)
    err1, r1, v1 = sat1.sgp4(jd1_int, fr1)
    
    if err0 != 0 or err1 != 0:
        return None
        
    r0, r1, v1 = np.array(r0), np.array(r1), np.array(v1)
    
    # RIC Local Orbital Frame Construction (Centered on Reference sat1)
    e_R = r1 / np.linalg.norm(r1)
    h_vec = np.cross(r1, v1)
    e_C = h_vec / np.linalg.norm(h_vec)
    e_I = np.cross(e_C, e_R)
    
    delta_r = r0 - r1
    delta_I_raw = abs(np.dot(delta_r, e_I))
    
    # QUADRATIC NORMALIZATION TO EXACT 72.0 HOURS: Delta_I proportional to t^2
    normalization_factor = (72.0 / dt_hours)**2
    delta_I_72_norm = delta_I_raw * normalization_factor
    
    return {
        "altitude_km": alt0,
        "eccentricity": sat0.ecco,
        "bstar": sat0.bstar,
        "dt_hours": dt_hours,
        "delta_a_m": delta_a_m,
        "delta_I_raw_km": delta_I_raw,
        "delta_I_72_km": delta_I_72_norm
    }

# ------------------------------------------------------------------------------
# 3. QUERY INGESTION & PROCESSING PIPELINE
# ------------------------------------------------------------------------------
dataset = []

for regime, (d_start, d_end) in WINDOWS.items():
    print(f"[*] Fetching {regime.upper()} telemetry from Space-Track...")
    query = (
        f"https://www.space-track.org/basicspacedata/query/class/gp_history/"
        f"NORAD_CAT_ID/{cat_ids}/EPOCH/{d_start}--{d_end}/"
        f"orderby/NORAD_CAT_ID,EPOCH ASC/format/tle"
    )
    res = session.get(query)
    if res.status_code != 200:
        print(f"[-] Query failed for {regime}: Status {res.status_code}")
        continue
        
    tles = parse_tle_stream(res.text)
    print(f"[+] Ingested {len(tles)} TLEs for {regime} window.")
    
    # Group by object
    by_cat = {}
    for t in tles:
        cat_id = int(t[0][2:7])
        by_cat.setdefault(cat_id, []).append(t)
        
    # Evaluate optimal 72h propagation arc
    for cat_id, t_list in by_cat.items():
        if len(t_list) < 2:
            continue
            
        sats = [Satrec.twoline2rv(t[0], t[1]) for t in t_list]
        epochs = [get_sat_epoch_jd(s) for s in sats]
        
        # Lock to anchor pair closest to 72 hours
        best_arc = None
        min_offset = float('inf')
        for i in range(len(t_list)):
            for j in range(i + 1, len(t_list)):
                dt_h = (epochs[j] - epochs[i]) * 24.0
                if 48.0 <= dt_h <= 96.0:  # Allow realistic radar gap window
                    offset = abs(dt_h - 72.0)
                    if offset < min_offset:
                        min_offset = offset
                        best_arc = (t_list[i], t_list[j])
                        
        if best_arc:
            res_arc = evaluate_arc(best_arc[0], best_arc[1])
            if res_arc:
                res_arc["norad_id"] = cat_id
                res_arc["name"] = TARGET_CATALOG.get(cat_id, "Unknown")
                res_arc["regime"] = regime
                dataset.append(res_arc)

df_all = pd.DataFrame(dataset)
if df_all.empty:
    raise RuntimeError("No valid orbital arcs computed. Verify network and catalog IDs.")

# Split storm vs quiet
df_storm = df_all[df_all["regime"] == "storm"].set_index("norad_id")
df_quiet = df_all[df_all["regime"] == "quiet"].set_index("norad_id")

audit = df_storm[["name", "altitude_km", "eccentricity", "bstar", "dt_hours", "delta_a_m", "delta_I_72_km"]].join(
    df_quiet[["delta_I_72_km"]], rsuffix="_quiet"
).dropna().reset_index()

audit.rename(columns={
    "delta_I_72_km": "storm_delta_I_72",
    "delta_I_72_km_quiet": "quiet_delta_I_72"
}, inplace=True)

audit["excess_delta_I"] = audit["storm_delta_I_72"] - audit["quiet_delta_I_72"]

# Ballistic-Normalized Error Metric: km / (B* [1/earth_radii])
audit["storm_delta_I_normalized"] = audit["storm_delta_I_72"] / audit["bstar"].abs()

print("\n" + "=" * 90)
print("                EMPIRICAL SGP4 ERROR AUDIT TABLE (NORMALIZED)")
print("=" * 90)
print(audit.to_string(index=False, formatters={
    "altitude_km": "{:.1f}".format,
    "bstar": "{:.6e}".format,
    "dt_hours": "{:.1f}".format,
    "delta_a_m": "{:.2f}".format,
    "storm_delta_I_72": "{:.2f}".format,
    "quiet_delta_I_72": "{:.2f}".format,
    "excess_delta_I": "{:.2f}".format,
    "storm_delta_I_normalized": "{:.1e}".format
}))

# ------------------------------------------------------------------------------
# 4. REGRESSION & DUAL-SCALE ANALYSIS
# ------------------------------------------------------------------------------
# Segregate circular targets (e < 0.005) from tumbling/eccentric debris
circular_cohort = audit[audit["eccentricity"] < 0.005].sort_values("altitude_km")

z_circ = circular_cohort["altitude_km"].values
y_circ = circular_cohort["storm_delta_I_72"].values
da_circ = circular_cohort["delta_a_m"].abs().values

# Noise floor: median quiet baseline of circular stages
noise_floor = circular_cohort["quiet_delta_I_72"].median()
z_ref = z_circ.min()

# 1. Physical Decay Scale Height (Delta a)
def da_model(z, H_da, A_da):
    return A_da * np.exp(-(z - z_ref) / H_da)

popt_da, _ = curve_fit(da_model, z_circ, da_circ, p0=[60.0, da_circ.max()])
H_da_fit, A_da_fit = popt_da

# 2. Along-Track Phase Dispersion Scale Height (Delta I)
def phase_model(z, H_eff, A_eff):
    return noise_floor + A_eff * np.exp(-(z - z_ref) / H_eff)

popt_phase, _ = curve_fit(phase_model, z_circ, y_circ, p0=[90.0, y_circ.max() - noise_floor])
H_eff_fit, A_eff_fit = popt_phase

print("\n" + "=" * 60)
print("             PHYSICAL VS. EMPIRICAL REGRESSION RESULTS")
print("=" * 60)
print(f"Reference Altitude (z_ref)   : {z_ref:.1f} km")
print(f"Baseline Noise Floor         : {noise_floor:.3f} km")
print(f"Semi-Major Axis Scale (H_da) : {H_da_fit:.2f} km")
print(f"Along-Track Phase Scale (Heff): {H_eff_fit:.2f} km")
print(f"Dual-Scale Ratio (Heff / H_da): {H_eff_fit / H_da_fit:.2f}")

# 5.0 km Operational Crossing Altitude (Parametric Inversion)
if A_eff_fit > (5.0 - noise_floor):
    z_5km_parametric = z_ref - H_eff_fit * np.log((5.0 - noise_floor) / A_eff_fit)
    print(f"Parametric 5.0 km Crossing   : {z_5km_parametric:.1f} km")
else:
    print("Along-track error remains below 5.0 km across entire upper-LEO domain.")
print("=" * 60)
