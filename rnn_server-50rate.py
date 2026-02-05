## multi user, multi carrier
## user 2 fixed at a wall
## integrate communication
# intend to use this file to train communication loss + crb loss
# com loss = rate
import sys
import os
from datetime import datetime

# Create output directory (if it does not exist)
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

# Create unique filename using current datetime
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f'output_{timestamp}.txt')

# Redirect standard output to file
sys.stdout = open(log_file, 'w', encoding='utf-8')
print(f"Start logging output to {log_file} - {datetime.now()}")
# Option to keep output to terminal while logging to file (optional)
class Logger:
    def __init__(self, file_path):
        self.terminal = sys.__stdout__
        self.log = open(file_path, 'w', encoding='utf-8')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
        
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# If you want to output to both terminal and file, please use the following line (uncomment):
sys.stdout = Logger(log_file)

# Limit external library thread usage to avoid low-level crashes (bus error)
import os as _os_env
_os_env.environ.setdefault('OMP_NUM_THREADS', '1')
_os_env.environ.setdefault('MKL_NUM_THREADS', '1')
_os_env.environ.setdefault('NUMEXPR_NUM_THREADS', '1')
_os_env.environ.setdefault('TF_CPP_MIN_LOG_LEVEL', '2')

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior() 
import numpy as np
import scipy.io as sio
import os
from keras.layers import BatchNormalization
from keras.layers import Dense
import time
import random
print(tf.config.list_physical_devices('GPU'))

# Import CRB calculation functions from the plotting script
from tf_crb_eval_style import (
    tf_crb_loss_eval_style_tx_fast,
    tf_crb_ue_peb_nomap_multi,
    tf_crb_ue_peb_map_multi,
    tf_crb_ue_peb_nomap_single,
    tf_crb_ue_peb_map_single,
)

print(tf.config.list_physical_devices('GPU'))

# === CRB-loss config ===
USE_CRB_LOSS = True  # Enable CRB component when True
CRB_COMPARE_PLOT_VS_TF = True  # Whether to compute py_func version in parallel for comparison during training (default off for speed)

# === CRB method selector (choose one of four) ===
# Options: 'nomap_multi' (K=10), 'map_multi' (K=10), 'nomap_single' (K=1), 'map_single' (K=1)
CRB_METHOD = 'map_multi'
print(f"[CRB Config] method={CRB_METHOD}")

def _crb_dispatch(theta_list, loc_input, P_lin):
    """Select corresponding CRB calculation function based on CRB_METHOD.
    All branches return PEB^2 mean of UE on batch (scalar tensor).
    """
    # Map P_lin to snr_db consistently (consistent with interface in tf_crb_eval_style)
    P_tf = tf.convert_to_tensor(P_lin, dtype=tf.float32)
    snr_db_tf = tf.constant(10.0, dtype=tf.float32) * tf.log(P_tf) / tf.log(tf.constant(10.0, dtype=tf.float32))
    if CRB_METHOD == 'nomap_multi':
        return tf_crb_ue_peb_nomap_multi(theta_list, loc_input, snr_db_tf)
    elif CRB_METHOD == 'map_multi':
        return tf_crb_ue_peb_map_multi(theta_list, loc_input, snr_db_tf)
    elif CRB_METHOD == 'nomap_single':
        return tf_crb_ue_peb_nomap_single(theta_list, loc_input, snr_db_tf)
    elif CRB_METHOD == 'map_single':
        return tf_crb_ue_peb_map_single(theta_list, loc_input, snr_db_tf)
    else:
        # Fallback to original total loss (UE+SP) version, for compatibility with old logic
        return tf_crb_loss_eval_style_tx_fast(theta_list, loc_input, P_lin)

# Unified entry point for training script
CRB_LOSS_FN = _crb_dispatch


# drive_save_path = 'RNN_SP'
snr_const = -1
drive_save_path = f'models_newcrb/snr{snr_const}_{CRB_METHOD}'
os.makedirs(drive_save_path, exist_ok=True)

'System Information'
N = 1   # Number of BS's antennas
delta_inv = 128 
delta = 1/delta_inv 
S = np.log2(delta_inv) 
OS_rate = 20 
delta_inv_OS = OS_rate*delta_inv 
delta_OS = 1/delta_inv_OS 

'Multicarrier system'
num_subcarriers = 1  # number of subcarriers
carrier_freq = 2.4e9  # （2.4GHz）
bandwidth = 100e6     # （100MHz）
subcarrier_freqs = carrier_freq + np.linspace(-bandwidth/2, bandwidth/2, num_subcarriers)
c = 3e8  

# Sensing parameters
tau = 8  # Pilot length

snr_const = np.array([snr_const])
Pvec = 10**(snr_const/10)

# Positions
location_bs_new = np.array([0, 0, 0])
location_ris_1 = np.array([-35, 40, -20])
num_ris = 1

# Channel parameters
mean_true_alpha = 0.0 + 0.0j
std_per_dim_alpha = np.sqrt(0.5) 
noiseSTD_per_dim = np.sqrt(0.5)

# RIS configuration
N_ris = 32
num_users = 2
params_system = (N, N_ris, num_users)
Rician_factor = 10

location_user = None

# ==== NEW: control how we sample training angles ====
# If True => sample (r, phi) with phi ~ Uniform[PHI_RANGE_DEG], r ~ Uniform[R_RANGE_M],
# then map back to (x,y) around the RIS. This gives near-uniform angular coverage.
UNIFORM_PHI = True
PHI_RANGE_DEG = (-10.0, 60.0)
# Choose a radius span that stays within the original rectangle around the RIS
R_RANGE_M = (10.0, 20.0)
# Grid test step (meters), used for uniform sampling in Cartesian coordinates within the sector area for final testing
GRID_STEP_M = 0.10

# ==== OVERFIT: small fixed-geometry dataset control ====
# When enabled, training/validation use a tiny dataset built by repeating
# the final-sample UE/SP geometry to deliberately overfit.
OVERFIT_FIXED_GEOM = False
OVERFIT_TRAIN_SAMPLES = 1000
OVERFIT_VAL_SAMPLES = 1000
# Use the same geometry as final test block below
FIXED_UE_XY = np.array([-18.0, 30.0], dtype=float)
FIXED_SP_XY = np.array([-28.074, 62.0], dtype=float)
FIXED_Z = -20.0

# New: choose overfit dataset mode
# 'fixed'  -> previous behavior: tile one UE/SP geometry for all samples
# 'random' -> prebuild a small random dataset (size OVERFIT_*_SAMPLES) and reuse it
OVERFIT_DATASET_MODE = 'random'  # set to 'fixed' to keep original behavior

# Learning Parameters
initial_run = 0 # 0: Continue training; 1: Starts from scratch
n_epochs =1
learning_rate = 1e-3
batch_per_epoch = 32*2 #100*8
batch_size_order = 1
val_size_order =32   #782
finial_sample = 1000

USE_FFT = False

# Loss weights
LOS_weight = 0.5
NLOS_weight = 0.5

# USE COMMUNICATION performance matrices
# Enable communication COM1 term (power of the last slot)
USE_COM1 = False
COM1_WEIGHT = 1  # Communication term weight
CRB_WEIGHT = 1   # CRB term weight
USE_COM2 = False

USE_WSTAR_LOSS = False
WSTAR_LOSS_WEIGHT = 1.0


tf.reset_default_graph()
he_init = tf.variance_scaling_initializer()

# Place Holders
loc_input = tf.placeholder(tf.float32, shape=(None, num_users, 3), name="loc_input")
channel_bs_irs_user = tf.placeholder(tf.float32, shape=(None, 2 * N_ris, 2 * N, num_users, num_subcarriers), name="channel_bs_irs_user")

# --- TF version of path loss (for CRB in-graph) ---
def tf_path_loss_r(d):
    # d: shape (batch,), returns loss in dB
    return 30.0 + 22.0 * tf.log(d + 1e-9) / tf.log(tf.constant(10.0, dtype=tf.float32))

# --- NumPy version of path loss (for channel generation) ---
def path_loss_r(d):
    # d: scalar or ndarray, returns loss in dB
    return 30.0 + 22.0 * np.log10(d + 1e-9)




# ==== NEW: support uniform angle/radius sampling for training ====
def _sample_point_uniform_phi(ris_xy, phi_range_deg=PHI_RANGE_DEG, r_range_m=R_RANGE_M, z_fixed=-20.0, y_max=None):
    while True:
        phi = np.random.uniform(np.deg2rad(phi_range_deg[0]), np.deg2rad(phi_range_deg[1]))
        r = np.random.uniform(r_range_m[0], r_range_m[1])
        x = ris_xy[0] + r * np.cos(phi)
        y = ris_xy[1] + r * np.sin(phi)
        if y_max is None or (y < y_max - 1e-6):
            return np.array([x, y, z_fixed], dtype=float)

def _build_rect_grid_in_wedge(desired_count,
                              ris_xy,
                              phi_range_deg=PHI_RANGE_DEG,
                              r_range_m=R_RANGE_M,
                              y_max=None,
                              step=GRID_STEP_M,
                              z_fixed=-20.0,
                              num_users=num_users):
    """
    Within the sector ring centered at RIS with angle range `phi_range_deg` and radius range `r_range_m`,
    construct a grid with uniform step in Cartesian coordinates, and filter to keep points satisfying angle/radius constraints and y<y_max.
    Return position array of shape (M, num_users, 3), truncate to desired_count if M >= desired_count.
    If M < desired_count, automatically reduce step size; if still insufficient, pad with random polar coordinate points.
    """
    phi_min = np.deg2rad(phi_range_deg[0])
    phi_max = np.deg2rad(phi_range_deg[1])
    r_min, r_max = float(r_range_m[0]), float(r_range_m[1])

    def build_with_step(step_m):
        # Calculate bounding box of the sector
        pts = np.array([
            [ris_xy[0] + r_min * np.cos(phi_min), ris_xy[1] + r_min * np.sin(phi_min)],
            [ris_xy[0] + r_max * np.cos(phi_min), ris_xy[1] + r_max * np.sin(phi_min)],
            [ris_xy[0] + r_min * np.cos(phi_max), ris_xy[1] + r_min * np.sin(phi_max)],
            [ris_xy[0] + r_max * np.cos(phi_max), ris_xy[1] + r_max * np.sin(phi_max)],
        ], dtype=float)
        x_min = np.floor(pts[:, 0].min() / step_m) * step_m
        x_max = np.ceil(pts[:, 0].max() / step_m) * step_m
        y_min = np.floor(pts[:, 1].min() / step_m) * step_m
        y_upper = np.ceil(pts[:, 1].max() / step_m) * step_m
        if y_max is not None:
            y_upper = min(y_upper, y_max - 1e-6)
        y_max_eff = y_upper
        xs = np.arange(x_min, x_max + 1e-9, step_m)
        ys = np.arange(y_min, y_max_eff + 1e-9, step_m)
        results = []
        for x in xs:
            for y in ys:
                dx = x - ris_xy[0]
                dy = y - ris_xy[1]
                r = np.hypot(dx, dy)
                if not (r_min - 1e-9 <= r <= r_max + 1e-9):
                    continue
                phi = np.arctan2(dy, dx)
                if not (phi_min - 1e-12 <= phi <= phi_max + 1e-12):
                    continue
                if (y_max is not None) and not (y < y_max - 1e-6):
                    continue
                ue = np.array([x, y, z_fixed], dtype=float)
                if num_users >= 2:
                    # Consistent with generation function: place scatter point on y=y_max wall based on mirror intersection
                    ris_mirror = np.array([ris_xy[0], 2.0 * y_max - ris_xy[1]], dtype=float)
                    ue_xy = np.array([x, y], dtype=float)
                    denom = (ris_mirror[1] - ue_xy[1])
                    if abs(denom) < 1e-9:
                        x_sp = ue_xy[0]
                    else:
                        t = (y_max - ue_xy[1]) / denom
                        x_sp = ue_xy[0] + t * (ris_mirror[0] - ue_xy[0])
                    sp = np.array([float(x_sp), y_max, z_fixed], dtype=float)
                    pair = np.stack([ue, sp], axis=0)
                else:
                    pair = ue[None, :]
                results.append(pair)
        return np.array(results, dtype=float)

    # Gradually reduce step size until desired count is reached
    step_cur = float(step)
    grid = build_with_step(step_cur)
    tries = 0
    while grid.shape[0] < desired_count and step_cur > 0.05 and tries < 12:
        step_cur *= 0.85
        grid = build_with_step(step_cur)
        tries += 1

    if grid.shape[0] >= desired_count:
        return grid[:desired_count]

    # If still insufficient: pad with random polar coordinate sampling
    needed = desired_count - grid.shape[0]
    extras = []
    for _ in range(needed):
        pt = _sample_point_uniform_phi(ris_xy, phi_range_deg, r_range_m, z_fixed, y_max)
        if num_users >= 2:
            ris_mirror = np.array([ris_xy[0], 2.0 * y_max - ris_xy[1]], dtype=float)
            ue_xy = pt[:2]
            denom = (ris_mirror[1] - ue_xy[1])
            if abs(denom) < 1e-9:
                x_sp = ue_xy[0]
            else:
                t = (y_max - ue_xy[1]) / denom
                x_sp = ue_xy[0] + t * (ris_mirror[0] - ue_xy[0])
            sp = np.array([float(x_sp), y_max, z_fixed], dtype=float)
            pair = np.stack([pt, sp], axis=0)
        else:
            pair = pt[None, :]
        extras.append(pair)
    if len(extras) > 0:
        extras = np.array(extras, dtype=float)
        return np.concatenate([grid, extras], axis=0)
    else:
        return grid

def _build_fixed_user_locations(num_samples,
                                ue_xy=FIXED_UE_XY,
                                sp_xy=FIXED_SP_XY,
                                z_fixed=FIXED_Z):
    """Construct a repeated (num_samples, num_users, 3) array of fixed UE/SP locations."""
    base = np.stack([
        np.array([ue_xy[0], ue_xy[1], z_fixed], dtype=float),
        np.array([sp_xy[0], sp_xy[1], z_fixed], dtype=float)
    ], axis=0)  # (2,3)
    return np.tile(base[None, :, :], (num_samples, 1, 1))

def generate_location(num_users):
    location_user = np.empty([num_users, 3], dtype=float)
    ris_xy = location_ris_1[:2]
    y_wall = float(ris_xy[1] + 5.5)
    if UNIFORM_PHI:
        location_user[0, :] = _sample_point_uniform_phi(ris_xy, y_max=y_wall)
    else:
        x1 = np.random.uniform(-34.5, -15)
        y1 = np.random.uniform(25, min(55.0, y_wall - 1e-6))
        location_user[0, :] = np.array([x1, y1, -20.0])
    if num_users >= 2:
        ue_xy = location_user[0, :2]
        ris_mirror = np.array([ris_xy[0], 2.0 * y_wall - ris_xy[1]], dtype=float)
        denom = (ris_mirror[1] - ue_xy[1])
        if abs(denom) < 1e-9:
            x_sp = ue_xy[0]
        else:
            t = (y_wall - ue_xy[1]) / denom
            x_sp = ue_xy[0] + t * (ris_mirror[0] - ue_xy[0])
        location_user[1, :] = np.array([float(x_sp), y_wall, -20.0])
    return location_user

# irs_Nh=N_ris number of antennas in y-direction
def generate_irs_user_channel(user_locations, location_irs, num_samples=1, Rician_factor=10, scale_factor=100, irs_Nh= N_ris, debug_power=False):
    num_elements_irs = N_ris
    if user_locations is None:
        num_user = num_users
    else:
        num_user = user_locations.shape[0] if user_locations.ndim == 2 else user_locations.shape[1]
    # --- diagnostics accumulators for quick power self-check ---
    total_P1 = 0.0
    total_P2 = 0.0
    power_count = 0

    # Create channel for each sample and each subcarrier
    channel_irs_user = []
    set_location_user = []
    
    for ii in range(num_samples):
        # Get user location
        if user_locations is None:
            location_user = generate_location(num_user)
        elif user_locations.ndim >= 3:
            location_user = user_locations[ii, :, :]
        else:
            location_user = user_locations
            
        set_location_user.append(location_user)
        
        # Create channel for each subcarrier
        sc_channels = []
        for sc_idx in range(num_subcarriers):
            freq = subcarrier_freqs[sc_idx]
            wavelength = c / freq
            
            # Channel 1: Direct channel from User1 to RIS
            d_1 = np.linalg.norm(location_user[0] - location_irs)
            pathloss_1 = path_loss_r(d_1) - scale_factor / 2
            pathloss_1 = np.sqrt(10 ** ((-pathloss_1) / 10))
            
            # Frequency dependent phase, key for distance estimation
            phase_shift_1 = np.exp(-1j * 2 * np.pi * freq * d_1 / c)
            
            aoa_irs_y_1 = (location_user[0][1] - location_irs[1]) / d_1
            aoa_irs_z_1 = (location_user[0][2] - location_irs[2]) / d_1
            
            i1 = np.mod(np.arange(num_elements_irs), irs_Nh)
            i2 = np.floor(np.arange(num_elements_irs) / irs_Nh)
            
            # User1 to RIS channel, adding frequency dependent phase
            tmp_1 = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs]) \
                  + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs])
            a_irs_user1 = np.exp(1j * np.pi * (i1 * aoa_irs_y_1 + i2 * aoa_irs_z_1))
            channel_1 = np.sqrt(Rician_factor/(1+Rician_factor)) * a_irs_user1 + np.sqrt(1/(1+Rician_factor))
            channel_1 = channel_1 * pathloss_1 * phase_shift_1
            
            # Channel 2: Cascaded channel from User1 to Scatter Point to RIS
            if num_user >= 2:
                # User1 to Scatter Point
                d_user1_scat = np.linalg.norm(location_user[0] - location_user[1])
                pathloss_user1_scat = path_loss_r(d_user1_scat) 
                pathloss_user1_scat = np.sqrt(10 ** ((-pathloss_user1_scat) / 10))
                phase_shift_user1_scat = np.exp(-1j * 2 * np.pi * freq * d_user1_scat / c)
                
                # Scatter Point to RIS
                d_scat_ris = np.linalg.norm(location_user[1] - location_irs)
                pathloss_scat_ris = path_loss_r(d_scat_ris) - scale_factor / 2
                pathloss_scat_ris = np.sqrt(10 ** ((-pathloss_scat_ris) / 10))
                phase_shift_scat_ris = np.exp(-1j * 2 * np.pi * freq * d_scat_ris / c)
                
                aoa_irs_y_scat = (location_user[1][1] - location_irs[1]) / d_scat_ris
                aoa_irs_z_scat = (location_user[1][2] - location_irs[2]) / d_scat_ris
                
                # Scatter Point to RIS channel
                tmp_scat = np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs]) \
                         + 1j * np.random.normal(loc=0, scale=np.sqrt(0.5), size=[num_elements_irs])
                a_irs_scat = np.exp(1j * np.pi * (i1 * aoa_irs_y_scat + i2 * aoa_irs_z_scat))
                channel_scat_ris = np.sqrt(Rician_factor/(1+Rician_factor)) * a_irs_scat + np.sqrt(1/(1+Rician_factor))
                channel_scat_ris = channel_scat_ris * pathloss_scat_ris * phase_shift_scat_ris
                #channel_scat_ris = channel_scat_ris * phase_shift_scat_ris
                
                # User1 to Scatter Point channel (adding frequency phase)
                #channel_user1_scat = 1.0 * pathloss_user1_scat * phase_shift_user1_scat
                channel_user1_scat = 1.0 * phase_shift_user1_scat

                #total pathloss
                pathloss_user1_scat_ris = path_loss_r(d_user1_scat + d_scat_ris) - scale_factor / 2
                #pathloss_user1_scat_ris = path_loss_r(d_scat_ris) - scale_factor / 2
                pathloss_user1_scat_ris = np.sqrt(10 ** ((-pathloss_user1_scat_ris) / 10))
                
                
                # Cascaded channel: User1 -> Scatter Point -> RIS
                channel_2 = pathloss_user1_scat_ris * channel_user1_scat * channel_scat_ris
            else:
                channel_2 = np.zeros(num_elements_irs, dtype=complex)

            # accumulate per-(sample,subcarrier) average element power for diagnostics
            p1 = float(np.mean(np.abs(channel_1)**2))
            p2 = float(np.mean(np.abs(channel_2)**2))
            total_P1 += p1
            total_P2 += p2
            power_count += 1
            
            # Combine channels, output strictly according to num_user count
            if num_user >= 2:
                tmp = np.column_stack([channel_1, channel_2])  # shape: [num_elements_irs, 2]
            else:
                # For single user debug, only output user1 channel as [num_elements_irs, 1]
                tmp = channel_1.reshape(-1, 1)
            sc_channels.append(tmp)
        
        # Add all subcarrier channels for current sample [num_elements_irs, num_user, num_subcarriers]
        channel_irs_user.append(np.stack(sc_channels, axis=-1))
    
    # Keep return format compatible, but add subcarrier dimension
    dummy_bs_user = np.zeros((num_samples, N, num_user, num_subcarriers), dtype=complex)
    dummy_bs_irs = np.zeros((num_samples, N, num_elements_irs, num_subcarriers), dtype=complex)

    # diagnostics print (averaged over all samples and subcarriers of this call)
    if debug_power and power_count > 0:
        mean_P1 = total_P1 / power_count
        mean_P2 = total_P2 / power_count
        ratio_db = 10.0 * np.log10(mean_P1 / (mean_P2 + 1e-12))
        print(f"[power] avg |ch1|^2={mean_P1:.3e}, avg |ch2|^2={mean_P2:.3e}, ratio(ch1/ch2)={ratio_db:.1f} dB, scale_factor={scale_factor}")

    channels = (dummy_bs_user, np.array(channel_irs_user), dummy_bs_irs)
    return channels, set_location_user

def channel_complex2real(channels):
    """complex = [real, imaginary], supports multiple subcarriers"""
    channel_bs_user, channel_irs_user, channel_bs_irs = channels
    
    # Handle new subcarrier dimension
    if channel_irs_user.ndim == 4:
        (num_sample, num_elements_irs, num_user, num_subcarriers) = channel_irs_user.shape
    else:
        (num_sample, num_elements_irs, num_user) = channel_irs_user.shape
        num_subcarriers = 1  # If no subcarrier dimension, default to 1
        
    num_antenna_bs = N
    
    # Modify shape to include subcarrier dimension
    A_T_real = np.zeros([num_sample, 2 * num_elements_irs, 2 * num_antenna_bs, num_user, num_subcarriers], dtype=np.float32)
    set_channel_combine_irs = np.zeros([num_sample, num_antenna_bs, num_elements_irs, num_user, num_subcarriers], dtype=np.complex64)

    for kk in range(num_user):
        for sc_idx in range(num_subcarriers):
            if num_subcarriers > 1:
                # Multi-subcarrier case
                channel_irs_user_k_sc = channel_irs_user[:, :, kk, sc_idx]
            else:
                # Single-subcarrier case
                channel_irs_user_k_sc = channel_irs_user[:, :, kk]
                
            # Process channel for current subcarrier
            channel_combine_irs = channel_irs_user_k_sc.reshape(num_sample, 1, num_elements_irs)
            set_channel_combine_irs[:, :, :, kk, sc_idx if num_subcarriers > 1 else 0] = channel_combine_irs
            
            A_tmp_tran = np.transpose(channel_combine_irs, (0, 2, 1))
            A_tmp_real1 = np.concatenate([A_tmp_tran.real.astype(np.float32), A_tmp_tran.imag.astype(np.float32)], axis=2)
            A_tmp_real2 = np.concatenate([-A_tmp_tran.imag.astype(np.float32), A_tmp_tran.real.astype(np.float32)], axis=2)
            A_tmp_real = np.concatenate([A_tmp_real1, A_tmp_real2], axis=1)
            A_T_real[:, :, :, kk, sc_idx if num_subcarriers > 1 else 0] = A_tmp_real
    
    return A_T_real, set_channel_combine_irs




##################### NETWORK
if __name__ == "__main__":

    with tf.name_scope("array_response_construction"):
        lay = {}
        lay['P'] = tf.placeholder(tf.float32, shape=(), name="snr_lin")  # SNR (linear) as a scalar placeholder
        from0toN = tf.cast(tf.range(0, N, 1), tf.float32)

    with tf.name_scope("channel_sensing"):
        # Increase number of hidden units
        hidden_size_1 = 768  # First LSTM layer
        hidden_size_2 = 512  # Second LSTM layer  
        hidden_size_3 = 256  # Third LSTM layer
        
        # Fully connected layer parameters - used for RIS phase prediction, using progressive dimensionality reduction structure
        A1 = tf.get_variable("A1", shape=[hidden_size_3, 512], dtype=tf.float32, initializer=he_init)
        A2 = tf.get_variable("A2", shape=[512, 256], dtype=tf.float32, initializer=he_init)
        A3 = tf.get_variable("A3", shape=[256, 128], dtype=tf.float32, initializer=he_init)
        A4 = tf.get_variable("A4", shape=[128, 2*N_ris], dtype=tf.float32, initializer=he_init)
        
        b1 = tf.get_variable("b1", shape=[512], dtype=tf.float32, initializer=he_init)
        b2 = tf.get_variable("b2", shape=[256], dtype=tf.float32, initializer=he_init)
        b3 = tf.get_variable("b3", shape=[128], dtype=tf.float32, initializer=he_init)
        b4 = tf.get_variable("b4", shape=[2*N_ris], dtype=tf.float32, initializer=he_init)
        
        # First LSTM layer
        layer_Ui_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Wi_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Uf_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Wf_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Uo_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Wo_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Uc_1 = Dense(units=hidden_size_1, activation='linear')
        layer_Wc_1 = Dense(units=hidden_size_1, activation='linear')
        
        # Second LSTM layer
        layer_Ui_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Wi_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Uf_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Wf_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Uo_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Wo_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Uc_2 = Dense(units=hidden_size_2, activation='linear')
        layer_Wc_2 = Dense(units=hidden_size_2, activation='linear')
        
        # Third LSTM layer
        layer_Ui_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Wi_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Uf_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Wf_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Uo_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Wo_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Uc_3 = Dense(units=hidden_size_3, activation='linear')
        layer_Wc_3 = Dense(units=hidden_size_3, activation='linear')
        
        def LSTM_layer(input_x, h_old, c_old, layer_Ui, layer_Wi, layer_Uf, layer_Wf, layer_Uo, layer_Wo, layer_Uc, layer_Wc):
            i_t = tf.sigmoid(layer_Ui(input_x) + layer_Wi(h_old))
            f_t = tf.sigmoid(layer_Uf(input_x) + layer_Wf(h_old))
            o_t = tf.sigmoid(layer_Uo(input_x) + layer_Wo(h_old))
            c_t = tf.tanh(layer_Uc(input_x) + layer_Wc(h_old))
            c = i_t * c_t + f_t * c_old
            h_new = o_t * tf.tanh(c)
            return h_new, c
        
        # Add projection layers to match dimensions of different LSTM layers
        projection_1_to_2 = Dense(units=hidden_size_2, activation='linear', name='proj_1_to_2')
        projection_2_to_3 = Dense(units=hidden_size_3, activation='linear', name='proj_2_to_3')
        
        def RNN(input_x, h_old_1, c_old_1, h_old_2, c_old_2, h_old_3, c_old_3):
            # First LSTM layer
            h_new_1, c_new_1 = LSTM_layer(input_x, h_old_1, c_old_1, 
                                         layer_Ui_1, layer_Wi_1, layer_Uf_1, layer_Wf_1, 
                                         layer_Uo_1, layer_Wo_1, layer_Uc_1, layer_Wc_1)
            
            # Second LSTM layer (input is output of first layer) + residual connection
            h_proj_1 = projection_1_to_2(h_new_1)  # Project to second layer dimension
            h_new_2, c_new_2 = LSTM_layer(h_new_1, h_old_2, c_old_2,
                                         layer_Ui_2, layer_Wi_2, layer_Uf_2, layer_Wf_2,
                                         layer_Uo_2, layer_Wo_2, layer_Uc_2, layer_Wc_2)
            h_new_2 = h_new_2 + h_proj_1  # Residual connection
            
            # Third LSTM layer (input is output of second layer) + residual connection
            h_proj_2 = projection_2_to_3(h_new_2)  # Project to third layer dimension
            h_new_3, c_new_3 = LSTM_layer(h_new_2, h_old_3, c_old_3,
                                         layer_Ui_3, layer_Wi_3, layer_Uf_3, layer_Wf_3,
                                         layer_Uo_3, layer_Wo_3, layer_Uc_3, layer_Wc_3)
            h_new_3 = h_new_3 + h_proj_2  # Residual connection
            
            return h_new_1, c_new_1, h_new_2, c_new_2, h_new_3, c_new_3
        
        snr = lay['P'] * tf.ones(shape=[tf.shape(loc_input)[0], 1], dtype=tf.float32)
        snr_dB = tf.log(snr) / np.log(10)
        snr_normal = (snr_dB - 1) / np.sqrt(1.6666) ##？？
        
        theta_list = []
        com_p1_terms = []
        com_p2_terms = []
        com2_terms = []
        com_tot_terms = []  # Collect |w^H(h1+h2)|^2 to calculate communication rate

        for t in range(tau):
            if t == 0:
                y_real = tf.ones([tf.shape(loc_input)[0], 2 * num_subcarriers])
                h_old_1 = tf.zeros([tf.shape(loc_input)[0], hidden_size_1])
                c_old_1 = tf.zeros([tf.shape(loc_input)[0], hidden_size_1])
                h_old_2 = tf.zeros([tf.shape(loc_input)[0], hidden_size_2])
                c_old_2 = tf.zeros([tf.shape(loc_input)[0], hidden_size_2])
                h_old_3 = tf.zeros([tf.shape(loc_input)[0], hidden_size_3])
                c_old_3 = tf.zeros([tf.shape(loc_input)[0], hidden_size_3])
            h_old_1, c_old_1, h_old_2, c_old_2, h_old_3, c_old_3 = RNN(tf.concat([y_real, snr_normal], axis=1), 
                                                                        h_old_1, c_old_1, h_old_2, c_old_2, h_old_3, c_old_3)

            # Use output of third LSTM layer for subsequent processing, adding moderate Dropout
            x1 = tf.nn.relu(h_old_3 @ A1 + b1)
            x1 = BatchNormalization()(x1)
            x1 = tf.keras.layers.Dropout(rate=0.1)(x1, training=True)  # Light Dropout
            
            x2 = tf.nn.relu(x1 @ A2 + b2)
            x2 = BatchNormalization()(x2)
            x2 = tf.keras.layers.Dropout(rate=0.1)(x2, training=True)  # Light Dropout
            
            x3 = tf.nn.relu(x2 @ A3 + b3)
            x3 = BatchNormalization()(x3)
            x3 = tf.keras.layers.Dropout(rate=0.05)(x3, training=True)  # Lighter Dropout
            
            # RIS phase design
            ris_her_unnorm = x3 @ A4 + b4
            ris_her_r = ris_her_unnorm[:, 0:N_ris]
            ris_her_i = ris_her_unnorm[:, N_ris:2*N_ris]
            theta_tmp = tf.sqrt(tf.square(ris_her_r) + tf.square(ris_her_i)) + 1e-6
            theta_real = ris_her_r / theta_tmp
            theta_imag = ris_her_i / theta_tmp
            theta = tf.concat([theta_real, theta_imag], axis=1)
            theta_T = tf.reshape(theta, [-1, 1, 2 * N_ris])
            theta_list.append(tf.complex(theta_real, theta_imag))
            if t == (tau - 1):
                h_sum_re = tf.zeros_like(ris_her_r)
                h_sum_im = tf.zeros_like(ris_her_r)
            
            # Process signals for each subcarrier
            y_real_combined = []
            for sc_idx in range(num_subcarriers):
                # Get channel for current subcarrier
                A_T_k1_sc = channel_bs_irs_user[:, :, :, 0, sc_idx]
                if num_users >= 2:
                    A_T_k2_sc = channel_bs_irs_user[:, :, :, 1, sc_idx]
                    A_T_k_sc = (A_T_k1_sc + A_T_k2_sc)
                else:
                    A_T_k2_sc = None
                    A_T_k_sc = A_T_k1_sc

                # === COM1: per-user received power on this subcarrier and timeslot ===
                # theta_T shape: (batch, 1, 2*N_ris); A_T_* shape: (batch, 2*N_ris, 2)
                theta_A_k1 = tf.matmul(theta_T, A_T_k1_sc)  # -> (batch,1,2)
                p1_sc = tf.square(theta_A_k1[:, :, 0]) + tf.square(theta_A_k1[:, :, 1])  # |w^H h1|^2
                com_p1_terms.append(p1_sc)
                if A_T_k2_sc is not None:
                    theta_A_k2 = tf.matmul(theta_T, A_T_k2_sc)  # -> (batch,1,2)
                    p2_sc = tf.square(theta_A_k2[:, :, 0]) + tf.square(theta_A_k2[:, :, 1])  # |w^H h2|^2
                    com_p2_terms.append(p2_sc)
                else:
                    # keep alignment; a zero tensor with same shape as p1_sc
                    com_p2_terms.append(tf.zeros_like(p1_sc))

                # Calculate received signal for current subcarrier
                theta_A_k_T_sc = tf.matmul(theta_T, A_T_k_sc)
                h_d_plus_h_cas_sc = theta_A_k_T_sc
                h_d_plus_h_cas_re_sc = h_d_plus_h_cas_sc[:, :, 0]
                h_d_plus_h_cas_im_sc = h_d_plus_h_cas_sc[:, :, 1]

                # Accumulate channel for w_star (last timeslot only)
                if t == (tau - 1):
                    h_col = A_T_k_sc[:, :, 0]  # (batch, 2*N_ris)
                    h_re_k = h_col[:, 0:N_ris]
                    h_im_k = h_col[:, N_ris:2*N_ris]
                    h_sum_re += h_re_k
                    h_sum_im += h_im_k

                # Add noise
                noise_sc = tf.complex(
                    tf.random_normal(tf.shape(h_d_plus_h_cas_re_sc), mean=0.0, stddev=noiseSTD_per_dim),
                    tf.random_normal(tf.shape(h_d_plus_h_cas_re_sc), mean=0.0, stddev=noiseSTD_per_dim)
                )
                y_complex_sc = tf.complex(tf.sqrt(lay['P']), 0.0) * tf.complex(h_d_plus_h_cas_re_sc, h_d_plus_h_cas_im_sc) + noise_sc

                # Process signal for current subcarrier
                if USE_FFT:
                    y_fft_sc = tf.signal.fft(y_complex_sc)
                    y_real_sc = tf.concat([tf.real(y_fft_sc), tf.imag(y_fft_sc)], axis=1) / tf.sqrt(lay['P'])
                else:
                    y_real_sc = tf.concat([tf.real(y_complex_sc), tf.imag(y_complex_sc)], axis=1) / tf.sqrt(lay['P'])
                y_real_combined.append(y_real_sc)


                y_t_norm = tf.stop_gradient(y_real_sc)          # (batch, 2) -> [Re, Im]
                z_re = h_d_plus_h_cas_re_sc                     # predicted Re{w^H(h1+h2)}
                z_im = h_d_plus_h_cas_im_sc                     # predicted Im{w^H(h1+h2)}
                com2_err_sc = tf.square(y_t_norm[:, 0] - z_re) + tf.square(y_t_norm[:, 1] - z_im)
                # shape (batch,)
                com2_terms.append(tf.reshape(com2_err_sc, [-1, 1]))

                # === Total signal power: |w^H(h1+h2)|^2 for communication rate ===
                p_tot_sc = tf.square(z_re) + tf.square(z_im)  # (batch,1)
                com_tot_terms.append(tf.reshape(p_tot_sc, [-1, 1]))
            
            # Combine signals from all subcarriers
            y_real = tf.concat(y_real_combined, axis=1)
            if t == (tau - 1):
                h_sum_c = tf.complex(h_sum_re, h_sum_im)
                ang = tf.atan2(tf.imag(h_sum_c), tf.real(h_sum_c))
                w_star_real = tf.cos(ang)
                w_star_imag = tf.sin(ang)
                w_star = tf.complex(w_star_real, w_star_imag)
                w_pred = theta_list[-1]
                # Phase-invariant similarity: |w_pred^H w_star| / N_ris (batch average)
                inner = tf.reduce_sum(tf.conj(w_star) * w_pred, axis=1)
                wstar_sim = tf.reduce_mean(tf.abs(inner) / tf.cast(N_ris, tf.float32))
                # Compatible with original monitoring: element-wise MSE
                wstar_mse = tf.reduce_mean(tf.reduce_mean(tf.square(tf.abs(w_pred - w_star)), axis=1))
        h_old_1, c_old_1, h_old_2, c_old_2, h_old_3, c_old_3 = RNN(tf.concat([y_real, snr_normal], axis=1), 
                                                                        h_old_1, c_old_1, h_old_2, c_old_2, h_old_3, c_old_3)
        # === COM1 objective: Changed to communication rate log2(1 + SNR), where SNR = P * |w^H(h1+h2)|^2 ===
        _log2_eps = tf.constant(1e-12, dtype=tf.float32, name="_log2_eps")
        if len(com_tot_terms) > 0:
            tot_stack = tf.concat(com_tot_terms, axis=1)  # (batch, T*K)
            start_idx = (tau - 1) * num_subcarriers
            end_idx = tau * num_subcarriers
            tot_last = tot_stack[:, start_idx:end_idx]  # (batch, K)
            snr_sum_last = lay['P'] * tf.reduce_sum(tot_last, axis=1, keepdims=True)  # (batch, 1)
            rate_sum = tf.log(1.0 + snr_sum_last) / tf.log(tf.constant(2.0, dtype=tf.float32))  # (batch, 1)
            rate_tot_mean = tf.reduce_mean(rate_sum)
            com_power_mean = rate_tot_mean
        else:
            rate_tot_mean = tf.constant(0.0, dtype=tf.float32)
            com_power_mean = rate_tot_mean

        # === COM2 objective: use only the last timeslot (all subcarriers) ===
        if len(com2_terms) > 0:
            com2_stack = tf.concat(com2_terms, axis=1)   # (batch, T*K)
            # slice last timeslot columns: indices [(tau-1)*K ... tau*K-1]
            start_idx = (tau - 1) * num_subcarriers
            end_idx   = tau * num_subcarriers
            com2_last = com2_stack[:, start_idx:end_idx]  # (batch, K)
            com2_loss_mean = tf.reduce_mean(com2_last)    # scalar
        else:
            com2_loss_mean = tf.constant(0.0, dtype=tf.float32)

        ## location estimator
        dense1 = Dense(units=200, activation='linear')
        dense2 = Dense(units=200, activation='linear')
        c_old_3 = dense1(c_old_3)  # Use cell state of third LSTM layer
        c_old_3 = dense2(c_old_3)
        # Output only 2D coordinates (x,y)
        loc_flat = Dense(units=2*num_users, activation='linear')(c_old_3)  # 2D coordinates * 2 users
        loc_hat = tf.reshape(loc_flat, [-1, num_users, 2])  # Changed to 2D

    # Improved loss function, adding adaptive weights and auxiliary loss
    with tf.name_scope("loss_computation"):
        # Original location loss
        xy_pred_u1 = loc_hat[:, 0, :]  # (batch, 2)
        xy_true_u1 = loc_input[:, 0, 0:2]  # Take first two dimensions from 3D input
        loss1 = tf.reduce_mean(tf.square(xy_pred_u1 - xy_true_u1))  # user-1 XY MSE
        if num_users >= 2:
            xy_pred_u2 = loc_hat[:, 1, :]
            xy_true_u2 = loc_input[:, 1, 0:2]
            loss2 = tf.reduce_mean(tf.square(xy_pred_u2 - xy_true_u2))  # user-2 XY MSE
        else:
            # single-user debug: define a zero placeholder loss for user-2 to keep logging shapes intact
            loss2 = tf.constant(0.0, dtype=tf.float32)

        if num_users >= 2:
            loss1_weight = 0.5  # User 1 weight 50%
            loss2_weight = 0.5  # User 2 weight 50%
        else:
            loss1_weight = 1.0
            loss2_weight = 0.0

        theta_smoothness_loss = tf.constant(0.0, dtype=tf.float32)
        if len(theta_list) > 1:
            for i in range(1, len(theta_list)):
                theta_diff = theta_list[i] - theta_list[i-1]
                theta_smoothness_loss += tf.reduce_mean(tf.square(tf.abs(theta_diff)))
            theta_smoothness_loss /= (len(theta_list) - 1)

        if USE_COM2:
            loss = com2_loss_mean
            # Still provide a placeholder for CRB component for user_loss dimension alignment
            loss_crb_tf = tf.constant(0.0, dtype=tf.float32)
        else:
            # Combined loss: Allow enabling both COM1 and CRB simultaneously
            loss_terms = []
            # Preset CRB component placeholder (remains 0 when not enabled)
            loss_crb_tf = tf.constant(0.0, dtype=tf.float32)

            # CRB component (using ground truth location, in-graph TF version)
            if USE_CRB_LOSS:
                loc_true_3d = loc_input
                # Unified entry: Dispatch four CRB variants based on CRB_METHOD or fallback to original
                loss_crb_tf = CRB_LOSS_FN(theta_list, loc_true_3d, lay['P'])
                loss_terms.append(CRB_WEIGHT * loss_crb_tf)

            # COM1 component: Use communication rate log2(1 + P*|w^H(h1+h2)|^2)
            if USE_COM1:
                # com1_term = - (com_p1_mean_log2 + com_p2_mean_log2)
                    # com1_term = - tf.minimum(com_p1_mean_log2, com_p2_mean_log2)
                    #com1_term = - (com_p1_mean * com_p2_mean)
                com1_term = - rate_tot_mean  # Maximizing rate is equivalent to minimizing its negative value
                loss_terms.append(COM1_WEIGHT * com1_term)

            if USE_WSTAR_LOSS:
                if 'wstar_sim' in locals():
                    wstar_term = 1.0 - wstar_sim
                elif 'wstar_mse' in locals():
                    wstar_term = wstar_mse
                else:
                    wstar_term = tf.constant(0.0, dtype=tf.float32)
                loss_terms.append(WSTAR_LOSS_WEIGHT * wstar_term)

            if len(loss_terms) > 0:
                loss = tf.add_n(loss_terms)
            else:
                # Fallback: Localization loss + smoothness penalty
                base_loss = loss1_weight * loss1 + loss2_weight * loss2
                smoothness_penalty = 0.01 * theta_smoothness_loss
                loss = base_loss + smoothness_penalty

        # Add CRB component, w_star error and similarity to user_loss, value is 0 when not enabled
        if 'wstar_mse' in locals():
            wstar_mse_out = wstar_mse
        else:
            wstar_mse_out = tf.constant(0.0, dtype=tf.float32)
        if 'wstar_sim' in locals():
            wstar_sim_out = wstar_sim
        else:
            wstar_sim_out = tf.constant(0.0, dtype=tf.float32)
        user_loss = tf.stack([loss1, loss2, com_power_mean, com2_loss_mean, loss_crb_tf, wstar_mse_out, wstar_sim_out], name='user_loss')

        # Improved optimizer configuration
        with tf.name_scope("optimizer"):
            # Use learning rate decay
            global_step = tf.Variable(0, trainable=False)
            learning_rate_decay = tf.train.exponential_decay(
                learning_rate=learning_rate,
                global_step=global_step,
                decay_steps=100,  # Decay every 100 steps
                decay_rate=0.96,  # Decay rate
                staircase=True
            )
            
            # Use Adam optimizer, and add gradient clipping
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate_decay, beta1=0.9, beta2=0.999)
            
            l2 = 1e-4
            reg_term = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
            loss_reg = loss + l2 * reg_term
            
            grads_vars = optimizer.compute_gradients(loss_reg)
            # Filter out None gradients (e.g., CRB from tf.py_func)
            grads_vars = [(g, v) for (g, v) in grads_vars if g is not None]
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                if len(grads_vars) > 0:
                    gradients, variables = zip(*grads_vars)
                    gradients, _ = tf.clip_by_global_norm(gradients, 1.0)
                    training_op = optimizer.apply_gradients(list(zip(gradients, variables)), global_step=global_step)
                else:
                    # No valid gradients, skip training step
                    training_op = tf.no_op()

        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

        # Validation set
        channel_true_val, set_location_user_val = generate_irs_user_channel(
            None, location_ris_1, num_samples=val_size_order*delta_inv, Rician_factor=Rician_factor, debug_power=True)
        # Allow overriding validation set with tiny dataset
        if OVERFIT_FIXED_GEOM:
            if OVERFIT_DATASET_MODE == 'fixed':
                locs_val = _build_fixed_user_locations(OVERFIT_VAL_SAMPLES)
                channel_true_val, set_location_user_val = generate_irs_user_channel(
                    user_locations=locs_val,
                    location_irs=location_ris_1,
                    num_samples=OVERFIT_VAL_SAMPLES,
                    Rician_factor=Rician_factor,
                    debug_power=True
                )
            else:  # 'random' small validation dataset
                channel_true_val, set_location_user_val = generate_irs_user_channel(
                    None,
                    location_ris_1,
                    num_samples=OVERFIT_VAL_SAMPLES,
                    Rician_factor=Rician_factor,
                    debug_power=True
                )
        A_T_1_real_val, _ = channel_complex2real(channel_true_val)

        feed_dict_val = {
            loc_input: np.array(set_location_user_val),
            channel_bs_irs_user: A_T_1_real_val,
            lay['P']: Pvec[0]
        }

        # Training process
        # Disable XLA/JIT, keep GPU memory growth, avoid macOS XLA platform error
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.OFF
        with tf.Session(config=config) as sess:
            # Initialize all variables first, then restore from checkpoint if initial_run==0
            init.run()
            if initial_run == 0:
                ckpt_path = f"{drive_save_path}/params_closeBS_fullRician_3D_1RIS_newcoordinateSISO_N_{N}_tau_{tau}_snr_{int(snr_const[0])}"
                try:
                    reader = tf.train.NewCheckpointReader(ckpt_path)
                    ckpt_vars = reader.get_variable_to_shape_map().keys()
                    var_list = {}
                    for v in tf.global_variables():
                        name = v.name.split(':')[0]
                        if name in ckpt_vars:
                            var_list[name] = v
                    if len(var_list) > 0:
                        tf.train.Saver(var_list=var_list).restore(sess, ckpt_path)
                        print('Restored {} vars from checkpoint: {}'.format(len(var_list), ckpt_path))
                    else:
                        print('No matching variables in checkpoint: {}'.format(ckpt_path))
                except Exception as e:
                    print('Checkpoint restore failed: {}'.format(e))
            
            print(tf.test.is_gpu_available())
            print('snr=', snr_const)

            # Record training/validation performance over epochs
            train_perf_list = []
            train_u1_list = []
            train_u2_list = []
            val_perf_list = []
            val_u1_list = []
            val_u2_list = []

            # If overfitting, prebuild a tiny training dataset (fixed or random)
            if OVERFIT_FIXED_GEOM:
                if OVERFIT_DATASET_MODE == 'fixed':
                    locs_train_small = _build_fixed_user_locations(OVERFIT_TRAIN_SAMPLES)
                    channel_true_train_small, set_location_user_small = generate_irs_user_channel(
                        user_locations=locs_train_small,
                        location_irs=location_ris_1,
                        num_samples=OVERFIT_TRAIN_SAMPLES,
                        Rician_factor=Rician_factor
                    )
                else:  # 'random' small training dataset
                    channel_true_train_small, set_location_user_small = generate_irs_user_channel(
                        None,
                        location_ris_1,
                        num_samples=OVERFIT_TRAIN_SAMPLES,
                        Rician_factor=Rician_factor
                    )
                A_T_1_real_small, _ = channel_complex2real(channel_true_train_small)

            for epoch in range(n_epochs):
                epoch_train_losses = []
                epoch_u1_losses = []
                epoch_u2_losses = []
                
                for rnd_indices in range(batch_per_epoch):
                    # Generate training data (optional: fixed small dataset for overfitting)
                    if OVERFIT_FIXED_GEOM:
                        set_location_user_train = set_location_user_small
                        A_T_1_real = A_T_1_real_small
                    else:
                        channel_true_train, set_location_user_train = generate_irs_user_channel(
                            None, location_ris_1, num_samples=batch_size_order*delta_inv, Rician_factor=Rician_factor)
                        A_T_1_real, _ = channel_complex2real(channel_true_train)
                    
                    feed_dict_batch = {
                        loc_input: np.array(set_location_user_train),
                        channel_bs_irs_user: A_T_1_real,
                        lay['P']: Pvec[0]
                    }
                    
                    if epoch == 0 and (rnd_indices % 4 == 0):
                        print(f"[epoch0] batch {rnd_indices}/{batch_per_epoch}", flush=True)
                    
                    _, train_loss_val, per_user_train = sess.run(
                        [training_op, loss, user_loss], feed_dict=feed_dict_batch
                    )
                    
                    epoch_train_losses.append(train_loss_val)
                    epoch_u1_losses.append(per_user_train[0])
                    epoch_u2_losses.append(per_user_train[1])
                
                avg_train_loss = np.mean(epoch_train_losses)
                loss_val, per_user_val = sess.run([loss, user_loss], feed_dict=feed_dict_val)

                # Record current epoch's training and validation performance
                train_perf_list.append(float(avg_train_loss))
                train_u1_list.append(float(np.mean(epoch_u1_losses)))
                train_u2_list.append(float(np.mean(epoch_u2_losses)))
                val_perf_list.append(float(loss_val))
                val_u1_list.append(float(per_user_val[0]))
                val_u2_list.append(float(per_user_val[1]))
                
                print('epoch', epoch,
                    '  train_loss:%2.5f' % avg_train_loss,
                    '  train_u1:%2.5f' % np.mean(epoch_u1_losses),
                    '  train_u2:%2.5f' % np.mean(epoch_u2_losses),
                    '  val_loss:%2.5f' % loss_val,
                    '  val_u1:%2.5f' % per_user_val[0],
                    '  val_u2:%2.5f' % per_user_val[1],
                    '  val_rate:%2.5f' % per_user_val[2],
                    '  val_COM2:%2.5f' % per_user_val[3],
                    '  val_CRB:%2.5f' % per_user_val[4],
                    '  val_wstar_mse:%2.5f' % per_user_val[5],
                    '  val_wstar_sim:%2.5f' % per_user_val[6],
                    flush=True)

                # Save checkpoint every 10 epochs
                if (epoch + 1) % 10 == 0:
                    save_prefix = f"{drive_save_path}/params_closeBS_fullRician_3D_1RIS_newcoordinateSISO_N_{N}_tau_{tau}_snr_{int(snr_const[0])}"
                    saver.save(sess, save_prefix)
                    print(f"[checkpoint] saved epoch {epoch+1} to {save_prefix}", flush=True)

            # Final test: if overfitting, use small dataset and sync sample number
            finial_sample = OVERFIT_TRAIN_SAMPLES if OVERFIT_FIXED_GEOM else 1000
            performance = np.zeros([finial_sample])  # Store total MSE for each sample
            performance_u1 = np.zeros([finial_sample])  # Store user1 MSE for each sample
            performance_u2 = np.zeros([finial_sample])  # Store user2 MSE for each sample
            all_locations = []  # Store location info for all samples
            all_theta_sequences = []  # Store theta sequence for all samples
            # Non-overfit: uniformly sample within the wedge defined by PHI_RANGE_DEG and R_RANGE_M
            if not OVERFIT_FIXED_GEOM:
                ris_xy = location_ris_1[:2]
                y_wall = float(ris_xy[1] + 5.5)
                grid_locations = _build_rect_grid_in_wedge(
                    desired_count=finial_sample,
                    ris_xy=ris_xy,
                    phi_range_deg=PHI_RANGE_DEG,
                    r_range_m=R_RANGE_M,
                    y_max=y_wall,
                    step=GRID_STEP_M,
                    z_fixed=-20.0,
                    num_users=num_users
                )
 
            for j in range(finial_sample): 
                #print(j) 
                if OVERFIT_FIXED_GEOM:
                    # Reuse pre-generated small dataset for overfitting
                    actual_location_for_feed = set_location_user_small[j]
                    # Pre-generated A_T_1_real_small is stacked by samples, so we take one and expand dims
                    A_T_1_real_test = np.expand_dims(A_T_1_real_small[j], axis=0)
                else:
                    # 1. Take grid-sampled position for the current test sample (satisfying PHI_RANGE_DEG and R_RANGE_M)
                    location_user_target = grid_locations[j]
                    # 2. Generate channel based on the new random position
                    channel_true_test, set_location_user_test_single = generate_irs_user_channel( 
                        user_locations=np.expand_dims(location_user_target, axis=0), 
                        location_irs=location_ris_1, 
                        num_samples=1, 
                        Rician_factor=Rician_factor, 
                        debug_power=(j == finial_sample - 1) 
                    ) 
                    actual_location_for_feed = set_location_user_test_single[0] 
                    A_T_1_real_test, _ = channel_complex2real(channel_true_test) 
                
                # 3. Feed the current test sample into the model
                feed_dict_test = { 
                    loc_input: np.expand_dims(actual_location_for_feed, axis=0), # loc_input  (batch, num_users, 3) 
                    channel_bs_irs_user: A_T_1_real_test,
                    lay['P']: Pvec[0] 
                } 
                
                # 4. Run the session to get loss and theta 
                mse_loss, loss1_val, loss2_val, phi_hat_test, theta_sequence = sess.run( 
                    [loss, loss1, loss2, loc_hat, theta_list], feed_dict=feed_dict_test 
                ) 
                
                # 5. Store the performance of the current sample 
                performance[j] = mse_loss 
                performance_u1[j] = loss1_val 
                performance_u2[j] = loss2_val 
                
                # 6. Store the location and theta sequence of the current sample
                all_locations.append(actual_location_for_feed.copy())
                all_theta_sequences.append(np.array(theta_sequence).copy())
                
                # 7. Save the theta_test and loc_true of the last sample as examples
                if j == finial_sample - 1: 
                    example_theta_test = np.array(theta_sequence) # theta_list  tau* (1, N_ris) 
                    example_loc_true = actual_location_for_feed 
 
            # 8. Calculate the average performance over all samples
            avg_mse = np.mean(performance) 
            avg_mse_u1 = np.mean(performance_u1) 
            avg_mse_u2 = np.mean(performance_u2) 
            print(f"\nFinal Test Average Performance over {finial_sample} samples:") 
            print(f"  Average Total MSE: {avg_mse:.5f}") 
            print(f"  Average User1 MSE: {avg_mse_u1:.5f}") 
            print(f"  Average User2 MSE: {avg_mse_u2:.5f}") 
 
            
            model_filename = os.path.join(drive_save_path, f'interpret_avg_N{N_ris}_tau{tau}_snr{int(snr_const[0])}.mat') 
            sio.savemat(model_filename, dict( 
                performance_samples=performance, # Save MSE of all samples
                performance_u1_samples=performance_u1, 
                performance_u2_samples=performance_u2, 
                avg_mse = avg_mse, # 保存平均MSE 
                avg_mse_u1 = avg_mse_u1, 
                avg_mse_u2 = avg_mse_u2, 
                snr_const=snr_const, 
                N=N, N_ris=N_ris, 
                epoch=n_epochs, 
                delta_inv=delta_inv, 
                mean_true_alpha=mean_true_alpha, 
                example_theta_test=example_theta_test, # Save a sample's theta sequence
                example_loc_true=example_loc_true,     # and its corresponding true location
                all_locations=np.array(all_locations), # Save location info of all samples
                all_theta_sequences=np.array(all_theta_sequences), # Save theta sequence of all samples
                std_per_dim_alpha=std_per_dim_alpha, 
                noiseSTD_per_dim=noiseSTD_per_dim, 
                tau=tau, 
                num_test_samples_avg = finial_sample 
            )) 
            print(f"Final test results saved to {model_filename}") 


            # sample from test set
            num_test_samples = 10
            if OVERFIT_FIXED_GEOM:
                sample_indices = random.sample(range(len(set_location_user_small)), num_test_samples)
            else:
                sample_indices = random.sample(range(len(set_location_user_train)), num_test_samples)
 
            train_losses = [] 
            train_losses_u1 = []   
            train_losses_u2 = []  
            theta_test_list = [] 
            location_list = [] 
 
            for sample_index in sample_indices:
                if OVERFIT_FIXED_GEOM:
                    location_user_target = set_location_user_small[sample_index]
                    A_T_1_real_test = A_T_1_real_small[sample_index]
                else:
                    location_user_target = set_location_user_train[sample_index]
                    A_T_1_real_test = A_T_1_real[sample_index]
                
                feed_dict_test = { 
                    loc_input: np.expand_dims(location_user_target, axis=0), 
                    channel_bs_irs_user: np.expand_dims(A_T_1_real_test, axis=0), 
                    lay['P']: Pvec[0] 
                } 
                
                
                mse_loss, loss1_val, loss2_val, phi_hat_test, theta_test = sess.run( 
                    [loss, loss1, loss2, loc_hat, theta_list], feed_dict=feed_dict_test) 
                train_losses.append(mse_loss) 
                train_losses_u1.append(loss1_val)  
                train_losses_u2.append(loss2_val)   
                theta_test = np.array(theta_test) 
                
                theta_test_list.append(theta_test.copy()) 
                location_list.append(location_user_target) 
 
            
            model_filename = os.path.join(drive_save_path, f'train_data_sample_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}.mat') 
            sio.savemat(model_filename, dict( 
                performance=np.array(train_losses), 
                performance_u1=np.array(train_losses_u1),   
                performance_u2=np.array(train_losses_u2),  
                snr_const=snr_const, 
                N=N, N_ris=N_ris, 
                epoch=n_epochs, 
                delta_inv=delta_inv, 
                mean_true_alpha=mean_true_alpha, 
                loc_true_list=location_list, 
                std_per_dim_alpha=std_per_dim_alpha, 
                noiseSTD_per_dim=noiseSTD_per_dim, 
                tau=tau, 
                theta_test_list=theta_test_list, 
                com_flag=np.array([int(USE_COM1)]), 
            )) 
            
            
            # sample form validation set
            num_test_samples = 10 
            sample_indices = random.sample(range(len(set_location_user_val)), num_test_samples) 
 
            train_losses = [] 
            val_losses_u1 = [] 
            val_losses_u2 = [] 
            theta_test_list = [] 
            location_list = [] 
 
            for sample_index in sample_indices: 
                location_user_target = set_location_user_val[sample_index] 
                A_T_1_real_test = A_T_1_real_val[sample_index] 
                
                feed_dict_test = { 
                    loc_input: np.expand_dims(location_user_target, axis=0), 
                    channel_bs_irs_user: np.expand_dims(A_T_1_real_test, axis=0), 
                    lay['P']: Pvec[0] 
                } 
                
                
                mse_loss, loss1_val, loss2_val, phi_hat_test, theta_test = sess.run( 
                    [loss, loss1, loss2, loc_hat, theta_list], feed_dict=feed_dict_test) 
                train_losses.append(mse_loss) 
                val_losses_u1.append(loss1_val)  # 
                val_losses_u2.append(loss2_val)  # 
                theta_test = np.array(theta_test) 
                theta_test_list.append(theta_test.copy()) 
                location_list.append(location_user_target) 
            
            model_filename = os.path.join(drive_save_path, f'validate_data_sample_N_{N_ris}_tau_{tau}_snr_{int(snr_const[0])}.mat') 
            sio.savemat(model_filename, dict( 
                performance=np.array(train_losses), 
                performance_u1=np.array(val_losses_u1),  # 
                performance_u2=np.array(val_losses_u2),  # 
                snr_const=snr_const, 
                N=N, N_ris=N_ris, 
                epoch=n_epochs, 
                delta_inv=delta_inv, 
                mean_true_alpha=mean_true_alpha, 
                loc_true_list=location_list, 
                std_per_dim_alpha=std_per_dim_alpha, 
                noiseSTD_per_dim=noiseSTD_per_dim, 
                tau=tau, 
                theta_test_list=theta_test_list, 
                com_flag=np.array([int(USE_COM1)]), 
            )) 
            
            # save finial checkpoint
            final_prefix = f"{drive_save_path}/params_closeBS_fullRician_3D_1RIS_newcoordinateSISO_N_{N}_tau_{tau}_snr_{int(snr_const[0])}"
            saver.save(sess, final_prefix)
            print(f"[checkpoint] final saved to {final_prefix}", flush=True)
