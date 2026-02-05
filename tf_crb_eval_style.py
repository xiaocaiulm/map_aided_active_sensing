"""TensorFlow CRB (eval-style, nuisance_gain=True) module.

This module provides a TensorFlow v1-compatible implementation that mirrors
matlab_style_crb.calculate_crb with nuisance_gain=True (tx SNR semantics).

Exports:
- tf_crb_loss_eval_style(theta_list, loc_input, snr_db)
- tf_crb_loss_eval_style_tx(theta_list, loc_input, P_lin)
 - tf_crb_loss_eval_style_fast(theta_list, loc_input, snr_db)
 - tf_crb_loss_eval_style_tx_fast(theta_list, loc_input, P_lin)

Inputs:
- theta_list: list of T tensors, each shape [1, N_ris], dtype tf.complex64
- loc_input: tensor shape [batch, 2, 3], dtype tf.float32 (UE xyz, SP xyz)
- snr_db: float scalar (tx SNR in dB) for tf_crb_loss_eval_style
- P_lin: float scalar (linear transmit power) for tf_crb_loss_eval_style_tx

Outputs:
- Mean of total position MSE (UE+SP) over batch.
"""

import numpy as np
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

# Constants aligned with evaluation code
N_RIS = 32
NUM_SUBCARRIERS = 10
CARRIER_FREQ = 2.4e9
BANDWIDTH = 100e6
SUBCARRIER_FREQS = CARRIER_FREQ + np.linspace(-BANDWIDTH/2, BANDWIDTH/2, NUM_SUBCARRIERS)
C_LIGHT = 3e8
RIS_POS = np.array([-35.0, 40.0, -20.0], dtype=np.float32)

# Numerical options for CRB inversion
CRB_USE_PINV = False  # Default to strict inverse semantics
EPS_JITTER = 1e-5     # Slightly higher absolute eigenvalue floor for stability
EPS_REL = 1e-2        # Relative floor (fraction of mean eigenvalue)
EPS_COND_FLOOR = 1e-12  # Singular value threshold for inverse fallback
EPS_DET_FLOOR = 1e-10   # Determinant floor to decide strict inverse safety
EPS_ADAPT_JITTER = 1e-3 # Extra per-batch jitter when near-singular
EPS_TRACE_FLOOR = 1e-9  # Floor for trace before sqrt to prevent NaN

# Robust SPD projection and inversion helpers
def robust_inv_spd_2d(M):
    """Return inverse of SPD-projected 2D matrix (shape [4,4])."""
    M_sym = 0.5 * (M + tf.transpose(M))
    M_sym = tf.where(tf.math.is_finite(M_sym), M_sym, tf.zeros_like(M_sym))
    eigvals, eigvecs = tf.linalg.eigh(M_sym)
    # Floor eigenvalues using absolute and relative thresholds
    mean_eig = tf.reduce_mean(eigvals)
    rel_floor = tf.maximum(mean_eig * tf.constant(EPS_REL, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
    eig_floor = tf.maximum(tf.constant(EPS_JITTER, dtype=tf.float32), rel_floor)
    eigvals_clamped = tf.maximum(eigvals, eig_floor)
    inv_diag = tf.linalg.diag(1.0 / eigvals_clamped)
    return tf.matmul(eigvecs, tf.matmul(inv_diag, tf.transpose(eigvecs)))

def robust_inv_spd_3d(M):
    """Return inverse of SPD-projected batched matrix (shape [B,4,4])."""
    M_sym = 0.5 * (M + tf.transpose(M, perm=[0, 2, 1]))
    M_sym = tf.where(tf.math.is_finite(M_sym), M_sym, tf.zeros_like(M_sym))
    eigvals, eigvecs = tf.linalg.eigh(M_sym)  # [B,4], [B,4,4]
    # Floor eigenvalues using absolute and relative thresholds per batch
    mean_eig = tf.reduce_mean(eigvals, axis=1, keepdims=True)  # [B,1]
    rel_floor = tf.maximum(mean_eig * tf.constant(EPS_REL, dtype=tf.float32), tf.zeros_like(mean_eig))
    eig_floor = tf.maximum(tf.constant(EPS_JITTER, dtype=tf.float32), rel_floor)  # [B,1]
    eigvals_clamped = tf.maximum(eigvals, eig_floor)  # [B,4]
    inv_diag = tf.linalg.diag(1.0 / eigvals_clamped)  # [B,4,4]
    return tf.matmul(eigvecs, tf.matmul(inv_diag, tf.transpose(eigvecs, perm=[0, 2, 1])))

def conditional_inv_2d(M):
    """Strict inverse with jitter; fall back to pseudo-inverse if near-singular.
    Uses determinant threshold to avoid building failing inverse ops.
    """
    I4 = tf.eye(4, dtype=tf.float32)
    M_base = tf.where(tf.math.is_finite(M), M, tf.zeros_like(M))
    M_plus = M_base + I4 * EPS_JITTER
    det_val = tf.linalg.det(M_plus)
    # If near-singular, add adaptive jitter
    add_jit = tf.where(tf.less_equal(tf.abs(det_val), tf.constant(EPS_DET_FLOOR, dtype=tf.float32)),
                       tf.constant(EPS_ADAPT_JITTER, dtype=tf.float32),
                       tf.constant(0.0, dtype=tf.float32))
    M_plus = M_plus + I4 * add_jit
    return robust_inv_spd_2d(M_plus)

def conditional_inv_3d(M):
    """Batched strict inverse with jitter; fall back to pseudo-inverse if ANY batch item is near-singular.
    Uses a global predicate (all det > floor) to avoid constructing failing inverse ops.
    """
    I4 = tf.eye(4, dtype=tf.float32)
    M_base = tf.where(tf.math.is_finite(M), M, tf.zeros_like(M))
    M_plus = M_base + I4 * EPS_JITTER
    dets = tf.linalg.det(M_plus)  # [B]
    # Per-batch adaptive jitter where det is small
    cond = tf.less_equal(tf.abs(dets), tf.constant(EPS_DET_FLOOR, dtype=tf.float32))  # [B]
    add_jit_vec = tf.where(cond,
                           tf.fill(tf.shape(dets), tf.constant(EPS_ADAPT_JITTER, dtype=tf.float32)),
                           tf.zeros_like(dets))  # [B]
    M_plus = M_plus + tf.expand_dims(tf.expand_dims(add_jit_vec, 1), 2) * I4
    return robust_inv_spd_3d(M_plus)

# Batched 2×2 SPD-projected inverse (for UE-only EFIMs)
def robust_inv_spd_2x2_batch(M):
    M_sym = 0.5 * (M + tf.transpose(M, perm=[0, 2, 1]))
    M_sym = tf.where(tf.math.is_finite(M_sym), M_sym, tf.zeros_like(M_sym))
    eigvals, eigvecs = tf.linalg.eigh(M_sym)  # [B,2], [B,2,2]
    mean_eig = tf.reduce_mean(eigvals, axis=1, keepdims=True)
    rel_floor = tf.maximum(mean_eig * tf.constant(EPS_REL, dtype=tf.float32), tf.zeros_like(mean_eig))
    eig_floor = tf.maximum(tf.constant(EPS_JITTER, dtype=tf.float32), rel_floor)
    eigvals_clamped = tf.maximum(eigvals, eig_floor)
    inv_diag = tf.linalg.diag(1.0 / eigvals_clamped)
    return tf.matmul(eigvecs, tf.matmul(inv_diag, tf.transpose(eigvecs, perm=[0, 2, 1])))


def _tf_path_loss_db(d):
    d_eff = tf.maximum(d, 0.1)
    return (30.0 + 22.0 * tf.log(d_eff) / tf.log(tf.constant(10.0, dtype=tf.float32)))


def _tf_alpha_fun(distance, scale_db=50.0):
    pl_db = _tf_path_loss_db(distance) - tf.constant(scale_db, dtype=tf.float32)
    return tf.pow(tf.constant(10.0, dtype=tf.float32), -pl_db / tf.constant(20.0, dtype=tf.float32))


def tf_crb_loss_eval_style(theta_list, loc_input, snr_db):
    """TensorFlow replica of calculate_crb nuisance_gain=True (tx SNR).

    Computes batch-mean of total position MSE (UE+SP).
    """
    T = len(theta_list)
    K = NUM_SUBCARRIERS
    # constants
    ris_pos = tf.constant(RIS_POS, dtype=tf.float32)
    i1 = tf.constant(np.arange(N_RIS), dtype=tf.float32)
    freqs = tf.constant(SUBCARRIER_FREQS, dtype=tf.float32)
    two_pi = tf.constant(2.0 * np.pi, dtype=tf.float32)
    c_tf = tf.constant(C_LIGHT, dtype=tf.float32)
    lam_center = tf.constant(C_LIGHT / CARRIER_FREQ, dtype=tf.float32)
    d_spacing = lam_center / 2.0
    j_complex = tf.complex(tf.constant(0.0, dtype=tf.float32), tf.constant(1.0, dtype=tf.float32))

    snr_db_tf = tf.convert_to_tensor(snr_db, dtype=tf.float32)
    snr_tx_lin = tf.pow(tf.constant(10.0, dtype=tf.float32), snr_db_tf / tf.constant(10.0, dtype=tf.float32))

    # Stack beams to [batch, T, N]
    theta_seq = tf.stack(theta_list, axis=1)

    def per_sample(loc_pair):
        loc_ue = loc_pair[0, :]
        loc_sp = loc_pair[1, :]

        # Geometry
        diff_ue = loc_ue - ris_pos
        diff_sp = loc_sp - ris_pos
        d_ue = tf.sqrt(tf.reduce_sum(tf.square(diff_ue)) + 1e-12)
        d_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_sp)) + 1e-12)
        diff_ue_sp = loc_ue - loc_sp
        d_ue_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_ue_sp)) + 1e-12)
        d_sum = d_sp + d_ue_sp
        theta_ue = tf.atan2(diff_ue[1], diff_ue[0])
        theta_sp = tf.atan2(diff_sp[1], diff_sp[0])

        # Steering vectors per frequency
        sin_ue = tf.sin(theta_ue)
        sin_sp = tf.sin(theta_sp)
        a_los_list = []
        a_sp_list = []
        phase_los_list = []
        phase_sp_list = []
        k_list = []
        for k_idx in range(K):
            f = freqs[k_idx]
            k_val = (two_pi * f) / c_tf
            k_list.append(k_val)
            k_val_c = tf.complex(k_val, tf.constant(0.0, dtype=tf.float32))
            sin_ue_c = tf.complex(sin_ue, tf.constant(0.0, dtype=tf.float32))
            sin_sp_c = tf.complex(sin_sp, tf.constant(0.0, dtype=tf.float32))
            d_spacing_c = tf.complex(d_spacing, tf.constant(0.0, dtype=tf.float32))
            i1_c = tf.complex(i1, tf.zeros_like(i1))
            elem_phase_ue = j_complex * k_val_c * d_spacing_c * i1_c * sin_ue_c
            elem_phase_sp = j_complex * k_val_c * d_spacing_c * i1_c * sin_sp_c
            a_los_list.append(tf.exp(elem_phase_ue))
            a_sp_list.append(tf.exp(elem_phase_sp))
            phase_los_list.append(tf.exp(-j_complex * k_val_c * tf.complex(d_ue, tf.constant(0.0, dtype=tf.float32))))
            phase_sp_list.append(tf.exp(-j_complex * k_val_c * tf.complex(d_sp, tf.constant(0.0, dtype=tf.float32))))
        a_los = tf.stack(a_los_list, axis=0)        # [K,N]
        a_sp = tf.stack(a_sp_list, axis=0)          # [K,N]
        phase_los = tf.stack(phase_los_list, axis=0)  # [K]
        phase_sp = tf.stack(phase_sp_list, axis=0)    # [K]
        k_vec = tf.stack(k_list, axis=0)              # [K]
        k_vec_c = tf.complex(k_vec, tf.zeros_like(k_vec))

        s_los = a_los * tf.expand_dims(phase_los, 1)  # [K,N]
        s_sp = a_sp * tf.expand_dims(phase_sp, 1)     # [K,N]

        g_los = _tf_alpha_fun(d_ue)
        g_sp = _tf_alpha_fun(d_sum)
        g_los_c = tf.complex(g_los, tf.constant(0.0, dtype=tf.float32))
        g_sp_c = tf.complex(g_sp, tf.constant(0.0, dtype=tf.float32))

        # Derivatives
        ds_los_ddue = (-j_complex * tf.expand_dims(k_vec_c, 1)) * s_los
        factor_ue = j_complex * tf.expand_dims(tf.complex(k_vec * d_spacing * tf.cos(theta_ue), tf.zeros_like(k_vec)), 1) * tf.complex(i1, tf.zeros_like(i1))
        ds_los_dtheta = s_los * factor_ue

        ds_sp_dphase = (-j_complex * tf.expand_dims(k_vec_c, 1)) * s_sp
        factor_sp = j_complex * tf.expand_dims(tf.complex(k_vec * d_spacing * tf.cos(theta_sp), tf.zeros_like(k_vec)), 1) * tf.complex(i1, tf.zeros_like(i1))
        ds_sp_dthetasp = s_sp * factor_sp

        # Phase distance derivative (RIS-only): ddphase/d(d_sp)=1; others 0
        ddphase_ddue_c = tf.complex(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        ddphase_dthetaue_c = tf.complex(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        ddphase_ddsp_c = tf.complex(tf.constant(1.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))
        ddphase_dthetasp_c = tf.complex(tf.constant(0.0, dtype=tf.float32), tf.constant(0.0, dtype=tf.float32))

        # Accumulate full J across t,f correctly: sum of per-(t,k) outer products
        J_sum = tf.zeros([8, 8], dtype=tf.float32)
        for t_idx in range(T):
            w_t = theta_seq[0, t_idx, :]  # [N]
            conj_w = tf.conj(w_t)

            def vdot(vec):
                return tf.reduce_sum(conj_w * vec)

            for k_idx in range(K):
                # Per-frequency slices
                s_los_k = s_los[k_idx, :]
                s_sp_k = s_sp[k_idx, :]
                ds_los_ddue_k = ds_los_ddue[k_idx, :]
                ds_los_dtheta_k = ds_los_dtheta[k_idx, :]
                ds_sp_dphase_k = ds_sp_dphase[k_idx, :]
                ds_sp_dthetasp_k = ds_sp_dthetasp[k_idx, :]

                # Interest gradients per (t,k)
                grad_due_k = vdot(g_los_c * ds_los_ddue_k) + vdot(g_sp_c * ds_sp_dphase_k) * ddphase_ddue_c
                grad_thetaue_k = vdot(g_los_c * ds_los_dtheta_k) + vdot(g_sp_c * ds_sp_dphase_k) * ddphase_dthetaue_c
                grad_dsp_k = vdot(g_sp_c * ds_sp_dphase_k) * ddphase_ddsp_c
                grad_thetasp_k = vdot(g_sp_c * (ds_sp_dphase_k * ddphase_dthetasp_c + ds_sp_dthetasp_k))

                # Gain gradients per (t,k)
                S_los_k = vdot(s_los_k)
                S_sp_k = vdot(s_sp_k)
                grad_gain_k = tf.stack([S_los_k, j_complex * S_los_k, S_sp_k, j_complex * S_sp_k])

                grad_interest_k = tf.stack([grad_due_k, grad_thetaue_k, grad_dsp_k, grad_thetasp_k])
                grad_full_k = tf.concat([grad_interest_k, grad_gain_k], axis=0)

                real_outer_k = tf.real(tf.tensordot(grad_full_k, tf.conj(grad_full_k), axes=0))
                J_sum = J_sum + real_outer_k

        # Scale by 2*SNR
        J_full = (2.0 * snr_tx_lin) * J_sum
        Jpp = J_full[0:4, 0:4]
        Jpq = J_full[0:4, 4:8]
        Jqq = J_full[4:8, 4:8]
        # Use pseudo-inverse with small jitter for numerical robustness
        # Strict inverse with jitter for nuisance block (fallback if near-singular)
        Jqq_inv = conditional_inv_2d(Jqq)
        J_eff = Jpp - tf.matmul(Jpq, tf.matmul(Jqq_inv, tf.transpose(Jpq)))
        # Effective Fisher block: symmetric + small jitter, strict inverse
        J_eff_sym = 0.5 * (J_eff + tf.transpose(J_eff))
        Finv = conditional_inv_2d(J_eff_sym)
        F11 = 0.5 * (Finv[0:2, 0:2] + tf.transpose(Finv[0:2, 0:2]))
        F22 = 0.5 * (Finv[2:4, 2:4] + tf.transpose(Finv[2:4, 2:4]))

        # Polar→Cartesian mapping
        J_ue = tf.stack([
            tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=0),
            tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=0)
        ], axis=0)
        J_sp = tf.stack([
            tf.stack([tf.cos(theta_sp), -d_sp * tf.sin(theta_sp)], axis=0),
            tf.stack([tf.sin(theta_sp),  d_sp * tf.cos(theta_sp)], axis=0)
        ], axis=0)
        Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(F11, tf.transpose(J_ue)))
        Sigma_xy_sp = tf.matmul(J_sp, tf.matmul(F22, tf.transpose(J_sp)))
        peb_ue = tf.sqrt(tf.linalg.trace(Sigma_xy_ue))
        peb_sp = tf.sqrt(tf.linalg.trace(Sigma_xy_sp))
        return tf.square(peb_ue) + tf.square(peb_sp)

    elems = loc_input[:, 0:2, :]
    pebs = tf.map_fn(per_sample, elems, dtype=tf.float32, parallel_iterations=16)
    return tf.reduce_mean(pebs)


def tf_crb_loss_eval_style_tx(theta_list, loc_input, P_lin):
    """Convenience wrapper: use linear transmit power (tx SNR) directly.

    Maps P_lin to snr_db = 10*log10(P_lin) and calls tf_crb_loss_eval_style.
    """
    P_tf = tf.convert_to_tensor(P_lin, dtype=tf.float32)
    snr_db_tf = tf.constant(10.0, dtype=tf.float32) * tf.log(P_tf) / tf.log(tf.constant(10.0, dtype=tf.float32))
    return tf_crb_loss_eval_style(theta_list, loc_input, snr_db_tf)


# === Fast, vectorized variant (GPU friendly) ===
def tf_crb_loss_eval_style_fast(theta_list, loc_input, snr_db):
    """Vectorized replica of calculate_crb nuisance_gain=True (tx SNR).

    Eliminates Python loops and tf.map_fn; computes over batch, T, K in one shot.
    """
    T = len(theta_list)
    K = NUM_SUBCARRIERS

    # Constants
    ris_pos = tf.constant(RIS_POS, dtype=tf.float32)
    i1 = tf.constant(np.arange(N_RIS), dtype=tf.float32)
    freqs = tf.constant(SUBCARRIER_FREQS, dtype=tf.float32)
    two_pi = tf.constant(2.0 * np.pi, dtype=tf.float32)
    c_tf = tf.constant(C_LIGHT, dtype=tf.float32)
    lam_center = tf.constant(C_LIGHT / CARRIER_FREQ, dtype=tf.float32)
    d_spacing = lam_center / 2.0
    j = tf.complex(tf.constant(0.0, tf.float32), tf.constant(1.0, tf.float32))

    snr_db_tf = tf.convert_to_tensor(snr_db, dtype=tf.float32)
    snr_tx_lin = tf.pow(tf.constant(10.0, dtype=tf.float32), snr_db_tf / tf.constant(10.0, dtype=tf.float32))

    # Stack beams to [T,N] and conj
    W = tf.stack([t[0, :] for t in theta_list], axis=0)  # [T,N]
    W_conj = tf.conj(W)                                  # [T,N]

    # Batch geometry
    loc_ue = loc_input[:, 0, :]  # [B,3]
    loc_sp = loc_input[:, 1, :]  # [B,3]
    diff_ue = loc_ue - ris_pos   # [B,3]
    diff_sp = loc_sp - ris_pos   # [B,3]
    d_ue = tf.sqrt(tf.reduce_sum(tf.square(diff_ue), axis=1) + 1e-12)  # [B]
    d_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_sp), axis=1) + 1e-12)  # [B]
    diff_ue_sp = loc_ue - loc_sp
    d_ue_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_ue_sp), axis=1) + 1e-12)  # [B]
    d_sum = d_sp + d_ue_sp  # [B]
    theta_ue = tf.atan2(diff_ue[:, 1], diff_ue[:, 0])
    theta_sp = tf.atan2(diff_sp[:, 1], diff_sp[:, 0])

    # Steering and phase per frequency (broadcast to [B,K,N])
    k_vec = (two_pi * freqs) / c_tf  # [K]
    sin_ue = tf.sin(theta_ue)        # [B]
    sin_sp = tf.sin(theta_sp)
    cos_ue = tf.cos(theta_ue)
    cos_sp = tf.cos(theta_sp)

    k_expanded = tf.reshape(k_vec, [1, K, 1])                 # [1,K,1]
    i1_expanded = tf.reshape(i1, [1, 1, N_RIS])               # [1,1,N]
    sin_ue_e = tf.reshape(sin_ue, [-1, 1, 1])                 # [B,1,1]
    sin_sp_e = tf.reshape(sin_sp, [-1, 1, 1])
    cos_ue_e = tf.reshape(cos_ue, [-1, 1, 1])
    cos_sp_e = tf.reshape(cos_sp, [-1, 1, 1])
    due_e = tf.reshape(d_ue, [-1, 1])                         # [B,1]
    dsp_e = tf.reshape(d_sp, [-1, 1])

    # Use complex k to avoid dtype mismatch in multiplications
    k_c = tf.complex(k_expanded, tf.zeros_like(k_expanded))
    elem_phase_ue = j * k_c * tf.complex(d_spacing * i1_expanded * sin_ue_e, tf.zeros_like(sin_ue_e))
    elem_phase_sp = j * k_c * tf.complex(d_spacing * i1_expanded * sin_sp_e, tf.zeros_like(sin_sp_e))
    a_los = tf.exp(elem_phase_ue)  # [B,K,N]
    a_sp = tf.exp(elem_phase_sp)   # [B,K,N]

    phase_los = tf.exp(-j * tf.complex(k_expanded[:, :, 0], tf.zeros_like(k_expanded[:, :, 0])) * tf.complex(due_e, tf.zeros_like(due_e)))  # [B,K]
    phase_sp = tf.exp(-j * tf.complex(k_expanded[:, :, 0], tf.zeros_like(k_expanded[:, :, 0])) * tf.complex(dsp_e, tf.zeros_like(dsp_e)))   # [B,K]
    s_los = a_los * tf.expand_dims(phase_los, axis=2)  # [B,K,N]
    s_sp = a_sp * tf.expand_dims(phase_sp, axis=2)     # [B,K,N]

    # Gains
    g_los = _tf_alpha_fun(d_ue)
    g_sp = _tf_alpha_fun(d_sum)
    g_los_c = tf.complex(g_los, tf.constant(0.0, tf.float32))  # [B]
    g_sp_c = tf.complex(g_sp, tf.constant(0.0, tf.float32))

    # Derivatives
    # Derivatives use complex k as well
    ds_los_ddue = (-j * k_c) * s_los
    ds_sp_dphase = (-j * k_c) * s_sp
    factor_ue = j * tf.complex(k_expanded * d_spacing * i1_expanded * cos_ue_e, tf.zeros_like(cos_ue_e))
    factor_sp = j * tf.complex(k_expanded * d_spacing * i1_expanded * cos_sp_e, tf.zeros_like(cos_sp_e))
    ds_los_dtheta = s_los * factor_ue
    ds_sp_dthetasp = s_sp * factor_sp

    # Dot with beams across N -> [B,K,T]
    S_los = tf.einsum('tn,bkn->bkt', W_conj, s_los)
    S_sp = tf.einsum('tn,bkn->bkt', W_conj, s_sp)
    grad_due = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_los_c, 1)[:, :, None] * ds_los_ddue)
    grad_thetaue = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_los_c, 1)[:, :, None] * ds_los_dtheta)
    grad_dsp = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_sp_c, 1)[:, :, None] * ds_sp_dphase)
    grad_thetasp = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_sp_c, 1)[:, :, None] * ds_sp_dthetasp)

    grad_interest = tf.stack([grad_due, grad_thetaue, grad_dsp, grad_thetasp], axis=-1)  # [B,K,T,4]
    grad_gain = tf.stack([S_los, j * S_los, S_sp, j * S_sp], axis=-1)                     # [B,K,T,4]
    grad_full = tf.concat([grad_interest, grad_gain], axis=-1)                             # [B,K,T,8]

    # Sum real outer products over (K,T)
    J_sum = tf.real(tf.einsum('bktp,bktq->bpq', grad_full, tf.conj(grad_full)))            # [B,8,8]
    J_full = (2.0 * snr_tx_lin) * J_sum
    Jpp = J_full[:, 0:4, 0:4]
    Jpq = J_full[:, 0:4, 4:8]
    Jqq = J_full[:, 4:8, 4:8]
    # Strict inverse with jitter for nuisance block (fallback if near-singular)
    
    Jqq_inv = robust_inv_spd_2x2_batch(Jqq)
    J_eff = Jpp - tf.matmul(Jpq, tf.matmul(Jqq_inv, tf.transpose(Jpq, perm=[0, 2, 1])))
    # Effective Fisher block: symmetric + small jitter, strict inverse
    J_eff_sym = 0.5 * (J_eff + tf.transpose(J_eff, perm=[0, 2, 1]))
    Finv = conditional_inv_3d(J_eff_sym)
    Finv = tf.where(tf.math.is_finite(Finv), Finv, tf.zeros_like(Finv))
    F11 = 0.5 * (Finv[:, 0:2, 0:2] + tf.transpose(Finv[:, 0:2, 0:2], perm=[0, 2, 1]))
    F22 = 0.5 * (Finv[:, 2:4, 2:4] + tf.transpose(Finv[:, 2:4, 2:4], perm=[0, 2, 1]))

    # Polar→Cartesian mapping per batch
    J_ue = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)  # [B,2,2]
    J_sp = tf.stack([
        tf.stack([tf.cos(theta_sp), -d_sp * tf.sin(theta_sp)], axis=1),
        tf.stack([tf.sin(theta_sp),  d_sp * tf.cos(theta_sp)], axis=1)
    ], axis=1)  # [B,2,2]

    Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(F11, tf.transpose(J_ue, perm=[0, 2, 1])))
    Sigma_xy_sp = tf.matmul(J_sp, tf.matmul(F22, tf.transpose(J_sp, perm=[0, 2, 1])))
    trace_ue = tf.linalg.trace(Sigma_xy_ue)
    trace_sp = tf.linalg.trace(Sigma_xy_sp)
    peb_ue = tf.sqrt(tf.maximum(trace_ue, tf.constant(EPS_TRACE_FLOOR, tf.float32)))  # [B]
    peb_sp = tf.sqrt(tf.maximum(trace_sp, tf.constant(EPS_TRACE_FLOOR, tf.float32)))  # [B]
    return tf.reduce_mean(tf.square(peb_ue) + tf.square(peb_sp))


def tf_crb_loss_eval_style_tx_fast(theta_list, loc_input, P_lin):
    """Fast convenience wrapper using linear transmit power (tx SNR)."""
    P_tf = tf.convert_to_tensor(P_lin, dtype=tf.float32)
    snr_db_tf = tf.constant(10.0, dtype=tf.float32) * tf.log(P_tf) / tf.log(tf.constant(10.0, dtype=tf.float32))
    return tf_crb_loss_eval_style_fast(theta_list, loc_input, snr_db_tf)


def tf_crb_loss_eval_style_fast_ue(theta_list, loc_input, snr_db):
    """Vectorized CRB loss returning only UE component (batch-mean of PEB_UE^2)."""
    T = len(theta_list)
    K = NUM_SUBCARRIERS
    ris_pos = tf.constant(RIS_POS, dtype=tf.float32)
    i1 = tf.constant(np.arange(N_RIS), dtype=tf.float32)
    freqs = tf.constant(SUBCARRIER_FREQS, dtype=tf.float32)
    two_pi = tf.constant(2.0 * np.pi, dtype=tf.float32)
    c_tf = tf.constant(C_LIGHT, dtype=tf.float32)
    lam_center = tf.constant(C_LIGHT / CARRIER_FREQ, dtype=tf.float32)
    d_spacing = lam_center / 2.0
    j = tf.complex(tf.constant(0.0, tf.float32), tf.constant(1.0, tf.float32))

    snr_db_tf = tf.convert_to_tensor(snr_db, dtype=tf.float32)
    snr_tx_lin = tf.pow(tf.constant(10.0, dtype=tf.float32), snr_db_tf / tf.constant(10.0, dtype=tf.float32))

    W = tf.stack([t[0, :] for t in theta_list], axis=0)
    W_conj = tf.conj(W)

    loc_ue = loc_input[:, 0, :]
    loc_sp = loc_input[:, 1, :]
    diff_ue = loc_ue - ris_pos
    diff_sp = loc_sp - ris_pos
    d_ue = tf.sqrt(tf.reduce_sum(tf.square(diff_ue), axis=1) + 1e-12)
    d_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_sp), axis=1) + 1e-12)
    diff_ue_sp = loc_ue - loc_sp
    d_ue_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_ue_sp), axis=1) + 1e-12)
    d_sum = d_sp + d_ue_sp
    theta_ue = tf.atan2(diff_ue[:, 1], diff_ue[:, 0])
    theta_sp = tf.atan2(diff_sp[:, 1], diff_sp[:, 0])

    k_vec = (two_pi * freqs) / c_tf
    k_expanded = tf.reshape(k_vec, [1, K, 1])
    i1_expanded = tf.reshape(i1, [1, 1, N_RIS])
    sin_ue_e = tf.reshape(tf.sin(theta_ue), [-1, 1, 1])
    sin_sp_e = tf.reshape(tf.sin(theta_sp), [-1, 1, 1])
    due_e = tf.reshape(d_ue, [-1, 1])
    dsp_e = tf.reshape(d_sp, [-1, 1])

    k_c = tf.complex(k_expanded, tf.zeros_like(k_expanded))
    elem_phase_ue = j * k_c * tf.complex(d_spacing * i1_expanded * sin_ue_e, tf.zeros_like(sin_ue_e))
    elem_phase_sp = j * k_c * tf.complex(d_spacing * i1_expanded * sin_sp_e, tf.zeros_like(sin_sp_e))
    a_los = tf.exp(elem_phase_ue)
    a_sp = tf.exp(elem_phase_sp)

    phase_los = tf.exp(-j * tf.complex(k_expanded[:, :, 0], tf.zeros_like(k_expanded[:, :, 0])) * tf.complex(due_e, tf.zeros_like(due_e)))
    phase_sp = tf.exp(-j * tf.complex(k_expanded[:, :, 0], tf.zeros_like(k_expanded[:, :, 0])) * tf.complex(dsp_e, tf.zeros_like(dsp_e)))
    s_los = a_los * tf.expand_dims(phase_los, axis=2)
    s_sp = a_sp * tf.expand_dims(phase_sp, axis=2)

    g_los = _tf_alpha_fun(d_ue)
    g_sp = _tf_alpha_fun(d_sum)
    g_los_c = tf.complex(g_los, tf.constant(0.0, tf.float32))
    g_sp_c = tf.complex(g_sp, tf.constant(0.0, tf.float32))

    ds_los_ddue = (-j * k_c) * s_los
    factor_ue = j * tf.complex(k_expanded * d_spacing * i1_expanded * tf.reshape(tf.cos(theta_ue), [-1, 1, 1]), tf.zeros_like(k_expanded))
    ds_los_dtheta = s_los * factor_ue

    S_los = tf.einsum('tn,bkn->bkt', W_conj, s_los)
    S_sp = tf.einsum('tn,bkn->bkt', W_conj, s_sp)
    grad_due = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_los_c, 1)[:, :, None] * ds_los_ddue)
    grad_thetaue = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_los_c, 1)[:, :, None] * ds_los_dtheta)
    grad_dsp = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_sp_c, 1)[:, :, None] * ((-j * k_c) * s_sp))
    grad_thetasp = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_sp_c, 1)[:, :, None] * (s_sp * j * tf.complex(k_expanded * d_spacing * i1_expanded * tf.reshape(tf.cos(theta_sp), [-1, 1, 1]), tf.zeros_like(k_expanded))))

    grad_interest = tf.stack([grad_due, grad_thetaue, grad_dsp, grad_thetasp], axis=-1)
    grad_gain = tf.stack([S_los, j * S_los, S_sp, j * S_sp], axis=-1)
    J_sum = tf.real(tf.einsum('bktp,bktq->bpq', tf.concat([grad_interest, grad_gain], axis=-1), tf.conj(tf.concat([grad_interest, grad_gain], axis=-1))))
    J_full = (2.0 * snr_tx_lin) * J_sum
    Jpp = J_full[:, 0:4, 0:4]
    Jpq = J_full[:, 0:4, 4:8]
    Jqq = J_full[:, 4:8, 4:8]
    # 单载波无映射同理：Jqq 为 2×2，使用批量 2×2 SPD 逆
    Jqq_inv = robust_inv_spd_2x2_batch(Jqq)
    J_eff = Jpp - tf.matmul(Jpq, tf.matmul(Jqq_inv, tf.transpose(Jpq, perm=[0, 2, 1])))
    J_eff_sym = 0.5 * (J_eff + tf.transpose(J_eff, perm=[0, 2, 1]))
    Finv = conditional_inv_3d(J_eff_sym)
    Finv = tf.where(tf.math.is_finite(Finv), Finv, tf.zeros_like(Finv))
    F11 = 0.5 * (Finv[:, 0:2, 0:2] + tf.transpose(Finv[:, 0:2, 0:2], perm=[0, 2, 1]))

    J_ue = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)
    Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(F11, tf.transpose(J_ue, perm=[0, 2, 1])))
    trace_ue = tf.linalg.trace(Sigma_xy_ue)
    peb_ue = tf.sqrt(tf.maximum(trace_ue, tf.constant(EPS_TRACE_FLOOR, tf.float32)))
    return tf.reduce_mean(tf.square(peb_ue))


def tf_crb_loss_eval_style_tx_fast_ue(theta_list, loc_input, P_lin):
    P_tf = tf.convert_to_tensor(P_lin, dtype=tf.float32)
    snr_db_tf = tf.constant(10.0, dtype=tf.float32) * tf.log(P_tf) / tf.log(tf.constant(10.0, dtype=tf.float32))
    return tf_crb_loss_eval_style_fast_ue(theta_list, loc_input, snr_db_tf)


# === Packaged CRB variants (nomap/map × K=10 and K=1) ===
# These wrappers reproduce compute_ue_crb’s four evaluation styles in TensorFlow
# without altering their definitions/calculation processes.

def _tf_interest_fim(theta_list, loc_input, snr_db, freqs_vec, include_gain):
    """Return batched interest FIM J_theta [B,4,4] and UE geometry.
    - include_gain=True: nuisance_gain semantics (gain treated as nuisance, eliminated).
    - include_gain=False: single-carrier, pathloss_known=True, no gain nuisance.
    """
    T = len(theta_list)
    K = int(freqs_vec.shape[0])

    # Constants
    ris_pos = tf.constant(RIS_POS, dtype=tf.float32)
    i1 = tf.constant(np.arange(N_RIS), dtype=tf.float32)
    two_pi = tf.constant(2.0 * np.pi, dtype=tf.float32)
    c_tf = tf.constant(C_LIGHT, dtype=tf.float32)
    lam_center = tf.constant(C_LIGHT / CARRIER_FREQ, dtype=tf.float32)
    d_spacing = lam_center / 2.0
    j = tf.complex(tf.constant(0.0, tf.float32), tf.constant(1.0, tf.float32))

    snr_db_tf = tf.convert_to_tensor(snr_db, dtype=tf.float32)
    snr_tx_lin = tf.pow(tf.constant(10.0, dtype=tf.float32), snr_db_tf / tf.constant(10.0, dtype=tf.float32))

    # Beams [T,N]
    W = tf.stack([t[0, :] for t in theta_list], axis=0)
    W_conj = tf.conj(W)

    # Geometry
    loc_ue = loc_input[:, 0, :]
    loc_sp = loc_input[:, 1, :]
    diff_ue = loc_ue - ris_pos
    diff_sp = loc_sp - ris_pos
    d_ue = tf.sqrt(tf.reduce_sum(tf.square(diff_ue), axis=1) + 1e-12)
    d_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_sp), axis=1) + 1e-12)
    diff_ue_sp = loc_ue - loc_sp
    d_ue_sp = tf.sqrt(tf.reduce_sum(tf.square(diff_ue_sp), axis=1) + 1e-12)
    d_sum = d_sp + d_ue_sp
    theta_ue = tf.atan2(diff_ue[:, 1], diff_ue[:, 0])
    theta_sp = tf.atan2(diff_sp[:, 1], diff_sp[:, 0])

    # Steering and phase per frequency (broadcast to [B,K,N])
    k_vec = (two_pi * freqs_vec) / c_tf  # [K]
    sin_ue = tf.sin(theta_ue)
    sin_sp = tf.sin(theta_sp)
    cos_ue = tf.cos(theta_ue)
    cos_sp = tf.cos(theta_sp)

    k_expanded = tf.reshape(k_vec, [1, K, 1])
    i1_expanded = tf.reshape(i1, [1, 1, N_RIS])
    sin_ue_e = tf.reshape(sin_ue, [-1, 1, 1])
    sin_sp_e = tf.reshape(sin_sp, [-1, 1, 1])
    cos_ue_e = tf.reshape(cos_ue, [-1, 1, 1])
    cos_sp_e = tf.reshape(cos_sp, [-1, 1, 1])
    due_e = tf.reshape(d_ue, [-1, 1])
    dsp_e = tf.reshape(d_sp, [-1, 1])

    k_c = tf.complex(k_expanded, tf.zeros_like(k_expanded))
    elem_phase_ue = j * k_c * tf.complex(d_spacing * i1_expanded * sin_ue_e, tf.zeros_like(sin_ue_e))
    elem_phase_sp = j * k_c * tf.complex(d_spacing * i1_expanded * sin_sp_e, tf.zeros_like(sin_sp_e))
    a_los = tf.exp(elem_phase_ue)
    a_sp = tf.exp(elem_phase_sp)

    phase_los = tf.exp(-j * tf.complex(k_expanded[:, :, 0], tf.zeros_like(k_expanded[:, :, 0])) * tf.complex(due_e, tf.zeros_like(due_e)))
    phase_sp = tf.exp(-j * tf.complex(k_expanded[:, :, 0], tf.zeros_like(k_expanded[:, :, 0])) * tf.complex(dsp_e, tf.zeros_like(dsp_e)))
    s_los = a_los * tf.expand_dims(phase_los, axis=2)
    s_sp = a_sp * tf.expand_dims(phase_sp, axis=2)

    # Gains (pathloss-known factors)
    g_los = _tf_alpha_fun(d_ue)
    g_sp = _tf_alpha_fun(d_sum)
    g_los_c = tf.complex(g_los, tf.constant(0.0, tf.float32))
    g_sp_c = tf.complex(g_sp, tf.constant(0.0, tf.float32))

    # Derivatives
    ds_los_ddue = (-j * k_c) * s_los
    ds_sp_dphase = (-j * k_c) * s_sp
    factor_ue = j * tf.complex(k_expanded * d_spacing * i1_expanded * cos_ue_e, tf.zeros_like(cos_ue_e))
    factor_sp = j * tf.complex(k_expanded * d_spacing * i1_expanded * cos_sp_e, tf.zeros_like(cos_sp_e))
    ds_los_dtheta = s_los * factor_ue
    ds_sp_dthetasp = s_sp * factor_sp

    # Dot with beams across N -> [B,K,T]
    S_los = tf.einsum('tn,bkn->bkt', W_conj, s_los)
    S_sp = tf.einsum('tn,bkn->bkt', W_conj, s_sp)
    grad_due = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_los_c, 1)[:, :, None] * ds_los_ddue)
    grad_thetaue = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_los_c, 1)[:, :, None] * ds_los_dtheta)
    grad_dsp = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_sp_c, 1)[:, :, None] * ds_sp_dphase)
    grad_thetasp = tf.einsum('tn,bkn->bkt', W_conj, tf.expand_dims(g_sp_c, 1)[:, :, None] * ds_sp_dthetasp)

    grad_interest = tf.stack([grad_due, grad_thetaue, grad_dsp, grad_thetasp], axis=-1)  # [B,K,T,4]

    if include_gain:
        # Treat complex gains as nuisance; eliminate them via block Schur complement
        grad_gain = tf.stack([S_los, j * S_los, S_sp, j * S_sp], axis=-1)                  # [B,K,T,4]
        grad_full = tf.concat([grad_interest, grad_gain], axis=-1)                          # [B,K,T,8]
        J_sum = tf.real(tf.einsum('bktp,bktq->bpq', grad_full, tf.conj(grad_full)))        # [B,8,8]
        J_full = (2.0 * snr_tx_lin) * J_sum
        Jpp = J_full[:, 0:4, 0:4]
        Jpq = J_full[:, 0:4, 4:8]
        Jqq = J_full[:, 4:8, 4:8]
        Jqq_inv = conditional_inv_3d(Jqq)
        J_theta = Jpp - tf.matmul(Jpq, tf.matmul(Jqq_inv, tf.transpose(Jpq, perm=[0, 2, 1])))  # [B,4,4]
    else:
        # No gain nuisance (single-carrier known pathloss); interest-only EFIM
        J_sum_interest = tf.real(tf.einsum('bktp,bktq->bpq', grad_interest, tf.conj(grad_interest)))  # [B,4,4]
        J_theta = (2.0 * snr_tx_lin) * J_sum_interest

    return J_theta, theta_ue, d_ue, theta_sp, d_sp


def _tf_map_R_theta(theta_ue, d_ue, theta_sp, d_sp, wall_n=None, wall_b=None, bs_xy=None, ris_xy=None):
    """Build batched R_theta [B,4,2] as in compute_ue_peb_map."""
    B = tf.shape(d_ue)[0]
    wall_n_np = np.array([0.0, 1.0], dtype=np.float32) if wall_n is None else np.asarray(wall_n, dtype=np.float32)
    wall_n_tf = tf.constant(wall_n_np / max(float(np.linalg.norm(wall_n_np)), 1e-12), dtype=tf.float32)
    bs_xy_np = RIS_POS[:2] if bs_xy is None else np.asarray(bs_xy, dtype=np.float32)
    ris_xy_np = RIS_POS[:2] if ris_xy is None else np.asarray(ris_xy, dtype=np.float32)
    bs_xy_tf = tf.constant(bs_xy_np, dtype=tf.float32)
    ris_xy_tf = tf.constant(ris_xy_np, dtype=tf.float32)

    # Reconstruct u, s in BS-relative coords
    u = tf.stack([d_ue * tf.cos(theta_ue), d_ue * tf.sin(theta_ue)], axis=1)  # [B,2]
    r = ris_xy_tf - bs_xy_tf  # [2]
    n = wall_n_tf  # [2]
    # Default wall_b to SP y in BS-relative coords if not provided
    s_rel = tf.stack([d_sp * tf.cos(theta_sp), d_sp * tf.sin(theta_sp)], axis=1)  # [B,2]
    if wall_b is None:
        b_tf = tf.reduce_sum(n * s_rel, axis=1)  # [B]
    else:
        b_tf = tf.fill([B], tf.constant(float(wall_b), dtype=tf.float32))

    # Mirror of RIS across wall: r' = r - 2 n (n^T r - b)
    nr = tf.reduce_sum(n * r)  # scalar
    rp = r - 2.0 * n * (nr - b_tf[:, None])  # [B,2]

    # tau(u) = (b - n^T u) / (n^T r' - n^T u)
    a = b_tf - tf.reduce_sum(n * u, axis=1)  # [B]
    c = tf.reduce_sum(n * rp, axis=1) - tf.reduce_sum(n * u, axis=1)  # [B]
    c_safe = tf.where(tf.abs(c) > tf.constant(1e-12, tf.float32), c, tf.sign(c) * tf.constant(1e-12, tf.float32))
    tau = a / c_safe  # [B]

    # s(u) = u + tau (r' - u)
    s_u = u + tau[:, None] * (rp - u)  # [B,2]

    # d tau / du = n (b - n^T r') / c^2
    bn_rp = b_tf - tf.reduce_sum(n * rp, axis=1)  # [B]
    dtau_du = n[None, :] * (bn_rp / tf.maximum(c * c, tf.constant(1e-12, tf.float32)))[:, None]  # [B,2]

    # ds/du = (1 - tau) I + (r' - u) (dtau_du)^T
    I2 = tf.eye(2, dtype=tf.float32)[None, :, :]  # [1,2,2]
    rp_minus_u = rp - u  # [B,2]
    outer = tf.einsum('bi,bj->bij', rp_minus_u, dtau_du)  # [B,2,2]
    ds_du = (1.0 - tau)[:, None, None] * I2 + outer  # [B,2,2]

    # T(θ,d)
    T = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)  # [B,2,2]

    # ∂d_SP/∂u = s^T / ||s|| ; ∂θ_SP/∂u = [-s_y/||s||^2, s_x/||s||^2]
    s_norm = tf.maximum(tf.norm(s_u, axis=1), tf.constant(1e-12, tf.float32))  # [B]
    ddsp_du = s_u / s_norm[:, None]  # [B,2]
    dthetasp_du = tf.stack([-s_u[:, 1] / (s_norm ** 2), s_u[:, 0] / (s_norm ** 2)], axis=1)  # [B,2]

    # Chain to (d_L, θ_UE)
    ddsp_d = tf.einsum('bi,bij,bjk->bk', ddsp_du, ds_du, T)         # [B,2]
    dthetasp_d = tf.einsum('bi,bij,bjk->bk', dthetasp_du, ds_du, T)  # [B,2]

    # R_theta: [B,4,2]， [d_L, θ_UE, d_SP, θ_SP] to (θ_UE, d_UE) jacobian
    e0 = tf.stack([tf.ones([B], tf.float32), tf.zeros([B], tf.float32)], axis=1)  # ∂(d_L,θ_UE)/∂(θ_UE,d_UE)  d_L 
    e1 = tf.stack([tf.zeros([B], tf.float32), tf.ones([B], tf.float32)], axis=1)  # ∂(d_L,θ_UE)/∂(θ_UE,d_UE) θ_UE 
    R = tf.stack([e0, e1, ddsp_d, dthetasp_d], axis=1)  # [B,4,2]

    return R


def tf_crb_ue_peb_nomap_multi(theta_list, loc_input, snr_db):
    """UE-only PEB (nomap), multi-carrier K=10, nuisance_gain=True."""
    freqs_vec = tf.constant(SUBCARRIER_FREQS, dtype=tf.float32)
    J_theta, theta_ue, d_ue, theta_sp, d_sp = _tf_interest_fim(theta_list, loc_input, snr_db, freqs_vec, include_gain=True)
    # Eliminate SP as nuisance from interest EFIM
    Jpp = J_theta[:, 0:2, 0:2]
    Jpq = J_theta[:, 0:2, 2:4]
    Jqq = J_theta[:, 2:4, 2:4]
    # 
    Jqq_inv = robust_inv_spd_2x2_batch(Jqq)
    J_eff_ue = Jpp - tf.matmul(Jpq, tf.matmul(Jqq_inv, tf.transpose(Jpq, perm=[0, 2, 1])))  # [B,2,2]
    # Invert to covariance in polar, map to Cartesian, return mean PEB^2
    cov_polar = robust_inv_spd_2x2_batch(J_eff_ue)
    J_ue = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)
    Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(cov_polar, tf.transpose(J_ue, perm=[0, 2, 1])))
    trace_ue = tf.linalg.trace(Sigma_xy_ue)
    peb_ue = tf.sqrt(tf.maximum(trace_ue, tf.constant(EPS_TRACE_FLOOR, tf.float32)))
    return tf.reduce_mean(tf.square(peb_ue))


def tf_crb_ue_peb_map_multi(theta_list, loc_input, snr_db, wall_n=None, wall_b=None, bs_xy=None, ris_xy=None):
    """UE-only PEB (map-aided), multi-carrier K=10, nuisance_gain=True."""
    freqs_vec = tf.constant(SUBCARRIER_FREQS, dtype=tf.float32)
    J_theta, theta_ue, d_ue, theta_sp, d_sp = _tf_interest_fim(theta_list, loc_input, snr_db, freqs_vec, include_gain=True)
    R = _tf_map_R_theta(theta_ue, d_ue, theta_sp, d_sp, wall_n=wall_n, wall_b=wall_b, bs_xy=bs_xy, ris_xy=ris_xy)
    J_eff_map = tf.matmul(tf.transpose(R, perm=[0, 2, 1]), tf.matmul(J_theta, R))  # [B,2,2]
    cov_polar = robust_inv_spd_2x2_batch(J_eff_map)
    J_ue = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)
    Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(cov_polar, tf.transpose(J_ue, perm=[0, 2, 1])))
    trace_ue = tf.linalg.trace(Sigma_xy_ue)
    peb_ue = tf.sqrt(tf.maximum(trace_ue, tf.constant(EPS_TRACE_FLOOR, tf.float32)))
    return tf.reduce_mean(tf.square(peb_ue))


def tf_crb_ue_peb_nomap_single(theta_list, loc_input, snr_db):
    """UE-only PEB (nomap), single-carrier K=1, pathloss_known=True, nuisance_gain=False."""
    freqs_vec = tf.constant([CARRIER_FREQ], dtype=tf.float32)
    J_theta, theta_ue, d_ue, theta_sp, d_sp = _tf_interest_fim(theta_list, loc_input, snr_db, freqs_vec, include_gain=False)
    Jpp = J_theta[:, 0:2, 0:2]
    Jpq = J_theta[:, 0:2, 2:4]
    Jqq = J_theta[:, 2:4, 2:4]
    # 
    Jqq_inv = robust_inv_spd_2x2_batch(Jqq)
    J_eff_ue = Jpp - tf.matmul(Jpq, tf.matmul(Jqq_inv, tf.transpose(Jpq, perm=[0, 2, 1])))
    cov_polar = robust_inv_spd_2x2_batch(J_eff_ue)
    J_ue = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)
    Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(cov_polar, tf.transpose(J_ue, perm=[0, 2, 1])))
    trace_ue = tf.linalg.trace(Sigma_xy_ue)
    peb_ue = tf.sqrt(tf.maximum(trace_ue, tf.constant(EPS_TRACE_FLOOR, tf.float32)))
    return tf.reduce_mean(tf.square(peb_ue))


def tf_crb_ue_peb_map_single(theta_list, loc_input, snr_db, wall_n=None, wall_b=None, bs_xy=None, ris_xy=None):
    """UE-only PEB (map-aided), single-carrier K=1, pathloss_known=True, nuisance_gain=False."""
    freqs_vec = tf.constant([CARRIER_FREQ], dtype=tf.float32)
    J_theta, theta_ue, d_ue, theta_sp, d_sp = _tf_interest_fim(theta_list, loc_input, snr_db, freqs_vec, include_gain=False)
    R = _tf_map_R_theta(theta_ue, d_ue, theta_sp, d_sp, wall_n=wall_n, wall_b=wall_b, bs_xy=bs_xy, ris_xy=ris_xy)
    J_eff_map = tf.matmul(tf.transpose(R, perm=[0, 2, 1]), tf.matmul(J_theta, R))
    cov_polar = robust_inv_spd_2x2_batch(J_eff_map)
    J_ue = tf.stack([
        tf.stack([tf.cos(theta_ue), -d_ue * tf.sin(theta_ue)], axis=1),
        tf.stack([tf.sin(theta_ue),  d_ue * tf.cos(theta_ue)], axis=1)
    ], axis=1)
    Sigma_xy_ue = tf.matmul(J_ue, tf.matmul(cov_polar, tf.transpose(J_ue, perm=[0, 2, 1])))
    trace_ue = tf.linalg.trace(Sigma_xy_ue)
    peb_ue = tf.sqrt(tf.maximum(trace_ue, tf.constant(EPS_TRACE_FLOOR, tf.float32)))
    return tf.reduce_mean(tf.square(peb_ue))
