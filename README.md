# Map-aided Active Sensing

This repository contains the `rnn_server-50rate.py` script, which implements a map-aided approach for active sensing in joint snensing and communication. It optimizes beamforming weights to minimize CRB while considering communication performance.

## Description

The main script `rnn_server-50rate.py` trains a Recurrent Neural Network (RNN/LSTM) to design the active sensing strategy. It supports:
- **Multi-user, Multi-carrier systems**: Configurable number of subcarriers and users.
- **CRB Optimization**: Minimizes the Cramer-Rao Bound for positioning accuracy.
- **Communication Constraints**: Options to include communication rate/power in the loss function.
- **Flexible Geometry**: Configurable TX and BS locations, and user distribution generation.

## Requirements

The code is compatible with **Python 3** and requires the following libraries:

- `tensorflow` (v1.x or v2.x with `compat.v1`)
- `numpy`
- `scipy`
- `keras`

## Usage

1.  Ensure all dependencies are installed.
2.  Run the script directly:

    ```bash
    python rnn_server-50rate.py
    ```

3.  Output logs are saved in the `logs/` directory. Models are saved in `models_newcrb/`.

## Configuration

You can modify key parameters at the top of `rnn_server-50rate.py`:

- **CRB Settings**:
    - `USE_CRB_LOSS`: Enable/disable CRB minimization.
    - `CRB_METHOD`: Choose the CRB calculation method (e.g., `'map_multi'`, `'nomap_multi'`).
    - `CRB_WEIGHT`: Weight of the CRB term in the loss function.

- **Communication Settings**:
    - `USE_COM1`: Enable power/rate optimization for the last slot.
    - `COM1_WEIGHT`: Weight of the communication term.

- **System Parameters**:
    - `N_ris`: Number of RIS elements.
    - `num_users`: Number of users.
    - `tau`: Pilot length.

- **Training**:
    - `n_epochs`: Number of training epochs.
    - `learning_rate`: Learning rate.
    - `OVERFIT_FIXED_GEOM`: Enable for debugging with fixed geometry.

## File Structure

- `rnn_server-50rate.py`: Main training script.
- `tf_crb_eval_style.py`: Contains TensorFlow implementations of CRB loss functions.
- `logs/`: Directory where execution logs are stored.

  
## Acknowledgments

Part of the code is derived from [Active-sensing-for-localization](https://github.com/dzhang0/Active-sensing-for-localization).
