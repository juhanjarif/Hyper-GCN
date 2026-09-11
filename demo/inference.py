import sys
import os
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

MODEL_VARIANT = 'large'

WEIGHTS_FOLDER = {
    'base': 'weights',
    'large': 'weights_large',
}

if MODEL_VARIANT == 'large':
    from model.hypergcn_large import Model
else:
    from model.hypergcn_base import Model

from demo.skeleton_data import NTU60_CLASSES
from feeders.bone_pairs import ntu_pairs

def load_model(weights_path: str, device: str = 'cpu') -> Model:
    model = Model(
        num_class=60,
        num_point=25,
        num_person=2,
        hyper_joints=3,
        graph='graph.ntu_rgb_d.Graph',
        graph_args={'labeling_mode': 'virtual_ensemble'}
    )

    weights = torch.load(weights_path, map_location=device)

    from collections import OrderedDict
    clean_weights = OrderedDict()
    for k, v in weights.items():
        name = k.replace('module.', '')
        clean_weights[name] = v

    model.load_state_dict(clean_weights, strict=False)
    model.to(device)
    model.eval()
    print(f"[inference.py] Loaded {os.path.basename(weights_path)}")
    return model

def load_ensemble_models(weights_dir: str = None, device: str = 'cpu'):
    if weights_dir is None:
        project_root = os.path.join(os.path.dirname(__file__), '..')
        weights_dir = os.path.join(project_root, WEIGHTS_FOLDER.get(MODEL_VARIANT, 'weights'))

    models = {}
    if not os.path.exists(weights_dir):
        print(f"Warning: Directory {weights_dir} does not exist!")
        return models

    for fname in os.listdir(weights_dir):
        if fname.endswith('.pt'):
            key = fname.replace('.pt', '')
            path = os.path.join(weights_dir, fname)
            try:
                models[key] = load_model(path, device=device)
            except Exception as e:
                print(f"Error loading {fname}: {e}")

    print(f"Ensemble ready with {len(models)} models: {list(models.keys())}")
    return models

def preprocess_joint(skeleton_seq: np.ndarray) -> torch.Tensor:
    T, V, C = skeleton_seq.shape
    target_T = 64

    # 0. Zero out Z axis — monocular camera depth (from phones/webcams) is unreliable
    #    and causes the model to see actions as "reaching forward" (Writing/Reading)
    skeleton_seq[:, :, 2] = 0.0

    # 1. NTU Spine Length Normalization (Scale Spine Base -> Chest = 1.0)
    for f in range(T):
        spine_base = skeleton_seq[f, 0, :]
        chest = skeleton_seq[f, 20, :]
        dist = np.linalg.norm(spine_base - chest)
        if dist > 1e-4:
            origin = skeleton_seq[f, 1, :].copy() # Mid-spine
            skeleton_seq[f] = (skeleton_seq[f] - origin) / dist + origin

    # 2. Center on mid-spine (joint 1) at frame 0
    center = skeleton_seq[0, 1, :].copy()
    skeleton_seq = skeleton_seq - center

    # 3. Linear interpolation to 64 frames
    if T != target_T:
        indices = np.linspace(0, T - 1, target_T)
        skeleton_resampled = np.zeros((target_T, V, C), dtype=np.float32)

        for v in range(V):
            for c in range(C):
                skeleton_resampled[:, v, c] = np.interp(
                    indices, np.arange(T), skeleton_seq[:, v, c]
                )
        skeleton_seq = skeleton_resampled

    # 4. Format tensor (1, C, T, V, 2)
    seq = skeleton_seq.transpose(2, 0, 1) # (3, 64, 25)
    seq = seq[np.newaxis, :, :, :, np.newaxis] # (1, 3, 64, 25, 1)

    second_person = np.zeros_like(seq)
    seq = np.concatenate([seq, second_person], axis=4) # (1, 3, 64, 25, 2)

    # 5. NTU Chest Centering (Joint 20)
    data_numpy = seq[0] # (C, T, V, M)
    trajectory = data_numpy[:, :, 20].copy()
    data_numpy = data_numpy - data_numpy[:, :, 20:21]
    data_numpy[:, :, 20] = trajectory
    seq[0] = data_numpy

    return torch.tensor(seq, dtype=torch.float32)

def convert_to_bone(x_joint_tensor: torch.Tensor) -> torch.Tensor:
    data_numpy = x_joint_tensor.numpy()[0] # (C, T, V, M)
    bone_data = np.zeros_like(data_numpy)

    for v1, v2 in ntu_pairs:
        bone_data[:, :, v1 - 1] = data_numpy[:, :, v1 - 1] - data_numpy[:, :, v2 - 1]
    bone_data[:, :, 20] = data_numpy[:, :, 20]

    return torch.tensor(bone_data[np.newaxis, ...], dtype=torch.float32)

def convert_to_motion(x_tensor: torch.Tensor) -> torch.Tensor:
    motion = torch.zeros_like(x_tensor)
    motion[:, :, :-1, :, :] = x_tensor[:, :, 1:, :, :] - x_tensor[:, :, :-1, :, :]
    motion[:, :, -1, :, :] = 0
    return motion

def predict_ensemble(models: dict, skeleton_seq: np.ndarray, top_k: int = 5, device: str = 'cpu'):
    x_joint = preprocess_joint(skeleton_seq)
    x_bone = convert_to_bone(x_joint)
    x_joint_motion = convert_to_motion(x_joint)
    x_bone_motion = convert_to_motion(x_bone)

    tensors = {
        'joint': x_joint.to(device),
        'bone': x_bone.to(device),
        'joint_motion': x_joint_motion.to(device),
        'bone_motion': x_bone_motion.to(device),
    }

    total_probs = None
    count = 0

    for key, model in models.items():
        if 'joint_motion' in key:
            t = tensors['joint_motion']
            weight = 1.0
        elif 'bone_motion' in key:
            t = tensors['bone_motion']
            weight = 1.0
        elif 'bone' in key:
            t = tensors['bone']
            weight = 3.0
        else:
            t = tensors['joint']
            weight = 3.0

        with torch.no_grad():
            out, _ = model(t)
            probs = torch.softmax(out, dim=1)[0]
            total_probs = probs * weight if total_probs is None else total_probs + probs * weight
            count += weight

    if total_probs is None:
        raise ValueError("No models loaded for prediction!")

    avg_probs = total_probs / count
    top_vals, top_idxs = torch.topk(avg_probs, top_k)

    results = []
    for val, idx in zip(top_vals.cpu().numpy(), top_idxs.cpu().numpy()):
        results.append((NTU60_CLASSES[idx], float(val) * 100))

    return results