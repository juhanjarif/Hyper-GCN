import numpy as np

NEUTRAL_POSE = np.array([
    [ 0.00,  0.00,  0.00],  # 0  base spine
    [ 0.00,  0.15,  0.00],  # 1  mid spine
    [ 0.00,  0.35,  0.00],  # 2  neck
    [ 0.00,  0.48,  0.00],  # 3  head
    [-0.20,  0.32,  0.00],  # 4  left shoulder
    [-0.38,  0.18,  0.00],  # 5  left elbow
    [-0.48,  0.02,  0.00],  # 6  left wrist
    [-0.52, -0.05,  0.00],  # 7  left hand
    [ 0.20,  0.32,  0.00],  # 8  right shoulder
    [ 0.38,  0.18,  0.00],  # 9  right elbow
    [ 0.48,  0.02,  0.00],  # 10 right wrist
    [ 0.52, -0.05,  0.00],  # 11 right hand
    [-0.10, -0.10,  0.00],  # 12 left hip
    [-0.10, -0.38,  0.00],  # 13 left knee
    [-0.10, -0.62,  0.00],  # 14 left ankle
    [-0.10, -0.70,  0.05],  # 15 left foot
    [ 0.10, -0.10,  0.00],  # 16 right hip
    [ 0.10, -0.38,  0.00],  # 17 right knee
    [ 0.10, -0.62,  0.00],  # 18 right ankle
    [ 0.10, -0.70,  0.05],  # 19 right foot
    [ 0.00,  0.25,  0.00],  # 20 chest (spine)
    [-0.56, -0.10,  0.00],  # 21 left hand tip
    [-0.52, -0.07,  0.02],  # 22 left thumb
    [ 0.56, -0.10,  0.00],  # 23 right hand tip
    [ 0.52, -0.07,  0.02],  # 24 right thumb
], dtype=np.float32);

# Return T copies of the neutral pose as starting point
def make_sequence(T=64):
    return np.tile(NEUTRAL_POSE[np.newaxis], (T, 1, 1)).copy()

# Clapping: both hands move toward each other and back, 3 times
def action_clapping(T=64):
    seq = make_sequence(T)

    for t in range(T):
        phase = np.sin(2 * np.pi * t / T * 3)
        offset = 0.25 * phase # max 25cm toward center

        # LEFT arm moving RIGHT (negative x becomes more negative)
        for j in [5, 6, 7, 21, 22]:
            seq[t, j, 0] += offset

        for k in [9, 10, 11, 23, 24]:
            seq[t, k, 0] -= offset

    return seq

# Drinking water: right hand raises from side to face level, pauses, lowers back
def action_drinking_water(T=64):
    seq = make_sequence(T)

    for t in range(T):
        alpha = t / T
        if alpha < 0.4:
            lift = alpha / 0.4              # rising phase
        elif alpha < 0.7:
            lift = 1.0                      # holding at face
        else:
            lift = (1.0 - alpha) / 0.3      # lowering phase

        lift_y = lift * 0.42 # max lift in metres

        for j in [9, 10, 11, 23, 24]:
            seq[t, j, 1] += lift_y      # move up in Y
            seq[t, j, 2] += lift * 0.10 # slight move forward

    return seq

# Jump up: whole body lifts, arms raise, then lands
def action_jumping(T=64):
    seq = make_sequence(T)

    for t in range(T):
        alpha = t / T

        # Parabolic jump arc: rises for first half, falls for second half
        jump_y = 0.4 * np.sin(np.pi * alpha)  # peak 40cm

        # Raise the whole body (all joints move up equally)
        seq[t, :, 1] += jump_y

        # Arms raise slightly during jump
        arm_raise = 0.15 * np.sin(np.pi * alpha)
        for j in [4, 5, 6, 7, 21, 22]:   # left arm
            seq[t, j, 1] += arm_raise
        for j in [8, 9, 10, 11, 23, 24]: # right arm
            seq[t, j, 1] += arm_raise

    return seq

# Hand waving: right arm raises to shoulder height, wrist waves side-to-side
def action_hand_waving(T=64):
    seq = make_sequence(T)
    for t in range(T):
        # Raise right arm to side-up position
        seq[t, 8, 1]  += 0.10   # shoulder up slightly
        seq[t, 9, 0]  += 0.05   # elbow out
        seq[t, 9, 1]  += 0.20   # elbow up
        seq[t, 10, 0] += 0.05
        seq[t, 10, 1] += 0.35   # wrist up to head height
        seq[t, 11, 1] += 0.38
        seq[t, 23, 1] += 0.38
        # Wrist oscillates left-right (waving motion)
        wave = 0.12 * np.sin(2 * np.pi * t / T * 4)  # 4 waves
        seq[t, 10, 0] += wave
        seq[t, 11, 0] += wave
        seq[t, 23, 0] += wave
        seq[t, 24, 0] += wave
    return seq

# Kicking: right leg extends forward, then returns
def action_kicking(T=64):
    seq = make_sequence(T)
    for t in range(T):
        alpha = t / T

        # Kick: leg goes forward during 0.2-0.6, returns 0.6-1.0
        if alpha < 0.2:
            kick = 0.0
        elif alpha < 0.6:
            kick = (alpha - 0.2) / 0.4   # 0 → 1
        else:
            kick = (1.0 - alpha) / 0.4   # 1 → 0

        kick_z = kick * 0.55  # forward extension in Z
        kick_y = kick * 0.20  # slight lift

        seq[t, 17, 2] += kick_z   # right knee forward
        seq[t, 17, 1] += kick_y
        seq[t, 18, 2] += kick_z * 1.5  # right ankle
        seq[t, 18, 1] += kick_y
        seq[t, 19, 2] += kick_z * 1.7  # right foot
        seq[t, 19, 1] += kick_y
    return seq

# name → (function, NTU class index, NTU class label)
ACTION_SAMPLES = {
    "Clapping":       (action_clapping,       9,  "clapping"),
    "Drinking Water": (action_drinking_water,  0,  "drink water"),
    "Jump Up":        (action_jumping,         26, "jump up"),
    "Hand Waving":    (action_hand_waving,     22, "hand waving"),
    "Kicking":        (action_kicking,         23, "kicking something"),
}

NTU60_CLASSES = [
    "drink water", "eat meal/snack", "brush teeth", "brush hair", "drop",
    "pick up", "throw", "sit down", "stand up", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket",
    "hopping (one foot jumping)", "jump up", "make a phone call/answer phone",
    "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie",
    "check time (from watch)", "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together",
    "cross hands in front (say stop)", "sneeze/cough", "staggering", "falling",
    "touch head (headache)", "touch chest (stomachache/heart pain)", "touch back (backache)",
    "touch neck (neck ache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person",
    "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket",
    "handshaking", "walking towards each other", "walking apart from each other"
]