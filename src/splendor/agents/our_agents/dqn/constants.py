"""
Constants relevant for the DQN agent and Q-learning based training.
"""

# Network architecture. The trunk mirrors the PPO MLP ([128 x 4] + LayerNorm)
# - the only architecture validated by actual training in this repository -
# minus the Dropout layer: dropout regularizes on-policy full-episode batches,
# while on off-policy minibatches (samples from different episodes & policies)
# it is a pure noise source.
HIDDEN_DIMS: tuple[int, ...] = (128, 128, 128, 128)

# Value assigned to illegal actions inside QNetwork.forward. Masking inside
# the forward pass guarantees that no code path (greedy act / training gather /
# bootstrap argmax) can ever observe an illegal action's value.
HUGE_NEG = -1e9

VERY_SMALL_EPSILON = 1e-8

# Momentum of the manual running-statistics update performed by
# QNetwork.observe (mirrors the EMA inside ppo/input_norm.py).
RUNNING_STATS_DECAY = 0.9

# Q-learning hyper-parameters (DQN_GUIDE §5.6).
DISCOUNT_FACTOR = 0.99
# NOT the PPO's 1e-6: with minibatch + replay updates 1e-6 would make training
# unusably slow, this is a "mirror the structure, never the hyper-parameters" case.
LEARNING_RATE = 1e-4
BATCH_SIZE = 512
BUFFER_SIZE = 500_000
# Purely random steps before the first gradient update, so that early updates
# sample a minimally diverse set of transitions.
WARMUP_STEPS = 5_000
# Rewards are sparse (most steps score 0, only buys / nobles / game end score);
# n-step returns shorten the credit assignment chain from the terminal signal.
N_STEP = 3

# Target network: soft update with tau by default. When TARGET_UPDATE_FREQ is
# set to a positive value, a hard update (full state_dict copy) is performed
# every TARGET_UPDATE_FREQ gradient steps instead.
TARGET_UPDATE_TAU = 0.005
TARGET_UPDATE_FREQ = 0

# Exploration: epsilon decays linearly from EPS_START to EPS_END across the
# first EPS_DECAY_FRACTION of the total training steps. Random actions are
# always sampled uniformly among *legal* actions.
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY_FRACTION = 0.2

# Terminal reward shaping. The engine's reward is a bare score delta without
# any win/loss signal; TD bootstrapping needs the terminal signal to propagate
# "chance of winning" backwards. The bonus must dominate single-card scores
# (0..5) without drowning the intermediate shaping.
WIN_BONUS = 10.0

# Global Gradient Norm Clipping as suggested by (bullet #11):
# https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
MAX_GRADIENT_NORM = 1.0

# Training loop bookkeeping.
TOTAL_STEPS = 200_000
EVAL_EVERY = 5_000
EVAL_GAMES = 20
SAVE_EVERY = 10_000

SEED = 1234
