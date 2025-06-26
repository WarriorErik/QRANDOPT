import sys
import os
from math import sqrt
import scipy.sparse as sp
from scipy.sparse import dok_matrix
from qci_client.optimization.enum import DeviceType
import time

# ──────────────────────────────────────────────────────────────────────────────
# Pointing to the repo root (one level up from this file) is on sys.path
# ──────────────────────────────────────────────────────────────────────────────
this_dir  = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(this_dir, os.pardir))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import requests
import binascii
import io
import gym
from gym import spaces

from qiskit_aer import AerSimulator
from qiskit import QuantumCircuit
from qiskit_aer.noise import NoiseModel, pauli_error


# Classical extractors and utilities
from extractors.von_neumann    import von_neumann
from extractors.elias          import elias
from extractors.universal_hash import universal_hash
from extractors.maurer_wolf    import maurer_wolf_extractor
from utils                     import compute_bias



# NIST SP 800-22 functions
from nistrng import (
    SP800_22R1A_BATTERY,
    check_eligibility_all_battery,
    run_all_battery
)

from qci_client import QciClient
import json
qci = QciClient(
 url=st.secrets.qci.url,
api_token=st.secrets.qci.token,
)

# ──────────────────────────────────────────────────────────────────────────────
# Helpers for Dirac-3 integer solver via polynomial upload
# ──────────────────────────────────────────────────────────────────────────────
# @st.cache_data(show_spinner=False)
# def upload_polynomial_file(
#     coeffs: list[float],
#     exp_counts: list[list[int]],
#     num_vars: int
# ) -> str:
#     """
#     Upload the polynomial in the exact REST schema Dirac-3 expects:
#       - num_variables: number of xi’s
#       - min_degree / max_degree: min/max of sum(exp_counts)
#       - data: list of {idx: [...], val: coef}, where idx is zero-padded
#     """
#     # compute total degree per term
#     degrees = [sum(v) for v in exp_counts]
#     min_deg, max_deg = min(degrees), max(degrees)

#     # build the QCI‐style data list
#     data = []
#     for coef, vec in zip(coeffs, exp_counts):
#         # repeat (j+1) p_j times
#         flat = []
#         for j, p in enumerate(vec):
#             flat += [j + 1] * p
#         # pad with leading zeros to length=max_deg
#         pad = [0] * (max_deg - len(flat))
#         idx_list = pad + flat
#         data.append({"idx": idx_list, "val": coef})

#     file_body = {
#         "file_name": "user_polynomial",
#         "file_config": {
#             "polynomial": {
#                 "num_variables": num_vars,
#                 "min_degree":    min_deg,
#                 "max_degree":    max_deg,
#                 "data":          data
#             }
#         }
#     }
#     resp = qci.upload_file(file=file_body)
#     return resp["file_id"]


# def submit_and_poll(
#     poly_file_id: str,
#     shots: int,
#     poll_interval: float = 2.0
# ) -> dict:
#     """
#     1) Submit a SAMPLE_HAMILTONIAN_INTEGER job  
#     2) Poll until status == COMPLETED/FAILED  
#     3) Return the final results dict
#     """
#     body = qci.build_job_body(
#         job_type="sample-hamiltonian-integer",
#         job_params={
#             "device_type": DeviceType.DIRAC3_QUDIT.value,
#             "num_samples": shots,
#             "num_levels":   [2],
#         },
#         polynomial_file_id=poly_file_id
#     )

#     # submit
#     submit_resp = qci.submit_job(job_body=body)
#     job_id = submit_resp["job_id"]

#     # poll
#     while True:
#         status = qci.get_job_status(job_id=job_id)["status"]
#         if status == "COMPLETED":
#             break
#         if status in ("FAILED", "CANCELLED"):
#             raise RuntimeError(f"Dirac-3 job {job_id} ended with status {status}")
#         time.sleep(poll_interval)

#     # fetch results
#     result = qci.get_job_results(job_id=job_id)
#     return result






# ──────────────────────────────────────────────────────────────────────────────
# Ensure a 1,000‐bit "sample_bitstream.txt" exists alongside dashboard.py
# ──────────────────────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sample_path = os.path.join(SCRIPT_DIR, "sample_bitstream.txt")

if not os.path.isfile(sample_path):
    bits = np.random.choice([0, 1], size=1000).tolist()
    with open(sample_path, "w") as f:
        f.write("".join(map(str, bits)))


# ──────────────────────────────────────────────────────────────────────────────
#  Bit‐generation helpers: drand → Aer fallback
# ──────────────────────────────────────────────────────────────────────────────
def generate_aer_bits(num_bits: int, batch_size: int = 256) -> list[int]:
    """
    Generate bits via Qiskit AerSimulator (Hadamard + measure).
    """
    backend = AerSimulator()
    qc = QuantumCircuit(batch_size, batch_size)
    qc.h(range(batch_size))
    qc.measure(range(batch_size), range(batch_size))

    shots = (num_bits + batch_size - 1) // batch_size
    result = backend.run(qc, shots=shots).result()

    bits: list[int] = []
    for bitstr, occ in result.get_counts().items():
        bits.extend([int(b) for b in bitstr] * occ)
        if len(bits) >= num_bits:
            break
    return bits[:num_bits]


def generate_drand_bits(num_bits: int) -> list[int]:
    """
    Fetch bits from the drand public randomness beacon (hex → bits).
    """
    bits: list[int] = []
    while len(bits) < num_bits:
        try:
            resp = requests.get("https://api.drand.sh/public/latest", timeout=5)
            resp.raise_for_status()
            data = resp.json().get('randomness', '')
            raw = binascii.unhexlify(data)
            arr = np.frombuffer(raw, dtype=np.uint8)
            bits.extend(np.unpackbits(arr).tolist())
        except Exception:
            break
    return bits[:num_bits]


@st.cache_data(show_spinner=False)
def generate_quantum_bits(num_bits: int, batch_size: int = 256) -> list[int]:
    """
    Attempt to fetch `num_bits` from drand. If insufficient or failure, fall back to AerSimulator.
    """
    drand_bits = generate_drand_bits(num_bits)
    if len(drand_bits) >= num_bits:
        return drand_bits[:num_bits]
    return generate_aer_bits(num_bits, batch_size)



# ──────────────────────────────────────────────────────────────────────────────
# Noisy bit‐generation helper using Kraus bit‐flip channel
# ──────────────────────────────────────────────────────────────────────────────
def apply_bit_flip_noise(bits: list[int], p: float) -> list[int]:
    """
    Independently flip each bit with probability p.
    This is the Kraus bit‐flip channel on a classical bitstream.
    """
    flips = np.random.rand(len(bits)) < p
    # XOR each bit with the flip mask
    return [b ^ int(f) for b, f in zip(bits, flips)]



# ──────────────────────────────────────────────────────────────────────────────
#  NIST SP 800-22 pass‐rate helper
# ──────────────────────────────────────────────────────────────────────────────
def nist_pass_rate(bits: list[int]) -> tuple[int, int, float]:
    """
    Run the SP800-22 R1A battery on `bits`. Returns (passed_count, total_tests, pass_fraction).
    """
    seq = np.array(bits, dtype=np.int8)
    eligibility = check_eligibility_all_battery(seq, SP800_22R1A_BATTERY)
    results = run_all_battery(seq, eligibility)

    passed = sum(1 for res, _ in results if res.passed)
    total = len(results)
    return passed, total, (passed / total if total > 0 else 0.0)


# ──────────────────────────────────────────────────────────────────────────────
#  Gym environment for Meta‐RL (ExtractorEnv)
# ──────────────────────────────────────────────────────────────────────────────
class ExtractorEnv(gym.Env):
    """
    Gym environment wrapping classical extractors.
    State: [bias, entropy] of the next window
    Action: choose extractor (0=VN, 1=Elias, 2=UHash, 3=Maurer–Wolf)
    Reward: shaped by QLearningAgent
    """
    metadata = {'render.modes': []}

    def __init__(self, raw_bits: list[int], window_size: int = 512, alpha: float = 0.0):
        super().__init__()
        self.raw_bits = raw_bits
        self.window_size = window_size
        self.alpha = alpha  # placeholder; agent computes shaped reward
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(0.0, 1.0, (2,), np.float32)
        self.reset()

    def reset(self):
        self.pos = 0
        return self._next_state()

    def step(self, action: int):
        start = self.pos
        end = min(self.pos + self.window_size, len(self.raw_bits))
        window = self.raw_bits[start:end]

        if not window:
            state = np.array([0.0, 0.0], dtype=np.float32)
            return state, 0.0, True, {'out_bits': []}

        if action == 0:
            out = von_neumann(window)
        elif action == 1:
            out = elias(window)
        elif action == 2:
            uh = universal_hash(window, seed="seed")
            out = uh if isinstance(uh, list) else self._bytes_to_bits(uh)
        else:
            rb = self._bits_to_bytes(window)
            mwb = maurer_wolf_extractor(rb, seed=b"seed", output_len=len(rb)//2)
            out = self._bytes_to_bits(mwb)

        self.pos = end
        done = (self.pos >= len(self.raw_bits))
        state = self._next_state()
        return state, 0.0, done, {'out_bits': out}

    def _next_state(self):
        end = min(self.pos + self.window_size, len(self.raw_bits))
        window = self.raw_bits[self.pos:end]
        if not window:
            return np.array([0.0, 0.0], dtype=np.float32)

        bias = compute_bias(window)
        p1 = sum(window) / len(window)
        p0 = 1 - p1
        entropy = -(p0 * np.log2(p0 + 1e-9) + p1 * np.log2(p1 + 1e-9))
        return np.array([bias, entropy], dtype=np.float32)

    @staticmethod
    def _bits_to_bytes(bits: list[int]) -> bytes:
        pad = (-len(bits)) % 8
        bits = bits + [0] * pad
        out = bytearray()
        for i in range(0, len(bits), 8):
            byte = 0
            for b in bits[i : i + 8]:
                byte = (byte << 1) | b
            out.append(byte)
        return bytes(out)

    @staticmethod
    def _bytes_to_bits(b: bytes) -> list[int]:
        bits = []
        for byte in b:
            for shift in range(7, -1, -1):
                bits.append((byte >> shift) & 1)
        return bits


# ──────────────────────────────────────────────────────────────────────────────
#  Q-Learning agent (Meta‐RL)
# ──────────────────────────────────────────────────────────────────────────────
class QLearningAgent:
    """
    Tabular Q-learning agent that now also rewards passing a simple Monobit
    (frequency) test on each extractor's output window.
    Reward = –bias_coef * (bias^2) + α * throughput + freq_bonus
    where freq_bonus = +1 if Monobit Z‐statistic within ±1.96, else –1.
    """
    def __init__(
        self,
        env: ExtractorEnv,
        lr: float = 5e-4,
        gamma: float = 0.99,
        epsilon: float = 0.2,
        bias_coef: float = 10.0,
        alpha_start: float = 0.0,
        alpha_end: float = 0.2,
        res: int = 50
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.bias_coef = bias_coef
        self.alpha_start = alpha_start
        self.alpha_end = alpha_end
        self.res = res
        self.q_table = np.zeros((res, res, env.action_space.n), dtype=np.float32)
        self.history_rewards: list[float] = []

    def discretize(self, state: np.ndarray) -> tuple[int,int]:
        return tuple((state * (self.res - 1)).astype(int))

    def select_action(self, state: np.ndarray) -> int:
        s_idx = self.discretize(state)
        if np.random.rand() < self.epsilon:
            return int(self.env.action_space.sample())
        return int(np.argmax(self.q_table[s_idx]))

    def train_one_episode(self, ep: int, total_episodes: int) -> float:
        """
        Run one full episode; returns total_reward.
        """
        α = self.alpha_start + (self.alpha_end - self.alpha_start) * (ep / total_episodes)
        state = self.env.reset()
        done = False
        total_reward = 0.0

        while not done:
            action = self.select_action(state)
            nxt_state, _, done, info = self.env.step(action)
            out_bits = info['out_bits']

            if not out_bits:
                base_reward = -self.bias_coef
                freq_bonus = -1
            else:
                b = compute_bias(out_bits)
                throughput = len(out_bits) / self.env.window_size
                base_reward = -self.bias_coef * (b ** 2) + α * throughput

                n_out = len(out_bits)
                sum_ones = sum(out_bits)
                z = (sum_ones - (n_out / 2)) / sqrt(n_out / 4)
                freq_bonus = 1 if abs(z) < 1.96 else -1

            reward = base_reward + freq_bonus

            s_idx = self.discretize(state)
            ns_idx = self.discretize(nxt_state)
            best_next = np.max(self.q_table[ns_idx])
            idx = s_idx + (action,)
            self.q_table[idx] += self.lr * (reward + self.gamma * best_next - self.q_table[idx])

            state = nxt_state
            total_reward += reward

        self.epsilon = max(0.01, self.epsilon * 0.999)
        self.history_rewards.append(total_reward)
        return total_reward


# ──────────────────────────────────────────────────────────────────────────────
#  Streamlit App
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="🔬 QRANDOPT Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🔬 QRANDOPT: Quantum/Classical Entropy Dashboard")
st.markdown(
    """
    Welcome!  
    Below you’ll find two main tabs:

    1. 🧮 Classical Extractor Metrics
       - Generate or upload a bitstream (drand → Aer fallback).  
       - Apply any combination of Von Neumann, Elias, Universal Hash, Maurer–Wolf.  
       - Compare bias, extraction rate, NIST SP 800-22 pass rates.  
       - New! Live Bitstream Explorer, sliding‐window entropy, autocorrelation, run‐length distribution, FFT, and more.  
       - New! Downloadable CSV reports.  

    2. 🤖 Meta-RL Extractor
       - Configure RL hyperparameters (learning rate, bias penalty, α schedule, grid size).  
       - Train episode‐by‐episode with a live progress bar.  
       - View learning curve and a 3D Q‐table heatmap.  
       - Compare RL output to classical extractors side by side.  

    3. ✨ Quantum Optimizer (Dirac-3)
    """
)

# ──────────────────────────────────────────────────────────────────────────────
# Sidebar: Global Settings
# ──────────────────────────────────────────────────────────────────────────────
st.sidebar.header("⚙️ Global Settings")
st.sidebar.write("Adjust global parameters for both tabs.")

num_bits = st.sidebar.number_input(
    "🔢 Number of raw bits to generate",
    min_value=1_000, max_value=200_000, value=20_000, step=1_000
)

batch_size = st.sidebar.selectbox(
    "📦 Batch size for Aer fallback",
    options=[64, 128, 256, 512], index=2
)

min_length_for_sp = st.sidebar.slider(
    "📏 Minimum length for SP 800-22 tests",
    min_value=100, max_value=10_000, value=1_000, step=100
)

noise_p = st.sidebar.slider(
    "🔊 Bit‐flip noise probability",
    0.0, 0.5, 0.0, 0.01,
    help="Apply a bit‐flip channel (Kraus) before extraction"
)


if "live_buffer" not in st.session_state:
    st.session_state.live_buffer = []

if "uploaded_bits" not in st.session_state:
    st.session_state.uploaded_bits = None

if "raw_bits" not in st.session_state:
    st.session_state.raw_bits = None

if "q_table" not in st.session_state:
    st.session_state.q_table = None

tab1, tab2, tab3, tab4 = st.tabs(["🧮 Classical Extractor Metrics", "🤖 Meta-RL Extractor", "👨‍🔬 Quantum Optimizer", " 📡QCI Data Analysis"])


# ──────────────────────────────────────────────────────────────────────────────
#  Classical Extractor Metrics
# ──────────────────────────────────────────────────────────────────────────────
with tab1:
    st.header("🧮 Classical Extractor Metrics")
    st.write("Generate or upload a bitstream, apply classical extractors, compare metrics, and view advanced visualizations.")

    col_upload, col_generate, col_custom = st.columns(3)
    with col_upload:
        uploaded_file = st.file_uploader(
            "📂 Upload your own bitstream (TXT/CSV of 0/1)", type=["txt", "csv"]
        )
        if uploaded_file is not None:
            raw_text = uploaded_file.read().decode("utf-8").strip().split()
            bits = []
            for token in raw_text:
                for char in token:
                    if char in ("0", "1"):
                        bits.append(int(char))
            if bits:
                st.session_state.uploaded_bits = bits
                st.success(f"Uploaded bitstream loaded (length = {len(bits)})")
            else:
                st.error("Could not parse any 0/1 bits from the uploaded file.")

    with col_generate:
        if st.button("▶️ Generate Raw Bits", key="btn_generate"):
            st.session_state.raw_bits = generate_quantum_bits(num_bits, batch_size)
            st.session_state.live_buffer = st.session_state.raw_bits.copy()
            st.success(f"Generated raw bitstream of length {len(st.session_state.raw_bits)}")

    with col_custom:
        with st.expander("➕ Generate a Biased Bitstream", expanded=False):
            st.write(
                "Choose a length and a bias `p` (for 1). This will create a synthetic stream\n"
                "where each bit is 1 with probability `p`, 0 otherwise."
            )
            custom_length = st.number_input(
                "Length of custom bitstream", min_value=100, max_value=100000, value=5000, step=100
            )
            bias_p = st.slider("Target bias (Probability of 1)", min_value=0.0, max_value=1.0,
                               value=0.5, step=0.01)
            if st.button("🛠 Create Biased Bitstream", key="btn_custom"):
                rng_bits = np.random.choice([0, 1], size=custom_length, p=[1 - bias_p, bias_p]).tolist()
                st.session_state.raw_bits = rng_bits
                st.session_state.live_buffer = rng_bits.copy()
                st.success(f"Custom bitstream created (length={custom_length}, p={bias_p:.2f})")

    if st.session_state.uploaded_bits is not None:
        raw_bits = st.session_state.uploaded_bits
        source_label = "Uploaded"
    elif st.session_state.raw_bits is not None:
        raw_bits = st.session_state.raw_bits
        source_label = "Generated"
    else:
        raw_bits = None
        source_label = None

        if raw_bits is not None:
        # ──────────────────────────────────────────────────────────────
        #  Inject Kraus bit‐flip noise if requested
        # ──────────────────────────────────────────────────────────────
            if noise_p > 0:
                raw_bits = apply_bit_flip_noise(raw_bits, noise_p)
                source_label += f" + noise(p={noise_p:.2f})"


    if raw_bits is not None:
        raw_bias = compute_bias(raw_bits)
        p1 = sum(raw_bits) / len(raw_bits)
        p0 = 1 - p1
        entropy_est = -(p1 * np.log2(p1 + 1e-9) + p0 * np.log2(p0 + 1e-9))

        st.subheader(f"🔎 {source_label} Bitstream Statistics")
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Bits", f"{len(raw_bits):,}")
        c2.metric("Raw Bias", f"{raw_bias:.4f}", delta=f"{raw_bias:.4f}")
        c3.metric("Entropy Estimate", f"{entropy_est:.4f}")

        if st.checkbox("📈 Show raw bit histogram (0 vs 1)", value=False):
            fig_rb = px.bar(
                x=[0, 1],
                y=[raw_bits.count(0), raw_bits.count(1)],
                labels={"x": "Bit", "y": "Count"},
                title="Raw Bits Histogram",
                color=[0, 1],
                color_continuous_scale=px.colors.qualitative.Prism
            )
            st.plotly_chart(fig_rb, use_container_width=True)

        st.markdown("---")

        # Live Bitstream Explorer
        st.subheader("🚀 Live Bitstream Explorer")
        st.write(
            """
            Start with the entire bitstream in a “live buffer.”  
            Click “➕ Add `batch_size` bits” to append the next chunk of generated bits (cycling if needed).  
            Watch how the rolling bias and sliding‐window entropy evolve in real time.
            """
        )
        live_col1, live_col2 = st.columns([2, 1])

        with live_col2:
            add_button = st.button(f"➕ Add {batch_size} bits to Live Buffer", key="btn_add_live")
            reset_button = st.button("🔄 Reset Live Buffer", key="btn_reset_live")
            st.write(f"Current buffer length: {len(st.session_state.live_buffer):,}")

            if add_button:
                buffer = st.session_state.live_buffer
                start_idx = len(buffer)
                for i in range(batch_size):
                    buffer.append(raw_bits[(start_idx + i) % len(raw_bits)])
                st.session_state.live_buffer = buffer
                st.success(f"Added {batch_size} bits. New buffer length = {len(buffer):,}")

            if reset_button:
                if source_label == "Uploaded":
                    st.session_state.live_buffer = st.session_state.uploaded_bits.copy()
                else:
                    st.session_state.live_buffer = st.session_state.raw_bits.copy()
                st.success("Live buffer reset to full bitstream.")

        buffer = st.session_state.live_buffer
        window_size = st.slider(
            "🔍 Rolling-window size (for bias/entropy)", 
            min_value=100, max_value=5000, value=1000, step=100
        )

        if len(buffer) >= window_size:
            rolling_bias = pd.Series(buffer).rolling(window=window_size).apply(
                lambda x: abs(x.mean() - 0.5)
            ).to_list()
        else:
            rolling_bias = [abs(np.array(buffer[:i+1]).mean() - 0.5) for i in range(len(buffer))]

        if len(buffer) >= window_size:
            def shannon_entropy(arr):
                p1_ = arr.mean()
                p0_ = 1 - p1_
                return -(p1_ * np.log2(p1_ + 1e-9) + p0_ * np.log2(p0_ + 1e-9))
            rolling_ent = pd.Series(buffer).rolling(window=window_size).apply(shannon_entropy).to_list()
        else:
            rolling_ent = [
                -( (np.array(buffer[:i+1]).mean()) * np.log2(np.array(buffer[:i+1]).mean() + 1e-9 )
                  + (1 - np.array(buffer[:i+1]).mean()) * np.log2((1 - np.array(buffer[:i+1]).mean()) + 1e-9) )
                for i in range(len(buffer))
            ]

        with live_col1:
            fig_live = go.Figure()
            fig_live.add_trace(go.Scatter(
                x=list(range(len(rolling_bias))),
                y=rolling_bias,
                mode='lines',
                name='Rolling Bias',
                line=dict(color='rgba(31,119,180,0.7)')
            ))
            fig_live.add_trace(go.Scatter(
                x=list(range(len(rolling_ent))),
                y=rolling_ent,
                mode='lines',
                name='Rolling Entropy',
                line=dict(color='rgba(255,127,14,0.7)')
            ))
            fig_live.update_layout(
                xaxis_title='Index in Live Buffer',
                yaxis_title='Value',
                title=f"Live Rolling Bias & Entropy (window={window_size})",
                template='plotly_white',
                height=350
            )
            st.plotly_chart(fig_live, use_container_width=True)

        st.markdown("---")

        # Classical Extractor Analysis
        st.subheader("🔍 Classical Extractor Analysis")

        extractor_options = ["Von Neumann", "Elias", "Universal Hash", "Maurer-Wolf", "None"]
        chosen_extractors = st.multiselect(
            "Select one or more extractors to apply:",
            options=extractor_options,
            default=["Von Neumann", "Elias"]
        )

        metrics_list = []
        streams_map = {}

        for ex in chosen_extractors:
            if ex == "None":
                out_bits = raw_bits.copy()
            elif ex == "Von Neumann":
                out_bits = von_neumann(raw_bits)
            elif ex == "Elias":
                out_bits = elias(raw_bits)
            elif ex == "Universal Hash":
                uh = universal_hash(raw_bits, seed="seed")
                out_bits = uh if isinstance(uh, list) else ExtractorEnv._bytes_to_bits(uh)
            else:
                rb = ExtractorEnv._bits_to_bytes(raw_bits)
                mwb = maurer_wolf_extractor(rb, seed=b"seed", output_len=len(rb)//2)
                out_bits = ExtractorEnv._bytes_to_bits(mwb)

            bias_after = compute_bias(out_bits)
            rate = (len(out_bits) / len(raw_bits)) if len(raw_bits) > 0 else 0.0
            passed, total, prate = (0, 0, 0.0)
            if len(out_bits) >= min_length_for_sp:
                passed, total, prate = nist_pass_rate(out_bits)

            metrics_list.append({
                "Extractor": ex,
                "Raw Bias": raw_bias,
                "Post Bias": bias_after,
                "Extracted Bits": len(out_bits),
                "Rate": rate,
                "SP800-22 Passed": passed,
                "SP800-22 Total": total,
                "PassRate": prate
            })
            streams_map[ex] = out_bits

        df_metrics = pd.DataFrame(metrics_list).set_index("Extractor")

        st.subheader("📊 Extractor Comparison Table")
        st.dataframe(
            df_metrics.style.format({
                "Raw Bias": "{:.4f}",
                "Post Bias": "{:.4f}",
                "Rate": "{:.4f}",
                "PassRate": "{:.4f}"
            }),
            use_container_width=True
        )

        # Download classical metrics as CSV
        csv_buffer = io.StringIO()
        df_metrics.to_csv(csv_buffer)
        st.download_button(
            label="⬇️ Download Classical Metrics as CSV",
            data=csv_buffer.getvalue().encode("utf-8"),
            file_name="classical_metrics.csv",
            mime="text/csv"
        )

        if not df_metrics.empty:
            best_pass = df_metrics["PassRate"].idxmax()
            best_rate = df_metrics["Rate"].idxmax()
            best_bias = df_metrics["Post Bias"].idxmin()

            st.subheader("🏆 Top Performers")
            mcol1, mcol2, mcol3 = st.columns(3)
            mcol1.metric(
                "Highest SP800-22 PassRate",
                best_pass,
                f"{df_metrics.loc[best_pass, 'PassRate']:.4f}"
            )
            mcol2.metric(
                "Highest Extraction Rate",
                best_rate,
                f"{df_metrics.loc[best_rate, 'Rate']:.4f}"
            )
            mcol3.metric(
                "Lowest Post-Bias",
                best_bias,
                f"{df_metrics.loc[best_bias, 'Post Bias']:.4f}"
            )

        st.markdown("---")

        # Per-extractor histogram & stats
        st.subheader("📈 Per-Extractor Bit Histogram & Stats")
        extractor_to_view = st.selectbox("Select extractor to view:", options=list(streams_map.keys()))
        if extractor_to_view:
            out_bits = streams_map[extractor_to_view]
            col_h1, col_h2 = st.columns([2, 1])
            with col_h1:
                fig_ex = px.bar(
                    x=[0, 1],
                    y=[out_bits.count(0), out_bits.count(1)],
                    labels={"x": "Bit", "y": "Count"},
                    title=f"{extractor_to_view}: Output Bit Distribution",
                    color=[0, 1],
                    color_continuous_scale=px.colors.qualitative.Prism
                )
                st.plotly_chart(fig_ex, use_container_width=True)
            with col_h2:
                post_bias = compute_bias(out_bits)
                post_rate = (len(out_bits) / len(raw_bits)) if len(raw_bits) > 0 else 0.0
                st.metric("Post-Bias", f"{post_bias:.4f}")
                st.metric("Extraction Rate", f"{post_rate:.4f}")
                if len(out_bits) >= min_length_for_sp:
                    passed_v, total_v, prate_v = nist_pass_rate(out_bits)
                    st.metric("SP800-22 PassRate", f"{prate_v:.4f}", delta=f"{passed_v}/{total_v} passed")
                else:
                    st.write("(Output too short for SP 800-22)")

        st.markdown("---")

        # Advanced Visualizations
        st.subheader("🔬 Advanced Visualizations")
        st.write("Explore additional randomness diagnostics on the raw or live buffer.")

        adv_col1, adv_col2 = st.columns(2)

        # Sliding-window Shannon Entropy (Raw)
        with adv_col1:
            st.write("1. Sliding-Window Shannon Entropy (Raw)")
            sw = st.slider("Window size for entropy (Raw)", min_value=100, max_value=5000, value=1000, step=100, key="sw_entropy_raw")
            if len(raw_bits) >= sw:
                def sh_entropy(arr):
                    p1_ = arr.mean()
                    p0_ = 1 - p1_
                    return -(p1_ * np.log2(p1_ + 1e-9) + p0_ * np.log2(p0_ + 1e-9))
                ent_series = pd.Series(raw_bits).rolling(window=sw).apply(sh_entropy).to_list()
            else:
                ent_series = [
                    -( (np.array(raw_bits[:i+1]).mean()) * np.log2(np.array(raw_bits[:i+1]).mean() + 1e-9 )
                      + (1 - np.array(raw_bits[:i+1]).mean()) * np.log2((1 - np.array(raw_bits[:i+1]).mean()) + 1e-9) )
                    for i in range(len(raw_bits))
                ]
            fig_ent_raw = go.Figure()
            fig_ent_raw.add_trace(go.Scatter(
                x=list(range(len(ent_series))),
                y=ent_series,
                mode='lines',
                name='Sliding Entropy',
                line=dict(color='rgba(44, 160, 44, 0.8)')
            ))
            fig_ent_raw.update_layout(
                xaxis_title='Index',
                yaxis_title='Shannon Entropy',
                title=f"Raw Bitstream: Sliding-Window Entropy (window={sw})",
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig_ent_raw, use_container_width=True)

        # Autocorrelation Plot (Raw)
        with adv_col2:
            st.write("2. Autocorrelation (Raw)")
            max_lag = st.slider("Max lag for autocorrelation", min_value=10, max_value=500, value=100, step=10, key="ac_lag_raw")
            def autocorr(x, lag):
                n = len(x)
                x_mean = np.mean(x)
                num = np.sum((x[:n-lag] - x_mean) * (x[lag:] - x_mean))
                den = np.sum((x - x_mean) ** 2)
                return num / den if den != 0 else 0.0
            ac_vals = [autocorr(np.array(raw_bits), lag) for lag in range(1, max_lag+1)]
            fig_ac_raw = px.line(
                x=list(range(1, max_lag+1)),
                y=ac_vals,
                labels={"x": "Lag", "y": "Autocorrelation"},
                title=f"Raw Bitstream: Autocorrelation (lags 1–{max_lag})",
                height=300
            )
            st.plotly_chart(fig_ac_raw, use_container_width=True)

        adv_col3, adv_col4 = st.columns(2)

        # Run-Length Distribution
        with adv_col3:
            st.write("3. Run-Length Distribution (Raw)")
            runs = []
            current_run = 1
            for i in range(1, len(raw_bits)):
                if raw_bits[i] == raw_bits[i-1]:
                    current_run += 1
                else:
                    runs.append(current_run)
                    current_run = 1
            runs.append(current_run)
            zero_runs = []
            one_runs = []
            idx = 0
            while idx < len(raw_bits):
                bit_val = raw_bits[idx]
                length = 1
                idx2 = idx + 1
                while idx2 < len(raw_bits) and raw_bits[idx2] == bit_val:
                    length += 1
                    idx2 += 1
                if bit_val == 0:
                    zero_runs.append(length)
                else:
                    one_runs.append(length)
                idx = idx2
            max_run = max(runs) if runs else 1
            histz = np.bincount(zero_runs, minlength=max_run+1)[1:] if zero_runs else [0]
            hist1 = np.bincount(one_runs, minlength=max_run+1)[1:] if one_runs else [0]
            fig_runs = go.Figure()
            fig_runs.add_trace(go.Bar(
                x=list(range(1, len(histz)+1)),
                y=histz,
                name='Zero Runs',
                marker_color='rgba(31,119,180,0.7)'
            ))
            fig_runs.add_trace(go.Bar(
                x=list(range(1, len(hist1)+1)),
                y=hist1,
                name='One Runs',
                marker_color='rgba(255,127,14,0.7)',
                opacity=0.75
            ))
            fig_runs.update_layout(
                barmode='overlay',
                xaxis_title='Run Length',
                yaxis_title='Count',
                title='Run-Length Distribution (Raw)',
                template='plotly_white',
                height=300
            )
            st.plotly_chart(fig_runs, use_container_width=True)

        # 2D Successive-Bit Scatter (Raw)
        with adv_col4:
            st.write("4. Successive-Bit Scatter (Raw)")
            if len(raw_bits) >= 2:
                x_vals = raw_bits[:-1]
                y_vals = raw_bits[1:]
                fig_scatter = px.scatter(
                    x=x_vals, y=y_vals,
                    labels={"x": "Bitₙ", "y": "Bitₙ₊₁"},
                    title="Raw Bitstream: Scatter of Successive Bits",
                    height=300
                )
                st.plotly_chart(fig_scatter, use_container_width=True)
            else:
                st.write("Need at least 2 bits for scatter plot.")

        st.markdown("---")

        adv_col5, adv_col6 = st.columns(2)

        # FFT Magnitude Spectrum (Raw)
        with adv_col5:
            st.write("5. FFT Magnitude Spectrum (Raw)")
            arr_fft = np.array(raw_bits) * 2 - 1
            freqs = np.fft.rfftfreq(len(arr_fft))
            mags = np.abs(np.fft.rfft(arr_fft))
            fig_fft = px.line(
                x=freqs,
                y=mags,
                labels={"x": "Normalized Frequency", "y": "Magnitude"},
                title="FFT Magnitude Spectrum (Raw)",
                height=300
            )
            st.plotly_chart(fig_fft, use_container_width=True)

        # Cumulative Sum Plot (Raw)
        with adv_col6:
            st.write("6. Cumulative Sum Plot (Raw)")
            cum_sum = np.cumsum(np.array(raw_bits)*2 - 1)
            fig_csum = px.line(
                x=list(range(len(cum_sum))),
                y=cum_sum,
                labels={"x": "Index", "y": "Cumulative Sum"},
                title="Raw Bitstream: Cumulative Sum (Martingale Plot)",
                height=300
            )
            st.plotly_chart(fig_csum, use_container_width=True)

        st.markdown("---")


    #─────────────────────────────────────────────────
    # Kraus Simulation Noise‐Sweep Benchmark
    # ─────────────────────────────────────────────────
    st.subheader("🎯 Noise‐Sweep Benchmark")
    sweep = st.multiselect(
        "Noise probabilities to test",
        options=[0.0, 0.01, 0.05, 0.1, 0.2],
        default=[0.0, 0.05, 0.1]
    )
    if st.button("🔄 Run Noise Sweep"):
        rows = []
        for p in sweep:
            # get a fresh stream and apply noise
            bits_s = generate_quantum_bits(num_bits, batch_size)
            if p > 0:
                bits_s = apply_bit_flip_noise(bits_s, p)

            for name, func in [
                ("Von Neumann", von_neumann),
                ("Elias", elias),
                ("Universal Hash", lambda b: 
                    (universal_hash(b, seed="seed") 
                     if isinstance(universal_hash(b, seed="seed"), list)
                     else ExtractorEnv._bytes_to_bits(universal_hash(b, seed="seed")))
                ),
                ("Maurer‐Wolf", lambda b:
                    ExtractorEnv._bytes_to_bits(
                        maurer_wolf_extractor(
                            ExtractorEnv._bits_to_bytes(b),
                            seed=b"seed",
                            output_len=len(b)//2
                        )
                    )
                ),
            ]:
                out = func(bits_s)
                _, _, pr = nist_pass_rate(out) if len(out) >= min_length_for_sp else (0,0,0.0)
                rows.append({"Extractor": name, "p_flip": p, "PassRate": pr})

        df_sweep = pd.DataFrame(rows)
        pivot = df_sweep.pivot(index="p_flip", columns="Extractor", values="PassRate")
        st.dataframe(pivot)
        st.line_chart(pivot)


        # “How to create your own bitstream” Tutorial
        st.subheader("📜 How to Create Your Own Bitstream")
        with st.expander("Show me Python & Shell code examples", expanded=False):
            st.write(
                """
                1. Manually in a Text Editor
                - Open Notepad (Windows) or any text editor and type a long sequence of `0` and `1`, e.g.:  
                  ```
                  010101010110110010101001...
                  ```  
                - Save as `my_bitstream.txt`—no spaces, just a continuous string of zeros and ones.  

                2. Programmatically in Python
                
                import numpy as np

                # Choose how many bits you want and desired bias
                N = 5000           # total number of bits
                p = 0.3            # probability of 1 (bias)
                bits = np.random.choice([0,1], size=N, p=[1-p, p])  

                bitstring = "".join(map(str, bits.tolist()))
                with open("my_bitstream.txt", "w") as f:
                    f.write(bitstring)
                print(f"Saved bitstream of length {N} to my_bitstream.txt")
                ```

                3. From the Command Line (Linux/macOS)
                ```bash
                # Generate 1000 random bits via /dev/urandom:
                head -c 10000 /dev/urandom | tr -dc '01' | head -c 1000 > my_bitstream.txt
                ```  

                4. Download This Sample Bitstream  
                Below is a link to download a 1,000-bit random sample.  
                Simply click to get `sample_bitstream.txt`, then upload it above.
                """
            )
            try:
                with open(sample_path, "rb") as f:
                    sample_data = f.read()
                st.download_button(
                    label="⬇️ Download sample_bitstream.txt",
                    data=sample_data,
                    file_name="sample_bitstream.txt",
                    mime="text/plain"
                )
            except FileNotFoundError:
                st.error(f"Could not find `{sample_path}`. It should be auto-generated at startup.")

    else:
        st.info("▶️ Generate or upload a bitstream to begin analysis.")

    


# ──────────────────────────────────────────────────────────────────────────────
# Meta-RL Extractor
# ──────────────────────────────────────────────────────────────────────────────
with tab2:
    st.header("🤖 Meta-RL Extractor (Q-Learning)")
    st.write("Configure RL hyperparameters, train episode by episode, and compare to classical extractors.")

    with st.sidebar.expander("⚙️ RL Hyperparameters", expanded=True):
        lr = st.number_input("Learning rate (lr)", min_value=1e-5, max_value=1e-1, value=5e-4, format="%g")
        gamma = st.slider("Discount factor (γ)", min_value=0.90, max_value=0.999, value=0.99, step=0.005)
        bias_coef = st.number_input("Bias penalty coefficient", min_value=0.0, max_value=100.0, value=10.0, step=0.5)
        alpha_start = st.slider("α (start)", min_value=0.0, max_value=1.0, value=0.0, step=0.05)
        alpha_end = st.slider("α (end)", min_value=0.0, max_value=1.0, value=0.2, step=0.05)
        res = st.select_slider("State discretization (res)", options=[20, 50, 100, 200], value=50)
        episodes = st.slider("Number of episodes", min_value=100, max_value=5000, value=2000, step=100)

    rl_run = st.button("▶️ Train Q-Learning Agent (Meta-RL)", key="run_meta_rl")
    if rl_run:
        with st.spinner("🏃 Setting up RL environment…"):
            if st.session_state.uploaded_bits is not None:
                raw_bits = st.session_state.uploaded_bits
            elif st.session_state.raw_bits is not None:
                raw_bits = st.session_state.raw_bits
            else:
                raw_bits = generate_quantum_bits(num_bits, batch_size)
                st.session_state.raw_bits = raw_bits

            env = ExtractorEnv(raw_bits, window_size=512, alpha=0.0)
            agent = QLearningAgent(
                env,
                lr=lr,
                gamma=gamma,
                epsilon=0.2,
                bias_coef=bias_coef,
                alpha_start=alpha_start,
                alpha_end=alpha_end,
                res=res
            )
            st.session_state.q_table = agent.q_table

        st.success("✅ RL environment ready")

        st.write("Training… watch the progress bar update below.")
        progress_bar = st.progress(0)
        train_status = st.empty()

        for ep in range(episodes):
            total_r = agent.train_one_episode(ep, episodes)
            progress_bar.progress((ep + 1) / episodes)
            if ep % max(1, episodes // 10) == 0:
                train_status.write(f"Episode {ep+1}/{episodes}  Reward = {total_r:.3f}")

        train_status.success("🎉 Training complete!")

        st.subheader("📈 Learning Curve")
        rewards = np.array(agent.history_rewards)
        ma = pd.Series(rewards).rolling(50, min_periods=1).mean()
        fig_lc = go.Figure()
        fig_lc.add_trace(go.Scatter(
            x=list(range(1, episodes + 1)),
            y=rewards,
            mode='lines',
            name='Per-Episode Reward',
            line=dict(color='rgba(31,119,180,0.5)')
        ))
        fig_lc.add_trace(go.Scatter(
            x=list(range(1, episodes + 1)),
            y=ma,
            mode='lines',
            name='50-Episode MA',
            line=dict(color='rgba(255,127,14,1)', width=3)
        ))
        fig_lc.update_layout(
            xaxis_title='Episode',
            yaxis_title='Total Reward',
            title='Q-Learning: Reward per Episode',
            template='plotly_white'
        )
        st.plotly_chart(fig_lc, use_container_width=True)

        st.subheader("🔽 Final ε (Exploration Rate)")
        st.metric("Epsilon", f"{agent.epsilon:.4f}")

        st.markdown("---")

        st.subheader("🔍 Q-Table Heatmap (Max over Actions)")
        q_max = np.max(agent.q_table, axis=2)
        xs = np.arange(res)
        ys = np.arange(res)
        Z = q_max

        fig_q = go.Figure(data=[
            go.Surface(
                z=Z,
                x=xs,
                y=ys,
                colorscale='Viridis',
                contours={
                    "z": {"show": True, "start": np.min(Z), "end": np.max(Z), "size": (np.max(Z)-np.min(Z))/10}
                }
            )
        ])
        fig_q.update_layout(
            scene=dict(
                xaxis_title='Bias Index',
                yaxis_title='Entropy Index',
                zaxis_title='Q-Value'
            ),
            title='3D Q-Table Surface',
            autosize=True,
            margin=dict(l=0, r=0, b=0, t=50)
        )
        st.plotly_chart(fig_q, use_container_width=True)

        q_df_list = []
        for i in range(res):
            for j in range(res):
                row = {"BiasIdx": i, "EntIdx": j}
                for a in range(env.action_space.n):
                    row[f"Q_{a}"] = agent.q_table[i, j, a]
                q_df_list.append(row)
        q_df = pd.DataFrame(q_df_list)
        csvbuf_q = io.StringIO()
        q_df.to_csv(csvbuf_q, index=False)
        st.download_button(
            label="⬇️ Download Q-Table as CSV",
            data=csvbuf_q.getvalue().encode("utf-8"),
            file_name="q_table.csv",
            mime="text/csv"
        )

        st.markdown("---")

        st.subheader("📊 Compare Meta-RL vs Classical Extractors")

        state, done = env.reset(), False
        agent.epsilon = 0.0
        meta_out_bits: list[int] = []
        while not done:
            a = int(np.argmax(agent.q_table[agent.discretize(state)]))
            state, _, done, info = env.step(a)
            meta_out_bits.extend(info["out_bits"])

        vn_out = von_neumann(raw_bits)
        el_out = elias(raw_bits)
        uh = universal_hash(raw_bits, seed="seed")
        uh_out = uh if isinstance(uh, list) else ExtractorEnv._bytes_to_bits(uh)
        rb = ExtractorEnv._bits_to_bytes(raw_bits)
        mwb = maurer_wolf_extractor(rb, seed=b"seed", output_len=len(rb)//2)
        mw_out = ExtractorEnv._bytes_to_bits(mwb)

        compare_streams = {
            "Meta-RL": meta_out_bits,
            "VN": vn_out,
            "Elias": el_out,
            "UHash": uh_out,
            "Maurer-Wolf": mw_out
        }

        rows = []
        for name, out_bits in compare_streams.items():
            b0 = compute_bias(raw_bits)
            b1 = compute_bias(out_bits)
            rate = (len(out_bits) / len(raw_bits)) if len(raw_bits) > 0 else 0.0
            passed, total, prate = (0, 0, 0.0)
            if len(out_bits) >= min_length_for_sp:
                passed, total, prate = nist_pass_rate(out_bits)
            rows.append({
                "Extractor": name,
                "Raw Bias": b0,
                "Post Bias": b1,
                "Extracted Bits": len(out_bits),
                "Rate": rate,
                "SP800-22 PassRate": prate
            })

        df_compare = pd.DataFrame(rows).set_index("Extractor")

        st.dataframe(
            df_compare.style.format({
                "Raw Bias": "{:.4f}",
                "Post Bias": "{:.4f}",
                "Rate": "{:.4f}",
                "SP800-22 PassRate": "{:.4f}"
            }),
            use_container_width=True
        )

        csvbuf_cmp = io.StringIO()
        df_compare.to_csv(csvbuf_cmp)
        st.download_button(
            label="⬇️ Download Comparison Table as CSV",
            data=csvbuf_cmp.getvalue().encode("utf-8"),
            file_name="compare_metrics.csv",
            mime="text/csv"
        )

        st.write("Side-by-Side Bar Charts")
        cb1, cb2 = st.columns(2)
        with cb1:
            fig_b = px.bar(
                df_compare.reset_index(),
                x="Extractor",
                y="Post Bias",
                title="Post Bias by Extractor",
                color="Extractor",
                color_discrete_sequence=px.colors.qualitative.Pastel
            )
            st.plotly_chart(fig_b, use_container_width=True)
        with cb2:
            fig_l = px.bar(
                df_compare.reset_index(),
                x="Extractor",
                y="Extracted Bits",
                title="Extracted Bits by Extractor",
                color="Extractor",
                color_discrete_sequence=px.colors.qualitative.Set2
            )
            st.plotly_chart(fig_l, use_container_width=True)

    else:
        st.info("▶️ Click the button above to train the Meta-RL agent.")

#___________________________________________________________________________
#Quantum Optimization Tab using the Dirac Hardware Model
#____________________________________________________________________________





with tab3:
    st.header("✨ Quantum Optimizer (Dirac-3)")

    with st.expander("❓ What is this tab and why it matters to QRANDOPT", expanded=True):
        st.markdown(
            """
            Overview  
            In this tab, we take your polynomial Hamiltonian (a cost function over binary variables)  
            and run it on Quantum Computing Inc.’s Dirac-3 qudit optimizer.  

            What it does  
            1. Uploads your polynomial (coeffs  exponent vectors) to the QCI REST API.  
            2. Submits a `sample-hamiltonian-integer` job targeting the Dirac-3 device.  
            3. Polls the job until completion, then retrieves samples of bit-strings and their energies.  
            4. Displays:  
               - The minimum (best) energy found  
               - The corresponding optimal bit-assignment  
               - A histogram of energies over all shots  
               - A table of every unique solution, its energy, and frequency  

            Why this matters for QRANDOPT 
            - Demonstrates how a quantum optimizer can be integrated _side-by-side_ with  
              classical extractors and meta-RL agents.  
            - Provides a concrete, hands-on example of leveraging real quantum hardware (qudits)  
              to solve discrete optimization problems arising in entropy-extraction workflows.  
            - Offers performance data (success‐rate, solution quality) for comparison against your  
              classical and RL baselines in Chapters 3–5 of your thesis.  
            """
        )



    with st.expander("🧑‍🏫 Quick Tutorial & Example Equations", expanded=True):
        st.markdown("""
        How to use:  
        1. Pick one of the example equations below (or write your own).  
        2. Copy the JSON under the equation and paste it into the Polynomial JSON box.  
        3. Set Shots and Variables (variables = number of xi’s, e.g. 3).  
        4. Click ▶️ Run Quantum Optimization and explore the results.

        ————————————————

        #1) Simple Linear  
        Minimize  
        
          H(x) = x_0 + x_1 + x_2
         
        JSON:
        
        {
          "coefficients": [1, 1, 1],
          "indices": [
            [1, 0, 0],   // x₀
            [0, 1, 0],   // x₁
            [0, 0, 1]    // x₂
          ]
        }
        

        ### 2) Pairwise Interaction (Max-Cut on 3 nodes)  
        Minimize (negative edge weights so maximizing cuts)  
        
          H(x) = -(x_0 x_1 + x_1 x_2 + x_0 x_2
         
        JSON:
        
        {
          "coefficients": [-1, -1, -1],
          "indices": [
            [1, 1, 0],   // x₀ x₁
            [0, 1, 1],   // x₁ x₂
            [1, 0, 1]    // x₀ x₂
          ]
        }
        

        #3) Mixed Linear & Quadratic  
        Minimize  
        
          H(x) = 2x_0 - 3x_1^2 + x_0 x_2
          
        (Here (x_1^2 = x_1) since (x_1 is in{0,1}).)  
        JSON:
        {
          "coefficients": [2, -3, 1],
          "indices": [
            [1, 0, 0],  // 2·x₀
            [0, 1, 0],  // -3·x₁^2
            [1, 0, 1]   // 1·(x₀ x₂)
          ]
        }
        
        """)


    # User inputs
    poly_json = st.text_area(
        "Polynomial JSON (coeffs & exponent-vectors):",
        height=120,
        placeholder='{\"coefficients\":[1,-2,1],\"indices\":[[3,0,0],[0,2,0],[0,0,1]]}'
    )
    shots    = st.number_input("Samples (shots)", 1, 10_000, 100, 1)
    num_vars = st.number_input("Variables",       1,    20,    3,  1)

    if st.button("▶️ Run Quantum Optimization"):
        # ────────────────────────────────────────────────────────────
        # Parse & sanitize
        # ────────────────────────────────────────────────────────────
        try:
            spec   = json.loads(poly_json)
            coeffs = spec["coefficients"]
            inds   = spec["indices"]
        except Exception as e:
            st.error(f"Invalid JSON: {e}")
            st.stop()

        if not (isinstance(coeffs, list) and isinstance(inds, list) and len(coeffs) == len(inds)):
            st.error("‘coefficients’ and ‘indices’ must be equally-long lists.")
            st.stop()

        # strip out any constant (deg=0) terms
        filtered = [(c,v) for c,v in zip(coeffs,inds) if sum(v)>0]
        if len(filtered) < len(coeffs):
            csum = sum(c for c,v in zip(coeffs,inds) if sum(v)==0)
            st.warning(f"Removed constant offset {csum}; Dirac-3 requires deg≥1.")
        if not filtered:
            st.error("No degree≥1 terms left.")
            st.stop()
        coeffs, inds = zip(*filtered)
        coeffs, inds = list(coeffs), list(inds)

        # ────────────────────────────────────────────────────────────
        # Build the polynomial‐upload payload
        # ────────────────────────────────────────────────────────────
        degrees = [sum(v) for v in inds]
        mn, mx  = min(degrees), max(degrees)
        data = []
        for c, vec in zip(coeffs, inds):
            flat   = [j+1 for j,p in enumerate(vec) for _ in range(p)]
            pad    = [0] * (mx - len(flat))
            data.append({"idx": pad + flat, "val": c})

        file_body = {
            "file_name": "user_polynomial",
            "file_config": {
                "polynomial": {
                    "num_variables": num_vars,
                    "min_degree":    mn,
                    "max_degree":    mx,
                    "data":          data
                }
            }
        }

        # placeholders for live feedback
        upload_ph = st.empty()
        submit_ph = st.empty()
        poll_ph   = st.empty()
        result_ph = st.empty()

        # ────────────────────────────────────────────────────────────
        # Upload
        # ────────────────────────────────────────────────────────────
        t0 = time.time()
        with st.spinner("Uploading polynomial…"):
            resp    = qci.upload_file(file=file_body)
            file_id = resp["file_id"]
        upload_ph.success(f"✔ Uploaded in {time.time()-t0:.1f}s (file_id={file_id})")

        # ────────────────────────────────────────────────────────────
        # Submit
        # ────────────────────────────────────────────────────────────
        t1 = time.time()
        with st.spinner("Submitting job…"):
            job_body    = qci.build_job_body(
                job_type="sample-hamiltonian-integer",
                job_params={
                    "device_type": DeviceType.DIRAC3_QUDIT.value,
                    "num_samples": shots,
                    "num_levels":   [2],
                },
                polynomial_file_id=file_id
            )
            submit_resp = qci.submit_job(job_body=job_body)
            job_id      = submit_resp["job_id"]
        submit_ph.success(f"✔ Submitted in {time.time()-t1:.1f}s (job_id={job_id})")

        # ────────────────────────────────────────────────────────────
        # Poll
        # ────────────────────────────────────────────────────────────
        t2 = time.time()
        with st.spinner("Polling status…"):
            while True:
                status = qci.get_job_status(job_id=job_id)["status"]
                poll_ph.info(f"Status: {status}")
                if status == "COMPLETED":
                    break
                if status in ("FAILED", "CANCELLED"):
                    poll_ph.error(f"Job {job_id} {status}")
                    st.stop()
                time.sleep(1)
        poll_ph.success(f"✔ Completed in {time.time()-t2:.1f}s")

        # ────────────────────────────────────────────────────────────
        # Fetch & Display
        # ────────────────────────────────────────────────────────────
        t3     = time.time()
        result = qci.get_job_results(job_id=job_id)
        result_ph.success(f"✔ Results retrieved in {time.time()-t3:.1f}s")

        # now pull out the real `results` object
        res = result.get("results", {})

        counts    = res.get("counts",    [])
        energies  = res.get("energies",  [])
        solutions = res.get("solutions", [])

        # Best‐energy metric
        if energies:
            best_idx     = int(min(range(len(energies)), key=lambda i: energies[i]))
            best_energy  = energies[best_idx]
            best_solution= solutions[best_idx]
            st.metric("Best Energy",   f"{best_energy:.4f}")
            st.metric("Best Solution", str(best_solution))

        # Energy histogram
        if energies:
            fig = px.histogram(energies, nbins=len(set(energies)), title="Energy Distribution")
            st.plotly_chart(fig, use_container_width=True)

        # Full samples table
        if solutions:
            rows = []
            for sol, ene, cnt in zip(solutions, energies, counts):
                row = {f"x{j}": sol[j] for j in range(len(sol))}
                row.update({"energy": ene, "count": cnt})
                rows.append(row)
            df = pd.DataFrame(rows)
            st.subheader("All Samples")
            st.dataframe(df, use_container_width=True)
            st.download_button(
                "⬇️ Download Samples as CSV",
                df.to_csv(index=False).encode("utf-8"),
                file_name="qci_samples.csv"
            )


with tab4:
    st.header("📡 QCI Data Analysis")
    st.write("Pull a raw bitstream from QCI, compute entropy metrics, apply every extractor (incl. RL), and run NIST.")

    # Pull from QCI
    job_id = st.text_input("▶️ Enter QCI Job ID:", "")
    if not job_id:
        st.info("Enter a QCI job ID above to begin.")
        st.stop()

    try:
        qci_res = qci.get_job_results(job_id=job_id).get("results", {})
        # flatten all solution bit-streams into one long list
        raw_bits = [bit for sol in qci_res.get("solutions", []) for bit in sol]
        st.success(f"✔ Pulled {len(raw_bits)} Qubits from job {job_id}")
    except Exception as e:
        st.error(f"Failed to fetch QCI job: {e}")
        st.stop()

    #  Raw-stream entropy metrics
    n    = len(raw_bits)
    p1   = sum(raw_bits) / n
    p0   = 1 - p1
    sh   = -(p0*np.log2(p0+1e-9) + p1*np.log2(p1+1e-9))
    pmax = max(p0, p1)
    hmin = -np.log2(pmax+1e-9)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Bits", f"{n:,}")
    c2.metric("Shannon Entropy", f"{sh:.4f}")
    c3.metric("Min Entropy", f"{hmin:.4f}")
    st.markdown("---")

    # NIST on Raw
    passed, total, pr = nist_pass_rate(raw_bits)
    st.metric("Raw SP800-22 PassRate", f"{pr:.4f}", delta=f"{passed}/{total}")
    st.markdown("---")

    # Prepare all extractors
    extractors = {
        "Von Neumann": von_neumann,
        "Elias":       elias,
        "Universal Hash": lambda b: (
            universal_hash(b, seed="seed")
            if isinstance(universal_hash(b, seed="seed"), list)
            else ExtractorEnv._bytes_to_bits(universal_hash(b, seed="seed"))
        ),
        "Maurer–Wolf": lambda b: ExtractorEnv._bytes_to_bits(
            maurer_wolf_extractor(
                ExtractorEnv._bits_to_bytes(b),
                seed=b"seed",
                output_len=len(b)//2
            )
        ),
    }

    # Quick Meta-RL agent rollout
    env   = ExtractorEnv(raw_bits, window_size=512)
    agent = QLearningAgent(env)
    for ep in range(200):
        agent.train_one_episode(ep, 200)
    state, done = env.reset(), False
    rl_stream = []
    agent.epsilon = 0.0
    while not done:
        a = int(np.argmax(agent.q_table[agent.discretize(state)]))
        state, _, done, info = env.step(a)
        rl_stream.extend(info["out_bits"])
    extractors["Meta-RL"] = lambda _: rl_stream

    # Gather & display per-extractor metrics
    rows = []
    for name, fn in extractors.items():
        out = fn(raw_bits) or []
        if out:
            # bias & rate
            b1   = compute_bias(out)
            rate = len(out) / n
            # SP800-22
            _, _, pr = nist_pass_rate(out) if len(out) >= min_length_for_sp else (0,0,0.0)
            # Shannon entropy
            p1_o = sum(out) / len(out)
            p0_o = 1 - p1_o
            sh_o   = -(p0_o*np.log2(p0_o+1e-9) + p1_o*np.log2(p1_o+1e-9))
            # min-entropy
            hmin_o = -np.log2(max(p0_o, p1_o) + 1e-9)
        else:
            b1, rate, pr, sh_o, hmin_o = 0.0, 0.0, 0.0, 0.0, 0.0

        rows.append({
            "Extractor":       name,
            "Shannon Entropy": f"{sh_o:.4f}",
            "Min Entropy":     f"{hmin_o:.4f}",
            "Post-Bias":       f"{b1:.4f}",
            "Rate":            f"{rate:.4f}",
            "SP800-22":        f"{pr:.4f}"
        })

    df = pd.DataFrame(rows).set_index("Extractor")
    st.subheader("📊 Extractor Performance on QCI Stream")
    st.dataframe(df, use_container_width=True)
    st.download_button(
        "⬇️ Download QCI-Extractor Metrics as CSV",
        df.to_csv().encode("utf-8"),
        file_name="qci_extractor_metrics.csv"
    )

# ──────────────────────────────────────────────────────────────────────────────
# End of dashboard.py
# ──────────────────────────────────────────────────────────────────────────────
