"""
Digital Communication Playground (DCP) — Simulation Engine
Pure NumPy signal-processing math. This module has zero AI involvement
and zero UI dependency (no Streamlit) — per the architecture, this
computation IS the ground truth for the DCP and AMC playgrounds. The
Sync Hub Protocol trusts whatever this module returns without a
tool-verification loop, so correctness here matters more than in the
AI-facing code.
Public entry point for the FastAPI layer / Hub is `simulate()`.
"""
from __future__ import annotations
from typing import Literal, TypedDict
import numpy as np

Channel = Literal["awgn", "rayleigh", "rician", "narrowband_jamming", "broadband_jamming", "spoofing"]
Modulation = Literal["bpsk", "qpsk", "16qam", "64qam", "bfsk", "gfsk"]

RICIAN_K = 5.0          # Rician K-factor (LOS power / scattered power)
EPSILON = 1e-10         # Avoids divide-by-zero in equalization
BFSK_CARRIER_FREQS = (5, 10)   # (f0, f1) Hz
BFSK_SAMPLE_RATE = 100         # samples per bit
GFSK_BT = 0.5           # GFSK bandwidth-time product
GFSK_SAMPLE_RATE = 16   # samples per symbol for GFSK

class SimResult(TypedDict):
    modulation: str
    channel: str
    snr_db: float
    n_bits: int
    ber: float
    errors: int
    constellation: list          # [[i, q], ...] — empty for BFSK/GFSK
    iq_samples: list             # [[I, Q], ...] for all modulations
    waveform_preview: list        # real-valued preview samples for plotting
    spectrum: dict                # {"freqs": [...], "magnitude_db": [...]}
    eye_diagram: dict             # {"samples_per_symbol": int, "traces": [[...], ...]}


# ---------------------------------------------------------------------------
# Shared channel / noise / equalization (previously duplicated 5x)
# ---------------------------------------------------------------------------
def _generate_channel(n: int, channel: Channel) -> np.ndarray:
    """Complex fading coefficients, one per symbol/sample."""
    if channel == "awgn":
        return np.ones(n, dtype=complex)
    
    if channel == "rayleigh":
        return (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
    
    if channel == "rician":
        los = np.sqrt(RICIAN_K / (RICIAN_K + 1))
        scattered = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
        return los + np.sqrt(1 / (RICIAN_K + 1)) * scattered
    
    # NEW CHANNEL MODELS
    if channel == "narrowband_jamming":
        h = np.ones(n, dtype=complex)
        jamming_ratio = 0.1
        jamming_power_db = 20
        jamming_indices = np.random.choice(n, size=int(n * jamming_ratio), replace=False)
        jamming_signal = np.exp(1j * 2 * np.pi * 0.1 * np.arange(len(jamming_indices)))
        h[jamming_indices] += 10**(jamming_power_db/20) * jamming_signal
        return h
    
    if channel == "broadband_jamming":
        h = np.ones(n, dtype=complex)
        jamming_power_db = 20
        jamming = (np.random.randn(n) + 1j * np.random.randn(n)) / np.sqrt(2)
        jamming_power = 10**(jamming_power_db/20)
        return h + jamming_power * jamming
    
    if channel == "spoofing":
        h = np.ones(n, dtype=complex)
        spoof_strength = 0.5
        spoof_signal = np.exp(1j * 2 * np.pi * np.random.rand(n))
        return (1 - spoof_strength) * h + spoof_strength * spoof_signal
    
    raise ValueError(f"Unknown channel '{channel}'. Must be 'awgn', 'rayleigh', 'rician', 'narrowband_jamming', 'broadband_jamming', or 'spoofing'.")


def _complex_awgn(n: int, snr_db: float, signal_power: float) -> np.ndarray:
    """Complex I/Q noise sized to hit the requested per-symbol SNR."""
    snr_linear = 10 ** (snr_db / 10)
    noise_power = signal_power / snr_linear
    noise_std = np.sqrt(noise_power / 2)
    return noise_std * (np.random.randn(n) + 1j * np.random.randn(n))


def _equalize(received: np.ndarray, h: np.ndarray) -> np.ndarray:
    """Zero-forcing equalization: undo the channel's complex gain."""
    return received * np.conj(h) / (np.abs(h) ** 2 + EPSILON)


def _transmit_symbols(symbols: np.ndarray, snr_db: float, channel: Channel) -> np.ndarray:
    """Apply channel + noise + equalization to a symbol-level signal."""
    n = len(symbols)
    h = _generate_channel(n, channel)
    signal_power = np.mean(np.abs(symbols) ** 2)
    noise = _complex_awgn(n, snr_db, signal_power)
    received = h * symbols + noise
    return _equalize(received, h)


# ---------------------------------------------------------------------------
# Modulation-specific mapping / demapping
# ---------------------------------------------------------------------------
def _map_bpsk(bits: np.ndarray) -> np.ndarray:
    return (2 * bits - 1).astype(complex)


def _demap_bpsk(rx: np.ndarray) -> np.ndarray:
    return (rx.real > 0).astype(int)


def _map_qpsk(bits: np.ndarray) -> np.ndarray:
    b1, b2 = bits[0::2], bits[1::2]
    i = np.where(b1 == 0, 1, -1)
    q = np.where(b2 == 0, 1, -1)
    return (i + 1j * q) / np.sqrt(2)


def _demap_qpsk(rx: np.ndarray, n_bits: int) -> np.ndarray:
    received_bits = np.empty(n_bits, dtype=int)
    received_bits[0::2] = (rx.real < 0).astype(int)
    received_bits[1::2] = (rx.imag < 0).astype(int)
    return received_bits


_QAM16_GRAY = np.array([0, 1, 3, 2])
_QAM16_LEVELS = np.array([-3, -1, 1, 3])
_QAM16_NORM = np.sqrt(10)

_QAM64_GRAY = np.array([0, 1, 3, 2, 6, 7, 5, 4])
_QAM64_LEVELS = np.array([-7, -5, -3, -1, 1, 3, 5, 7])
_QAM64_NORM = np.sqrt(42)


def _map_qam(bits: np.ndarray, bits_per_axis: int, gray: np.ndarray,
             levels: np.ndarray, norm: float) -> np.ndarray:
    bits_per_symbol = 2 * bits_per_axis
    groups = bits.reshape(-1, bits_per_symbol)
    i_bits, q_bits = groups[:, :bits_per_axis], groups[:, bits_per_axis:]
    weights = 2 ** np.arange(bits_per_axis - 1, -1, -1)
    i_bin = i_bits @ weights
    q_bin = q_bits @ weights
    i_val = levels[gray[i_bin]]
    q_val = levels[gray[q_bin]]
    return (i_val + 1j * q_val) / norm


def _demap_qam(rx: np.ndarray, bits_per_axis: int, gray: np.ndarray,
               levels: np.ndarray, norm: float, n_bits: int) -> np.ndarray:
    rx_descaled = rx * norm
    gray_to_binary = np.argsort(gray)
    i_gray = np.argmin(np.abs(rx_descaled.real[:, None] - levels), axis=1)
    q_gray = np.argmin(np.abs(rx_descaled.imag[:, None] - levels), axis=1)
    i_bin = gray_to_binary[i_gray]
    q_bin = gray_to_binary[q_gray]
    bits_per_symbol = 2 * bits_per_axis
    out = np.empty((len(rx), bits_per_symbol), dtype=int)
    for k in range(bits_per_axis):
        out[:, k] = (i_bin >> (bits_per_axis - 1 - k)) & 1
        out[:, bits_per_axis + k] = (q_bin >> (bits_per_axis - 1 - k)) & 1
    return out.reshape(-1)[:n_bits]


# ---------------------------------------------------------------------------
# GFSK-specific functions
# ---------------------------------------------------------------------------
def _gaussian_filter(bt: float, sps: int) -> np.ndarray:
    """Generate Gaussian filter impulse response for GFSK."""
    b = bt / np.sqrt(2 / np.log(2))
    t = np.arange(-3, 3.0, 1.0/sps)
    h_gaussian = np.exp(-2 * np.pi**2 * b**2 * t**2)
    h_gaussian /= np.sum(h_gaussian)
    return h_gaussian


# ---------------------------------------------------------------------------
# GFSK-specific functions (CORRECTED)
# ---------------------------------------------------------------------------
def _gaussian_filter(bt: float, sps: int) -> np.ndarray:
    """Generate Gaussian filter impulse response for GFSK."""
    # Standard deviation in symbol periods
    sigma = np.sqrt(np.log(2)) / (2.0 * np.pi * bt)
    # Time vector from -3 to +3 symbol periods
    t = np.arange(-3 * sps, 3 * sps + 1) / sps
    h_gaussian = np.exp(- (t ** 2) / (2 * sigma ** 2))
    h_gaussian /= np.sum(h_gaussian)  # Normalize area to 1
    return h_gaussian


def _map_gfsk(bits: np.ndarray, bt: float = GFSK_BT, sps: int = GFSK_SAMPLE_RATE) -> np.ndarray:
    """Generate GFSK complex baseband modulated signal."""
    # Frequency deviation: +1 for bit 1, -1 for bit 0
    freq_dev = np.where(bits == 0, -1.0, 1.0)
    
    # Upsample to sample rate
    upsampled = np.repeat(freq_dev, sps)
    
    # Apply Gaussian filter
    h_gauss = _gaussian_filter(bt, sps)
    filtered = np.convolve(upsampled, h_gauss, mode='same')
    
    # Integrate to get phase. 
    # We want a total phase shift of +/- pi/2 (modulation index h=0.5) per bit.
    # Since 'filtered' sums to ~sps over one bit period, we scale by (pi/2) / sps.
    phase = np.cumsum(filtered) * (np.pi / (2.0 * sps))
    
    # Generate COMPLEX baseband signal (critical for differential demodulation)
    modulated = np.exp(1j * phase)
    return modulated


def _demap_gfsk(rx: np.ndarray, sps: int = GFSK_SAMPLE_RATE) -> np.ndarray:
    """Non-coherent demodulation of GFSK using differential phase detection."""
    # Calculate instantaneous phase difference between consecutive complex samples
    phase_diff = np.angle(rx[1:] * np.conj(rx[:-1]))
    
    # Downsample to symbol rate by averaging over the symbol period
    n_symbols = len(phase_diff) // sps
    if n_symbols == 0:
        return np.array([], dtype=int)
        
    phase_diff = phase_diff[:n_symbols * sps]
    phase_diff = phase_diff.reshape(n_symbols, sps).mean(axis=1)
    
    # Decision: positive phase diff -> bit 1, negative -> bit 0
    received_bits = (phase_diff > 0).astype(int)
    return received_bits

# ---------------------------------------------------------------------------
# Per-modulation simulators
# ---------------------------------------------------------------------------
def _sim_symbol_level(bits: np.ndarray, symbols: np.ndarray, snr_db: float,
                     channel: Channel, demap) -> dict:
    equalized = _transmit_symbols(symbols, snr_db, channel)
    received_bits = demap(equalized)
    return {
        "bits": bits,
        "received_bits": received_bits,
        "equalized": equalized,
    }


def simulate_bpsk(snr_db: float, channel: Channel = "awgn", n: int = 100_000) -> dict:
    bits = np.random.randint(0, 2, n)
    symbols = _map_bpsk(bits)
    return _sim_symbol_level(bits, symbols, snr_db, channel, _demap_bpsk)


def simulate_qpsk(snr_db: float, channel: Channel = "awgn", n: int = 100_000) -> dict:
    n -= n % 2
    bits = np.random.randint(0, 2, n)
    symbols = _map_qpsk(bits)
    return _sim_symbol_level(bits, symbols, snr_db, channel,
                            lambda rx: _demap_qpsk(rx, n))


def simulate_16qam(snr_db: float, channel: Channel = "awgn", n: int = 100_000) -> dict:
    n -= n % 4
    bits = np.random.randint(0, 2, n)
    symbols = _map_qam(bits, 2, _QAM16_GRAY, _QAM16_LEVELS, _QAM16_NORM)
    return _sim_symbol_level(bits, symbols, snr_db, channel,
                            lambda rx: _demap_qam(rx, 2, _QAM16_GRAY, _QAM16_LEVELS, _QAM16_NORM, n))


def simulate_64qam(snr_db: float, channel: Channel = "awgn", n: int = 100_000) -> dict:
    n -= n % 6
    bits = np.random.randint(0, 2, n)
    symbols = _map_qam(bits, 3, _QAM64_GRAY, _QAM64_LEVELS, _QAM64_NORM)
    return _sim_symbol_level(bits, symbols, snr_db, channel,
                            lambda rx: _demap_qam(rx, 3, _QAM64_GRAY, _QAM64_LEVELS, _QAM64_NORM, n))


def simulate_bfsk(snr_db: float, channel: Channel = "awgn", n: int = 10_000) -> dict:
    """Non-coherent-style BFSK over a correlation receiver."""
    f0, f1 = BFSK_CARRIER_FREQS
    fs = BFSK_SAMPLE_RATE
    samples_per_bit = fs
    t = np.arange(samples_per_bit) / fs
    
    bits = np.random.randint(0, 2, n)
    waveform0 = np.cos(2 * np.pi * f0 * t)
    waveform1 = np.cos(2 * np.pi * f1 * t)
    
    tx = np.where(bits[:, None] == 0, waveform0, waveform1).reshape(-1).astype(complex)
    
    h = _generate_channel(len(tx), channel)
    signal_power = np.mean(np.abs(tx) ** 2)
    noise = _complex_awgn(len(tx), snr_db, signal_power)
    received = h * tx + noise
    
    equalized_complex = _equalize(received, h)
    equalized = equalized_complex.real
    
    received_bits = np.empty(n, dtype=int)
    for i in range(n):
        seg = equalized[i * samples_per_bit:(i + 1) * samples_per_bit]
        corr0 = np.sum(seg * waveform0)
        corr1 = np.sum(seg * waveform1)
        received_bits[i] = 0 if corr0 > corr1 else 1
    
    return {
        "bits": bits,
        "received_bits": received_bits,
        "equalized": equalized,
        "equalized_complex": equalized_complex,
        "samples_per_bit": samples_per_bit,
    }


def simulate_gfsk(snr_db: float, channel: Channel = "awgn", n: int = 10_000) -> dict:
    """GFSK modulation with non-coherent detection."""
    sps = GFSK_SAMPLE_RATE
    bits = np.random.randint(0, 2, n)
    
    # Modulate
    tx = _map_gfsk(bits)
    
    # Apply channel
    h = _generate_channel(len(tx), channel)
    signal_power = np.mean(np.abs(tx) ** 2)
    noise = _complex_awgn(len(tx), snr_db, signal_power)
    received = h * tx + noise
    
    # Equalize
    equalized_complex = _equalize(received, h)
    
    # Demodulate
    received_bits = _demap_gfsk(equalized_complex, sps)
    
    # Ensure same length
    min_len = min(len(bits), len(received_bits))
    
    return {
        "bits": bits[:min_len],
        "received_bits": received_bits[:min_len],
        "equalized": equalized_complex.real,
        "equalized_complex": equalized_complex,
        "samples_per_bit": sps,
    }


_SIMULATORS = {
    "bpsk": simulate_bpsk,
    "qpsk": simulate_qpsk,
    "16qam": simulate_16qam,
    "64qam": simulate_64qam,
    "bfsk": simulate_bfsk,
    "gfsk": simulate_gfsk,
}

SUPPORTED_MODULATIONS = ["BPSK", "QPSK", "16-QAM", "64-QAM", "BFSK", "GFSK"]
SUPPORTED_CHANNELS = ["AWGN", "Rayleigh", "Rician", "Narrowband Jamming", "Broadband Jamming", "Spoofing"]


# ---------------------------------------------------------------------------
# JSON-serializable result builder
# ---------------------------------------------------------------------------
def _spectrum(waveform_real: np.ndarray, fs: float, max_samples: int = 10_000) -> dict:
    sig = waveform_real[:max_samples]
    spectrum = np.fft.fftshift(np.fft.fft(sig))
    freqs = np.fft.fftshift(np.fft.fftfreq(len(sig), d=1 / fs))
    magnitude_db = 20 * np.log10(np.abs(spectrum) / len(sig) + 1e-12)
    return {"freqs": freqs.tolist(), "magnitude_db": magnitude_db.tolist()}


def _eye_diagram(waveform_real: np.ndarray, samples_per_symbol: int,
                 n_traces: int = 100) -> dict:
    eye_len = 2 * samples_per_symbol
    traces = []
    for i in range(n_traces):
        start = i * samples_per_symbol
        end = start + eye_len
        if end <= len(waveform_real):
            traces.append(waveform_real[start:end].tolist())
    return {"samples_per_symbol": samples_per_symbol, "traces": traces}


def _build_result(modulation: Modulation, channel: Channel, snr_db: float,
                  n_bits: int, raw: dict) -> SimResult:
    is_bfsk = modulation in ["bfsk", "gfsk"]
    samples_per_symbol = raw.get("samples_per_bit", 16)
    equalized = raw["equalized"]
    waveform_real = equalized if is_bfsk else equalized.real
    fs = raw.get("samples_per_bit", 16)
    
    errors = int(np.sum(raw["bits"] != raw["received_bits"]))
    ber = errors / n_bits
    
    iq_source = raw.get("equalized_complex", equalized)
    preview_n = min(3000, len(iq_source))
    iq_samples = np.column_stack(
        [iq_source.real[:preview_n], iq_source.imag[:preview_n]]
    ).tolist()
    
    constellation = [] if is_bfsk else iq_samples
    
    return {
        "modulation": modulation,
        "channel": channel,
        "snr_db": snr_db,
        "n_bits": n_bits,
        "ber": ber,
        "errors": errors,
        "constellation": constellation,
        "iq_samples": iq_samples,
        "waveform_preview": waveform_real[:2000].tolist(),
        "spectrum": _spectrum(waveform_real, fs),
        "eye_diagram": _eye_diagram(waveform_real, samples_per_symbol),
    }


_BITS_PER_SYMBOL = {"bpsk": 1, "qpsk": 2, "16qam": 4, "64qam": 6, "bfsk": 1, "gfsk": 1}


def generate_iq_window(modulation: Modulation, snr_db: float, channel: Channel,
                       window_len: int = 128) -> np.ndarray:
    """Generate one independent (2, window_len) I/Q example for AMC training."""
    modulation = modulation.lower().replace("-", "")
    if modulation not in _SIMULATORS:
        raise ValueError(f"Unknown modulation '{modulation}'. Must be one of {sorted(_SIMULATORS)}.")

    # FIX: Force all modulations to share the same time-domain structure
    sps = 16  # Samples per symbol

    if modulation in ["bfsk", "gfsk"]:
        n_bits = max(1, window_len // sps)
        raw = _SIMULATORS[modulation](snr_db=snr_db, channel=channel, n=n_bits)
        sig = raw["equalized_complex"]
    else:
        bits_per_symbol = _BITS_PER_SYMBOL[modulation]
        n_symbols = window_len // sps
        n_bits = n_symbols * bits_per_symbol
        raw = _SIMULATORS[modulation](snr_db=snr_db, channel=channel, n=n_bits)
        sig = raw["equalized"]
        
        # CRITICAL FIX: Upsample discrete symbols to match FSK time-domain shape
        sig = np.repeat(sig, sps)

    # Ensure exact length
    if len(sig) < window_len:
        sig = np.pad(sig, (0, window_len - len(sig)))
    else:
        sig = sig[:window_len]

    return np.stack([sig.real, sig.imag], axis=0).astype(np.float32)


def simulate(modulation: Modulation, snr_db: float, channel: Channel = "awgn",
             n: int = 100_000) -> SimResult:
    """Single entry point for the FastAPI layer / Sync Hub."""
    modulation = modulation.lower().replace("-", "")
    if modulation not in _SIMULATORS:
        raise ValueError(
            f"Unknown modulation '{modulation}'. Must be one of {sorted(_SIMULATORS)}."
        )
    channel = channel.lower()
    raw = _SIMULATORS[modulation](snr_db=snr_db, channel=channel, n=n)
    return _build_result(modulation, channel, snr_db, len(raw["bits"]), raw)


if __name__ == "__main__":
    # Quick manual sanity check
    result = simulate("qpsk", snr_db=10, channel="rayleigh", n=50_000)
    print(f"QPSK / Rayleigh / 10dB -> BER = {result['ber']:.6g} "
          f"({result['errors']} errors)")
