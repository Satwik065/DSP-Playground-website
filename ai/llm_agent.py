"""
AI narration layer over the deterministic DSP ground truth.
The model never computes anything — it only explains numbers that the
NumPy engine already produced. Prompts are deliberately fact-heavy and
constraint-heavy because the local model is small (qwen2.5-coder:3b):
give it the truth, tell it exactly what not to do.
"""
import json
import requests
from typing import Dict, List, Optional

SYSTEM_RULES = (
    "You are the narration layer of a deterministic DSP simulation platform. "
    "Every number in the context below was computed by a verified NumPy ground-truth engine. "
    "Hard rules: "
    "(1) Never invent numbers, formulas, or measurements that are not in the context. "
    "(2) The platform SNR range is -20 to +20 dB; every recommendation must stay inside it. "
    "(3) Write 4-6 short, precise sentences for a communications-engineering reader. "
    "You may use **bold** and '- ' bullet lists; never use '#' headers or code blocks. "
    "(4) If the measured BER is 0, say the link was error-free over the simulated bits; "
    "do not call the system 'perfect' and do not suggest raising SNR further when already high."
)

_MODULATION_FACTS = {
    "bpsk":  "BPSK maps 1 bit/symbol onto two antipodal points (+/-1 on the I axis); the most robust scheme here.",
    "qpsk":  "QPSK maps 2 bits/symbol onto four phase points (45/135/225/315 degrees); same per-bit robustness as BPSK at double the rate.",
    "16qam": "16-QAM maps 4 bits/symbol onto a 4x4 square grid (levels +/-1, +/-3 on each axis); needs roughly 4-5 dB more SNR than QPSK for the same BER.",
    "64qam": "64-QAM maps 6 bits/symbol onto an 8x8 square grid; the dense packing makes it very sensitive to noise and fading.",
    "bfsk":  "BFSK maps 1 bit/symbol onto one of two carrier frequencies, detected by a correlation receiver; power-robust but bandwidth-inefficient, no discrete constellation.",
    "gfsk":  "GFSK is BFSK with Gaussian-filtered frequency pulses (BT=0.5), giving a compact spectrum (Bluetooth-style); constant envelope, no discrete constellation.",
}

_CHANNEL_FACTS = {
    "awgn":               "AWGN adds white Gaussian noise only (channel gain = 1), so all constellation spread is pure noise.",
    "rayleigh":           "Rayleigh fading multiplies each symbol by a complex Gaussian gain (no line-of-sight); deep fades cause burst errors and a much slower BER decay with SNR.",
    "rician":             "Rician fading (K=5) mixes a strong line-of-sight path with scattered multipath; performance sits between AWGN and Rayleigh.",
    "narrowband_jamming": "Narrowband jamming corrupts ~10% of the symbols with a tone 20 dB above the signal; errors concentrate on the jammed symbols.",
    "broadband_jamming":  "Broadband jamming adds wideband noise 20 dB above signal power, effectively collapsing the SNR seen by the receiver.",
    "spoofing":           "Spoofing blends 50% of the signal with a fake unit-magnitude constellation, pulling symbols away from their true positions.",
}

_IDEAL_SHAPE = {
    "bpsk":  "two clean dots on the I axis at +1 and -1",
    "qpsk":  "four clean dots on a circle at 45/135/225/315 degrees",
    "16qam": "a clean 4x4 square grid",
    "64qam": "a clean 8x8 square grid",
    "bfsk":  "two dense clusters after frequency discrimination (FSK has no discrete I/Q constellation)",
    "gfsk":  "a constant-envelope phase trajectory (no discrete I/Q constellation)",
}

# Approximate AWGN SNR for BER ~ 1e-3 (reference only; fading/jamming need more).
_AWGN_KNEE_DB = {"bpsk": 7, "qpsk": 7, "16qam": 14, "64qam": 18, "bfsk": 10, "gfsk": 10}


def _ber_regime(ber: float) -> str:
    if ber == 0:    return "error-free over the simulated bits"
    if ber < 1e-5:  return "excellent, far below the common 1e-3 link target"
    if ber < 1e-3:  return "good, at or below the common 1e-3 link target"
    if ber < 1e-2:  return "marginal, above typical link targets (FEC would be needed)"
    return "poor, the link is effectively unusable at this SNR"


class AIAgentEngine:
    def __init__(self, ollama_model: str = "qwen2.5-coder:3b"):
        self.model = ollama_model
        self.ollama_url = "http://localhost:11434/api/generate"
        print(f"🤖 AI Engine initialized with model: {self.model}")

    # ------------------------------------------------------------
    def _narrate(self, context: str, task: str, fallback: str) -> Dict[str, str]:
        """Send a fact-grounded prompt to Ollama; deterministic fallback if offline."""
        prompt = f"{SYSTEM_RULES}\n\nGROUND-TRUTH CONTEXT (verified, do not alter):\n{context}\n\nTASK:\n{task}"
        try:
            response = requests.post(
                self.ollama_url,
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.2, "top_p": 0.9, "num_predict": 400},
                },
                timeout=90,
            )
            if response.status_code == 200:
                text = response.json().get("response", "").strip()
                if text:
                    return {"source": f"Ollama ({self.model})", "response": text}
        except Exception as e:
            print(f"⚠️ Ollama request failed: {e}")
        return {"source": "Deterministic Heuristic Engine (Ollama offline)", "response": fallback}

    # ------------------------------------------------------------
    def ask_ai_explanation(self, prompt: str, dsp_context: Dict) -> Dict[str, str]:
        """Free-form question over ground-truth data."""
        context = json.dumps(dsp_context, default=str)
        fallback = (
            f"Question received: \"{prompt}\". "
            f"Ground truth: {context}. "
            f"Ollama ('{self.model}') is offline - run 'ollama run {self.model}' for live AI narration."
        )
        return self._narrate(context, prompt, fallback)

    # ------------------------------------------------------------
    def analyze_snr_impact(self, modulation: str, channel: str,
                           snr_values: List[float], ber_values: List[float]) -> Dict[str, str]:
        """AI narrates the SNR-vs-BER waterfall produced by the ground-truth sweep."""
        pairs = "; ".join(f"{snr} dB -> {ber:.2e}" for snr, ber in zip(snr_values, ber_values))
        threshold = next((s for s, b in zip(snr_values, ber_values) if b < 1e-3), None)
        ch_key = channel.lower().replace(" ", "_")
        context = (
            f"- Modulation: {modulation} — {_MODULATION_FACTS.get(modulation.lower().replace('-', ''), '')}\n"
            f"- Channel: {ch_key} — {_CHANNEL_FACTS.get(ch_key, '')}\n"
            f"- Sweep: {min(snr_values)} to {max(snr_values)} dB (platform range -20..+20 dB)\n"
            f"- Measured BER per SNR: {pairs}\n"
            f"- First SNR below the 1e-3 target: {threshold if threshold is not None else 'not reached in range'}"
        )
        task = (
            "Summarize this BER waterfall in 4-6 sentences: where the knee is, how steep the decay is, "
            "how this channel shapes it compared to AWGN, and one recommended operating SNR inside -20..+20 dB."
        )
        if threshold is not None:
            fb = (
                f"Sweeping {modulation} over {ch_key} from {min(snr_values)} to {max(snr_values)} dB shows the "
                f"classic BER waterfall, from {ber_values[0]:.2e} down to {ber_values[-1]:.2e}. The 1e-3 target is "
                f"first met at {threshold} dB, so operating a few dB above that gives comfortable margin. "
                f"(Ollama '{self.model}' offline - run 'ollama run {self.model}' for live narration.)"
            )
        else:
            fb = (
                f"Over {ch_key}, {modulation} never reaches the 1e-3 BER target between {min(snr_values)} and "
                f"{max(snr_values)} dB (best: {ber_values[-1]:.2e}); the channel is too hostile or the modulation "
                f"too dense for this range. (Ollama '{self.model}' offline - run 'ollama run {self.model}' for live narration.)"
            )
        return self._narrate(context, task, fb)

    # ------------------------------------------------------------
    def explain_constellation_quality(self, modulation: str, snr_db: float, channel: str,
                                      ber: float, errors: Optional[int] = None,
                                      n_bits: Optional[int] = None) -> Dict[str, str]:
        """AI narrates constellation quality, grounded in verified facts."""
        mod_key = modulation.lower().replace("-", "")
        ch_key = channel.lower().replace(" ", "_")
        knee = _AWGN_KNEE_DB.get(mod_key)
        bits_line = (f"{errors} errors over {n_bits} bits"
                     if errors is not None and n_bits else f"BER {ber:.3e}")
        context = (
            f"- Modulation: {modulation} — {_MODULATION_FACTS.get(mod_key, '')}\n"
            f"- Ideal shape: {_IDEAL_SHAPE.get(mod_key, 'modulation-specific point set')}\n"
            f"- Channel: {ch_key} — {_CHANNEL_FACTS.get(ch_key, '')}\n"
            f"- SNR: {snr_db:.1f} dB (platform range -20..+20 dB)\n"
            f"- Measured BER: {ber:.3e} ({bits_line}) — {_ber_regime(ber)}\n"
            f"- Reference: {modulation} needs roughly {knee} dB over AWGN for BER~1e-3; fading/jamming/spoofing need several dB more."
        )
        task = (
            "Explain in 4-6 sentences: (1) the ideal constellation shape; (2) how this channel at this SNR "
            "produces the observed symbol spread; (3) whether the measured BER meets the 1e-3 link target; "
            "(4) one concrete recommendation inside -20..+20 dB (raise SNR, lower it if there is excess margin, or keep as is)."
        )
        fallback = (
            f"{modulation} over {ch_key} at {snr_db:.1f} dB measured BER {ber:.3e} ({_ber_regime(ber)}). "
            f"Ideal shape: {_IDEAL_SHAPE.get(mod_key, 'see docs')}. "
            f"Ollama ('{self.model}') is offline - run 'ollama run {self.model}' for live AI narration."
        )
        return self._narrate(context, task, fallback)