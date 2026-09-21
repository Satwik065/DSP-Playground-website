# DSP Signal & Verification Playground(v1)

This is a website which aims to demonstrate DSP visualizations of various modulation schemes and adversial channels, along with an ambitious addition of Digital Verfication in the future,  it will have two modes, open and restricted with AI and user having different level of control in generation of UVM testbench components. The AI used in the website not just the DV playground will have the protocol Synchronization Hub in the backend as a security and verification layer. It is open to public at zenodo, linked below.

Sync Hub Protocol: https://doi.org/10.5281/zenodo.22863926

## Features
- **DCP** – Digital Communication Playground (BPSK, QPSK, 16-QAM, 64-QAM, BFSK, GFSK modulation schemes with AWGN, Rician, Rayleigh, narrowband jamming, broadband jamming and spoofing channels)
- **AMC** – Automatic Modulation Classification
- **DV** – Digital Verification (slang → Verilator)(under development)

## Stack
- FastAPI backend
- NumPy ground-truth DSP engine
- Ollama (qwen2.5-coder:3b) for AI narration
- slang + Verilator for RTL verification

## Run locally
pip install -r requirements.txt
uvicorn main:app --reload --port 8000

Then open http://localhost:8000

run the index.html file in the main root of dsp_sync_hub(1) folder
