import io
import base64
import numpy as np
import matplotlib
matplotlib.use('Agg')  # No GUI, just saves to bytes
import matplotlib.pyplot as plt

def fig_to_base64(fig):
    """Convert a matplotlib figure to base64 string for HTML embedding."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=100)
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return img_base64

def plot_constellation(iq_complex, title="Constellation"):
    """Plot I/Q constellation diagram. Expects a 1D complex array."""
    fig, ax = plt.subplots(figsize=(5, 5))
    # Sample down to avoid overplotting (max 5000 points for clarity)
    step = max(1, len(iq_complex) // 5000)
    samples = iq_complex[::step]
    
    ax.scatter(samples.real, samples.imag, s=1, alpha=0.6, color='cyan')
    ax.axhline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.axvline(0, color='white', linewidth=0.5, linestyle='--', alpha=0.3)
    ax.set_title(title, color='white')
    ax.set_xlabel('In-Phase (I)', color='gray')
    ax.set_ylabel('Quadrature (Q)', color='gray')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.tick_params(colors='gray')
    return fig_to_base64(fig)

def plot_spectrum(waveform, title="Spectrum"):
    """Plot power spectral density using FFT. Expects a 1D real/complex array."""
    fig, ax = plt.subplots(figsize=(6, 3))
    n = len(waveform)
    fft_vals = np.fft.fft(waveform)
    fft_shifted = np.fft.fftshift(fft_vals)
    power = np.abs(fft_shifted) ** 2
    freq = np.fft.fftshift(np.fft.fftfreq(n, d=1))
    # Take log for dB scale
    power_db = 10 * np.log10(power + 1e-12)
    
    ax.plot(freq, power_db, color='orange', linewidth=0.8)
    ax.set_title(title, color='white')
    ax.set_xlabel('Frequency (Normalized)', color='gray')
    ax.set_ylabel('Power (dB)', color='gray')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.tick_params(colors='gray')
    return fig_to_base64(fig)

def plot_eye_diagram(waveform, samples_per_symbol=16, title="Eye Diagram"):
    """Generate an eye diagram from the real part of the signal. Expects 1D real array."""
    fig, ax = plt.subplots(figsize=(6, 3.5))
    i_signal = np.real(waveform)
    
    # We want to slice the signal into segments of exactly 2 symbol periods.
    seg_len = 2 * samples_per_symbol  # This is 32 if samples_per_symbol=16
    total_len = len(i_signal)
    
    if total_len < seg_len * 2:
        ax.text(0.5, 0.5, "Not enough samples for eye diagram", 
                color='gray', ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title, color='white')
        ax.set_facecolor('#1e1e1e')
        fig.patch.set_facecolor('#1e1e1e')
        return fig_to_base64(fig)
        
    # Drop any trailing samples that don't fit into a full segment.
    usable_len = (total_len // seg_len) * seg_len
    i_signal = i_signal[:usable_len]
    
    num_segments = len(i_signal) // seg_len
    eye_matrix = i_signal.reshape(num_segments, seg_len)
    
    # Cap at 100 traces so it doesn't get too crowded.
    max_traces = min(num_segments, 100)
    time_base = np.arange(seg_len)
    
    for row in range(max_traces):
        ax.plot(time_base, eye_matrix[row, :], color='green', alpha=0.15, linewidth=0.8)
        
    ax.set_title(title, color='white')
    ax.set_xlabel('Sample Index (2 Symbol Periods)', color='gray')
    ax.set_ylabel('Amplitude', color='gray')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.tick_params(colors='gray')
    return fig_to_base64(fig)

def plot_waveform(waveform, title="Time Domain Waveform"):
    """Plot time-domain waveform. Expects 1D real array."""
    fig, ax = plt.subplots(figsize=(8, 3))
    
    # Sample down if too many points
    max_samples = 2000
    step = max(1, len(waveform) // max_samples)
    samples = waveform[::step]
    time_axis = np.arange(len(samples))
    
    ax.plot(time_axis, samples, color='cyan', linewidth=0.8)
    ax.set_title(title, color='white')
    ax.set_xlabel('Sample Index', color='gray')
    ax.set_ylabel('Amplitude', color='gray')
    ax.grid(True, alpha=0.2)
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.tick_params(colors='gray')
    return fig_to_base64(fig)

def plot_ber_curve(modulation, channel, snr_range, n_per_point=2000):
    """Generate BER vs SNR curve by running multiple simulations."""
    from dcp.digital_comm import simulate  # Import here to avoid circular import
    
    fig, ax = plt.subplots(figsize=(6, 4))
    ber_values = []
    
    for snr in snr_range:
        try:
            res = simulate(modulation=modulation, snr_db=snr, channel=channel, n=n_per_point)
            # Avoid log(0)
            ber_values.append(max(res['ber'], 1e-7))
        except Exception:
            ber_values.append(0.5)  # Worst case fallback
            
    ax.semilogy(snr_range, ber_values, 'o-', color='cyan', linewidth=2, markersize=4)
    ax.set_title(f'BER vs SNR ({modulation.upper()}, {channel.upper()})', color='white')
    ax.set_xlabel('SNR (dB)', color='gray')
    ax.set_ylabel('Bit Error Rate (BER)', color='gray')
    ax.grid(True, alpha=0.3, which='both')
    ax.set_facecolor('#1e1e1e')
    fig.patch.set_facecolor('#1e1e1e')
    for spine in ax.spines.values():
        spine.set_color('gray')
    ax.tick_params(colors='gray')
    ax.set_ylim(1e-7, 1)
    return fig_to_base64(fig)
def generate_ber_table(modulation, channel, snr_range, n_per_point=10000):
    """Generate BER data table for multiple SNR values."""
    from dcp.digital_comm import simulate
    
    table_data = []
    for snr in snr_range:
        try:
            res = simulate(modulation=modulation, snr_db=snr, channel=channel, n=n_per_point)
            ber = res['ber']
            errors = res['errors']
            n_bits = res['n_bits']
            
            table_data.append({
                'SNR_dB': snr,
                'BER': ber,
                'BER_scientific': f"{ber:.2e}",
                'Errors': errors,
                'Total_Bits': n_bits,
                'Error_Rate': f"{errors}/{n_bits}"
            })
        except Exception as e:
            table_data.append({
                'SNR_dB': snr,
                'BER': 1.0,
                'BER_scientific': "1.00e+0",
                'Errors': n_per_point,
                'Total_Bits': n_per_point,
                'Error_Rate': f"{n_per_point}/{n_per_point}"
            })
    
    return table_data

def ber_table_to_html(table_data):
    """Convert BER table data to HTML."""
    html = """
    <table class="ber-table">
        <thead>
            <tr>
                <th>SNR (dB)</th>
                <th>BER</th>
                <th>BER (Scientific)</th>
                <th>Errors</th>
                <th>Total Bits</th>
                <th>Error Rate</th>
            </tr>
        </thead>
        <tbody>
    """
    
    for row in table_data:
        html += f"""
            <tr>
                <td>{row['SNR_dB']}</td>
                <td>{row['BER']:.6f}</td>
                <td>{row['BER_scientific']}</td>
                <td>{row['Errors']}</td>
                <td>{row['Total_Bits']}</td>
                <td>{row['Error_Rate']}</td>
            </tr>
        """
    
    html += """
        </tbody>
    </table>
    """
    return html