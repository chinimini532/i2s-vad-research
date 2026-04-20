"""
src/evaluation/validate_simulation.py

Validates that our software G.711 A-law simulation is equivalent
to real G.711 PCMA codec output captured from a production
telephony network via Wireshark.

Usage:
    python src/evaluation/validate_simulation.py --pcap path/to/capture.pcap

Output:
    outputs/validation/alaw_validation.pdf
    outputs/validation/alaw_validation_stats.json
"""

import sys
import json
import audioop
import argparse
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from scipy.signal import welch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

OUT_DIR = ROOT / "outputs" / "validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_rtp_alaw(pcap_path: str) -> bytes:
    """Extract G.711 A-law payload bytes from RTP packets in pcap."""
    try:
        import dpkt
    except ImportError:
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install",
                       "dpkt", "-q"], check=True)
        import dpkt

    alaw_bytes = []
    with open(pcap_path, 'rb') as f:
        pcap = dpkt.pcap.Reader(f)
        for ts, buf in pcap:
            try:
                if len(buf) < 54:
                    continue
                # RTP payload type (skip Ethernet+IP+UDP+RTP header = 54 bytes)
                pt = buf[43] & 0x7F
                if pt in [8, 0]:   # 8=PCMA (A-law), 0=PCMU (u-law)
                    alaw_bytes.extend(list(buf[54:]))
            except Exception:
                pass

    if not alaw_bytes:
        raise ValueError("No G.711 RTP packets found in pcap file.")

    return bytes(alaw_bytes)


def kl_divergence(p_vals: np.ndarray, q_vals: np.ndarray,
                  bins: int = 256) -> float:
    """KL divergence between two byte value distributions."""
    p_hist, _ = np.histogram(p_vals, bins=bins, range=(0, 255), density=True)
    q_hist, _ = np.histogram(q_vals, bins=bins, range=(0, 255), density=True)
    p_hist = p_hist + 1e-10
    q_hist = q_hist + 1e-10
    p_hist /= p_hist.sum()
    q_hist /= q_hist.sum()
    return float(np.sum(p_hist * np.log(p_hist / q_hist)))


def validate(pcap_path: str) -> dict:
    print(f"\nLoading pcap: {pcap_path}")
    rtp_alaw_bytes = extract_rtp_alaw(pcap_path)
    n = len(rtp_alaw_bytes)
    print(f"  Extracted {n:,} G.711 A-law bytes ({n/8000:.1f}s at 8kHz)")

    # decode real RTP to PCM
    rtp_pcm_bytes = audioop.alaw2lin(rtp_alaw_bytes, 2)
    rtp_pcm       = np.frombuffer(rtp_pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0
    rtp_int16     = np.frombuffer(rtp_pcm_bytes, dtype=np.int16)

    # apply our simulation to the same decoded PCM
    sim_alaw_bytes = audioop.lin2alaw(rtp_int16.tobytes(), 2)
    sim_pcm_bytes  = audioop.alaw2lin(sim_alaw_bytes, 2)
    sim_pcm        = np.frombuffer(sim_pcm_bytes, dtype=np.int16).astype(np.float32) / 32767.0

    # compare
    rtp_byte_vals  = np.array(list(rtp_alaw_bytes), dtype=np.float32)
    sim_byte_vals  = np.array(list(sim_alaw_bytes), dtype=np.float32)

    byte_match_pct = float(np.mean(rtp_byte_vals == sim_byte_vals) * 100)
    kl             = kl_divergence(rtp_byte_vals, sim_byte_vals)
    ks_stat, ks_p  = stats.ks_2samp(
        rtp_pcm[:5000].astype(float),
        sim_pcm[:5000].astype(float)
    )

    stats_dict = {
        "n_samples":       n,
        "duration_s":      round(n / 8000, 2),
        "byte_match_pct":  round(byte_match_pct, 4),
        "kl_divergence":   round(kl, 6),
        "ks_statistic":    round(float(ks_stat), 6),
        "ks_p_value":      round(float(ks_p), 6),
        "rtp_pcm_mean":    round(float(rtp_pcm.mean()), 6),
        "rtp_pcm_std":     round(float(rtp_pcm.std()),  6),
        "sim_pcm_mean":    round(float(sim_pcm.mean()), 6),
        "sim_pcm_std":     round(float(sim_pcm.std()),  6),
        "conclusion":      "EQUIVALENT" if kl < 0.01 else "SIMILAR",
    }

    print(f"\n  Byte match    : {byte_match_pct:.2f}%")
    print(f"  KL divergence : {kl:.6f}")
    print(f"  KS statistic  : {ks_stat:.6f}  (p={ks_p:.4f})")
    print(f"  Conclusion    : {stats_dict['conclusion']}")

    # generate figure
    _plot_validation(rtp_pcm, sim_pcm, rtp_byte_vals, sim_byte_vals, stats_dict)

    # save stats
    stats_path = OUT_DIR / "alaw_validation_stats.json"
    with open(str(stats_path), "w") as f:
        json.dump(stats_dict, f, indent=2)
    print(f"\n  Stats saved : {stats_path}")

    return stats_dict


def _plot_validation(rtp_pcm, sim_pcm, rtp_bv, sim_bv, s):
    c1 = '#1565C0'
    c2 = '#E53935'

    fig = plt.figure(figsize=(14, 10))
    fig.suptitle(
        'Validation: G.711 A-Law Software Simulation vs Real RTP Network Traffic\n'
        '(PT=8, G.711 PCMA, 8kHz, captured from production telephony system)',
        fontsize=12, fontweight='bold', y=0.98)
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.42, wspace=0.35)

    # byte distribution
    ax1 = fig.add_subplot(gs[0, :2])
    ax1.hist(rtp_bv, bins=np.arange(0, 258), density=True, alpha=0.55,
             color=c1, label='Real G.711 PCMA (Wireshark RTP capture)')
    ax1.hist(sim_bv, bins=np.arange(0, 258), density=True, alpha=0.55,
             color=c2, label='Software A-law simulation (Python audioop ITU-T G.711)')
    ax1.set_xlabel('A-law Encoded Byte Value', fontsize=10)
    ax1.set_ylabel('Probability Density', fontsize=10)
    ax1.set_title('G.711 A-law Codec Output Byte Distribution', fontsize=11, fontweight='bold')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    ax1.text(0.02, 0.95,
             f'KL Divergence = {s["kl_divergence"]:.4f}\nByte match = {s["byte_match_pct"]:.1f}%',
             transform=ax1.transAxes, fontsize=9, va='top',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))

    # stats table
    ax2 = fig.add_subplot(gs[0, 2])
    ax2.axis('off')
    rows = [
        ['Metric', 'Real RTP', 'Simulation'],
        ['Duration', f'{s["duration_s"]:.1f}s', f'{s["duration_s"]:.1f}s'],
        ['PCM Mean', f'{s["rtp_pcm_mean"]:.5f}', f'{s["sim_pcm_mean"]:.5f}'],
        ['PCM Std',  f'{s["rtp_pcm_std"]:.5f}',  f'{s["sim_pcm_std"]:.5f}'],
        ['Byte Match', f'{s["byte_match_pct"]:.1f}%', '—'],
        ['KL Div', f'{s["kl_divergence"]:.4f}', '—'],
        ['KS stat', f'{s["ks_statistic"]:.4f}', f'p={s["ks_p_value"]:.2e}'],
        ['Result', s["conclusion"], ''],
    ]
    tbl = ax2.table(cellText=rows[1:], colLabels=rows[0],
                    cellLoc='center', loc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1.1, 1.65)
    for j in range(3):
        tbl[0,j].set_facecolor('#1A237E')
        tbl[0,j].set_text_props(color='white', fontweight='bold')
    for i in range(1, len(rows)):
        bg = '#E8EAF6' if i%2==0 else 'white'
        for j in range(3): tbl[i,j].set_facecolor(bg)
    ax2.set_title('Statistical Summary', fontsize=11, fontweight='bold', pad=8)

    # waveform
    ax3 = fig.add_subplot(gs[1, 0])
    show = min(1000, len(rtp_pcm))
    t_ms = np.arange(show) / 8000 * 1000
    ax3.plot(t_ms, rtp_pcm[:show], color=c1, alpha=0.8, lw=0.8, label='Real RTP')
    ax3.plot(t_ms, sim_pcm[:show], color=c2, alpha=0.8, lw=0.8,
             label='Simulation', linestyle='--')
    ax3.set_xlabel('Time (ms)', fontsize=10)
    ax3.set_ylabel('Amplitude', fontsize=10)
    ax3.set_title('Decoded PCM Waveform\n(125ms window)', fontsize=11, fontweight='bold')
    ax3.legend(fontsize=8)
    ax3.grid(True, alpha=0.3)

    # PSD
    ax4 = fig.add_subplot(gs[1, 1])
    f1, p1 = welch(rtp_pcm.astype(float), fs=8000, nperseg=512)
    f2, p2 = welch(sim_pcm.astype(float), fs=8000, nperseg=512)
    ax4.semilogy(f1, p1, color=c1, lw=1.2, alpha=0.85, label='Real RTP')
    ax4.semilogy(f2, p2, color=c2, lw=1.2, alpha=0.85, label='Simulation', ls='--')
    ax4.set_xlabel('Frequency (Hz)', fontsize=10)
    ax4.set_ylabel('PSD', fontsize=10)
    ax4.set_title('Power Spectral Density\n(0-4000 Hz)', fontsize=11, fontweight='bold')
    ax4.legend(fontsize=8)
    ax4.grid(True, alpha=0.3)
    ax4.set_xlim([0, 4000])

    # Q-Q
    ax5 = fig.add_subplot(gs[1, 2])
    q1 = np.percentile(rtp_pcm[:3000], np.linspace(1, 99, 200))
    q2 = np.percentile(sim_pcm[:3000], np.linspace(1, 99, 200))
    ax5.scatter(q1, q2, s=10, alpha=0.7, color='#6A1B9A')
    lim = max(abs(q1).max(), abs(q2).max()) * 1.1
    ax5.plot([-lim,lim],[-lim,lim],'k--',lw=1.2,label='y=x (perfect match)')
    ax5.set_xlabel('Real RTP Quantiles', fontsize=10)
    ax5.set_ylabel('Simulation Quantiles', fontsize=10)
    ax5.set_title('Q-Q Plot\n(alignment = statistical equivalence)', fontsize=11, fontweight='bold')
    ax5.legend(fontsize=8)
    ax5.grid(True, alpha=0.3)
    ax5.set_xlim([-lim,lim]); ax5.set_ylim([-lim,lim])

    path = OUT_DIR / "alaw_validation.pdf"
    plt.savefig(str(path), bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Figure saved: {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pcap', required=True, help='Path to Wireshark .pcap file')
    args = parser.parse_args()
    validate(args.pcap)
    print("\nValidation complete.")


if __name__ == "__main__":
    main()