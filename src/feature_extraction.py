
import numpy as np
from scipy.signal import welch


def extract_advanced_features(signal, prefix):
    signal = np.asarray(signal, dtype=np.float64).ravel()
    signal = signal[np.isfinite(signal)]

    if len(signal) < 16:
        return {}

    centered = signal - np.mean(signal)
    eps = 1e-12
    abs_signal = np.abs(centered)

    rms = np.sqrt(np.mean(centered ** 2))
    mean_abs = np.mean(abs_signal)
    peak = np.max(abs_signal)

    features = {
        f"{prefix}_crest_factor": float(peak / (rms + eps)),
        f"{prefix}_shape_factor": float(rms / (mean_abs + eps)),
        f"{prefix}_impulse_factor": float(peak / (mean_abs + eps)),
        f"{prefix}_peak_amplitude": float(peak),
        f"{prefix}_variance": float(np.var(centered)),
        f"{prefix}_zero_crossing_rate": float(
            np.mean(np.diff(np.signbit(centered)) != 0)
        ),
    }

    frequencies, power = welch(
        centered, fs=1.0, nperseg=min(2048, len(centered))
    )
    power = np.maximum(power, 0)
    total_power = np.sum(power)

    if total_power <= eps:
        features.update({
            f"{prefix}_spectral_centroid": 0.0,
            f"{prefix}_spectral_entropy": 0.0,
            f"{prefix}_spectral_flatness": 0.0,
            f"{prefix}_spectral_high_band_ratio": 0.0,
        })
        return features

    probabilities = power / total_power
    nonzero = probabilities > 0

    centroid = np.sum(frequencies * probabilities)
    entropy = (
        -np.sum(probabilities[nonzero] * np.log(probabilities[nonzero]))
        / np.log(len(probabilities))
    )
    flatness = (
        np.exp(np.mean(np.log(power + eps)))
        / (np.mean(power) + eps)
    )

    high_band = frequencies >= 0.25
    high_ratio = np.sum(power[high_band]) / total_power

    features.update({
        f"{prefix}_spectral_centroid": float(centroid),
        f"{prefix}_spectral_entropy": float(entropy),
        f"{prefix}_spectral_flatness": float(flatness),
        f"{prefix}_spectral_high_band_ratio": float(high_ratio),
    })

    return features
