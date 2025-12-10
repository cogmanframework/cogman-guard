import numpy as np


class TextSignalPhysics:
    """แปลงข้อความเป็นสัญญาณ"""

    def __init__(self, sampling_rate=1000, bit_duration=0.01):
        self.sampling_rate = sampling_rate
        self.bit_duration = bit_duration
        self.samples_per_bit = int(sampling_rate * bit_duration)

    def text_to_signal(self, text: str) -> np.ndarray:
        """แปลงข้อความเป็นสัญญาณ"""
        bits = ''.join(format(ord(c), '08b') for c in text)

        # 4-level encoding
        voltages = []
        for i in range(0, len(bits), 2):
            pair = bits[i:i + 2].ljust(2, '0')
            if pair == '00':
                voltages.append(0.0)
            elif pair == '01':
                voltages.append(1.67)
            elif pair == '10':
                voltages.append(3.33)
            elif pair == '11':
                voltages.append(5.0)

        voltages = np.array(voltages)

        # สร้างสัญญาณ
        total_samples = len(voltages) * self.samples_per_bit
        t = np.linspace(0, len(voltages) * self.bit_duration, total_samples)

        signal = np.zeros_like(t)
        for i, voltage in enumerate(voltages):
            start_idx = i * self.samples_per_bit
            end_idx = (i + 1) * self.samples_per_bit

            freq = 100 + (voltage / 5.0) * 900
            phase = (i * np.pi) / (len(voltages) + 1)

            time_segment = t[start_idx:end_idx]
            segment_wave = np.sin(2 * np.pi * freq * time_segment + phase)

            amplitude = 0.1 + (voltage / 5.0) * 0.4
            signal[start_idx:end_idx] = segment_wave * amplitude

        return signal