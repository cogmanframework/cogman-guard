import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from scipy import signal, fft, stats
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import plotly.graph_objects as go
import plotly.express as px
from typing import Dict, List, Tuple, Optional, Union
try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False


class EmbeddingQualityInspector:
    """เครื่องมือวิเคราะห์คุณภาพและโครงสร้างของ embedding vectors"""

    def __init__(self, 
                 sampling_rate: float = 1000,
                 fs_method: str = 'fft',
                 device: str = None):
        """
        Args:
            sampling_rate: อัตราสุ่มตัวอย่างสำหรับแปลง embedding เป็นสัญญาณ
            fs_method: วิธีแปลง embedding เป็นสัญญาณ ('fft', 'pca', 'direct')
            device: device สำหรับ torch
        """
        self.sampling_rate = sampling_rate
        self.fs_method = fs_method
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # โหลดโมเดล reference (ถ้ามี)
        self.reference_embeddings = {}
        
    def embedding_to_signal(self, embedding: Union[torch.Tensor, np.ndarray], 
                           method: str = None) -> np.ndarray:
        """
        แปลง embedding เป็นสัญญาณ 1D
        """
        if method is None:
            method = self.fs_method
        
        # แปลงเป็น numpy ถ้าเป็น tensor
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.detach().cpu().numpy()
        
        if len(embedding.shape) > 2:
            # ถ้าเป็น batch, ใช้ embedding แรก
            embedding = embedding[0]
        
        # Flatten ถ้ามีหลายมิติ
        if len(embedding.shape) > 1:
            flat_embed = embedding.flatten()
        else:
            flat_embed = embedding
        
        if method == 'direct':
            # ใช้ embedding โดยตรง
            signal_wave = flat_embed
            
        elif method == 'fft':
            # ใช้ FFT magnitude
            fft_result = fft.fft(flat_embed)
            signal_wave = np.abs(fft_result)
            
        elif method == 'pca':
            # ลดมิติด้วย PCA
            if len(embedding.shape) > 1:
                pca = PCA(n_components=1)
                signal_wave = pca.fit_transform(embedding).flatten()
            else:
                signal_wave = flat_embed
                
        elif method == 'random_projection':
            # Random projection
            n_components = min(100, len(flat_embed))
            proj_matrix = np.random.randn(n_components, len(flat_embed))
            signal_wave = proj_matrix @ flat_embed
            
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # Normalize
        if np.std(signal_wave) > 0:
            signal_wave = (signal_wave - np.mean(signal_wave)) / np.std(signal_wave)
        
        return signal_wave
    
    def analyze_embedding(self, 
                         embedding: Union[torch.Tensor, np.ndarray],
                         original_text: str = None) -> Dict:
        """
        วิเคราะห์คุณภาพและโครงสร้างของ embedding
        
        Returns:
            Dict with quality metrics:
            - information_strength (I): ความหนาแน่นของข้อมูลที่ใช้จริง
            - signal_quality (S): เสถียรภาพและความเรียบของ representation
            - distribution_entropy (H): ความกระจายของค่า embedding
            - embedding_quality_index (EQI): คะแนนคุณภาพโดยรวม
        """
        return self._calculate_embedding_quality(embedding, original_text)
    
    def _calculate_embedding_quality(self, 
                                    embedding: Union[torch.Tensor, np.ndarray],
                                    original_text: str = None) -> Dict:
        """
        คำนวณคุณภาพของ embedding (internal method)
        """
        # Backward compatibility alias
        return self.calculate_embedding_physics(embedding, original_text)
    
    def calculate_embedding_physics(self, 
                                   embedding: Union[torch.Tensor, np.ndarray],
                                   original_text: str = None) -> Dict:
        """
        คำนวณคุณภาพของ embedding (backward compatibility)
        """
        
        # แปลง embedding เป็นสัญญาณ
        signal_wave = self.embedding_to_signal(embedding)
        
        # 1. Information (I) - effective dimensions
        # ใช้ SVD เพื่อหาค่า singular values
        if isinstance(embedding, torch.Tensor):
            emb_tensor = embedding
        else:
            emb_tensor = torch.tensor(embedding)
        
        if len(emb_tensor.shape) > 1:
            # Matrix embedding
            U, S, V = torch.svd(emb_tensor)
            # Effective rank (จำนวน singular values ที่มีนัยสำคัญ)
            singular_sum = torch.sum(S)
            if singular_sum > 0:
                cumulative = torch.cumsum(S, dim=0)
                mask = cumulative / singular_sum < 0.95  # 95% ของพลังงาน
                I = torch.sum(mask).item() + 1
            else:
                I = 1
        else:
            # Vector embedding
            I = len(emb_tensor)
        
        # Extract values early for circuit calculations
        if isinstance(emb_tensor, torch.Tensor):
            values = emb_tensor.flatten().cpu().numpy()
        else:
            values = emb_tensor.flatten()
        
        # 2. Power (P) - energy of the signal
        # ใช้ทั้ง signal energy และ circuit energy (ถ้า enable)
        P_signal = np.mean(signal_wave**2) if len(signal_wave) > 0 else 0
        
        # Circuit-based energy (optional enhancement)
        P_circuit = self._calculate_circuit_energy(values) if len(values) > 1 else 0
        
        # รวม energy (weighted average)
        P = 0.7 * P_signal + 0.3 * P_circuit if P_circuit > 0 else P_signal
        
        # 3. Signal Quality (S) - วัดคุณภาพ embedding จริงๆ
        # Good embedding: กระจายตัวดี, ไม่ collapse, ไม่ sparse, ไม่มี outlier
        # Bad embedding: collapse (all same), sparse (mostly zeros), มี outlier
        
        if len(values) > 1:
            # Component 1: Distribution Uniformity (ไม่ collapse)
            # Good: values กระจายหลายค่า (unique_ratio สูง)
            # Bad: values ซ้ำกันมาก (unique_ratio ต่ำ)
            unique_values = len(np.unique(np.round(values, 4)))
            S_uniformity = min(1.0, unique_values / len(values))
            
            # Component 2: Active Dimensions (ไม่ sparse)
            # Good: ส่วนใหญ่ของ dimensions มีค่า (active_ratio สูง)
            # Bad: mostly zeros (active_ratio ต่ำ)
            active_dims = np.sum(np.abs(values) > 1e-6)
            S_active = active_dims / len(values)
            
            # Component 3: Distribution Shape (ไม่มี outlier มาก)
            # Good: kurtosis ใกล้ 0 (normal-like)
            # Bad: kurtosis สูง (มี outlier)
            kurt = stats.kurtosis(values)
            S_shape = 1 / (1 + abs(kurt) / 3)  # Penalty for high kurtosis
            
            # Component 4: Variance Consistency (มี variance พอเหมาะ)
            # Good: std > 0 และไม่สูงเกินไป
            # Bad: std = 0 (collapsed) หรือสูงมาก (unstable)
            std_val = np.std(values)
            mean_val = np.mean(np.abs(values))
            if mean_val > 0:
                cv = std_val / (mean_val + 1e-8)  # Coefficient of variation
                S_variance = np.exp(-abs(cv - 1) / 2)  # Optimal CV ~ 1
            else:
                S_variance = 0.1  # Very low if mean is zero
            
            # Combine components (weighted average)
            S = (0.30 * S_uniformity + 
                 0.25 * S_active + 
                 0.25 * S_shape + 
                 0.20 * S_variance)
            
            # Ensure S is in [0, 1]
            S = np.clip(S, 0, 1)
            
            # Store components for debugging
            S_components_detail = {
                'uniformity': S_uniformity,
                'active_ratio': S_active,
                'shape_quality': S_shape,
                'variance_quality': S_variance
            }
        else:
            S = 0.5
            S_components_detail = {}
        
        # 4. Entropy (H) - จาก distribution ของ embedding values
        # ใช้ normalized Shannon entropy (0-1)
        hist, _ = np.histogram(values, bins=50)
        hist = hist[hist > 0]  # Remove zero bins
        if len(hist) > 1:
            # Normalize to probability
            prob = hist / np.sum(hist)
            # Shannon entropy
            entropy = -np.sum(prob * np.log2(prob))
            # Normalize to 0-1 range
            max_entropy = np.log2(len(hist))
            H = entropy / max_entropy if max_entropy > 0 else 0
            H = np.clip(H, 0, 1)  # Ensure 0-1 range
        else:
            H = 0
        
        # 5. คำนวณ Embedding Quality Index (EQI)
        # สูตรใหม่: EQI สูง = ดี, EQI ต่ำ = ไม่ดี
        # - Signal Quality สูง = ดี
        # - Entropy ปานกลาง-สูง = ดี (กระจายตัวดี)
        # - Entropy ต่ำมาก = ไม่ดี (collapsed หรือ all zeros)
        
        # Compute information ratio
        total_dims = len(values)
        info_ratio = I / total_dims if total_dims > 0 else 0
        
        # Entropy penalty for extreme values (too low or too high)
        # Optimal entropy is around 0.6-0.8
        H_score = 1 - abs(H - 0.7) * 2  # Peak at H=0.7
        H_score = np.clip(H_score, 0, 1)
        
        # EQI: weighted combination (0-100 scale)
        EQI = (0.4 * S + 0.35 * H_score + 0.25 * info_ratio) * 100
        EQI = np.clip(EQI, 0, 100)
        
        # Legacy metrics (backward compatibility)
        ΔEΨ_without_H = I * P * S
        ΔEΨ_with_H = EQI  # Use new EQI
        
        # 6. คำนวณสูตรเพิ่มเติม
        QΨ = (I * H) / (S + 0.01)
        TΨ = QΨ ** 2
        mΨ = (H + (1 - S)) / 2
        
        # 7. Force (FΨ) - gradient magnitude
        if len(values) > 1:
            grad = np.gradient(values)
            FΨ = np.mean(np.abs(grad)) * I
        else:
            FΨ = 0
        
        # 8. Acceleration (aΨ)
        aΨ = FΨ / (mΨ + 1e-8)
        
        # 9. Additional embedding-specific metrics
        if isinstance(emb_tensor, torch.Tensor) and len(emb_tensor.shape) > 1:
            # Matrix properties
            matrix = emb_tensor.cpu().numpy()
            rank = np.linalg.matrix_rank(matrix)
            cond_number = np.linalg.cond(matrix) if rank == matrix.shape[1] else np.inf
            
            # Cosine similarity statistics
            if matrix.shape[0] > 1:
                norms = np.linalg.norm(matrix, axis=1, keepdims=True)
                normalized = matrix / (norms + 1e-8)
                cosine_sim = normalized @ normalized.T
                np.fill_diagonal(cosine_sim, 0)
                avg_cosine_sim = np.mean(cosine_sim)
            else:
                avg_cosine_sim = 0
        else:
            rank = 1
            cond_number = 1
            avg_cosine_sim = 0
        
        # EQI already calculated above
        
        result = {
            'embedding_shape': emb_tensor.shape if isinstance(emb_tensor, torch.Tensor) else embedding.shape,
            'signal_wave': signal_wave,
            'I': I,
            'P': P,
            'S': S,
            'H': H,
            'ΔEΨ_without_H': ΔEΨ_without_H,
            'ΔEΨ_with_H': ΔEΨ_with_H,
            'QΨ': QΨ,
            'TΨ': TΨ,
            'mΨ': mΨ,
            'FΨ': FΨ,
            'aΨ': aΨ,
            'rank': rank,
            'condition_number': cond_number,
            'avg_cosine_similarity': avg_cosine_sim,
            'original_text': original_text,
            
            # New engineering-friendly names (aliases)
            'information_strength': I,
            'signal_quality': S,
            'distribution_entropy': H,
            'embedding_quality_index': EQI,
            'energy': P,
            
            # Circuit-based metrics (if calculated)
            'circuit_energy': P_circuit if 'P_circuit' in locals() else 0,
            
            # Signal Quality Component breakdown
            'S_components': S_components_detail if 'S_components_detail' in locals() else {}
        }
        return result
    
    def _calculate_circuit_energy(self, values: np.ndarray) -> float:
        """
        คำนวณ energy จากวงจรไฟฟ้า (RLC circuit model)
        ใช้แนวคิดจาก embedding_analyzer.py
        """
        if len(values) < 2:
            return 0.0
        
        # คำนวณ R, L, C จากสถิติ
        std_val = np.std(values)
        skew_val = stats.skew(values) if len(values) > 2 else 0
        kurt_val = stats.kurtosis(values) if len(values) > 3 else 0
        
        # RLC parameters
        R = 1.0 / (std_val + 1e-6)
        L = abs(skew_val) * 0.1 + 1e-6
        C = abs(kurt_val) * 0.01 + 1e-6
        
        # สร้าง voltage signal (embedding values)
        voltage = values
        
        # คำนวณ current จาก RLC circuit (simplified)
        # i ≈ v/R สำหรับ steady state
        current = voltage / (R + 1e-6)
        
        # พลังงาน = ∫ P dt = ∫ V*I dt
        power = voltage * current
        energy = np.trapz(np.abs(power)) if len(power) > 1 else np.mean(np.abs(power))
        
        # Normalize
        energy = energy / (len(values) + 1e-6)
        
        return float(energy)
    
    def _calculate_circuit_quality(self, values: np.ndarray) -> float:
        """
        คำนวณ quality จาก circuit impedance analysis
        ใช้แนวคิดจาก embedding_analyzer.py
        """
        if len(values) < 2:
            return 0.5
        
        # สร้าง resistance network (simplified)
        # ความต้านทานระหว่างโหนด = 1/|value[i] - value[j]|
        D = len(values)
        
        # คำนวณ average resistance
        resistances = []
        for i in range(min(100, D)):  # Sample เพื่อความเร็ว
            for j in range(i + 1, min(i + 10, D)):
                diff = abs(values[i] - values[j])
                if diff > 1e-6:
                    resistance = 1.0 / diff
                    resistances.append(resistance)
        
        if not resistances:
            return 0.5
        
        # Impedance quality = inverse of resistance variance
        # Low variance = high quality (consistent impedance)
        resistance_std = np.std(resistances)
        resistance_mean = np.mean(resistances)
        
        if resistance_mean > 0:
            impedance_quality = 1.0 / (1.0 + resistance_std / resistance_mean)
        else:
            impedance_quality = 0.5
        
        # Normalize to [0, 1]
        impedance_quality = np.tanh(impedance_quality)
        
        return float(impedance_quality)
    
    def compare_embeddings(self, embeddings: List[Union[torch.Tensor, np.ndarray]], 
                          labels: List[str] = None) -> Dict:
        """
        เปรียบเทียบ embedding หลายๆ ตัว
        """
        if labels is None:
            labels = [f'Embedding_{i}' for i in range(len(embeddings))]
        
        results = {}
        for i, (embedding, label) in enumerate(zip(embeddings, labels)):
            results[label] = self.analyze_embedding(embedding)
        
        # คำนวณ statistics เปรียบเทียบ
        comparison_stats = {
            'mean_ΔEΨ': np.mean([r['ΔEΨ_with_H'] for r in results.values()]),
            'std_ΔEΨ': np.std([r['ΔEΨ_with_H'] for r in results.values()]),
            'mean_H': np.mean([r['H'] for r in results.values()]),
            'mean_S': np.mean([r['S'] for r in results.values()]),
            'correlation_matrix': self._calculate_correlation_matrix(results),
            'anomaly_scores': self._detect_anomalies(results)
        }
        
        return {'per_embedding': results, 'comparison': comparison_stats}
    
    def visualize(self, quality_dict: Dict, save_path: str = None):
        """
        แสดง visualization สำหรับ embedding quality analysis
        """
        return self.visualize_embedding_physics(quality_dict, save_path)
    
    def visualize_embedding_physics(self, physics_dict: Dict, 
                                   save_path: str = None):
        """
        แสดง visualization สำหรับ embedding quality (backward compatibility)
        """
        fig, axes = plt.subplots(3, 4, figsize=(18, 12))
        
        # 1. Embedding Signal
        axes[0, 0].plot(physics_dict['signal_wave'][:500])
        axes[0, 0].set_title('Embedding Signal Trace (first 500 samples)')
        axes[0, 0].set_xlabel('Sample')
        axes[0, 0].set_ylabel('Amplitude')
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. System Health Metrics
        metrics = ['EQI', 'Info Strength', 'Signal Quality', 'Entropy']
        values = [
            physics_dict.get('embedding_quality_index', physics_dict['ΔEΨ_with_H']),
            physics_dict.get('information_strength', physics_dict['I']) / 100 if physics_dict.get('I', 0) > 100 else physics_dict.get('information_strength', physics_dict['I']) / 10,
            physics_dict.get('signal_quality', physics_dict['S']),
            physics_dict.get('distribution_entropy', physics_dict['H'])
        ]
        bars = axes[0, 1].bar(metrics, values)
        for bar, val in zip(bars, values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                           f'{val:.2f}', ha='center', va='bottom', fontsize=8)
        axes[0, 1].set_title('System Health Metrics')
        axes[0, 1].tick_params(axis='x', rotation=45)
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Energy Components
        components = ['I/100', 'P', 'S', '1-H']
        comp_values = [
            physics_dict['I']/100 if physics_dict['I'] > 100 else physics_dict['I']/10,
            physics_dict['P'],
            physics_dict['S'],
            1 - physics_dict['H']
        ]
        axes[0, 2].bar(components, comp_values)
        axes[0, 2].set_title('Energy Components')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4. Signal Integrity
        if 'S_components' in physics_dict:
            s_comps = list(physics_dict['S_components'].keys())
            s_vals = list(physics_dict['S_components'].values())
            axes[0, 3].bar(s_comps, s_vals)
            axes[0, 3].set_title('Signal Integrity')
            axes[0, 3].tick_params(axis='x', rotation=45)
            axes[0, 3].grid(True, alpha=0.3)
        
        # 5. Entropy Distribution
        if 'original_text' in physics_dict and physics_dict['original_text']:
            # Histogram of embedding values
            if isinstance(physics_dict.get('embedding_values', None), np.ndarray):
                values = physics_dict['embedding_values']
            else:
                # Need to extract from embedding
                pass
        
        # 6. Frequency Spectrum
        if len(physics_dict['signal_wave']) > 1:
            fft_result = fft.fft(physics_dict['signal_wave'])
            freqs = fft.fftfreq(len(fft_result), 1/self.sampling_rate)
            positive_mask = freqs >= 0
            axes[1, 0].plot(freqs[positive_mask], np.abs(fft_result[positive_mask]))
            axes[1, 0].set_title('Frequency Spectrum')
            axes[1, 0].set_xlabel('Frequency')
            axes[1, 0].set_ylabel('Magnitude')
            axes[1, 0].grid(True, alpha=0.3)
        
        # 7. Entropy vs Signal Quality (Phase Space)
        axes[1, 1].scatter(physics_dict['H'], physics_dict['S'], s=100)
        axes[1, 1].set_xlabel('Entropy (H)')
        axes[1, 1].set_ylabel('Signal Quality (S)')
        axes[1, 1].set_title('Entropy vs Signal Quality (Phase Space)')
        axes[1, 1].grid(True, alpha=0.3)
        
        # 8. Quality vs Stability
        axes[1, 2].scatter(physics_dict['QΨ'], physics_dict['TΨ'], s=100)
        axes[1, 2].set_xlabel('Quality Factor')
        axes[1, 2].set_ylabel('Stability Index')
        axes[1, 2].set_title('Quality vs Stability')
        axes[1, 2].grid(True, alpha=0.3)
        
        # 9. Information Metrics
        axes[1, 3].axis('off')
        info_text = f"""
        Embedding Shape: {physics_dict['embedding_shape']}
        Rank: {physics_dict.get('rank', 'N/A')}
        Condition Number: {physics_dict.get('condition_number', 'N/A'):.2f}
        Avg Cosine Sim: {physics_dict.get('avg_cosine_similarity', 'N/A'):.3f}
        """
        axes[1, 3].text(0.1, 0.5, info_text, fontsize=10)
        
        # 10. Health Indicators
        health_metrics = self._calculate_health_indicators(physics_dict)
        health_names = list(health_metrics.keys())
        health_values = list(health_metrics.values())
        
        colors = ['green' if v > 0.7 else 'orange' if v > 0.4 else 'red' 
                 for v in health_values]
        
        axes[2, 0].barh(health_names, health_values, color=colors)
        axes[2, 0].set_xlim(0, 1)
        axes[2, 0].set_title('Health Indicators')
        axes[2, 0].grid(True, alpha=0.3)
        
        # 11. Summary
        axes[2, 1].axis('off')
        eqi = physics_dict.get('embedding_quality_index', physics_dict['ΔEΨ_with_H'])
        info_strength = physics_dict.get('information_strength', physics_dict['I'])
        signal_quality = physics_dict.get('signal_quality', physics_dict['S'])
        entropy = physics_dict.get('distribution_entropy', physics_dict['H'])
        
        summary = f"""
        Embedding Quality Index (EQI): {eqi:.2f}
        Information Strength: {info_strength:.0f}
        Signal Quality Score: {signal_quality:.3f}
        Distribution Entropy: {entropy:.3f}
        Energy: {physics_dict['P']:.4f}
        """
        axes[2, 1].text(0.1, 0.5, summary, fontsize=10)
        
        # 12. Recommendations
        axes[2, 2].axis('off')
        recommendations = self._generate_recommendations(physics_dict)
        rec_text = "\n".join([f"• {rec}" for rec in recommendations[:3]])
        axes[2, 2].text(0.1, 0.5, f"Recommendations:\n{rec_text}", fontsize=9)
        
        # 13. Anomaly Detection
        axes[2, 3].axis('off')
        anomalies = self._detect_embedding_anomalies(physics_dict)
        if anomalies:
            anomaly_text = "⚠️ Anomalies Detected:\n" + \
                          "\n".join([f"• {a}" for a in anomalies[:2]])
        else:
            anomaly_text = "✅ No anomalies detected"
        axes[2, 3].text(0.1, 0.5, anomaly_text, fontsize=9)
        
        plt.suptitle('Embedding Quality Analysis', fontsize=16, y=1.02)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved visualization to {save_path}")
        
        plt.show()
    
    def _calculate_health_indicators(self, physics_dict: Dict) -> Dict:
        """คำนวณ health indicators"""
        return {
            'Information Density': min(1.0, physics_dict['I'] / 1000),
            'Signal Clarity': physics_dict['S'],
            'Energy Efficiency': min(1.0, physics_dict['ΔEΨ_with_H'] / 100),
            'Stability': 1 - physics_dict['H'],  # Low entropy = stable
            'Quality Score': min(1.0, physics_dict['QΨ'] / 50),
            'Temperature Balance': 1 / (1 + abs(physics_dict['TΨ'] - 25) / 25)  # Around 25 ideal
        }
    
    def _generate_recommendations(self, physics_dict: Dict) -> List[str]:
        """สร้างคำแนะนำจากผลวิเคราะห์"""
        recs = []
        
        if physics_dict['H'] > 0.8:
            recs.append("Entropy too high - consider regularization")
        
        if physics_dict['S'] < 0.3:
            recs.append("Signal quality low - check for noise or artifacts")
        
        if physics_dict['ΔEΨ_with_H'] < 10:
            recs.append("Information energy low - embedding may be too sparse")
        
        if physics_dict.get('condition_number', 1) > 1000:
            recs.append("High condition number - embedding may be ill-conditioned")
        
        if physics_dict['QΨ'] > 100:
            recs.append("Quality factor very high - may indicate overfitting")
        
        if physics_dict['TΨ'] > 100:
            recs.append("Temperature very high - embedding may be unstable")
        
        if not recs:
            recs.append("Embedding looks healthy - no major issues detected")
        
        return recs
    
    def _detect_embedding_anomalies(self, physics_dict: Dict) -> List[str]:
        """ตรวจจับความผิดปกติใน embedding"""
        anomalies = []
        
        # Check for NaN or Inf
        for key, value in physics_dict.items():
            if isinstance(value, (int, float)):
                if np.isnan(value) or np.isinf(value):
                    anomalies.append(f"{key} is {value}")
        
        # Check physics constraints
        if not (0 <= physics_dict['H'] <= 1):
            anomalies.append(f"Entropy H={physics_dict['H']} outside [0,1]")
        
        if physics_dict['P'] < 0:
            anomalies.append(f"Negative power P={physics_dict['P']}")
        
        if physics_dict['ΔEΨ_with_H'] < 0:
            anomalies.append(f"Negative ΔEΨ={physics_dict['ΔEΨ_with_H']}")
        
        # Check for extreme values
        if physics_dict['TΨ'] > 1000:
            anomalies.append(f"Extreme temperature TΨ={physics_dict['TΨ']}")
        
        if physics_dict['QΨ'] > 1000:
            anomalies.append(f"Extreme quality factor QΨ={physics_dict['QΨ']}")
        
        return anomalies
    
    def _calculate_correlation_matrix(self, results: Dict) -> np.ndarray:
        """คำนวณ correlation matrix ระหว่าง embeddings"""
        metrics = ['ΔEΨ_with_H', 'H', 'S', 'QΨ', 'TΨ']
        n_metrics = len(metrics)
        n_embeddings = len(results)
        
        # สร้าง matrix ของ metric values
        metric_matrix = np.zeros((n_embeddings, n_metrics))
        
        for i, (label, result) in enumerate(results.items()):
            for j, metric in enumerate(metrics):
                metric_matrix[i, j] = result[metric]
        
        # คำนวณ correlation
        corr_matrix = np.corrcoef(metric_matrix, rowvar=False)
        
        return corr_matrix
    
    def _detect_anomalies(self, results: Dict) -> Dict:
        """ตรวจจับ embeddings ที่ผิดปกติ"""
        # คำนวณ Z-scores สำหรับ metrics สำคัญ
        metrics = ['ΔEΨ_with_H', 'H', 'S', 'QΨ']
        anomaly_scores = {}
        
        for label, result in results.items():
            z_scores = []
            for metric in metrics:
                values = [r[metric] for r in results.values()]
                mean = np.mean(values)
                std = np.std(values)
                if std > 0:
                    z = abs((result[metric] - mean) / std)
                    z_scores.append(z)
            
            anomaly_score = np.max(z_scores) if z_scores else 0
            anomaly_scores[label] = {
                'score': anomaly_score,
                'is_anomaly': anomaly_score > 3.0  # 3 standard deviations
            }
        
        return anomaly_scores
    
    def interactive_3d_visualization(self, embeddings: List[np.ndarray], 
                                    labels: List[str] = None):
        """
        แสดง visualization 3D แบบ interactive
        """
        try:
            # ลดมิติเป็น 3D ด้วย PCA
            all_embeddings = np.vstack([e.flatten() for e in embeddings])
            
            # PCA to 3D
            pca = PCA(n_components=3)
            embeddings_3d = pca.fit_transform(all_embeddings)
            
            # คำนวณ physics สำหรับแต่ละ embedding
            physics_results = [self.analyze_embedding(emb) for emb in embeddings]
            
            # สร้าง 3D scatter plot
            fig = go.Figure()
            
            colors = ['red', 'blue', 'green', 'orange', 'purple']
            
            for i, (emb_3d, physics, label) in enumerate(zip(embeddings_3d, physics_results, labels)):
                color = colors[i % len(colors)]
                
                # Size based on ΔEΨ
                size = 10 + physics['ΔEΨ_with_H'] / 10
                
                fig.add_trace(go.Scatter3d(
                    x=[emb_3d[0]],
                    y=[emb_3d[1]],
                    z=[emb_3d[2]],
                    mode='markers+text',
                    marker=dict(
                        size=size,
                        color=color,
                        opacity=0.8,
                        symbol='circle'
                    ),
                    text=[label],
                    textposition="bottom center",
                    name=label,
                    hovertemplate=f"""
                    <b>{label}</b><br>
                    EQI: {physics.get('embedding_quality_index', physics['ΔEΨ_with_H']):.2f}<br>
                    Info Strength: {physics.get('information_strength', physics['I']):.0f}<br>
                    Signal Quality: {physics.get('signal_quality', physics['S']):.3f}<br>
                    Entropy: {physics.get('distribution_entropy', physics['H']):.3f}
                    """
                ))
            
            fig.update_layout(
                title="Embedding Quality 3D Space",
                scene=dict(
                    xaxis_title='PC1',
                    yaxis_title='PC2',
                    zaxis_title='PC3'
                ),
                showlegend=True
            )
            
            fig.show()
            
        except Exception as e:
            print(f"Could not create 3D visualization: {e}")
    
    def generate_report(self, embeddings: Union[torch.Tensor, np.ndarray, List],
                       save_path: str = None) -> str:
        """
        สร้างรายงานสรุป
        """
        if not isinstance(embeddings, list):
            embeddings = [embeddings]
        
        results = self.compare_embeddings(embeddings)
        
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("EMBEDDING QUALITY INSPECTION REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {np.datetime64('now')}")
        report_lines.append(f"Number of embeddings analyzed: {len(embeddings)}")
        report_lines.append("")
        
        # Summary statistics
        report_lines.append("SUMMARY STATISTICS:")
        report_lines.append("-" * 40)
        
        for label, result in results['per_embedding'].items():
            report_lines.append(f"\nEmbedding: {label}")
            report_lines.append(f"  Shape: {result['embedding_shape']}")
            eqi = result.get('embedding_quality_index', result['ΔEΨ_with_H'])
            info_strength = result.get('information_strength', result['I'])
            signal_quality = result.get('signal_quality', result['S'])
            entropy = result.get('distribution_entropy', result['H'])
            
            report_lines.append(f"  Embedding Quality Index (EQI): {eqi:.4f}")
            report_lines.append(f"  Information Strength: {info_strength:.0f}")
            report_lines.append(f"  Distribution Entropy: {entropy:.4f}")
            report_lines.append(f"  Signal Quality Score: {signal_quality:.4f}")
        
        # Anomaly detection
        report_lines.append("\n\nANOMALY DETECTION:")
        report_lines.append("-" * 40)
        
        for label, anomaly_info in results['comparison']['anomaly_scores'].items():
            status = "⚠️ ANOMALY" if anomaly_info['is_anomaly'] else "✅ Normal"
            report_lines.append(f"{label}: {status} (score: {anomaly_info['score']:.2f})")
        
        # Health assessment
        report_lines.append("\n\nHEALTH ASSESSMENT:")
        report_lines.append("-" * 40)
        
        health_categories = {
            'Excellent': {'min_ΔEΨ': 50, 'max_H': 0.3, 'min_S': 0.7},
            'Good': {'min_ΔEΨ': 20, 'max_H': 0.5, 'min_S': 0.5},
            'Fair': {'min_ΔEΨ': 10, 'max_H': 0.7, 'min_S': 0.3},
            'Poor': {'min_ΔEΨ': 0, 'max_H': 1.0, 'min_S': 0.0}
        }
        
        for label, result in results['per_embedding'].items():
            for category, criteria in health_categories.items():
                if (result['ΔEΨ_with_H'] >= criteria['min_ΔEΨ'] and
                    result['H'] <= criteria['max_H'] and
                    result['S'] >= criteria['min_S']):
                    report_lines.append(f"{label}: {category}")
                    break
        
        # Recommendations
        report_lines.append("\n\nRECOMMENDATIONS:")
        report_lines.append("-" * 40)
        
        for label, result in results['per_embedding'].items():
            recs = self._generate_recommendations(result)
            report_lines.append(f"\n{label}:")
            for rec in recs[:2]:  # Top 2 recommendations
                report_lines.append(f"  • {rec}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report


# ตัวอย่างการใช้งาน
def demo_embedding_inspection():
    """Demo การใช้งาน Embedding Quality Inspector"""
    
    print("🔍 Embedding Quality Inspector Demo")
    print("="*50)
    
    # 1. สร้าง inspector
    inspector = EmbeddingQualityInspector(sampling_rate=1000)
    
    # 2. สร้างตัวอย่าง embeddings
    print("\n📊 Creating sample embeddings...")
    
    # Healthy embedding
    healthy_emb = torch.randn(10, 256) * 0.5  # Normal distribution
    
    # Noisy embedding
    noisy_emb = healthy_emb + torch.randn(10, 256) * 2.0
    
    # Sparse embedding
    sparse_emb = torch.zeros(10, 256)
    sparse_emb[torch.rand(10, 256) > 0.9] = 1.0
    
    # Collapsed embedding (low variance)
    collapsed_emb = torch.ones(10, 256) * 0.1 + torch.randn(10, 256) * 0.01
    
    embeddings = [healthy_emb, noisy_emb, sparse_emb, collapsed_emb]
    labels = ['Healthy', 'Noisy', 'Sparse', 'Collapsed']
    
    # 3. วิเคราะห์แต่ละ embedding
    print("\n🔍 Analyzing embeddings...")
    
    for emb, label in zip(embeddings, labels):
        print(f"\n{label} Embedding:")
        result = inspector.analyze_embedding(emb)
        print(f"  Embedding Quality Index (EQI): {result.get('embedding_quality_index', result['ΔEΨ_with_H']):.2f}")
        print(f"  Information Strength: {result.get('information_strength', result['I']):.0f}")
        print(f"  Distribution Entropy: {result.get('distribution_entropy', result['H']):.3f}")
        print(f"  Signal Quality Score: {result.get('signal_quality', result['S']):.3f}")
    
    # 4. เปรียบเทียบทั้งหมด
    print("\n📈 Comparing all embeddings...")
    comparison = inspector.compare_embeddings(embeddings, labels)
    
    print(f"\nMean ΔEΨ: {comparison['comparison']['mean_ΔEΨ']:.2f}")
    print(f"Std ΔEΨ: {comparison['comparison']['std_ΔEΨ']:.2f}")
    
    # 5. ตรวจสอบ anomalies
    print("\n🚨 Anomaly Detection:")
    for label, anomaly_info in comparison['comparison']['anomaly_scores'].items():
        if anomaly_info['is_anomaly']:
            print(f"  ⚠️ {label}: ANOMALY DETECTED (score: {anomaly_info['score']:.2f})")
        else:
            print(f"  ✅ {label}: Normal (score: {anomaly_info['score']:.2f})")
    
    # 6. Visualization สำหรับ embedding แรก
    print("\n📊 Generating visualization for 'Healthy' embedding...")
    healthy_result = inspector.analyze_embedding(healthy_emb)
    inspector.visualize(healthy_result)
    
    # 7. สร้างรายงาน
    print("\n📝 Generating report...")
    import os
    report_path = os.path.join('outputs', 'reports', 'embedding_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = inspector.generate_report(embeddings[:2], save_path=report_path)
    
    print("\n✅ Demo completed!")
    
    return inspector, comparison


# ฟังก์ชันสำหรับตรวจสอบ embedding ของโมเดล
def inspect_model_embeddings(model: nn.Module, 
                           sample_texts: List[str] = None,
                           tokenizer = None):
    """
    ตรวจสอบ embedding layer ของโมเดล
    """
    
    print("🧠 Inspecting Model Embeddings")
    print("="*50)
    
    inspector = EmbeddingQualityInspector()
    
    # หา embedding layer ในโมเดล
    embedding_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Embedding):
            embedding_layers.append((name, module))
    
    if not embedding_layers:
        print("No embedding layers found in the model.")
        return None
    
    print(f"Found {len(embedding_layers)} embedding layer(s):")
    for name, layer in embedding_layers:
        print(f"  - {name}: {layer.weight.shape}")
    
    # วิเคราะห์แต่ละ embedding layer
    results = {}
    for name, layer in embedding_layers:
        print(f"\n📊 Analyzing {name}...")
        
        # ดึง weight matrix
        weights = layer.weight.detach().cpu().numpy()
        
        # วิเคราะห์คุณภาพ
        quality_result = inspector.analyze_embedding(weights)
        results[name] = quality_result
        
        print(f"  Shape: {weights.shape}")
        print(f"  EQI: {quality_result.get('embedding_quality_index', quality_result['ΔEΨ_with_H']):.2f}")
        print(f"  Information Strength: {quality_result.get('information_strength', quality_result['I']):.0f}")
        print(f"  Distribution Entropy: {quality_result.get('distribution_entropy', quality_result['H']):.3f}")
        print(f"  Rank: {quality_result['rank']}")
        
        # ตรวจสอบ anomalies
        anomalies = inspector._detect_embedding_anomalies(quality_result)
        if anomalies:
            print(f"  🚨 Anomalies: {', '.join(anomalies[:2])}")
        else:
            print(f"  ✅ No anomalies detected")
    
    # วิเคราะห์เฉพาะสำหรับ sample texts ถ้ามี
    if sample_texts and tokenizer:
        print("\n📝 Analyzing embeddings for sample texts...")
        
        # สร้าง embeddings สำหรับตัวอย่างข้อความ
        sample_embeddings = []
        for text in sample_texts[:3]:  # ใช้ 3 ข้อความแรก
            inputs = tokenizer(text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                # ได้最后一层的隐藏状态作为 embedding
                if hasattr(outputs, 'last_hidden_state'):
                    emb = outputs.last_hidden_state.mean(dim=1)  # average pooling
                else:
                    emb = outputs[0].mean(dim=1)
                
                sample_embeddings.append(emb)
        
        # เปรียบเทียบ sample embeddings
        sample_labels = [f'Text_{i}' for i in range(len(sample_embeddings))]
        sample_comparison = inspector.compare_embeddings(sample_embeddings, sample_labels)
        
        print(f"\nSample Text Embeddings Comparison:")
        for label, result in sample_comparison['per_embedding'].items():
            eqi = result.get('embedding_quality_index', result['ΔEΨ_with_H'])
            entropy = result.get('distribution_entropy', result['H'])
            print(f"  {label}: EQI={eqi:.2f}, Entropy={entropy:.3f}")
    
    return results


if __name__ == "__main__":
    # รัน demo
    inspector, comparison = demo_embedding_inspection()
    
    print("\n🎯 Try it with your own embeddings:")
    print("""
    # สร้าง inspector
    inspector = EmbeddingQualityInspector()
    
    # วิเคราะห์ embedding ของคุณ
    your_embedding = ...  # torch.Tensor หรือ numpy array
    result = inspector.analyze_embedding(your_embedding)
    
    # แสดง visualization
    inspector.visualize(result)
    
    # สร้างรายงาน
    report = inspector.generate_report([your_embedding], save_path='my_embedding_report.txt')
    """)

# Backward compatibility alias
EmbeddingPhysicsInspector = EmbeddingQualityInspector

