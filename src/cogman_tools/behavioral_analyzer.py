"""
Baseline Behavioral Analysis Specification Implementation
Version: 0.1
Status: Community Draft

ระบบวิเคราะห์พฤติกรรมสำหรับ Model / Embedding / Multimodal Systems
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from datetime import datetime
from collections import defaultdict
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
import warnings

EPS = 1e-8  # numeric stability guard


@dataclass
class OperationalStatus:
    """สถานะการทำงานของระบบ"""
    status: str  # NORMAL, WARNING, DEGRADED, UNSAFE
    confidence: float
    reasons: List[str]
    timestamp: datetime


class BehavioralAnalyzer:
    """
    เครื่องมือวิเคราะห์พฤติกรรมตาม Baseline Behavioral Analysis Specification
    
    หลักการ:
    1. Neutrality - ไม่ผูกกับโมเดล/สถาปัตยกรรม
    2. Measurability - ทุกข้อวัดซ้ำได้
    3. Replaceability - สูตรเปลี่ยนได้
    4. Accountability - บอกผู้ใช้ได้ว่าระบบยังใช้งานได้หรือไม่
    5. Modality-agnostic - ใช้ได้กับ text/image/audio/multimodal
    """
    
    def __init__(self, 
                 baseline_embeddings: Optional[List[np.ndarray]] = None,
                 similarity_threshold: float = 0.7,
                 anomaly_threshold: float = 3.0,
                 drift_threshold: float = 0.15):
        """
        Args:
            baseline_embeddings: ตัวอย่าง embeddings สำหรับ baseline
            similarity_threshold: threshold สำหรับ similarity analysis
            anomaly_threshold: threshold สำหรับ anomaly detection (z-score)
            drift_threshold: threshold สำหรับ drift detection
        """
        self.baseline_embeddings = baseline_embeddings or []
        self.similarity_threshold = similarity_threshold
        self.anomaly_threshold = anomaly_threshold
        self.drift_threshold = drift_threshold
        
        # เก็บประวัติการวิเคราะห์
        self.history: List[Dict] = []
        self.trend_data: Dict[str, List[Tuple[datetime, float]]] = defaultdict(list)
        
        # Baseline statistics
        self.baseline_stats: Optional[Dict] = None
        if self.baseline_embeddings:
            self._compute_baseline_stats()
    
    def _compute_baseline_stats(self):
        """คำนวณสถิติ baseline"""
        if not self.baseline_embeddings:
            return
        
        embeddings_array = np.vstack([e.flatten() for e in self.baseline_embeddings])
        
        self.baseline_stats = {
            'mean': np.mean(embeddings_array, axis=0),
            'std': np.std(embeddings_array, axis=0),
            'count': len(self.baseline_embeddings),
            'dimension': embeddings_array.shape[1],
            'mean_norm': np.mean([np.linalg.norm(e.flatten()) for e in self.baseline_embeddings]),
            'std_norm': np.std([np.linalg.norm(e.flatten()) for e in self.baseline_embeddings])
        }
    
    # ==================== 4.1 Similarity Analysis ====================
    
    def similarity_analysis(self, 
                           embedding_a: Union[torch.Tensor, np.ndarray],
                           embedding_b: Union[torch.Tensor, np.ndarray],
                           method: str = 'cosine') -> Dict:
        """
        วิเคราะห์ความใกล้เคียงเชิงบริบท
        
        Args:
            embedding_a: Embedding A
            embedding_b: Embedding B
            method: 'cosine', 'dot', 'euclidean'
        
        Returns:
            Dict with similarity metrics
        """
        # แปลงเป็น numpy
        if isinstance(embedding_a, torch.Tensor):
            emb_a = embedding_a.detach().cpu().numpy().flatten()
        else:
            emb_a = np.array(embedding_a).flatten()
        
        if isinstance(embedding_b, torch.Tensor):
            emb_b = embedding_b.detach().cpu().numpy().flatten()
        else:
            emb_b = np.array(embedding_b).flatten()
        
        # Normalize (guard zero norm)
        norm_a = np.linalg.norm(emb_a)
        norm_b = np.linalg.norm(emb_b)
        
        if norm_a < EPS or norm_b < EPS:
            return {
                'similarity': 0.0,
                'distance': float('inf'),
                'method': method,
                'warning': 'Zero vector detected'
            }
        
        # คำนวณ similarity
        if method == 'cosine':
            similarity = np.dot(emb_a, emb_b) / (norm_a * norm_b)
            distance = 1 - similarity
        elif method == 'dot':
            similarity = np.dot(emb_a, emb_b)
            distance = np.linalg.norm(emb_a - emb_b)
        elif method == 'euclidean':
            distance = euclidean(emb_a, emb_b)
            similarity = 1 / (1 + distance)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # คำนวณ distance deviation จาก baseline
        distance_deviation = None
        if self.baseline_stats:
            baseline_distances = []
            for baseline_emb in self.baseline_embeddings:
                baseline_flat = baseline_emb.flatten()
                if method == 'cosine':
                    base_sim = np.dot(emb_a, baseline_flat) / (norm_a * np.linalg.norm(baseline_flat))
                    base_dist = 1 - base_sim
                else:
                    base_dist = euclidean(emb_a, baseline_flat)
                baseline_distances.append(base_dist)
            
            if baseline_distances:
                mean_baseline_dist = np.mean(baseline_distances)
                std_baseline_dist = np.std(baseline_distances)
                if std_baseline_dist > EPS:
                    distance_deviation = (distance - mean_baseline_dist) / std_baseline_dist
        
        # Interpretation
        is_low_similarity = similarity < self.similarity_threshold
        interpretation = {
            'is_low_similarity': is_low_similarity,
            'out_of_domain_signal': is_low_similarity,
            'note': 'Low similarity may indicate out-of-domain input, but does not imply correctness'
        }
        
        return {
            'similarity': float(similarity),
            'distance': float(distance),
            'distance_deviation': float(distance_deviation) if distance_deviation is not None else None,
            'method': method,
            'interpretation': interpretation
        }
    
    # ==================== 4.2 Cluster Analysis ====================
    
    def cluster_analysis(self, 
                        embeddings: List[Union[torch.Tensor, np.ndarray]],
                        method: str = 'kmeans',
                        n_clusters: Optional[int] = None) -> Dict:
        """
        วิเคราะห์โครงสร้างพฤติกรรมของ output
        
        Args:
            embeddings: List of embeddings
            method: 'kmeans', 'dbscan'
            n_clusters: จำนวน clusters (ถ้า None จะ auto-detect)
        
        Returns:
            Dict with cluster metrics
        """
        if len(embeddings) < 2:
            return {
                'cluster_count': 1,
                'cluster_density': 0.0,
                'distribution_shift': 0.0,
                'warning': 'Insufficient embeddings for cluster analysis'
            }
        
        # แปลงเป็น numpy array
        embeddings_array = np.vstack([self._to_numpy(e).flatten() for e in embeddings])
        
        # Auto-detect n_clusters ถ้าไม่ระบุ
        if n_clusters is None:
            # ใช้ elbow method หรือ silhouette score
            max_clusters = min(10, len(embeddings) // 2)
            if max_clusters < 2:
                n_clusters = 1
            else:
                best_score = -1
                best_k = 1
                for k in range(2, max_clusters + 1):
                    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                    labels = kmeans.fit_predict(embeddings_array)
                    if len(np.unique(labels)) > 1:
                        score = silhouette_score(embeddings_array, labels)
                        if score > best_score:
                            best_score = score
                            best_k = k
                n_clusters = best_k
        
        # Clustering
        if method == 'kmeans':
            clusterer = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            labels = clusterer.fit_predict(embeddings_array)
            centers = clusterer.cluster_centers_
        elif method == 'dbscan':
            clusterer = DBSCAN(eps=0.5, min_samples=2)
            labels = clusterer.fit_predict(embeddings_array)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            centers = None
        else:
            raise ValueError(f"Unknown method: {method}")
        
        # คำนวณ cluster density
        cluster_densities = []
        for cluster_id in range(n_clusters):
            cluster_points = embeddings_array[labels == cluster_id]
            if len(cluster_points) > 1:
                # คำนวณ average distance within cluster
                distances = []
                for i in range(len(cluster_points)):
                    for j in range(i + 1, len(cluster_points)):
                        distances.append(euclidean(cluster_points[i], cluster_points[j]))
                if distances:
                    avg_distance = np.mean(distances)
                    density = 1 / (1 + avg_distance)  # Inverse of distance
                    cluster_densities.append(density)
        
        cluster_density = np.mean(cluster_densities) if cluster_densities else 0.0
        
        # คำนวณ distribution shift จาก baseline
        distribution_shift = 0.0
        if self.baseline_stats and len(embeddings_array) > 0:
            # เปรียบเทียบ mean และ std
            current_mean = np.mean(embeddings_array, axis=0)
            current_std = np.std(embeddings_array, axis=0)
            
            mean_shift = np.linalg.norm(current_mean - self.baseline_stats['mean'])
            std_shift = np.linalg.norm(current_std - self.baseline_stats['std'])
            
            # Normalize
            baseline_norm = np.linalg.norm(self.baseline_stats['mean'])
            if baseline_norm > EPS:
                distribution_shift = mean_shift / baseline_norm
        
        # Interpretation
        interpretation = {
            'new_clusters_detected': n_clusters > (self.baseline_stats.get('cluster_count', 0) if self.baseline_stats else 0),
            'clusters_merged': n_clusters < (self.baseline_stats.get('cluster_count', 0) if self.baseline_stats else n_clusters),
            'behavior_change': distribution_shift > self.drift_threshold,
            'over_constraint_risk': cluster_density > 0.9 and n_clusters == 1
        }
        
        return {
            'cluster_count': int(n_clusters),
            'cluster_density': float(cluster_density),
            'distribution_shift': float(distribution_shift),
            'labels': labels.tolist(),
            'method': method,
            'interpretation': interpretation
        }
    
    # ==================== 4.3 Anomaly Detection ====================
    
    def anomaly_detection(self, 
                         embeddings: List[Union[torch.Tensor, np.ndarray]],
                         baseline_embeddings: Optional[List[np.ndarray]] = None) -> Dict:
        """
        ตรวจจับพฤติกรรมที่ออกนอก pattern ปกติ
        
        Args:
            embeddings: Embeddings to check
            baseline_embeddings: Baseline embeddings (ถ้า None ใช้ self.baseline_embeddings)
        
        Returns:
            Dict with anomaly metrics
        """
        if not embeddings:
            return {
                'anomaly_score': 0.0,
                'anomaly_density': 0.0,
                'stress_index': 0.0,
                'warning': 'No embeddings provided'
            }
        
        baseline = baseline_embeddings or self.baseline_embeddings
        
        # แปลงเป็น numpy
        embeddings_array = np.vstack([self._to_numpy(e).flatten() for e in embeddings])
        
        if not baseline:
            # ไม่มี baseline: ใช้ self-consistency check
            # ตรวจสอบว่า embeddings มีความสม่ำเสมอหรือไม่
            if len(embeddings_array) < 3:
                # ข้อมูลน้อยเกินไป - ถือว่าปกติ
                return {
                    'anomaly_score': 0.0,
                    'anomaly_density': 0.0,
                    'stress_index': 0.0,
                    'individual_scores': [],
                    'interpretation': {
                        'anomaly_detected': False,
                        'degradation_risk': False,
                        'note': 'No baseline available, assuming normal'
                    }
                }
            
            # ใช้ mean และ std ของ embeddings เอง
            baseline_mean = np.mean(embeddings_array, axis=0)
            baseline_std = np.std(embeddings_array, axis=0)
        else:
            baseline_array = np.vstack([self._to_numpy(e).flatten() for e in baseline])
            baseline_mean = np.mean(baseline_array, axis=0)
            baseline_std = np.std(baseline_array, axis=0)
        
        # Avoid division by zero and handle constant dimensions
        baseline_std = np.where(baseline_std < 1e-6, 1.0, baseline_std)
        
        # คำนวณ anomaly scores (z-scores)
        anomaly_scores = []
        for emb in embeddings_array:
            z_scores = np.abs((emb - baseline_mean) / baseline_std)
            
            # ใช้ percentile-based threshold แทน max
            # เพื่อลด false positive จาก outlier dimensions
            p95_z_score = np.percentile(z_scores, 95)
            mean_z_score = np.mean(z_scores)
            
            # Anomaly ต้องมี p95 > threshold หรือ mean > threshold * 0.5
            is_anomaly = (p95_z_score > self.anomaly_threshold) or (mean_z_score > self.anomaly_threshold * 0.5)
            
            anomaly_scores.append({
                'max_z': float(np.max(z_scores)),
                'p95_z': float(p95_z_score),
                'mean_z': float(mean_z_score),
                'is_anomaly': is_anomaly
            })
        
        # คำนวณ anomaly density
        anomaly_count = sum(1 for a in anomaly_scores if a['is_anomaly'])
        anomaly_density = anomaly_count / len(anomaly_scores) if anomaly_scores else 0.0
        
        # คำนวณ stress index (weighted combination)
        p95_scores = [a['p95_z'] for a in anomaly_scores]
        mean_scores = [a['mean_z'] for a in anomaly_scores]
        
        # Stress index: normalized to 0-1 range (approximately)
        raw_stress = (
            0.6 * np.mean(p95_scores) + 
            0.4 * np.mean(mean_scores)
        )
        # Normalize: threshold = 1.0, double threshold = 2.0
        stress_index = raw_stress / self.anomaly_threshold
        stress_index = np.clip(stress_index, 0, 10)  # Cap at 10 for extreme cases
        
        # Interpretation
        interpretation = {
            'anomaly_detected': anomaly_density > 0.1,  # >10% anomalies
            'degradation_risk': anomaly_density > 0.3,  # >30% anomalies
            'note': 'Anomaly does not imply error, but indicates operational warning'
        }
        
        return {
            'anomaly_score': float(np.mean([a['max_z'] for a in anomaly_scores])),
            'anomaly_density': float(anomaly_density),
            'stress_index': float(stress_index),
            'anomaly_details': anomaly_scores,
            'interpretation': interpretation
        }
    
    # ==================== 4.4 Trend Analysis ====================
    
    def trend_analysis(self, 
                      metric_name: str,
                      time_window: Optional[int] = None) -> Dict:
        """
        ติดตามสุขภาพระบบตามเวลา
        
        Args:
            metric_name: ชื่อ metric ที่ต้องการวิเคราะห์
            time_window: จำนวนจุดข้อมูลล่าสุดที่ต้องการวิเคราะห์ (None = ทั้งหมด)
        
        Returns:
            Dict with trend metrics
        """
        if metric_name not in self.trend_data or not self.trend_data[metric_name]:
            return {
                'drift_slope': 0.0,
                'stability_variance': 0.0,
                'pattern_persistence': 0.0,
                'warning': 'No trend data available'
            }
        
        data = self.trend_data[metric_name]
        if time_window:
            data = data[-time_window:]
        
        if len(data) < 2:
            return {
                'drift_slope': 0.0,
                'stability_variance': 0.0,
                'pattern_persistence': 0.0,
                'warning': 'Insufficient data points'
            }
        
        # แยกเวลาและค่า
        timestamps = [d[0] for d in data]
        values = np.array([d[1] for d in data])
        
        # คำนวณ drift slope (linear regression)
        time_numeric = np.array([(t - timestamps[0]).total_seconds() for t in timestamps])
        if np.std(time_numeric) > 0:
            slope, intercept, r_value, p_value, std_err = stats.linregress(time_numeric, values)
            drift_slope = slope
        else:
            drift_slope = 0.0
        
        # คำนวณ stability variance
        stability_variance = np.var(values)
        
        # คำนวณ pattern persistence (autocorrelation)
        if len(values) > 1:
            # ป้องกัน warning เมื่อค่าคงที่หรือมีความแปรปรวนเป็นศูนย์
            prev = values[:-1]
            nxt = values[1:]
            if np.std(prev) < 1e-8 or np.std(nxt) < 1e-8:
                pattern_persistence = 0.0
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    autocorr = np.corrcoef(prev, nxt)[0, 1]
                pattern_persistence = abs(autocorr) if not np.isnan(autocorr) else 0.0
        else:
            pattern_persistence = 0.0
        
        # Interpretation
        interpretation = {
            'is_stable': abs(drift_slope) < 0.01 and stability_variance < 0.1,
            'silent_failure_risk': abs(drift_slope) > self.drift_threshold and pattern_persistence < 0.3,
            'safe_operation': abs(drift_slope) < 0.05 and pattern_persistence > 0.5
        }
        
        return {
            'drift_slope': float(drift_slope),
            'stability_variance': float(stability_variance),
            'pattern_persistence': float(pattern_persistence),
            'data_points': len(data),
            'interpretation': interpretation
        }
    
    def record_metric(self, metric_name: str, value: float, timestamp: Optional[datetime] = None):
        """บันทึก metric สำหรับ trend analysis"""
        if timestamp is None:
            timestamp = datetime.now()
        self.trend_data[metric_name].append((timestamp, value))
    
    # ==================== 4.5 Cross-modal Analysis ====================
    
    def cross_modal_analysis(self, 
                            modal_embeddings: Dict[str, List[Union[torch.Tensor, np.ndarray]]]) -> Dict:
        """
        ตรวจสอบความสอดคล้องข้าม modality
        
        Args:
            modal_embeddings: Dict mapping modality name to list of embeddings
                e.g., {'text': [emb1, emb2], 'image': [emb1, emb2], 'audio': [emb1, emb2]}
        
        Returns:
            Dict with cross-modal metrics
        """
        if len(modal_embeddings) < 2:
            return {
                'cross_modal_alignment': 0.0,
                'modality_divergence': 0.0,
                'warning': 'Need at least 2 modalities for cross-modal analysis'
            }
        
        modalities = list(modal_embeddings.keys())
        
        # คำนวณ alignment scores ระหว่างทุกคู่ของ modalities
        alignment_scores = []
        divergence_scores = []
        
        for i, mod_a in enumerate(modalities):
            for mod_b in modalities[i+1:]:
                embs_a = [self._to_numpy(e).flatten() for e in modal_embeddings[mod_a]]
                embs_b = [self._to_numpy(e).flatten() for e in modal_embeddings[mod_b]]
                
                # ต้องมีจำนวน embeddings เท่ากัน
                min_len = min(len(embs_a), len(embs_b))
                if min_len == 0:
                    continue
                
                embs_a = embs_a[:min_len]
                embs_b = embs_b[:min_len]
                
                # คำนวณ pairwise similarities
                similarities = []
                for emb_a, emb_b in zip(embs_a, embs_b):
                    norm_a = np.linalg.norm(emb_a)
                    norm_b = np.linalg.norm(emb_b)
                    if norm_a > 0 and norm_b > 0:
                        sim = np.dot(emb_a, emb_b) / (norm_a * norm_b)
                        similarities.append(sim)
                
                if similarities:
                    alignment = np.mean(similarities)
                    alignment_scores.append(alignment)
                    
                    # Divergence = 1 - alignment
                    divergence = 1 - alignment
                    divergence_scores.append(divergence)
        
        cross_modal_alignment = np.mean(alignment_scores) if alignment_scores else 0.0
        modality_divergence = np.mean(divergence_scores) if divergence_scores else 0.0
        
        # ตรวจสอบ modality ที่ผิดปกติ
        abnormal_modalities = []
        for mod, embs in modal_embeddings.items():
            if not embs:
                abnormal_modalities.append(mod)
                continue
            
            # ตรวจสอบ norm ของ embeddings
            norms = [np.linalg.norm(self._to_numpy(e).flatten()) for e in embs]
            mean_norm = np.mean(norms)
            std_norm = np.std(norms)
            
            # ถ้า norm ผิดปกติ (0 หรือ infinity)
            if mean_norm == 0 or np.isinf(mean_norm) or std_norm / (mean_norm + 1e-8) > 2.0:
                abnormal_modalities.append(mod)
        
        # Interpretation
        interpretation = {
            'modality_abnormal': abnormal_modalities,
            'cross_modal_broken': modality_divergence > 0.5,
            'sensor_pipeline_issue': len(abnormal_modalities) > 0,
            'note': 'Abnormal modality may indicate sensor or pipeline issue'
        }
        
        return {
            'cross_modal_alignment': float(cross_modal_alignment),
            'modality_divergence': float(modality_divergence),
            'abnormal_modalities': abnormal_modalities,
            'interpretation': interpretation
        }
    
    # ==================== 5. Operational Status Indicators ====================
    
    def assess_operational_status(self, 
                                  embeddings: List[Union[torch.Tensor, np.ndarray]],
                                  include_trends: bool = True) -> OperationalStatus:
        """
        ประเมินสถานะการทำงานของระบบ
        
        Returns:
            OperationalStatus with status (NORMAL, WARNING, DEGRADED, UNSAFE)
        """
        reasons = []
        warning_count = 0
        degraded_count = 0
        unsafe_count = 0
        
        # 1. Anomaly Detection
        anomaly_result = self.anomaly_detection(embeddings)
        anomaly_density = anomaly_result.get('anomaly_density', 0)
        stress_index = anomaly_result.get('stress_index', 0)
        
        # ใช้ทั้ง anomaly_density และ stress_index ในการตัดสิน
        if anomaly_density > 0.5 or stress_index > 3.0:
            unsafe_count += 1
            reasons.append(f"High anomaly density: {anomaly_density:.2%}")
        elif anomaly_density > 0.3 or stress_index > 2.0:
            degraded_count += 1
            reasons.append(f"Moderate anomaly density: {anomaly_density:.2%}")
        elif anomaly_density > 0.15 or stress_index > 1.5:
            warning_count += 1
            reasons.append(f"Low anomaly density: {anomaly_density:.2%}")
        
        # 2. Cluster Analysis
        cluster_result = self.cluster_analysis(embeddings)
        distribution_shift = cluster_result.get('distribution_shift', 0)
        
        # ปรับ threshold สำหรับ distribution shift
        if distribution_shift > 5.0:  # Very large shift
            unsafe_count += 1
            reasons.append(f"Large distribution shift: {distribution_shift:.3f}")
        elif distribution_shift > 2.0:  # Moderate shift
            degraded_count += 1
            reasons.append(f"Moderate distribution shift: {distribution_shift:.3f}")
        elif distribution_shift > 1.0:  # Small shift
            warning_count += 1
            reasons.append(f"Small distribution shift: {distribution_shift:.3f}")
        
        # 3. Trend Analysis (if available)
        if include_trends:
            for metric_name in ['anomaly_density', 'distribution_shift']:
                trend_result = self.trend_analysis(metric_name, time_window=20)
                if trend_result.get('silent_failure_risk', False):
                    unsafe_count += 1
                    reasons.append(f"Silent failure risk detected in {metric_name}")
                elif abs(trend_result.get('drift_slope', 0)) > self.drift_threshold:
                    degraded_count += 1
                    reasons.append(f"Drift detected in {metric_name}")
        
        # 4. Determine Status
        if unsafe_count > 0:
            status = 'UNSAFE'
            confidence = min(0.9, 0.5 + unsafe_count * 0.1)
        elif degraded_count > 0:
            status = 'DEGRADED'
            confidence = min(0.8, 0.4 + degraded_count * 0.1)
        elif warning_count > 0:
            status = 'WARNING'
            confidence = min(0.7, 0.3 + warning_count * 0.1)
        else:
            status = 'NORMAL'
            confidence = 0.9
        
        if not reasons:
            reasons.append("All metrics within normal range")
        
        return OperationalStatus(
            status=status,
            confidence=confidence,
            reasons=reasons,
            timestamp=datetime.now()
        )
    
    # ==================== Helper Methods ====================
    
    def _to_numpy(self, embedding: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """แปลง embedding เป็น numpy array"""
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().cpu().numpy()
        return np.array(embedding)
    
    def comprehensive_analysis(self, 
                              embeddings: List[Union[torch.Tensor, np.ndarray]],
                              labels: Optional[List[str]] = None) -> Dict:
        """
        วิเคราะห์ครบทุก module
        
        Returns:
            Dict with all analysis results
        """
        results = {
            'timestamp': datetime.now().isoformat(),
            'embedding_count': len(embeddings)
        }
        
        # Similarity Analysis (pairwise)
        if len(embeddings) >= 2:
            similarity_results = []
            for i in range(len(embeddings) - 1):
                sim_result = self.similarity_analysis(embeddings[i], embeddings[i+1])
                similarity_results.append(sim_result)
            results['similarity_analysis'] = {
                'pairwise_results': similarity_results,
                'mean_similarity': np.mean([s['similarity'] for s in similarity_results])
            }
        
        # Cluster Analysis
        results['cluster_analysis'] = self.cluster_analysis(embeddings)
        
        # Anomaly Detection
        results['anomaly_detection'] = self.anomaly_detection(embeddings)
        
        # Operational Status
        results['operational_status'] = self.assess_operational_status(embeddings)
        
        # บันทึก metrics สำหรับ trend analysis
        if results.get('anomaly_detection'):
            self.record_metric('anomaly_density', results['anomaly_detection']['anomaly_density'])
        if results.get('cluster_analysis'):
            self.record_metric('distribution_shift', results['cluster_analysis']['distribution_shift'])
        
        # เก็บประวัติ
        self.history.append(results)
        
        return results
    
    def generate_report(self, save_path: Optional[str] = None) -> str:
        """สร้างรายงานสรุป"""
        report_lines = []
        report_lines.append("=" * 60)
        report_lines.append("BASELINE BEHAVIORAL ANALYSIS REPORT")
        report_lines.append("=" * 60)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        if not self.history:
            report_lines.append("No analysis history available.")
        else:
            latest = self.history[-1]
            report_lines.append("LATEST ANALYSIS:")
            report_lines.append("-" * 40)
            
            if 'operational_status' in latest:
                status = latest['operational_status']
                report_lines.append(f"Status: {status.status}")
                report_lines.append(f"Confidence: {status.confidence:.2%}")
                report_lines.append("Reasons:")
                for reason in status.reasons:
                    report_lines.append(f"  • {reason}")
            
            report_lines.append("")
            report_lines.append("METRICS:")
            report_lines.append("-" * 40)
            
            if 'anomaly_detection' in latest:
                ad = latest['anomaly_detection']
                report_lines.append(f"Anomaly Density: {ad['anomaly_density']:.2%}")
                report_lines.append(f"Stress Index: {ad['stress_index']:.3f}")
            
            if 'cluster_analysis' in latest:
                ca = latest['cluster_analysis']
                report_lines.append(f"Cluster Count: {ca['cluster_count']}")
                report_lines.append(f"Distribution Shift: {ca['distribution_shift']:.3f}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"Report saved to {save_path}")
        
        return report


# ==================== Example Usage ====================

def demo_behavioral_analysis():
    """Demo การใช้งาน Behavioral Analyzer"""
    print("🔍 Baseline Behavioral Analysis Demo")
    print("=" * 50)
    
    # 1. สร้าง baseline embeddings
    print("\n📊 Creating baseline embeddings...")
    baseline_embeddings = [np.random.randn(768) * 0.5 for _ in range(30)]
    
    # 2. สร้าง analyzer
    print("\n🔧 Initializing Behavioral Analyzer...")
    analyzer = BehavioralAnalyzer(
        baseline_embeddings=baseline_embeddings,
        similarity_threshold=0.7,
        anomaly_threshold=3.0,
        drift_threshold=0.15
    )
    print(f"  Baseline embeddings: {len(baseline_embeddings)}")
    print(f"  Similarity threshold: {analyzer.similarity_threshold}")
    print(f"  Anomaly threshold: {analyzer.anomaly_threshold}")
    
    # 3. สร้าง test embeddings หลายประเภท
    print("\n🔬 Creating test embeddings...")
    
    # Normal embeddings (คล้าย baseline)
    normal_embeddings = [np.random.randn(768) * 0.5 for _ in range(10)]
    
    # Noisy embeddings (anomaly)
    noisy_embeddings = [np.random.randn(768) * 2.0 for _ in range(5)]
    
    # Drifted embeddings (distribution shift)
    drifted_embeddings = [np.random.randn(768) * 0.5 + np.array([1.0] * 768) for _ in range(5)]
    
    # Sparse embeddings
    sparse_embeddings = []
    for _ in range(5):
        emb = np.zeros(768)
        emb[np.random.choice(768, size=100, replace=False)] = np.random.randn(100) * 0.5
        sparse_embeddings.append(emb)
    
    all_test_embeddings = normal_embeddings + noisy_embeddings + drifted_embeddings
    labels = ['Normal'] * 10 + ['Noisy'] * 5 + ['Drifted'] * 5
    
    # 4. Similarity Analysis
    print("\n🔗 Running Similarity Analysis...")
    print("  Comparing first two normal embeddings:")
    sim_result = analyzer.similarity_analysis(normal_embeddings[0], normal_embeddings[1])
    print(f"    Similarity: {sim_result['similarity']:.3f}")
    print(f"    Distance: {sim_result['distance']:.3f}")
    print(f"    Out-of-domain signal: {sim_result['interpretation']['out_of_domain_signal']}")
    
    # 5. Cluster Analysis
    print("\n📊 Running Cluster Analysis...")
    cluster_result = analyzer.cluster_analysis(all_test_embeddings)
    print(f"  Cluster Count: {cluster_result['cluster_count']}")
    print(f"  Cluster Density: {cluster_result['cluster_density']:.3f}")
    print(f"  Distribution Shift: {cluster_result['distribution_shift']:.3f}")
    print(f"  Behavior Change: {cluster_result['interpretation']['behavior_change']}")
    print(f"  Over-constraint Risk: {cluster_result['interpretation']['over_constraint_risk']}")
    
    # 6. Anomaly Detection
    print("\n🚨 Running Anomaly Detection...")
    anomaly_result = analyzer.anomaly_detection(all_test_embeddings)
    print(f"  Anomaly Score: {anomaly_result['anomaly_score']:.3f}")
    print(f"  Anomaly Density: {anomaly_result['anomaly_density']:.2%}")
    print(f"  Stress Index: {anomaly_result['stress_index']:.3f}")
    print(f"  Anomalies Detected: {anomaly_result['interpretation']['anomaly_detected']}")
    print(f"  Degradation Risk: {anomaly_result['interpretation']['degradation_risk']}")
    
    # แสดงรายละเอียด anomalies
    anomaly_count = sum(1 for a in anomaly_result['anomaly_details'] if a['is_anomaly'])
    print(f"  Total Anomalous Embeddings: {anomaly_count}/{len(all_test_embeddings)}")
    
    # 7. Trend Analysis (จำลองข้อมูลตามเวลา)
    print("\n📈 Running Trend Analysis...")
    from datetime import timedelta
    base_time = datetime.now()
    
    # บันทึก metrics ตามเวลา
    for i in range(20):
        timestamp = base_time + timedelta(seconds=i*10)
        # จำลอง anomaly density ที่เพิ่มขึ้น
        value = 0.05 + i * 0.01
        analyzer.record_metric('anomaly_density', value, timestamp)
    
    trend_result = analyzer.trend_analysis('anomaly_density', time_window=20)
    print(f"  Drift Slope: {trend_result['drift_slope']:.6f}")
    print(f"  Stability Variance: {trend_result['stability_variance']:.3f}")
    print(f"  Pattern Persistence: {trend_result['pattern_persistence']:.3f}")
    print(f"  Is Stable: {trend_result['interpretation']['is_stable']}")
    print(f"  Silent Failure Risk: {trend_result['interpretation']['silent_failure_risk']}")
    
    # 8. Cross-modal Analysis
    print("\n🌐 Running Cross-modal Analysis...")
    modal_embeddings = {
        'text': normal_embeddings[:5],
        'image': [np.random.randn(768) * 0.5 for _ in range(5)],
        'audio': [np.random.randn(768) * 0.5 for _ in range(5)]
    }
    
    cross_modal_result = analyzer.cross_modal_analysis(modal_embeddings)
    print(f"  Cross-modal Alignment: {cross_modal_result['cross_modal_alignment']:.3f}")
    print(f"  Modality Divergence: {cross_modal_result['modality_divergence']:.3f}")
    print(f"  Abnormal Modalities: {cross_modal_result['abnormal_modalities']}")
    print(f"  Cross-modal Broken: {cross_modal_result['interpretation']['cross_modal_broken']}")
    
    # 9. Comprehensive Analysis
    print("\n📋 Running Comprehensive Analysis...")
    comprehensive_results = analyzer.comprehensive_analysis(all_test_embeddings)
    
    # 10. Operational Status Assessment
    print("\n⚡ Assessing Operational Status...")
    status = comprehensive_results['operational_status']
    print(f"  Status: {status.status}")
    print(f"  Confidence: {status.confidence:.2%}")
    print(f"  Timestamp: {status.timestamp.isoformat()}")
    print(f"  Reasons:")
    for reason in status.reasons:
        print(f"    • {reason}")
    
    # 11. เปรียบเทียบ embeddings หลายกลุ่ม
    print("\n📊 Comparing Different Embedding Groups...")
    groups = {
        'Normal': normal_embeddings,
        'Noisy': noisy_embeddings,
        'Drifted': drifted_embeddings
    }
    
    for group_name, group_embeddings in groups.items():
        group_status = analyzer.assess_operational_status(group_embeddings, include_trends=False)
        group_anomaly = analyzer.anomaly_detection(group_embeddings)
        print(f"\n  {group_name} Group:")
        print(f"    Status: {group_status.status}")
        print(f"    Anomaly Density: {group_anomaly['anomaly_density']:.2%}")
        print(f"    Stress Index: {group_anomaly['stress_index']:.3f}")
    
    # 12. Generate Report
    print("\n📝 Generating report...")
    report = analyzer.generate_report(save_path='behavioral_analysis_report.txt')
    print("  Report saved to 'behavioral_analysis_report.txt'")
    
    # 13. Summary
    print("\n" + "=" * 50)
    print("📊 SUMMARY")
    print("=" * 50)
    print(f"Total embeddings analyzed: {len(all_test_embeddings)}")
    print(f"Operational Status: {status.status}")
    print(f"Anomaly Density: {anomaly_result['anomaly_density']:.2%}")
    print(f"Cluster Count: {cluster_result['cluster_count']}")
    print(f"Distribution Shift: {cluster_result['distribution_shift']:.3f}")
    
    print("\n✅ Demo completed!")
    print("\n🎯 Try it with your own embeddings:")
    print("""
    from .behavioral_analyzer import BehavioralAnalyzer
    import numpy as np
    
    # สร้าง analyzer
    baseline = [np.random.randn(768) for _ in range(20)]
    analyzer = BehavioralAnalyzer(baseline_embeddings=baseline)
    
    # วิเคราะห์ embeddings ของคุณ
    your_embeddings = [...]  # List of embeddings
    results = analyzer.comprehensive_analysis(your_embeddings)
    
    # ตรวจสอบสถานะ
    status = analyzer.assess_operational_status(your_embeddings)
    print(f"Status: {status.status}")
    """)
    
    return analyzer, comprehensive_results


# ฟังก์ชันสำหรับตรวจสอบ behavioral analysis ของโมเดล
def analyze_model_behavior(model: torch.nn.Module,
                           sample_inputs: List[Any] = None,
                           embedding_extractor = None):
    """
    วิเคราะห์พฤติกรรมของโมเดล
    
    Args:
        model: PyTorch model
        sample_inputs: ตัวอย่าง inputs สำหรับทดสอบ
        embedding_extractor: ฟังก์ชันสำหรับดึง embeddings จาก model output
                           ถ้า None จะพยายามดึงจาก last_hidden_state
    
    Returns:
        Dict with behavioral analysis results
    """
    print("🧠 Analyzing Model Behavior")
    print("=" * 50)
    
    # สร้าง analyzer
    analyzer = BehavioralAnalyzer()
    
    # หา embedding layer ในโมเดล (ถ้ามี)
    embedding_layers = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Embedding):
            embedding_layers.append((name, module))
    
    if embedding_layers:
        print(f"Found {len(embedding_layers)} embedding layer(s):")
        for name, layer in embedding_layers:
            print(f"  - {name}: {layer.weight.shape}")
        
        # วิเคราะห์ embedding weights
        print("\n📊 Analyzing embedding weights...")
        for name, layer in embedding_layers:
            weights = layer.weight.detach().cpu().numpy()
            
            # แปลงเป็น list of embeddings (แต่ละ row เป็น embedding)
            weight_embeddings = [weights[i] for i in range(min(100, len(weights)))]
            
            # วิเคราะห์
            cluster_result = analyzer.cluster_analysis(weight_embeddings)
            anomaly_result = analyzer.anomaly_detection(weight_embeddings)
            
            print(f"\n  {name}:")
            print(f"    Cluster Count: {cluster_result['cluster_count']}")
            print(f"    Anomaly Density: {anomaly_result['anomaly_density']:.2%}")
    
    # วิเคราะห์จาก sample inputs ถ้ามี
    if sample_inputs and embedding_extractor:
        print("\n📝 Analyzing embeddings from sample inputs...")
        
        sample_embeddings = []
        with torch.no_grad():
            for input_data in sample_inputs[:10]:  # ใช้ 10 ตัวอย่างแรก
                if isinstance(input_data, dict):
                    outputs = model(**input_data)
                else:
                    outputs = model(input_data)
                
                # ดึง embedding
                emb = embedding_extractor(outputs)
                if isinstance(emb, torch.Tensor):
                    emb = emb.detach().cpu().numpy()
                
                if len(emb.shape) > 1:
                    # ถ้าเป็น batch, ใช้ตัวแรก
                    emb = emb[0] if len(emb.shape) == 2 else emb.flatten()
                
                sample_embeddings.append(emb)
        
        if sample_embeddings:
            # Comprehensive analysis
            results = analyzer.comprehensive_analysis(sample_embeddings)
            
            print(f"\n  Sample Embeddings Analysis:")
            print(f"    Operational Status: {results['operational_status'].status}")
            print(f"    Anomaly Density: {results['anomaly_detection']['anomaly_density']:.2%}")
            print(f"    Cluster Count: {results['cluster_analysis']['cluster_count']}")
            
            return results
    
    return None


if __name__ == "__main__":
    analyzer, results = demo_behavioral_analysis()

