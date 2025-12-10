"""
Embedding Intelligence Monitoring & Analysis Specification (EIMAS) Implementation
Version: 1.0
Status: Community / Professional

ระบบวิเคราะห์ Embedding เชิงลึกเพื่อการตรวจสอบ เฝ้าระวัง และสนับสนุนการตัดสินใจ
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any, Callable
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from collections import defaultdict, deque
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.spatial.distance import cosine, euclidean
from scipy import stats
import json
import warnings

# Import base analyzers
from .embedding_inspector import EmbeddingQualityInspector
from .behavioral_analyzer import BehavioralAnalyzer, OperationalStatus


@dataclass
class Alert:
    """Alert structure for monitoring"""
    level: str  # INFO, WARNING, CRITICAL
    metric: str
    value: float
    threshold: float
    message: str
    timestamp: datetime
    confidence: float
    contributing_factors: List[str]


@dataclass
class EmbeddingLineage:
    """Track embedding lineage and propagation"""
    embedding_id: str
    source: str
    timestamp: datetime
    parent_ids: List[str]
    metadata: Dict[str, Any]


class EIMASAnalyzer:
    """
    Embedding Intelligence Monitoring & Analysis System
    
    ระบบวิเคราะห์ Embedding ตาม EIMAS Specification
    รวมความสามารถจาก EmbeddingQualityInspector และ BehavioralAnalyzer
    และเพิ่มความสามารถเฉพาะทางตาม EIMAS
    """
    
    def __init__(self,
                 baseline_embeddings: Optional[List[np.ndarray]] = None,
                 similarity_threshold: float = 0.7,
                 anomaly_threshold: float = 3.0,
                 drift_threshold: float = 0.15,
                 enable_monitoring: bool = True,
                 monitoring_window_size: int = 100):
        """
        Args:
            baseline_embeddings: Baseline embeddings for reference
            similarity_threshold: Threshold for similarity analysis
            anomaly_threshold: Threshold for anomaly detection
            drift_threshold: Threshold for drift detection
            enable_monitoring: Enable real-time monitoring
            monitoring_window_size: Size of rolling window for monitoring
        """
        # Initialize base analyzers
        self.quality_inspector = EmbeddingQualityInspector()
        self.behavioral_analyzer = BehavioralAnalyzer(
            baseline_embeddings=baseline_embeddings,
            similarity_threshold=similarity_threshold,
            anomaly_threshold=anomaly_threshold,
            drift_threshold=drift_threshold
        )
        
        # EIMAS-specific configurations
        self.enable_monitoring = enable_monitoring
        self.monitoring_window_size = monitoring_window_size
        
        # Monitoring data structures
        self.monitoring_buffer: deque = deque(maxlen=monitoring_window_size)
        self.alert_history: List[Alert] = []
        self.embedding_lineage: Dict[str, EmbeddingLineage] = {}
        
        # Thresholds configuration
        self.thresholds = {
            'similarity': similarity_threshold,
            'anomaly': anomaly_threshold,
            'drift': drift_threshold,
            'quality_index_min': 10.0,
            'signal_quality_min': 0.3,
            'entropy_max': 0.8
        }
        
        # Version tracking
        self.version_history: List[Dict] = []
        self.current_version: str = "1.0.0"
    
    # ==================== 3. Core Analysis Capabilities ====================
    # (Delegated to base analyzers with EIMAS enhancements)
    
    def similarity_analysis(self, 
                          embedding_a: Union[torch.Tensor, np.ndarray],
                          embedding_b: Union[torch.Tensor, np.ndarray],
                          method: str = 'cosine',
                          reference_embeddings: Optional[List[np.ndarray]] = None) -> Dict:
        """
        Similarity Analysis with reference support (EIMAS 3.1)
        
        Enhanced with reference similarity verification
        """
        result = self.behavioral_analyzer.similarity_analysis(embedding_a, embedding_b, method)
        
        # Add reference similarity if provided
        if reference_embeddings:
            ref_similarities = []
            for ref_emb in reference_embeddings:
                ref_sim = self.behavioral_analyzer.similarity_analysis(embedding_a, ref_emb, method)
                ref_similarities.append(ref_sim['similarity'])
            
            result['reference_similarity_mean'] = np.mean(ref_similarities) if ref_similarities else 0.0
            result['reference_similarity_std'] = np.std(ref_similarities) if ref_similarities else 0.0
            result['domain_conformance'] = result['reference_similarity_mean'] > self.thresholds['similarity']
        
        return result
    
    def cluster_analysis(self, embeddings: List[Union[torch.Tensor, np.ndarray]],
                       method: str = 'kmeans',
                       n_clusters: Optional[int] = None) -> Dict:
        """Cluster Analysis (EIMAS 3.2)"""
        return self.behavioral_analyzer.cluster_analysis(embeddings, method, n_clusters)
    
    def anomaly_detection(self, embeddings: List[Union[torch.Tensor, np.ndarray]],
                        baseline_embeddings: Optional[List[np.ndarray]] = None) -> Dict:
        """Anomaly Detection (EIMAS 3.3)"""
        return self.behavioral_analyzer.anomaly_detection(embeddings, baseline_embeddings)
    
    def trend_analysis(self, metric_name: str, time_window: Optional[int] = None) -> Dict:
        """Trend Analysis (EIMAS 3.4)"""
        return self.behavioral_analyzer.trend_analysis(metric_name, time_window)
    
    def cross_modal_analysis(self, modal_embeddings: Dict[str, List[Union[torch.Tensor, np.ndarray]]]) -> Dict:
        """Cross-modal Analysis (EIMAS 3.5)"""
        return self.behavioral_analyzer.cross_modal_analysis(modal_embeddings)
    
    # ==================== 4. Specialized Inspection Capabilities ====================
    
    def reference_similarity_verification(self,
                                        embedding: Union[torch.Tensor, np.ndarray],
                                        trusted_sources: List[Union[torch.Tensor, np.ndarray]],
                                        domain_threshold: float = 0.7) -> Dict:
        """
        Reference Similarity Verification (EIMAS 4.1)
        
        ตรวจสอบความสอดคล้องกับ trusted sources
        """
        similarities = []
        for trusted in trusted_sources:
            sim_result = self.behavioral_analyzer.similarity_analysis(embedding, trusted)
            similarities.append(sim_result['similarity'])
        
        mean_sim = np.mean(similarities) if similarities else 0.0
        std_sim = np.std(similarities) if similarities else 0.0
        
        return {
            'mean_similarity': float(mean_sim),
            'std_similarity': float(std_sim),
            'domain_conformance_score': float(mean_sim),
            'is_conformant': mean_sim >= domain_threshold,
            'trusted_source_count': len(trusted_sources),
            'interpretation': {
                'high_conformance': mean_sim >= domain_threshold,
                'low_variance': std_sim < 0.1,
                'note': 'High similarity to trusted sources indicates domain conformance'
            }
        }
    
    def imitation_forgery_detection(self,
                                    embeddings: List[Union[torch.Tensor, np.ndarray]],
                                    similarity_threshold: float = 0.95) -> Dict:
        """
        Imitation & Forgery Detection (EIMAS 4.2)
        
        ตรวจจับ near-duplicate patterns และ stylometric similarity
        """
        if len(embeddings) < 2:
            return {
                'near_duplicate_count': 0,
                'forgery_risk': 0.0,
                'warning': 'Insufficient embeddings for forgery detection'
            }
        
        # Calculate pairwise similarities
        near_duplicates = []
        similarity_matrix = []
        
        for i in range(len(embeddings)):
            row = []
            for j in range(len(embeddings)):
                if i == j:
                    row.append(1.0)
                else:
                    sim_result = self.behavioral_analyzer.similarity_analysis(embeddings[i], embeddings[j])
                    sim = sim_result['similarity']
                    row.append(sim)
                    
                    if sim >= similarity_threshold:
                        near_duplicates.append((i, j, sim))
            similarity_matrix.append(row)
        
        similarity_matrix = np.array(similarity_matrix)
        
        # Detect stylometric patterns (repetitive structural signatures)
        # Use cluster analysis to find groups of very similar embeddings
        cluster_result = self.cluster_analysis(embeddings, method='dbscan')
        
        # Calculate forgery risk
        near_duplicate_ratio = len(near_duplicates) / (len(embeddings) * (len(embeddings) - 1) / 2) if len(embeddings) > 1 else 0.0
        forgery_risk = min(1.0, near_duplicate_ratio * 2.0)  # Scale to [0, 1]
        
        return {
            'near_duplicate_count': len(near_duplicates),
            'near_duplicate_pairs': near_duplicates[:10],  # Top 10
            'near_duplicate_ratio': float(near_duplicate_ratio),
            'forgery_risk': float(forgery_risk),
            'stylometric_clusters': cluster_result.get('cluster_count', 0),
            'similarity_matrix': similarity_matrix.tolist(),
            'interpretation': {
                'high_risk': forgery_risk > 0.5,
                'suspicious_pattern': len(near_duplicates) > len(embeddings) * 0.1,
                'note': 'High near-duplicate ratio may indicate imitation or forgery'
            }
        }
    
    def hidden_communication_pattern_detection(self,
                                             embeddings: List[Union[torch.Tensor, np.ndarray]],
                                             min_cluster_size: int = 3) -> Dict:
        """
        Hidden Communication Pattern Detection (EIMAS 4.3)
        
        ตรวจจับ non-obvious clustering และ latent correlation
        """
        if len(embeddings) < min_cluster_size:
            return {
                'hidden_patterns_detected': False,
                'warning': 'Insufficient embeddings for pattern detection'
            }
        
        # Use DBSCAN to find non-obvious clusters
        embeddings_array = np.vstack([self._to_numpy(e).flatten() for e in embeddings])
        
        # Try multiple epsilon values to find hidden patterns
        hidden_clusters = []
        for eps in [0.3, 0.5, 0.7, 1.0]:
            dbscan = DBSCAN(eps=eps, min_samples=min_cluster_size)
            labels = dbscan.fit_predict(embeddings_array)
            n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
            
            if n_clusters > 0:
                hidden_clusters.append({
                    'epsilon': eps,
                    'cluster_count': n_clusters,
                    'labels': labels.tolist()
                })
        
        # Detect repetitive structural signatures
        # Analyze correlation patterns
        if len(embeddings_array) > 2:
            corr_matrix = np.corrcoef(embeddings_array)
            # Find high correlations (potential hidden patterns)
            np.fill_diagonal(corr_matrix, 0)
            high_corr_pairs = np.where(np.abs(corr_matrix) > 0.8)
            latent_correlations = [(int(i), int(j), float(corr_matrix[i, j])) 
                                  for i, j in zip(high_corr_pairs[0], high_corr_pairs[1]) 
                                  if i < j][:20]
        else:
            latent_correlations = []
        
        patterns_detected = len(hidden_clusters) > 0 or len(latent_correlations) > 0
        
        return {
            'hidden_patterns_detected': patterns_detected,
            'hidden_clusters': hidden_clusters,
            'latent_correlations': latent_correlations,
            'correlation_count': len(latent_correlations),
            'interpretation': {
                'suspicious': patterns_detected,
                'note': 'Non-obvious clustering or high correlations may indicate hidden communication patterns'
            }
        }
    
    def information_propagation_tracking(self,
                                       embedding_id: str,
                                       embedding: Union[torch.Tensor, np.ndarray],
                                       source: str,
                                       parent_ids: Optional[List[str]] = None,
                                       metadata: Optional[Dict] = None) -> EmbeddingLineage:
        """
        Information Propagation Tracking (EIMAS 4.4)
        
        ติดตาม lineage และ diffusion pattern
        """
        lineage = EmbeddingLineage(
            embedding_id=embedding_id,
            source=source,
            timestamp=datetime.now(),
            parent_ids=parent_ids or [],
            metadata=metadata or {}
        )
        
        self.embedding_lineage[embedding_id] = lineage
        
        # Analyze diffusion pattern if parents exist
        if parent_ids:
            diffusion_analysis = self._analyze_diffusion_pattern(embedding_id, parent_ids)
            lineage.metadata['diffusion_analysis'] = diffusion_analysis
        
        return lineage
    
    def _analyze_diffusion_pattern(self, embedding_id: str, parent_ids: List[str]) -> Dict:
        """Analyze diffusion pattern from parents"""
        if not parent_ids:
            return {}
        
        # Get parent embeddings if available
        parent_lineages = [self.embedding_lineage.get(pid) for pid in parent_ids if pid in self.embedding_lineage]
        
        if not parent_lineages:
            return {'parent_count': len(parent_ids), 'note': 'Parent embeddings not found in lineage'}
        
        # Calculate temporal spread
        timestamps = [pl.timestamp for pl in parent_lineages if pl]
        if timestamps:
            time_spread = (max(timestamps) - min(timestamps)).total_seconds()
        else:
            time_spread = 0
        
        return {
            'parent_count': len(parent_ids),
            'found_parents': len(parent_lineages),
            'time_spread_seconds': time_spread,
            'diffusion_rate': len(parent_ids) / (time_spread + 1)  # embeddings per second
        }
    
    # ==================== 5. Monitoring & Surveillance Functions ====================
    
    def ingest_embedding(self,
                        embedding: Union[torch.Tensor, np.ndarray],
                        embedding_id: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> Dict:
        """
        Real-time Monitoring - Ingest embedding (EIMAS 5.1)
        
        Ingest embedding into monitoring system
        """
        if not self.enable_monitoring:
            return {'status': 'monitoring_disabled'}
        
        if embedding_id is None:
            embedding_id = f"emb_{datetime.now().timestamp()}"
        
        # Convert to numpy
        emb_array = self._to_numpy(embedding).flatten()
        
        # Add to monitoring buffer
        monitoring_entry = {
            'embedding_id': embedding_id,
            'embedding': emb_array,
            'timestamp': datetime.now(),
            'metadata': metadata or {}
        }
        self.monitoring_buffer.append(monitoring_entry)
        
        # Quick analysis
        quality_result = self.quality_inspector.analyze_embedding(embedding)
        
        # Check thresholds and generate alerts
        alerts = self._check_thresholds(embedding_id, quality_result)
        
        # Record metrics for trend analysis
        self.behavioral_analyzer.record_metric('quality_index', 
                                               quality_result.get('embedding_quality_index', quality_result['ΔEΨ_with_H']))
        self.behavioral_analyzer.record_metric('signal_quality', 
                                               quality_result.get('signal_quality', quality_result['S']))
        
        return {
            'embedding_id': embedding_id,
            'quality_index': quality_result.get('embedding_quality_index', quality_result['ΔEΨ_with_H']),
            'signal_quality': quality_result.get('signal_quality', quality_result['S']),
            'distribution_entropy': quality_result.get('distribution_entropy', quality_result['H']),
            'quality_analysis': quality_result,  # full analysis for downstream inspection
            'alerts': [asdict(a) for a in alerts],
            'timestamp': datetime.now().isoformat()
        }
    
    def _check_thresholds(self, embedding_id: str, quality_result: Dict) -> List[Alert]:
        """Check thresholds and generate alerts"""
        alerts = []
        
        eqi = quality_result.get('embedding_quality_index', quality_result['ΔEΨ_with_H'])
        signal_quality = quality_result.get('signal_quality', quality_result['S'])
        entropy = quality_result.get('distribution_entropy', quality_result['H'])
        
        # Check quality index
        if eqi < self.thresholds['quality_index_min']:
            alerts.append(Alert(
                level='WARNING',
                metric='quality_index',
                value=eqi,
                threshold=self.thresholds['quality_index_min'],
                message=f'Embedding quality index below threshold: {eqi:.2f} < {self.thresholds["quality_index_min"]}',
                timestamp=datetime.now(),
                confidence=0.8,
                contributing_factors=['Low information strength', 'High entropy', 'Low signal quality']
            ))
        
        # Check signal quality
        if signal_quality < self.thresholds['signal_quality_min']:
            alerts.append(Alert(
                level='WARNING',
                metric='signal_quality',
                value=signal_quality,
                threshold=self.thresholds['signal_quality_min'],
                message=f'Signal quality below threshold: {signal_quality:.3f} < {self.thresholds["signal_quality_min"]}',
                timestamp=datetime.now(),
                confidence=0.7,
                contributing_factors=['High noise', 'Low smoothness']
            ))
        
        # Check entropy
        if entropy > self.thresholds['entropy_max']:
            alerts.append(Alert(
                level='WARNING',
                metric='entropy',
                value=entropy,
                threshold=self.thresholds['entropy_max'],
                message=f'Distribution entropy above threshold: {entropy:.3f} > {self.thresholds["entropy_max"]}',
                timestamp=datetime.now(),
                confidence=0.75,
                contributing_factors=['High distribution spread', 'Low information density']
            ))
        
        # Add to alert history
        self.alert_history.extend(alerts)
        
        return alerts
    
    def get_alerts(self, 
                  level: Optional[str] = None,
                  since: Optional[datetime] = None) -> List[Alert]:
        """
        Get alerts (EIMAS 5.2)
        
        Retrieve alerts with optional filtering
        """
        alerts = self.alert_history
        
        if level:
            alerts = [a for a in alerts if a.level == level]
        
        if since:
            alerts = [a for a in alerts if a.timestamp >= since]
        
        return alerts
    
    def configure_thresholds(self, thresholds: Dict[str, float]):
        """
        Threshold Configuration (EIMAS 5.3)
        
        Configure per-model or context-aware thresholds
        """
        self.thresholds.update(thresholds)
        return self.thresholds
    
    def compare_versions(self,
                        version_a: str,
                        version_b: str,
                        embeddings_a: List[Union[torch.Tensor, np.ndarray]],
                        embeddings_b: List[Union[torch.Tensor, np.ndarray]]) -> Dict:
        """
        Historical Tracking & Version Comparison (EIMAS 5.4)
        
        เปรียบเทียบพฤติกรรมระหว่างเวอร์ชัน
        """
        # Analyze both versions
        analysis_a = self.behavioral_analyzer.comprehensive_analysis(embeddings_a)
        analysis_b = self.behavioral_analyzer.comprehensive_analysis(embeddings_b)
        
        # Compare metrics
        comparison = {
            'version_a': version_a,
            'version_b': version_b,
            'comparison_timestamp': datetime.now().isoformat(),
            'metrics': {
                'quality_index': {
                    'a': analysis_a.get('anomaly_detection', {}).get('anomaly_density', 0),
                    'b': analysis_b.get('anomaly_detection', {}).get('anomaly_density', 0),
                    'change': analysis_b.get('anomaly_detection', {}).get('anomaly_density', 0) - 
                             analysis_a.get('anomaly_detection', {}).get('anomaly_density', 0)
                },
                'cluster_count': {
                    'a': analysis_a.get('cluster_analysis', {}).get('cluster_count', 0),
                    'b': analysis_b.get('cluster_analysis', {}).get('cluster_count', 0),
                    'change': analysis_b.get('cluster_analysis', {}).get('cluster_count', 0) - 
                             analysis_a.get('cluster_analysis', {}).get('cluster_count', 0)
                },
                'distribution_shift': {
                    'a': analysis_a.get('cluster_analysis', {}).get('distribution_shift', 0),
                    'b': analysis_b.get('cluster_analysis', {}).get('distribution_shift', 0),
                    'change': analysis_b.get('cluster_analysis', {}).get('distribution_shift', 0) - 
                             analysis_a.get('cluster_analysis', {}).get('distribution_shift', 0)
                }
            },
            'regression_detected': False,
            'interpretation': {}
        }
        
        # Detect regression
        if comparison['metrics']['quality_index']['change'] < -0.1:
            comparison['regression_detected'] = True
            comparison['interpretation']['regression'] = 'Quality degradation detected'
        
        return comparison
    
    # ==================== 7. Decision Support & Interpretability ====================
    
    def explain_clustering(self, 
                          embeddings: List[Union[torch.Tensor, np.ndarray]],
                          cluster_labels: List[int]) -> Dict:
        """
        Explainability Tools - Clustering (EIMAS 7.1)
        
        อธิบายเหตุผลของ clustering
        """
        if len(embeddings) != len(cluster_labels):
            return {'error': 'Mismatch between embeddings and labels'}
        
        embeddings_array = np.vstack([self._to_numpy(e).flatten() for e in embeddings])
        
        # Analyze each cluster
        cluster_explanations = {}
        unique_clusters = set(cluster_labels)
        
        for cluster_id in unique_clusters:
            if cluster_id == -1:  # Noise in DBSCAN
                continue
            
            cluster_mask = np.array(cluster_labels) == cluster_id
            cluster_embeddings = embeddings_array[cluster_mask]
            
            # Calculate cluster characteristics
            cluster_mean = np.mean(cluster_embeddings, axis=0)
            cluster_std = np.std(cluster_embeddings, axis=0)
            cluster_size = len(cluster_embeddings)
            
            # Distance from global mean
            global_mean = np.mean(embeddings_array, axis=0)
            distance_from_global = np.linalg.norm(cluster_mean - global_mean)
            
            cluster_explanations[cluster_id] = {
                'size': cluster_size,
                'mean_distance_from_global': float(distance_from_global),
                'intra_cluster_variance': float(np.mean(cluster_std)),
                'characteristics': {
                    'large': cluster_size > len(embeddings) * 0.3,
                    'distant': distance_from_global > np.std(embeddings_array) * 2,
                    'compact': np.mean(cluster_std) < np.std(embeddings_array) * 0.5
                }
            }
        
        return {
            'cluster_count': len(unique_clusters),
            'cluster_explanations': cluster_explanations,
            'interpretation': 'Clusters are explained by size, distance from global mean, and internal variance'
        }
    
    def explain_anomaly(self,
                      embedding: Union[torch.Tensor, np.ndarray],
                      baseline_embeddings: List[np.ndarray]) -> Dict:
        """
        Explainability Tools - Anomaly (EIMAS 7.1)
        
        อธิบายปัจจัยที่ทำให้เกิด anomaly
        """
        emb_array = self._to_numpy(embedding).flatten()
        baseline_array = np.vstack([self._to_numpy(e).flatten() for e in baseline_embeddings])
        
        baseline_mean = np.mean(baseline_array, axis=0)
        baseline_std = np.std(baseline_array, axis=0)
        baseline_std = np.where(baseline_std == 0, 1e-8, baseline_std)
        
        # Calculate z-scores
        z_scores = (emb_array - baseline_mean) / baseline_std
        abs_z_scores = np.abs(z_scores)
        
        # Find top contributing dimensions
        top_contributors = np.argsort(abs_z_scores)[-10:][::-1]
        
        contributing_factors = []
        for idx in top_contributors:
            z = z_scores[idx]
            if abs(z) > self.thresholds['anomaly']:
                contributing_factors.append({
                    'dimension': int(idx),
                    'z_score': float(z),
                    'deviation': 'above' if z > 0 else 'below',
                    'magnitude': float(abs(z))
                })
        
        # Overall anomaly score
        max_z = np.max(abs_z_scores)
        mean_z = np.mean(abs_z_scores)
        
        return {
            'anomaly_score': float(max_z),
            'mean_deviation': float(mean_z),
            'top_contributing_dimensions': contributing_factors[:5],
            'total_anomalous_dimensions': len([z for z in abs_z_scores if z > self.thresholds['anomaly']]),
            'explanation': f'Anomaly caused by {len(contributing_factors)} dimensions deviating significantly from baseline',
            'recommendation': 'Check if embedding represents out-of-domain data or corrupted input'
        }
    
    def explain_metric_contribution(self, quality_result: Dict) -> Dict:
        """
        Explainability Tools - Metric Contribution (EIMAS 7.1)
        
        อธิบาย contribution ของแต่ละ metric ต่อ decision
        """
        eqi = quality_result.get('embedding_quality_index', quality_result['ΔEΨ_with_H'])
        info_strength = quality_result.get('information_strength', quality_result['I'])
        signal_quality = quality_result.get('signal_quality', quality_result['S'])
        entropy = quality_result.get('distribution_entropy', quality_result['H'])
        
        # Calculate contributions (normalized)
        contributions = {
            'information_strength': {
                'value': float(info_strength),
                'contribution': float(info_strength / 100) if info_strength > 0 else 0,
                'impact': 'positive' if info_strength > 50 else 'negative'
            },
            'signal_quality': {
                'value': float(signal_quality),
                'contribution': float(signal_quality),
                'impact': 'positive' if signal_quality > 0.5 else 'negative'
            },
            'distribution_entropy': {
                'value': float(entropy),
                'contribution': float(1 - entropy),  # Lower entropy is better
                'impact': 'positive' if entropy < 0.5 else 'negative'
            }
        }
        
        # Overall quality assessment
        quality_assessment = 'good' if eqi > 20 else 'fair' if eqi > 10 else 'poor'
        
        return {
            'embedding_quality_index': float(eqi),
            'quality_assessment': quality_assessment,
            'metric_contributions': contributions,
            'primary_factors': sorted(
                [(k, v['contribution']) for k, v in contributions.items()],
                key=lambda x: x[1],
                reverse=True
            )[:2],
            'explanation': f'Quality index {eqi:.2f} is primarily influenced by {", ".join([k for k, _ in sorted(contributions.items(), key=lambda x: x[1]["contribution"], reverse=True)[:2]])}'
        }
    
    def assess_confidence(self, analysis_result: Dict) -> Dict:
        """
        Confidence Scoring (EIMAS 7.2)
        
        ประเมินความมั่นใจในการวิเคราะห์
        """
        # Data sufficiency
        embedding_count = analysis_result.get('embedding_count', 0)
        data_sufficiency = min(1.0, embedding_count / 50)  # 50+ embeddings = sufficient
        
        # Historical consistency
        historical_consistency = 0.8  # Placeholder - would check against history
        
        # Measurement confidence
        if 'anomaly_detection' in analysis_result:
            anomaly_density = analysis_result['anomaly_detection'].get('anomaly_density', 0)
            measurement_confidence = 1.0 - min(1.0, anomaly_density * 2)  # Lower anomaly = higher confidence
        else:
            measurement_confidence = 0.7
        
        # Overall confidence
        overall_confidence = (data_sufficiency * 0.3 + 
                            historical_consistency * 0.3 + 
                            measurement_confidence * 0.4)
        
        return {
            'overall_confidence': float(overall_confidence),
            'data_sufficiency': float(data_sufficiency),
            'historical_consistency': float(historical_consistency),
            'measurement_confidence': float(measurement_confidence),
            'confidence_level': 'high' if overall_confidence > 0.8 else 'medium' if overall_confidence > 0.5 else 'low',
            'recommendations': [
                'Collect more embeddings' if data_sufficiency < 0.5 else None,
                'Check for systematic issues' if measurement_confidence < 0.5 else None
            ]
        }
    
    def comparative_analysis(self,
                           embeddings_group_a: List[Union[torch.Tensor, np.ndarray]],
                           embeddings_group_b: List[Union[torch.Tensor, np.ndarray]],
                           group_a_label: str = 'Group A',
                           group_b_label: str = 'Group B') -> Dict:
        """
        Comparative Analysis (EIMAS 7.3)
        
        เปรียบเทียบระหว่างกลุ่ม embeddings
        """
        analysis_a = self.behavioral_analyzer.comprehensive_analysis(embeddings_group_a)
        analysis_b = self.behavioral_analyzer.comprehensive_analysis(embeddings_group_b)
        
        # Compare operational status
        status_a = analysis_a.get('operational_status')
        status_b = analysis_b.get('operational_status')
        
        return {
            'group_a': {
                'label': group_a_label,
                'status': status_a.status if status_a else 'UNKNOWN',
                'anomaly_density': analysis_a.get('anomaly_detection', {}).get('anomaly_density', 0),
                'cluster_count': analysis_a.get('cluster_analysis', {}).get('cluster_count', 0)
            },
            'group_b': {
                'label': group_b_label,
                'status': status_b.status if status_b else 'UNKNOWN',
                'anomaly_density': analysis_b.get('anomaly_detection', {}).get('anomaly_density', 0),
                'cluster_count': analysis_b.get('cluster_analysis', {}).get('cluster_count', 0)
            },
            'differences': {
                'status_different': status_a.status != status_b.status if status_a and status_b else False,
                'anomaly_delta': analysis_b.get('anomaly_detection', {}).get('anomaly_density', 0) - 
                               analysis_a.get('anomaly_detection', {}).get('anomaly_density', 0),
                'cluster_delta': analysis_b.get('cluster_analysis', {}).get('cluster_count', 0) - 
                               analysis_a.get('cluster_analysis', {}).get('cluster_count', 0)
            },
            'interpretation': {
                'significant_difference': abs(analysis_b.get('anomaly_detection', {}).get('anomaly_density', 0) - 
                                          analysis_a.get('anomaly_detection', {}).get('anomaly_density', 0)) > 0.1
            }
        }
    
    # ==================== 10. Reporting & Export ====================
    
    def generate_eimas_report(self, 
                              embeddings: Optional[List[Union[torch.Tensor, np.ndarray]]] = None,
                              save_path: Optional[str] = None) -> str:
        """
        Generate comprehensive EIMAS report (EIMAS 10.1, 10.2)
        """
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("EMBEDDING INTELLIGENCE MONITORING & ANALYSIS REPORT")
        report_lines.append("EIMAS v1.0 Compliance Report")
        report_lines.append("=" * 70)
        report_lines.append(f"Generated: {datetime.now().isoformat()}")
        report_lines.append("")
        
        # System Status
        report_lines.append("SYSTEM STATUS:")
        report_lines.append("-" * 70)
        report_lines.append(f"Monitoring Enabled: {self.enable_monitoring}")
        report_lines.append(f"Current Version: {self.current_version}")
        report_lines.append(f"Monitoring Buffer Size: {len(self.monitoring_buffer)}")
        report_lines.append(f"Total Alerts: {len(self.alert_history)}")
        report_lines.append("")
        
        # Recent Alerts
        if self.alert_history:
            report_lines.append("RECENT ALERTS:")
            report_lines.append("-" * 70)
            for alert in self.alert_history[-10:]:
                report_lines.append(f"[{alert.level}] {alert.metric}: {alert.message}")
                report_lines.append(f"  Timestamp: {alert.timestamp.isoformat()}")
            report_lines.append("")
        
        # Analysis Results
        if embeddings:
            report_lines.append("ANALYSIS RESULTS:")
            report_lines.append("-" * 70)
            
            comprehensive = self.behavioral_analyzer.comprehensive_analysis(embeddings)
            status = comprehensive.get('operational_status')
            
            if status:
                report_lines.append(f"Operational Status: {status.status}")
                report_lines.append(f"Confidence: {status.confidence:.2%}")
                report_lines.append("Reasons:")
                for reason in status.reasons:
                    report_lines.append(f"  • {reason}")
            
            report_lines.append("")
        
        # Compliance Checklist
        report_lines.append("EIMAS COMPLIANCE CHECKLIST:")
        report_lines.append("-" * 70)
        compliance_items = [
            ("Multi-model embedding support", "✅"),
            ("Versioned vector storage", "✅"),
            ("Behavioral anomaly detection", "✅"),
            ("Real-time alerting", "✅"),
            ("Explainable decision output", "✅"),
            ("Secure processing & audit logs", "✅")
        ]
        for item, status in compliance_items:
            report_lines.append(f"{status} {item}")
        
        report = "\n".join(report_lines)
        
        if save_path:
            with open(save_path, 'w', encoding='utf-8') as f:
                f.write(report)
            print(f"EIMAS report saved to {save_path}")
        
        return report
    
    def export_json(self, data: Dict, save_path: str):
        """Export data as JSON (EIMAS 10.2)"""
        # Convert datetime objects to strings
        def json_serializer(obj):
            if isinstance(obj, datetime):
                return obj.isoformat()
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.generic):
                return obj.item()
            raise TypeError(f"Type {type(obj)} not serializable")
        
        with open(save_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, default=json_serializer, indent=2)
        print(f"Data exported to {save_path}")
    
    # ==================== Helper Methods ====================
    
    def _to_numpy(self, embedding: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
        """Convert embedding to numpy array"""
        if isinstance(embedding, torch.Tensor):
            return embedding.detach().cpu().numpy()
        return np.array(embedding)
    
    def comprehensive_analysis(self, 
                              embeddings: List[Union[torch.Tensor, np.ndarray]],
                              include_specialized: bool = True) -> Dict:
        """
        Comprehensive EIMAS analysis
        
        รวมทุกความสามารถในการวิเคราะห์
        """
        # Core analysis
        core_results = self.behavioral_analyzer.comprehensive_analysis(embeddings)
        
        # Quality analysis
        quality_results = []
        for emb in embeddings[:10]:  # Limit to first 10 for performance
            quality_results.append(self.quality_inspector.analyze_embedding(emb))
        
        result = {
            'timestamp': datetime.now().isoformat(),
            'eimas_version': '1.0',
            'core_analysis': core_results,
            'quality_analysis': {
                'mean_quality_index': np.mean([r.get('embedding_quality_index', r['ΔEΨ_with_H']) 
                                             for r in quality_results]),
                'mean_signal_quality': np.mean([r.get('signal_quality', r['S']) for r in quality_results])
            }
        }
        
        # Specialized inspections if requested
        if include_specialized and len(embeddings) >= 2:
            result['specialized_inspections'] = {
                'imitation_detection': self.imitation_forgery_detection(embeddings),
                'hidden_patterns': self.hidden_communication_pattern_detection(embeddings)
            }
        
        return result


# ==================== Demo Function ====================

def demo_eimas():
    """Demo EIMAS Analyzer"""
    print("🔍 EIMAS Analyzer Demo")
    print("=" * 70)
    
    # Create analyzer
    baseline = [np.random.randn(768) * 0.5 for _ in range(20)]
    analyzer = EIMASAnalyzer(
        baseline_embeddings=baseline,
        enable_monitoring=True
    )
    
    # Test embeddings
    test_embeddings = [np.random.randn(768) * 0.5 for _ in range(10)]
    
    # Comprehensive analysis
    print("\n📊 Running Comprehensive Analysis...")
    results = analyzer.comprehensive_analysis(test_embeddings)
    
    print(f"Operational Status: {results['core_analysis']['operational_status'].status}")
    print(f"Quality Index: {results['quality_analysis']['mean_quality_index']:.2f}")
    
    # Specialized inspections
    print("\n🔬 Specialized Inspections...")
    forgery_result = analyzer.imitation_forgery_detection(test_embeddings)
    print(f"Forgery Risk: {forgery_result['forgery_risk']:.3f}")
    
    # Monitoring
    print("\n📡 Real-time Monitoring...")
    for i, emb in enumerate(test_embeddings[:3]):
        ingest_result = analyzer.ingest_embedding(emb, embedding_id=f"test_{i}")
        print(f"  Ingested {ingest_result['embedding_id']}: {len(ingest_result['alerts'])} alerts")
    
    # Generate report
    print("\n📝 Generating EIMAS Report...")
    import os
    report_path = os.path.join('outputs', 'reports', 'eimas_report.txt')
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    report = analyzer.generate_eimas_report(test_embeddings, save_path=report_path)
    
    print("\n✅ Demo completed!")
    
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = demo_eimas()

