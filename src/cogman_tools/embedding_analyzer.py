# ตัวอย่างใช้งานกับ BERT embedding
from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, fft, stats
import networkx as nx
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler

class EmbeddingCircuitAnalyzer:
    """เครื่องมือวิเคราะห์ Embedding ด้วยโมเดลวงจรไฟฟ้า"""

    def __init__(self, embedding_dim=768):
        self.embedding_dim = embedding_dim
        self.electric_params = {}

    def embedding_to_circuit(self, embedding_vector, method='resistance_network'):
        """
        แปลง embedding vector เป็นวงจรไฟฟ้า
        """
        # Normalize embedding
        embedding_norm = StandardScaler().fit_transform(
            embedding_vector.reshape(-1, 1)
        ).flatten()

        if method == 'resistance_network':
            return self._create_resistance_network(embedding_norm)
        elif method == 'capacitor_grid':
            return self._create_capacitor_grid(embedding_norm)
        elif method == 'rlc_circuit':
            return self._create_rlc_circuit(embedding_norm)
        else:
            return self._create_transmission_line(embedding_norm)

    def _create_resistance_network(self, embedding):
        """สร้างเครือข่ายความต้านทานจาก embedding"""
        D = len(embedding)

        # สร้าง adjacency matrix สำหรับกราฟ
        adj_matrix = np.zeros((D, D))

        # ความต้านทานระหว่างโหนด i,j = 1/|embedding[i] - embedding[j]|
        for i in range(D):
            for j in range(i + 1, D):
                if abs(embedding[i] - embedding[j]) > 1e-6:
                    resistance = 1.0 / abs(embedding[i] - embedding[j])
                    adj_matrix[i, j] = resistance
                    adj_matrix[j, i] = resistance

        # แปลงเป็น conductance matrix
        conductance_matrix = np.zeros((D, D))
        for i in range(D):
            total_resistance = np.sum(adj_matrix[i, :])
            if total_resistance > 0:
                for j in range(D):
                    if adj_matrix[i, j] > 0:
                        conductance_matrix[i, j] = 1.0 / adj_matrix[i, j]

        # คำนวณวงจรเทียบเท่า
        circuit_params = {
            'adjacency': adj_matrix,
            'conductance': conductance_matrix,
            'node_voltages': embedding,  # ใช้ embedding เป็นแรงดันเริ่มต้น
            'node_currents': np.zeros(D),
            'type': 'resistance_network'
        }

        return circuit_params

    def _create_capacitor_grid(self, embedding):
        """สร้างกริดตัวเก็บประจุจาก embedding"""
        D = len(embedding)

        # สร้าง grid 2D (ถ้า embedding เป็น 1D ให้ reshape)
        if D <= 1024:  # ถ้าไม่ใหญ่เกินไป
            side = int(np.sqrt(D))
            if side * side < D:
                side += 1

            grid = embedding[:side * side].reshape(side, side)
        else:
            # ใช้ PCA ลดมิติก่อน
            from sklearn.decomposition import PCA
            pca = PCA(n_components=256)
            reduced = pca.fit_transform(embedding.reshape(1, -1))
            side = 16
            grid = reduced.reshape(side, side)

        # คำนวณ capacitance จาก gradient
        grad_x = np.gradient(grid, axis=0)
        grad_y = np.gradient(grid, axis=1)

        # Capacitance ∝ 1/|gradient|
        capacitance = 1.0 / (np.sqrt(grad_x ** 2 + grad_y ** 2) + 1e-6)

        return {
            'grid': grid,
            'capacitance_map': capacitance,
            'gradient_x': grad_x,
            'gradient_y': grad_y,
            'type': 'capacitor_grid'
        }

    def _create_rlc_circuit(self, embedding):
        """สร้างวงจร RLC จาก embedding"""
        D = len(embedding)

        # ค่า R, L, C จากสถิติของ embedding
        mean_val = np.mean(embedding)
        std_val = np.std(embedding)
        skew_val = stats.skew(embedding)
        kurt_val = stats.kurtosis(embedding)

        # สร้างวงจร RLC แบบง่าย
        R = 1.0 / (std_val + 1e-6)  # ความต้านทาน
        L = abs(skew_val) * 0.1  # ความเหนี่ยวนำ
        C = abs(kurt_val) * 0.01  # ความจุ

        # สัญญาณไฟฟ้าจาก embedding
        time = np.linspace(0, 10, D)
        voltage_signal = embedding

        # แก้สมการวงจร RLC
        current_signal = self._solve_rlc_circuit(
            voltage_signal, R, L, C, time
        )

        return {
            'R': R, 'L': L, 'C': C,
            'voltage': voltage_signal,
            'current': current_signal,
            'time': time,
            'type': 'rlc_circuit',
            'resonant_frequency': 1.0 / np.sqrt(L * C) if L * C > 0 else 0
        }

    def _solve_rlc_circuit(self, voltage, R, L, C, time):
        """แก้สมการวงจร RLC"""
        # สมการ: L*d²i/dt² + R*di/dt + i/C = dv/dt
        dt = time[1] - time[0] if len(time) > 1 else 0.1

        # แบบจำลองอย่างง่าย
        current = np.zeros_like(voltage)

        for i in range(1, len(voltage)):
            dv_dt = (voltage[i] - voltage[i - 1]) / dt

            # อินทิเกรตอย่างง่าย
            if i == 1:
                current[i] = dv_dt * dt / (R + 1e-6)
            else:
                di_dt = (current[i - 1] - current[i - 2]) / dt
                current[i] = current[i - 1] + (dv_dt - R * current[i - 1] - (1 / C) * current[i - 1]) * dt / L

        return current

    def analyze_hidden_patterns(self, circuit_data, embedding):
        """วิเคราะห์ pattern ที่ซ่อนอยู่"""
        analysis = {}

        # 1. วิเคราะห์ Frequency Domain
        fft_result = fft.fft(embedding)
        freqs = fft.fftfreq(len(embedding))

        # หาความถี่ที่โดดเด่น
        magnitude = np.abs(fft_result)
        dominant_freq_idx = np.argmax(magnitude[1:len(magnitude) // 2]) + 1
        dominant_freq = freqs[dominant_freq_idx]

        analysis['frequency_analysis'] = {
            'dominant_frequency': dominant_freq,
            'bandwidth': np.std(magnitude),
            'harmonic_count': len(signal.find_peaks(magnitude[:len(magnitude) // 2])[0])
        }

        # 2. วิเคราะห์ Impedance
        if circuit_data['type'] == 'rlc_circuit':
            R, L, C = circuit_data['R'], circuit_data['L'], circuit_data['C']
            frequencies = np.logspace(-2, 2, 100)

            impedance = []
            for f in frequencies:
                Z = R + 1j * (2 * np.pi * f * L - 1 / (2 * np.pi * f * C + 1e-6))
                impedance.append(np.abs(Z))

            analysis['impedance_analysis'] = {
                'impedance_curve': impedance,
                'frequencies': frequencies,
                'resonance_point': np.argmin(impedance) if impedance else 0
            }

        # 3. วิเคราะห์ Graph Properties (ถ้าเป็น network)
        if circuit_data['type'] == 'resistance_network':
            adj_matrix = circuit_data['adjacency']

            # สร้างกราฟ
            G = nx.from_numpy_array(adj_matrix)

            analysis['graph_analysis'] = {
                'clustering_coefficient': nx.average_clustering(G),
                'average_path_length': nx.average_shortest_path_length(G)
                if nx.is_connected(G) else float('inf'),
                'degree_centrality': dict(nx.degree_centrality(G)),
                'betweenness_centrality': dict(nx.betweenness_centrality(G)),
                'connected_components': nx.number_connected_components(G)
            }

        # 4. วิเคราะห์ Energy Distribution
        if 'voltage' in circuit_data:
            voltage = circuit_data['voltage']
            current = circuit_data.get('current', np.zeros_like(voltage))

            # พลังงานในระบบ
            power = voltage * current
            total_energy = np.trapz(np.abs(power))

            analysis['energy_analysis'] = {
                'total_energy': total_energy,
                'avg_power': np.mean(np.abs(power)),
                'power_factor': np.mean(power) / (np.std(voltage) * np.std(current) + 1e-6),
                'energy_distribution': power
            }

        # 5. Detect Anomalies (จุดที่ผิดปกติ)
        anomalies = self._detect_anomalies(circuit_data, embedding)
        analysis['anomalies'] = anomalies

        return analysis

    def _detect_anomalies(self, circuit_data, embedding):
        """ตรวจจับความผิดปกติใน embedding"""
        anomalies = {
            'voltage_spikes': [],
            'current_leakage': [],
            'impedance_mismatch': [],
            'frequency_outliers': [],
            'hidden_nodes': []
        }

        # 1. Detect voltage spikes (ค่าสูงผิดปกติ)
        mean_v = np.mean(embedding)
        std_v = np.std(embedding)
        spike_threshold = mean_v + 3 * std_v

        voltage_spikes = np.where(np.abs(embedding) > spike_threshold)[0]
        anomalies['voltage_spikes'] = voltage_spikes.tolist()

        # 2. Detect current leakage (ถ้ามีข้อมูล current)
        if 'current' in circuit_data:
            current = circuit_data['current']
            mean_current = np.mean(np.abs(current))

            # หาจุดที่ current ต่ำผิดปกติ (อาจเกิด leakage)
            leakage_points = np.where(np.abs(current) < 0.1 * mean_current)[0]
            anomalies['current_leakage'] = leakage_points.tolist()

        # 3. Detect impedance mismatch ในเครือข่าย
        if 'adjacency' in circuit_data:
            adj_matrix = circuit_data['adjacency']

            # คำนวณ impedance ของแต่ละโหนด
            node_impedances = []
            for i in range(len(adj_matrix)):
                connected = adj_matrix[i, :] > 0
                if np.any(connected):
                    avg_impedance = np.mean(adj_matrix[i, connected])
                    node_impedances.append(avg_impedance)

            if node_impedances:
                mean_imp = np.mean(node_impedances)
                std_imp = np.std(node_impedances)

                mismatch_nodes = []
                for i, imp in enumerate(node_impedances):
                    if abs(imp - mean_imp) > 2 * std_imp:
                        mismatch_nodes.append(i)

                anomalies['impedance_mismatch'] = mismatch_nodes

        # 4. Detect frequency outliers
        fft_vals = fft.fft(embedding)
        freqs = fft.fftfreq(len(embedding))
        magnitudes = np.abs(fft_vals)

        # หาความถี่ที่ผิดปกติ (นอกช่วงความถี่หลัก)
        main_freqs = freqs[(magnitudes > 0.5 * np.max(magnitudes))]
        if len(main_freqs) > 0:
            main_band = [np.min(main_freqs), np.max(main_freqs)]

            outlier_freqs = []
            for i, (freq, mag) in enumerate(zip(freqs, magnitudes)):
                if mag > 0.1 * np.max(magnitudes) and not (main_band[0] <= freq <= main_band[1]):
                    outlier_freqs.append(i)

            anomalies['frequency_outliers'] = outlier_freqs

        # 5. Detect hidden nodes (โหนดที่เชื่อมต่อน้อยแต่สำคัญ)
        if 'adjacency' in circuit_data:
            adj_matrix = circuit_data['adjacency']
            degrees = np.sum(adj_matrix > 0, axis=1)

            # โหนดที่มี degree ต่ำแต่ embedding value สูง
            hidden_nodes = []
            for i in range(len(degrees)):
                if degrees[i] < 0.1 * len(adj_matrix) and abs(embedding[i]) > 0.5 * np.max(np.abs(embedding)):
                    hidden_nodes.append(i)

            anomalies['hidden_nodes'] = hidden_nodes

        return anomalies


class EmbeddingCircuitVisualizer:
    """เครื่องมือแสดงผลวงจรไฟฟ้าจาก embedding"""

    def __init__(self):
        self.fig = None

    def plot_3d_circuit(self, circuit_data, embedding, anomalies=None):
        """แสดงวงจร 3D แบบอินเตอร์แอคทีฟ"""
        fig = go.Figure()

        if circuit_data['type'] == 'resistance_network':
            adj_matrix = circuit_data['adjacency']
            node_values = embedding

            # สร้างโหนด
            node_trace = go.Scatter3d(
                x=np.arange(len(node_values)),
                y=node_values,
                z=np.zeros_like(node_values),
                mode='markers',
                marker=dict(
                    size=10,
                    color=node_values,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title="Node Value")
                ),
                text=[f"Node {i}<br>Value: {v:.3f}" for i, v in enumerate(node_values)],
                hoverinfo='text',
                name='Nodes'
            )

            # สร้างเส้นเชื่อม (edges)
            edge_x, edge_y, edge_z = [], [], []
            for i in range(len(adj_matrix)):
                for j in range(i + 1, len(adj_matrix)):
                    if adj_matrix[i, j] > 0:
                        edge_x.extend([i, j, None])
                        edge_y.extend([node_values[i], node_values[j], None])
                        edge_z.extend([0, 0, None])

            edge_trace = go.Scatter3d(
                x=edge_x, y=edge_y, z=edge_z,
                mode='lines',
                line=dict(width=1, color='gray'),
                opacity=0.5,
                name='Connections'
            )

            fig.add_trace(node_trace)
            fig.add_trace(edge_trace)

            # แสดง anomalies ถ้ามี
            if anomalies and 'voltage_spikes' in anomalies:
                spike_nodes = anomalies['voltage_spikes']
                if len(spike_nodes) > 0:
                    spike_trace = go.Scatter3d(
                        x=spike_nodes,
                        y=embedding[spike_nodes],
                        z=np.zeros(len(spike_nodes)),
                        mode='markers',
                        marker=dict(size=15, color='red', symbol='x'),
                        name='Voltage Spikes'
                    )
                    fig.add_trace(spike_trace)

        elif circuit_data['type'] == 'capacitor_grid':
            grid = circuit_data['grid']
            capacitance = circuit_data['capacitance_map']

            # สร้าง surface plot
            x, y = np.meshgrid(np.arange(grid.shape[1]), np.arange(grid.shape[0]))

            fig.add_trace(go.Surface(
                z=grid,
                surfacecolor=capacitance,
                colorscale='Plasma',
                colorbar=dict(title="Capacitance"),
                name='Capacitor Grid'
            ))

            # แสดง gradient vectors
            grad_x = circuit_data['gradient_x']
            grad_y = circuit_data['gradient_y']

            # สุ่มแสดงบาง vectors
            step = max(1, grid.shape[0] // 10)
            for i in range(0, grid.shape[0], step):
                for j in range(0, grid.shape[1], step):
                    fig.add_trace(go.Cone(
                        x=[j], y=[i], z=[grid[i, j]],
                        u=[grad_x[i, j]], v=[grad_y[i, j]], w=[0],
                        sizemode="absolute",
                        sizeref=0.5,
                        colorscale='Blues',
                        showscale=False,
                        name=f'Gradient at ({i},{j})'
                    ))

        fig.update_layout(
            title="3D Embedding Circuit Visualization",
            scene=dict(
                xaxis_title="Node Index / X",
                yaxis_title="Embedding Value / Y",
                zaxis_title="Z / Additional Dimension",
                camera=dict(
                    eye=dict(x=1.5, y=1.5, z=1.5)
                )
            ),
            showlegend=True
        )

        return fig

    def plot_frequency_analysis(self, embedding):
        """แสดงการวิเคราะห์ความถี่"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # 1. Original embedding
        axes[0, 0].plot(embedding, alpha=0.7)
        axes[0, 0].set_title('Original Embedding')
        axes[0, 0].set_xlabel('Dimension Index')
        axes[0, 0].set_ylabel('Value')
        axes[0, 0].grid(True, alpha=0.3)

        # 2. FFT magnitude
        fft_vals = fft.fft(embedding)
        freqs = fft.fftfreq(len(embedding))
        magnitude = np.abs(fft_vals)

        axes[0, 1].plot(freqs[:len(freqs) // 2], magnitude[:len(magnitude) // 2])
        axes[0, 1].set_title('Frequency Spectrum')
        axes[0, 1].set_xlabel('Frequency')
        axes[0, 1].set_ylabel('Magnitude')
        axes[0, 1].grid(True, alpha=0.3)

        # 3. Spectrogram
        axes[1, 0].specgram(embedding, Fs=1, NFFT=64, noverlap=32)
        axes[1, 0].set_title('Spectrogram')
        axes[1, 0].set_xlabel('Time (dimension index)')
        axes[1, 0].set_ylabel('Frequency')

        # 4. Phase plot
        phase = np.angle(fft_vals)
        axes[1, 1].scatter(magnitude[:len(magnitude) // 2], phase[:len(phase) // 2],
                           alpha=0.5, s=10)
        axes[1, 1].set_title('Phase-Magnitude Plot')
        axes[1, 1].set_xlabel('Magnitude')
        axes[1, 1].set_ylabel('Phase (radians)')
        axes[1, 1].grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_interactive_dashboard(self, circuit_data, embedding, analysis):
        """สร้าง dashboard แบบอินเตอร์แอคทีฟ"""
        import dash
        from dash import dcc, html
        import plotly.express as px

        app = dash.Dash(__name__)

        # 3D circuit plot
        circuit_fig = self.plot_3d_circuit(circuit_data, embedding, analysis.get('anomalies'))

        # Heatmap of adjacency matrix
        if 'adjacency' in circuit_data:
            adj_heatmap = px.imshow(
                circuit_data['adjacency'],
                title='Admittance Matrix Heatmap',
                labels=dict(x="Node", y="Node", color="Conductance")
            )
        else:
            adj_heatmap = go.Figure()

        # Energy distribution plot
        if 'energy_analysis' in analysis:
            energy_data = analysis['energy_analysis']
            energy_fig = go.Figure()
            energy_fig.add_trace(go.Scatter(
                y=energy_data.get('energy_distribution', []),
                mode='lines',
                name='Power Distribution'
            ))
            energy_fig.update_layout(
                title='Energy Distribution',
                xaxis_title='Dimension Index',
                yaxis_title='Power'
            )
        else:
            energy_fig = go.Figure()

        # Anomalies table
        anomalies_table = []
        if 'anomalies' in analysis:
            for anomaly_type, indices in analysis['anomalies'].items():
                if indices:
                    anomalies_table.append(html.Tr([
                        html.Td(anomaly_type),
                        html.Td(str(len(indices))),
                        html.Td(str(indices[:10]) + ("..." if len(indices) > 10 else ""))
                    ]))

        app.layout = html.Div([
            html.H1("Embedding Circuit Analysis Dashboard"),

            html.Div([
                html.Div([
                    dcc.Graph(figure=circuit_fig, style={'height': '600px'})
                ], className='six columns'),

                html.Div([
                    dcc.Graph(figure=adj_heatmap, style={'height': '600px'})
                ], className='six columns')
            ], className='row'),

            html.Div([
                html.Div([
                    dcc.Graph(figure=energy_fig, style={'height': '400px'})
                ], className='six columns'),

                html.Div([
                    html.H3("Detected Anomalies"),
                    html.Table([
                        html.Thead(html.Tr([
                            html.Th("Type"),
                            html.Th("Count"),
                            html.Th("Indices")
                        ])),
                        html.Tbody(anomalies_table)
                    ], style={'width': '100%', 'margin-top': '20px'})
                ], className='six columns')
            ], className='row'),

            html.Div([
                html.H3("Circuit Parameters"),
                html.Pre(str({
                    k: v for k, v in circuit_data.items()
                    if not isinstance(v, np.ndarray) or v.size < 10
                }))
            ])
        ])

        return app

class EmbeddingDetective:
    """เครื่องมือตรวจจับสิ่งที่ซ่อนอยู่ใน embedding"""

    def __init__(self, model_name='bert-base-uncased'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.circuit_analyzer = EmbeddingCircuitAnalyzer()
        self.visualizer = EmbeddingCircuitVisualizer()

    def analyze_text(self, text):
        """วิเคราะห์ข้อความด้วยวงจรไฟฟ้า"""
        # 1. สร้าง embedding
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # ใช้ [CLS] token embedding
        embedding = outputs.last_hidden_state[0, 0, :].numpy()

        # 2. แปลงเป็นวงจรไฟฟ้า
        circuit_data = self.circuit_analyzer.embedding_to_circuit(
            embedding, method='resistance_network'
        )

        # 3. วิเคราะห์หา pattern ที่ซ่อนอยู่
        analysis = self.circuit_analyzer.analyze_hidden_patterns(circuit_data, embedding)

        # 4. แสดงผล
        self._print_analysis_report(text, embedding, circuit_data, analysis)

        # 5. สร้าง visualization
        fig_3d = self.visualizer.plot_3d_circuit(circuit_data, embedding, analysis['anomalies'])
        fig_freq = self.visualizer.plot_frequency_analysis(embedding)

        return {
            'embedding': embedding,
            'circuit': circuit_data,
            'analysis': analysis,
            'visualizations': {
                '3d_circuit': fig_3d,
                'frequency_analysis': fig_freq
            }
        }

    def _print_analysis_report(self, text, embedding, circuit, analysis):
        """พิมพ์รายงานการวิเคราะห์"""
        print("=" * 80)
        print("🔍 EMBEDDING CIRCUIT ANALYSIS REPORT")
        print("=" * 80)
        print(f"\n📝 Text: {text[:100]}..." if len(text) > 100 else f"📝 Text: {text}")
        print(f"📏 Embedding Dimension: {len(embedding)}")
        print(f"⚡ Circuit Type: {circuit['type']}")

        print("\n📊 STATISTICAL ANALYSIS:")
        print(f"  • Mean: {np.mean(embedding):.4f}")
        print(f"  • Std: {np.std(embedding):.4f}")
        print(f"  • Min/Max: {np.min(embedding):.4f} / {np.max(embedding):.4f}")
        print(f"  • Skewness: {stats.skew(embedding):.4f}")
        print(f"  • Kurtosis: {stats.kurtosis(embedding):.4f}")

        if 'frequency_analysis' in analysis:
            freq_info = analysis['frequency_analysis']
            print(f"\n📡 FREQUENCY ANALYSIS:")
            print(f"  • Dominant Frequency: {freq_info['dominant_frequency']:.4f}")
            print(f"  • Bandwidth: {freq_info['bandwidth']:.4f}")
            print(f"  • Harmonic Count: {freq_info['harmonic_count']}")

        if 'anomalies' in analysis:
            anomalies = analysis['anomalies']
            print(f"\n🚨 DETECTED ANOMALIES:")

            total_anomalies = sum(len(v) for v in anomalies.values())
            print(f"  • Total Anomaly Points: {total_anomalies}")

            for anomaly_type, indices in anomalies.items():
                if indices:
                    print(f"  • {anomaly_type.replace('_', ' ').title()}: {len(indices)} points")
                    if len(indices) <= 10:
                        print(f"    Indices: {indices}")

        if 'graph_analysis' in analysis:
            graph_info = analysis['graph_analysis']
            print(f"\n🕸️  GRAPH ANALYSIS:")
            print(f"  • Clustering Coefficient: {graph_info['clustering_coefficient']:.4f}")
            print(f"  • Connected Components: {graph_info['connected_components']}")

        if 'energy_analysis' in analysis:
            energy_info = analysis['energy_analysis']
            print(f"\n⚡ ENERGY ANALYSIS:")
            print(f"  • Total Energy: {energy_info['total_energy']:.4f}")
            print(f"  • Average Power: {energy_info['avg_power']:.4f}")
            print(f"  • Power Factor: {energy_info['power_factor']:.4f}")

        print("\n" + "=" * 80)


# ตัวอย่างการใช้งาน
if __name__ == "__main__":
    detective = EmbeddingDetective()

    # ทดสอบกับข้อความต่างๆ
    test_texts = [
        "I love artificial intelligence and machine learning",
        "The quick brown fox jumps over the lazy dog",
        "This is a secret message that contains hidden patterns",
        "Quantum physics reveals the mysteries of the universe",
        "Deep learning models can discover hidden representations"
    ]

    for text in test_texts:
        print(f"\n{'=' * 80}")
        print(f"Analyzing: '{text}'")
        print('=' * 80)

        result = detective.analyze_text(text)

        # บันทึก visualization
        result['visualizations']['3d_circuit'].write_html(f"circuit_{hash(text)}.html")

        # แสดง anomalies ที่พบ
        anomalies = result['analysis']['anomalies']
        print(f"\nFound {sum(len(v) for v in anomalies.values())} anomaly points")