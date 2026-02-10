from pathlib import Path
from typing import List, Any
import pandas as pd
from langchain_core.documents import Document


def load_network_documents(data_dir: str = "../data") -> List[Document]:
    """
    Load CSV network performance data and convert to LangChain document structure.
    Following RAG-Tutorials pattern.
    """
    data_path = Path(data_dir).resolve()
    print(f"[DEBUG] Data path: {data_path}")
    documents = []
    
    # Find CSV files
    csv_files = list(data_path.glob('**/*.csv'))
    print(f"[DEBUG] Found {len(csv_files)} CSV files: {[str(f) for f in csv_files]}")
    
    for csv_file in csv_files:
        print(f"[DEBUG] Loading CSV: {csv_file}")
        try:
            # Use pandas to get more control over CSV loading
            df = pd.read_csv(csv_file)
            print(f"[DEBUG] Loaded {len(df)} records from {csv_file}")
            
            # Convert each row to a Document with meaningful content
            for idx, row in df.iterrows():
                # Create comprehensive text representation
                content = f"""Network Performance Record {idx}

DEPLOYMENT CONFIGURATION:
- Layer: {row['assigned_layer']}
- Model: {row['model']}
- Server ID: {row['server_id']}
- Status: {'Degraded' if row['degraded'] else 'Normal'}

NETWORK METRICS:
- Data Rate: {row['datarate']:,.0f} bps ({row['datarate']/1e6:.1f} Mbps)
- SINR: {row['sinr']:.2f} dB
- Ping: {row['ping_ms']:.2f} ms
- Speed: {row['speed']:.2f} m/s
- RSRP: {row['rsrp_dbm']:.1f} dBm
- RSSI: {row['rssi_dbm']:.1f} dBm
- Jitter: {row['jitter']:.6f}

PERFORMANCE RESULTS:
- Latency: {row['latency_ms']:.2f} ms
- Processing Latency: {row['processing_latency_ms']:.3f} ms
- Base Latency: {row['base_latency_ms']:.2f} ms
- Distance: {row['distance']:.1f} m
- Energy Consumption: {row['energy_consumption']:.6f} J
- Cost: ${row['cost']:.6f}

RESOURCE UTILIZATION:
- CPU Demand: {row['cpu_demand']}%
- Memory Demand: {row['memory_demand']} MB

This record represents a {row['assigned_layer'].lower()} deployment configuration."""
                
                # Create metadata for filtering
                metadata = {
                    'row_id': idx,
                    'assigned_layer': row['assigned_layer'],
                    'model': row['model'],
                    'server_id': row['server_id'],
                    'degraded': row['degraded'],
                    'datarate': row['datarate'],
                    'latency_ms': row['latency_ms'],
                    'energy_consumption': row['energy_consumption'],
                    'cost': row['cost']
                }
                
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
                
        except Exception as e:
            print(f"[ERROR] Failed to load CSV {csv_file}: {e}")
    
    print(f"[DEBUG] Total loaded documents: {len(documents)}")
    return documents