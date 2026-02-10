"""
Network Data Loader - Load real network scenarios from CSV files
Supports: ai4mobile, EdgeSimPy results, and test scenarios
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any

class Document:
    """Simple Document class compatible with LangChain."""
    def __init__(self, page_content: str, metadata: Dict[str, Any]):
        self.page_content = page_content
        self.metadata = metadata


class NetworkDataLoader:
    """Load and convert network CSV data into documents."""
    
    def __init__(self):
        self.workspace_root = Path(__file__).parent.parent.parent.parent
        
    def load_csv_network_data(self, csv_path: str) -> List[Document]:
        """
        Load network scenarios from CSV and convert to documents.
        
        Supports:
        - ai4mobile test results
        - EdgeSimPy simulation results  
        - AI agent test scenarios
        """
        csv_file = Path(csv_path)
        
        if not csv_file.exists():
            print(f"⚠️  CSV file not found: {csv_file}")
            return []
        
        print(f"📂 Loading network data from: {csv_file.name}")
        df = pd.read_csv(csv_file)
        print(f"✅ Loaded {len(df)} network scenarios")
        
        documents = []
        
        for idx, row in df.iterrows():
            # Create document from each scenario
            doc = self._row_to_document(row, idx)
            documents.append(doc)
        
        return documents
    
    def _row_to_document(self, row: pd.Series, idx: int) -> Document:
        """Convert CSV row to Document."""
        
        # Extract key metrics
        scenario_text = self._format_scenario(row)
        
        # Metadata for retrieval
        metadata = {
            "scenario_id": idx,
            "source": "network_data",
        }
        
        # Add deployment layer if present
        if "deployment_layer" in row and pd.notna(row["deployment_layer"]):
            metadata["deployment_layer"] = row["deployment_layer"]
        
        # Add actual layer if different from assigned
        if "assigned_layer" in row and pd.notna(row["assigned_layer"]):
            metadata["assigned_layer"] = row["assigned_layer"]
        
        # Add key metrics for filtering
        for col in ["datarate", "datarate_mbps", "ping_ms", "latency_ms", 
                   "sinr_db", "rsrp_dbm", "total_cost", "reliability", "cpu_demand", "memory_demand"]:
            if col in row and pd.notna(row[col]):
                try:
                    metadata[col] = float(row[col])
                except (ValueError, TypeError):
                    pass
        
        return Document(page_content=scenario_text, metadata=metadata)
    
    def _format_scenario(self, row: pd.Series) -> str:
        """Format scenario row as readable text."""
        text = "Network Deployment Scenario:\n"
        
        # Network conditions
        if "datarate_mbps" in row and pd.notna(row["datarate_mbps"]):
            text += f"- Data Rate: {row['datarate_mbps']:.2f} Mbps\n"
        elif "datarate" in row and pd.notna(row["datarate"]):
            datarate = row['datarate']
            if datarate > 1000:  # Convert from bps to Mbps
                datarate = datarate / 1e6
            text += f"- Data Rate: {datarate:.2f} Mbps\n"
        
        if "ping_ms" in row and pd.notna(row["ping_ms"]):
            text += f"- Ping Latency: {row['ping_ms']:.2f} ms\n"
        elif "latency_ms" in row and pd.notna(row["latency_ms"]):
            text += f"- Latency: {row['latency_ms']:.2f} ms\n"
        
        if "sinr_db" in row and pd.notna(row["sinr_db"]):
            text += f"- SINR: {row['sinr_db']:.2f} dB\n"
        
        if "rsrp_dbm" in row and pd.notna(row["rsrp_dbm"]):
            text += f"- RSRP: {row['rsrp_dbm']:.2f} dBm\n"
        
        # Resource requirements
        if "cpu_demand" in row and pd.notna(row["cpu_demand"]):
            text += f"- CPU Demand: {row['cpu_demand']:.0f}%\n"
        
        if "memory_demand" in row and pd.notna(row["memory_demand"]):
            text += f"- Memory: {row['memory_demand']:.0f} MB\n"
        elif "memory_demand_mb" in row and pd.notna(row["memory_demand_mb"]):
            text += f"- Memory: {row['memory_demand_mb']:.0f} MB\n"
        
        # Deployment decision
        if "deployment_layer" in row and pd.notna(row["deployment_layer"]):
            text += f"- Recommended Layer: {row['deployment_layer']}\n"
        elif "assigned_layer" in row and pd.notna(row["assigned_layer"]):
            text += f"- Assigned Layer: {row['assigned_layer']}\n"
        
        # Performance metrics
        if "total_cost" in row and pd.notna(row["total_cost"]):
            text += f"- Total Cost: ${row['total_cost']:.6f}\n"
        
        if "total_energy_mw" in row and pd.notna(row["total_energy_mw"]):
            text += f"- Energy: {row['total_energy_mw']:.2f} mW\n"
        
        if "reliability" in row and pd.notna(row["reliability"]):
            text += f"- Reliability: {row['reliability']:.4f}\n"
        
        return text
    
    def load_all_network_data(self) -> List[Document]:
        """Load all available network data from workspace."""
        all_documents = []
        
        # Paths to search
        csv_paths = [
            self.workspace_root / "data" / "edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv",
            self.workspace_root / "AI_agent" / "ai4mobile" / "results" / "test_scenarios_1000_results_20260128_064008.csv",
        ]
        
        for csv_path in csv_paths:
            if csv_path.exists():
                print(f"\n📂 Found: {csv_path.name}")
                docs = self.load_csv_network_data(str(csv_path))
                all_documents.extend(docs)
                print(f"✅ Added {len(docs)} documents")
        
        if not all_documents:
            print("\n⚠️  No CSV files found!")
        
        return all_documents
    
    def filter_scenarios_by_condition(
        self, 
        documents: List[Document],
        latency_min: float = None,
        latency_max: float = None,
        datarate_min: float = None,
        datarate_max: float = None,
        deployment_layer: str = None
    ) -> List[Document]:
        """Filter scenarios by network conditions."""
        filtered = documents
        
        if latency_min is not None:
            filtered = [
                d for d in filtered 
                if ("latency_ms" in d.metadata and d.metadata["latency_ms"] >= latency_min)
                or ("ping_ms" in d.metadata and d.metadata["ping_ms"] >= latency_min)
            ]
        
        if latency_max is not None:
            filtered = [
                d for d in filtered 
                if ("latency_ms" in d.metadata and d.metadata["latency_ms"] <= latency_max)
                or ("ping_ms" in d.metadata and d.metadata["ping_ms"] <= latency_max)
            ]
        
        if datarate_min is not None:
            filtered = [
                d for d in filtered 
                if "datarate_mbps" in d.metadata and d.metadata["datarate_mbps"] >= datarate_min
            ]
        
        if deployment_layer is not None:
            filtered = [
                d for d in filtered 
                if "deployment_layer" in d.metadata and d.metadata["deployment_layer"] == deployment_layer
            ]
        
        return filtered


class RealNetworkRAGSearch:
    """RAG Search using real network scenarios."""
    
    def __init__(self, vectorstore, llm):
        self.vectorstore = vectorstore
        self.llm = llm
    
    def retrieve_similar_scenarios(self, query: str, top_k: int = 5) -> List[Document]:
        """Retrieve similar real network scenarios from vector store."""
        results = self.vectorstore.query(query, top_k=top_k)
        return results
    
    def analyze_scenario_with_context(
        self, 
        query_scenario: Dict[str, float],
        context_docs: List[Document]
    ) -> str:
        """Analyze scenario using real historical data as context."""
        
        # Build context from retrieved scenarios
        context_text = "Similar Real Network Scenarios from Database:\n\n"
        
        for i, doc in enumerate(context_docs[:5], 1):
            context_text += f"Example {i}:\n{doc.page_content}\n"
        
        # Generate analysis prompt
        prompt = f"""Based on the following real network scenarios from our database:

{context_text}

---

Analyze this NEW scenario:
- Data Rate: {query_scenario.get('datarate_mbps', 'N/A')} Mbps
- Ping Latency: {query_scenario.get('ping_ms', 'N/A')} ms
- SINR: {query_scenario.get('sinr_db', 'N/A')} dB
- RSRP: {query_scenario.get('rsrp_dbm', 'N/A')} dBm
- CPU Demand: {query_scenario.get('cpu_demand', 'N/A')}%

Provide:
1. Recommended deployment layer (Edge/Fog/Cloud)
2. Confidence score (0-1)
3. Reasoning based on similar real cases
4. Expected performance metrics
"""
        
        response = self.llm.invoke(prompt)
        return response.content


if __name__ == "__main__":
    loader = NetworkDataLoader()
    
    # Load real data
    docs = loader.load_all_network_data()
    print(f"\n✅ Total documents loaded: {len(docs)}")
    
    # Show sample
    if docs:
        print(f"\n📄 Sample Document:")
        print(docs[0].page_content)
        print(f"\n[Metadata]")
        for key, val in docs[0].metadata.items():
            print(f"  {key}: {val}")
