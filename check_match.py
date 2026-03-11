"""Check if RAG rules match EdgeSimPy output."""
import pandas as pd

df = pd.read_csv(r'c:\Users\Sahil Padole\Videos\AI_agent_ml_threshold\data\edgesimpy_failure_ml_+_thresh_(gb)_no_failure_20251223_075347_results.csv')

print('=== EdgeSimPy vs RAG Rules Comparison ===')
print()

# User's export shows Tasks 3-12, let's check those
for i in range(3, 13):
    row = df.iloc[i]
    dr = row['datarate']/1e6
    sinr = row['sinr']
    lat = row['latency_ms']
    edgesimpy = row['assigned_layer']
    
    # Our RAG rule (SAME as EdgeSimPy training labels but using latency_ms)
    if lat < 20 and dr < 16.6:
        rag = 'Edge'
    elif 9.6 <= dr < 16.6 and sinr > 10:
        rag = 'Fog'
    else:
        rag = 'Cloud'
    
    match = 'OK' if edgesimpy == rag else 'DIFF'
    print(f'Row {i}: DR={dr:5.1f}Mbps | SINR={sinr:5.1f}dB | Lat={lat:6.1f}ms | EdgeSimPy={edgesimpy:5s} | RAG={rag:5s} | {match}')

print()
print('=== Full Dataset Match Rate ===')
correct = 0
for i in range(len(df)):
    row = df.iloc[i]
    dr = row['datarate']/1e6
    sinr = row['sinr']
    lat = row['latency_ms']
    
    if lat < 20 and dr < 16.6:
        rag = 'Edge'
    elif 9.6 <= dr < 16.6 and sinr > 10:
        rag = 'Fog'
    else:
        rag = 'Cloud'
    
    if row['assigned_layer'] == rag:
        correct += 1

print(f'Match: {correct}/1000 = {correct/10:.1f}%')
