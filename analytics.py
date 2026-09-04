from bertopic import BERTopic
import networkx as nx
import pandas as pd

# Component D: Real-Time Trend Detection
def detect_trends(docs: list[str], timestamps: list[str]):
    """
    Groups posts into trending narratives over time.
    Requires a list of text documents and their corresponding timestamps.
    """
    topic_model = BERTopic(language="english", calculate_probabilities=False)
    topics, probs = topic_model.fit_transform(docs)
    
    # Track how topics evolve chronologically
    topics_over_time = topic_model.topics_over_time(docs, timestamps)
    return topic_model.get_topic_info(), topics_over_time

# Component E: Link Analysis & Network Topology
def build_influence_graph(interactions: list[dict]):
    """
    interactions format: [{'source': 'userA', 'target': 'userB', 'weight': 1}]
    source = person replying/retweeting, target = original author
    """
    G = nx.DiGraph()
    
    for interaction in interactions:
        G.add_edge(interaction['source'], interaction['target'], weight=interaction.get('weight', 1))
        
    # Calculate PageRank to find Key Opinion Leaders (KOLs)
    # The higher the PageRank, the more influence the node exerts in the network
    pagerank_scores = nx.pagerank(G, weight='weight')
    
    # Sort users by influence
    influential_nodes = sorted(pagerank_scores.items(), key=lambda x: x[1], reverse=True)
    return influential_nodes[:10]