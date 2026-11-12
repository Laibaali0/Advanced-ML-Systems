"""Small, hand-authored fixtures for deterministic engineering demonstrations."""
from .retrieval import Document


def retrieval_fixture():
    documents = [
        Document("d1", "Deploying quantized transformer models with packed four bit weights"),
        Document("d2", "Ranking ecommerce product search using lexical relevance and embeddings"),
        Document("d3", "Offline evaluation of contextual bandit recommendation policies"),
        Document("d4", "Reducing GPU inference latency and memory with mixed precision"),
        Document("d5", "Detecting data leakage in fraud detection time series models"),
        Document("d6", "Understanding tokenizer vocabulary and natural language processing"),
        Document("d7", "Reinforcement learning from logged recommendation interactions"),
        Document("d8", "Vector similarity indexes for retrieving product listings"),
        Document("d9", "Monitoring large language model serving performance"),
        Document("d10", "Improving mobile accessibility for merchants"),
    ]
    # Graded labels were written for this toy corpus, not collected from users.
    labels = {
        "GPU memory optimization": {"d4": 3, "d1": 2, "d9": 1},
        "product search ranking": {"d2": 3, "d8": 2},
        "recommendation policy evaluation": {"d3": 3, "d7": 2},
        "fraud time leakage": {"d5": 3},
        "language tokenizer": {"d6": 3},
    }
    return documents, labels
