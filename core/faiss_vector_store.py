"""
FAISS-based Vector Store for Fast Similarity Search

Replaces basic PyTorch cosine similarity with industrial-grade FAISS.
Enables:
- Ultra-fast similarity search (millions of vectors)
- Approximate nearest neighbors (ANN)
- GPU acceleration (optional)
- Memory-efficient indexing

This is the RAG backbone for discovery!
"""

import faiss
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import pickle
import os


@dataclass
class VectorEntry:
    """A single entry in the vector store."""
    id: str
    vector: np.ndarray
    metadata: Dict


class FAISSVectorStore:
    """
    High-performance vector store using FAISS.

    Features:
    - Fast ANN search (approximate nearest neighbors)
    - Batch operations
    - Persistent storage
    - Memory-efficient
    """

    def __init__(
        self,
        dimension: int = 384,  # sentence-transformers default
        index_type: str = "IndexFlatIP",  # Inner product (cosine similarity)
        storage_path: str = "./faiss_index"
    ):
        self.dimension = dimension
        self.storage_path = storage_path

        # Create FAISS index
        if index_type == "IndexFlatIP":
            # Exact search using inner product (for cosine similarity)
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == "IndexIVFFlat":
            # Approximate search with inverted file index (faster for large datasets)
            nlist = 100  # Number of clusters
            quantizer = faiss.IndexFlatIP(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
        elif index_type == "IndexHNSWFlat":
            # Hierarchical Navigable Small World graph (best for < 10M vectors)
            M = 32  # Number of connections per element
            self.index = faiss.IndexHNSWFlat(dimension, M)
        else:
            raise ValueError(f"Unknown index type: {index_type}")

        # Metadata storage (FAISS only stores vectors, we store metadata separately)
        self.id_to_metadata: Dict[str, Dict] = {}
        self.id_to_index: Dict[str, int] = {}  # Map ID to FAISS index
        self.index_to_id: Dict[int, str] = {}  # Map FAISS index to ID
        self.next_index = 0

        # Load existing index if available
        self.load()

        print(f"[FAISS] Vector store initialized: {type(self.index).__name__}")
        print(f"[FAISS] Dimension: {dimension}, Total vectors: {self.index.ntotal}")

    def add(
        self,
        id: str,
        vector: np.ndarray,
        metadata: Dict = None
    ):
        """
        Add a single vector to the store.

        Args:
            id: Unique identifier
            vector: Embedding vector (dimension must match)
            metadata: Additional data to store
        """
        if id in self.id_to_index:
            # Update existing entry
            idx = self.id_to_index[id]
            # FAISS doesn't support in-place updates, so we'd need to rebuild
            # For now, just update metadata
            self.id_to_metadata[id] = metadata or {}
            return

        # Ensure vector is right shape
        if vector.ndim == 1:
            vector = vector.reshape(1, -1)

        # Normalize for cosine similarity (if using IP index)
        if isinstance(self.index, (faiss.IndexFlatIP, faiss.IndexHNSWFlat)):
            faiss.normalize_L2(vector)

        # Add to FAISS
        self.index.add(vector.astype('float32'))

        # Store metadata
        idx = self.next_index
        self.id_to_index[id] = idx
        self.index_to_id[idx] = id
        self.id_to_metadata[id] = metadata or {}
        self.next_index += 1

    def add_batch(
        self,
        ids: List[str],
        vectors: np.ndarray,
        metadatas: List[Dict] = None
    ):
        """
        Add multiple vectors at once (more efficient).

        Args:
            ids: List of unique identifiers
            vectors: Array of shape (n, dimension)
            metadatas: List of metadata dicts
        """
        if metadatas is None:
            metadatas = [{}] * len(ids)

        # Normalize for cosine similarity
        if isinstance(self.index, (faiss.IndexFlatIP, faiss.IndexHNSWFlat)):
            faiss.normalize_L2(vectors)

        # Add to FAISS
        self.index.add(vectors.astype('float32'))

        # Store metadata
        for i, (id, metadata) in enumerate(zip(ids, metadatas)):
            idx = self.next_index + i
            self.id_to_index[id] = idx
            self.index_to_id[idx] = id
            self.id_to_metadata[id] = metadata

        self.next_index += len(ids)

    def search(
        self,
        query_vector: np.ndarray,
        k: int = 5,
        threshold: float = 0.0
    ) -> List[Tuple[str, float, Dict]]:
        """
        Search for similar vectors.

        Args:
            query_vector: Query embedding
            k: Number of results to return
            threshold: Minimum similarity score (0-1)

        Returns:
            List of (id, similarity, metadata) tuples
        """
        if self.index.ntotal == 0:
            return []

        # Ensure vector is right shape
        if query_vector.ndim == 1:
            query_vector = query_vector.reshape(1, -1)

        # Normalize for cosine similarity
        if isinstance(self.index, (faiss.IndexFlatIP, faiss.IndexHNSWFlat)):
            faiss.normalize_L2(query_vector)

        # Search
        k = min(k, self.index.ntotal)  # Can't return more than we have
        similarities, indices = self.index.search(query_vector.astype('float32'), k)

        # Convert to results
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx == -1:  # FAISS returns -1 for not found
                continue

            if sim < threshold:
                continue

            id = self.index_to_id[idx]
            metadata = self.id_to_metadata[id]
            results.append((id, float(sim), metadata))

        return results

    def get(self, id: str) -> Optional[Dict]:
        """Get metadata for a specific ID."""
        return self.id_to_metadata.get(id)

    def delete(self, id: str):
        """
        Delete an entry.

        Note: FAISS doesn't support deletion, so we just mark as deleted in metadata.
        Rebuild index periodically to actually remove.
        """
        if id in self.id_to_metadata:
            self.id_to_metadata[id]['_deleted'] = True

    def save(self):
        """Save index and metadata to disk."""
        try:
            os.makedirs(self.storage_path, exist_ok=True)

            # Save FAISS index
            index_path = os.path.join(self.storage_path, "faiss.index")
            faiss.write_index(self.index, index_path)

            # Save metadata
            metadata_path = os.path.join(self.storage_path, "metadata.pkl")
            with open(metadata_path, 'wb') as f:
                pickle.dump({
                    'id_to_metadata': self.id_to_metadata,
                    'id_to_index': self.id_to_index,
                    'index_to_id': self.index_to_id,
                    'next_index': self.next_index
                }, f)

            print(f"[FAISS] Saved index with {self.index.ntotal} vectors to {self.storage_path}")
        except Exception as e:
            print(f"[FAISS] Failed to save: {e}")

    def load(self):
        """Load index and metadata from disk."""
        try:
            index_path = os.path.join(self.storage_path, "faiss.index")
            metadata_path = os.path.join(self.storage_path, "metadata.pkl")

            if not os.path.exists(index_path) or not os.path.exists(metadata_path):
                return

            # Load FAISS index
            self.index = faiss.read_index(index_path)

            # Load metadata
            with open(metadata_path, 'rb') as f:
                data = pickle.load(f)
                self.id_to_metadata = data['id_to_metadata']
                self.id_to_index = data['id_to_index']
                self.index_to_id = data['index_to_id']
                self.next_index = data['next_index']

            print(f"[FAISS] Loaded index with {self.index.ntotal} vectors from {self.storage_path}")
        except Exception as e:
            print(f"[FAISS] Failed to load: {e}")

    def size(self) -> int:
        """Get number of vectors in index."""
        return self.index.ntotal

    def clear(self):
        """Clear all vectors and metadata."""
        # Create new empty index of same type
        if isinstance(self.index, faiss.IndexFlatIP):
            self.index = faiss.IndexFlatIP(self.dimension)
        elif isinstance(self.index, faiss.IndexHNSWFlat):
            M = 32
            self.index = faiss.IndexHNSWFlat(self.dimension, M)

        self.id_to_metadata.clear()
        self.id_to_index.clear()
        self.index_to_id.clear()
        self.next_index = 0

        print(f"[FAISS] Cleared vector store")


def demo_faiss():
    """Demonstrate FAISS vector store."""
    print("="*70)
    print("FAISS VECTOR STORE - Demo")
    print("="*70)

    # Create store
    store = FAISSVectorStore(dimension=384)

    # Add some vectors (normally from sentence-transformers)
    np.random.seed(42)

    # Simulate concept embeddings
    concepts = [
        ("prime_numbers", "Numbers only divisible by 1 and themselves"),
        ("composite_numbers", "Numbers with more than two divisors"),
        ("even_numbers", "Numbers divisible by 2"),
        ("odd_numbers", "Numbers not divisible by 2"),
        ("pythagorean_theorem", "a² + b² = c² in right triangles"),
    ]

    vectors = np.random.randn(len(concepts), 384)
    ids = [c[0] for c in concepts]
    metadatas = [{"definition": c[1]} for c in concepts]

    store.add_batch(ids, vectors, metadatas)

    # Search
    print("\n[Search] Looking for concepts similar to 'prime_numbers'...")
    query = vectors[0]  # prime_numbers
    results = store.search(query, k=3)

    for id, sim, metadata in results:
        print(f"  {id}: {sim:.3f} - {metadata.get('definition', '')}")

    # Save
    store.save()

    # Load
    store2 = FAISSVectorStore(dimension=384)
    print(f"\n[Load] Loaded {store2.size()} vectors")

    print("\n" + "="*70)


if __name__ == "__main__":
    demo_faiss()
