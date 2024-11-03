import numpy as np
import faiss


class FAISSIndex:
    def __init__(self, dimension, index_type='l2'):
        """
        Initialize FAISS index

        Args:
            dimension: dimensionality of the vectors
            index_type: 'l2' for L2 distance, 'ip' for inner product,
                       'cosine' for cosine similarity
        """
        if index_type == 'l2':
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == 'ip':
            self.index = faiss.IndexFlatIP(dimension)
        elif index_type == 'cosine':
            # For cosine similarity, we need to normalize vectors
            self.index = faiss.IndexFlatIP(dimension)
        else:
            raise ValueError("index_type must be 'l2', 'ip', or 'cosine'")

        self.index_type = index_type
        self.dimension = dimension
        self.id_map = {}  # Map to store vector IDs and their metadata

    def add_vectors(self, vectors, ids=None, metadata=None):
        """
        Add vectors to the index

        Args:
            vectors: numpy array of shape (n, dimension)
            ids: optional list of IDs for the vectors
            metadata: optional list of metadata for each vector
        """
        if len(vectors.shape) != 2 or vectors.shape[1] != self.dimension:
            raise ValueError(f"vectors must be shape (n, {self.dimension})")

        # Convert to float32 if needed
        vectors = vectors.astype(np.float32)

        # Normalize vectors for cosine similarity
        if self.index_type == 'cosine':
            faiss.normalize_L2(vectors)

        # Generate sequential IDs if none provided
        if ids is None:
            ids = list(range(len(self.id_map), len(self.id_map) + len(vectors)))

        # Store metadata
        for i, vid in enumerate(ids):
            self.id_map[vid] = metadata[i] if metadata else None

        # Add to index
        self.index.add(vectors)

    def search(self, query_vectors, k=5):
        """
        Search for similar vectors

        Args:
            query_vectors: numpy array of shape (n, dimension)
            k: number of nearest neighbors to return

        Returns:
            distances: numpy array of shape (n, k)
            indices: numpy array of shape (n, k)
            metadata: list of metadata for each returned vector
        """
        if len(query_vectors.shape) != 2 or query_vectors.shape[1] != self.dimension:
            raise ValueError(f"query_vectors must be shape (n, {self.dimension})")

        # Convert to float32 if needed
        query_vectors = query_vectors.astype(np.float32)

        # Normalize query vectors for cosine similarity
        if self.index_type == 'cosine':
            faiss.normalize_L2(query_vectors)

        # Perform search
        distances, indices = self.index.search(query_vectors, k)

        # Get metadata for results
        metadata = []
        for query_results in indices:
            query_metadata = []
            for idx in query_results:
                if idx != -1:  # FAISS returns -1 for not enough results
                    query_metadata.append(self.id_map.get(idx))
                else:
                    query_metadata.append(None)
            metadata.append(query_metadata)

        return distances, indices, metadata

if __name__ == "__main__":
    # Create sample embeddings (100 vectors of dimension 64)
    dimension = 64
    num_vectors = 100
    vectors = np.random.random((num_vectors, dimension)).astype('float32')

    # Initialize index
    index = FAISSIndex(dimension, index_type='cosine')

    # Add vectors with metadata
    metadata = [f"document_{i}" for i in range(num_vectors)]
    index.add_vectors(vectors, metadata=metadata)

    # Create query vectors
    query_vectors = np.random.random((5, dimension)).astype('float32')

    # Search
    distances, indices, metadata = index.search(query_vectors, k=3)

    # Print results for first query
    print("Results for first query:")
    for i in range(len(distances[0])):
        print(f"Distance: {distances[0][i]:.4f}, Document: {metadata[0][i]}")