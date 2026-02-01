def build_index(embeddings):
    try:
        import faiss

        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index
    except Exception:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=min(10, len(embeddings)), metric="euclidean")
        nn.fit(embeddings)
        return nn


def main():
    print("Index build scaffold")

if __name__ == "__main__":
    main()
