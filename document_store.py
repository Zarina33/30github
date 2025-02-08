import numpy as np
from typing import List, Dict, Optional
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

class DocumentStore:
    def __init__(self, embedding_model_name: str = 'all-MiniLM-L6-v2'):
        """Инициализация хранилища документов"""
        self.documents: List[Dict] = []
        self.embeddings = None
        self.embedding_model = SentenceTransformer(embedding_model_name)

    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> int:
        """Добавление документов в хранилище"""
        if not texts:
            return 0

        if metadata is None:
            metadata = [{} for _ in texts]

        # Создаем эмбеддинги для новых документов
        new_embeddings = self.embedding_model.encode(texts)

        # Добавляем документы и их метаданные
        for text, meta in zip(texts, metadata):
            self.documents.append({
                'text': text,
                'metadata': meta
            })

        # Обновляем массив эмбеддингов
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

        return len(texts)

    def search(self, query: str, top_k: int = 3, min_similarity: float = 0.0) -> List[Dict]:
        """Поиск релевантных документов"""
        if not self.documents:
            return []

        # Создаем эмбеддинг для запроса
        query_embedding = self.embedding_model.encode([query])[0]

        # Вычисляем схожесть с каждым документом
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Фильтруем результаты по минимальной схожести
        relevant_indices = np.where(similarities >= min_similarity)[0]

        # Сортируем по убыванию схожести и берем top_k
        top_indices = relevant_indices[np.argsort(similarities[relevant_indices])[-top_k:][::-1]]

        # Формируем результаты
        results = []
        for idx in top_indices:
            results.append({
                'document': self.documents[idx],
                'similarity': float(similarities[idx])
            })

        return results