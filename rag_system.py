from typing import List, Dict, Optional
from llama_cpp import Llama
from document_store import DocumentStore

class LocalRAGSystem:
    def __init__(
        self,
        model_path: str,
        embedding_model_name: str = "all-MiniLM-L6-v2",
        n_ctx: int = 2048,
        n_batch: int = 512
    ):
        """
        Инициализация RAG системы оптимизированной для Mac M1
        
        Args:
            model_path: путь к модели в формате GGUF
            embedding_model_name: название модели для эмбеддингов
            n_ctx: размер контекстного окна
            n_batch: размер батча для инференса
        """
        # Инициализируем хранилище документов
        self.document_store = DocumentStore(embedding_model_name)
        
        # Загружаем модель с оптимизациями для M1
        self.model = Llama(
            model_path=model_path,
            n_ctx=n_ctx,
            n_batch=n_batch,
            n_threads=6,            # Ограничиваем количество потоков
            seed=-1,               # Случайный сид
            n_gpu_layers=0,        # Отключаем GPU слои для Metal
            use_mlock=False,       # Отключаем блокировку памяти
            use_mmap=True,         # Включаем memory mapping
            vocab_only=False,
            verbose=True
        )
        
    def generate_prompt(self, query: str, context: List[Dict]) -> str:
        """Генерация промпта для модели"""
        # Формируем контекст из релевантных документов
        context_texts = [doc['document']['text'] for doc in context]
        context_str = "\n\n".join(context_texts)
        
        # Создаем промпт
        prompt = f"""<s>[INST] На основе предоставленного контекста ответьте на вопрос.
Если в контексте нет необходимой информации, так и скажите.

Контекст:
{context_str}

Вопрос: {query} [/INST]"""
        
        return prompt
        
    def generate_response(
        self,
        query: str,
        top_k: int = 3,
        min_similarity: float = 0.0,
        max_tokens: int = 512,
        temperature: float = 0.7
    ) -> Dict:
        """Генерация ответа на вопрос"""
        # Ищем релевантные документы
        relevant_docs = self.document_store.search(
            query,
            top_k=top_k,
            min_similarity=min_similarity
        )
        
        if not relevant_docs:
            return {
                'response': 'Не найдено релевантных документов для ответа на ваш вопрос.',
                'documents': []
            }
        
        try:
            # Генерируем промпт
            prompt = self.generate_prompt(query, relevant_docs)
            
            # Генерируем ответ с оптимизированными параметрами для M1
            response = self.model.create_completion(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=0.9,
                top_k=40,
                repeat_penalty=1.1,
                stop=["[INST]", "[/INST]", "</s>"],
                echo=False
            )
            
            # Извлекаем ответ
            answer = response['choices'][0]['text'].strip()
            
        except Exception as e:
            return {
                'response': f'Произошла ошибка при генерации ответа: {str(e)}',
                'documents': relevant_docs
            }
        
        return {
            'response': answer,
            'documents': relevant_docs
        }

    def add_documents(self, texts: List[str], metadata: Optional[List[Dict]] = None) -> int:
        """Добавление документов"""
        return self.document_store.add_documents(texts, metadata)