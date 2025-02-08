import os
from typing import Optional, Dict
from rag_system import LocalRAGSystem

class RAGInterface:
    def __init__(self, model_path: str):
        """Инициализация интерфейса RAG системы"""
        self.rag = LocalRAGSystem(
            model_path="/Users/zarinamacbook/Desktop/rag_system/mistral-7b-v0.1.Q4_K_M.gguf",
            n_ctx=1024,  # Размер контекстного окна
            n_batch=512  # Оптимальный размер батча для M1
        )
        print("RAG система успешно инициализирована")

    def add_text(self, text: str, metadata: Optional[Dict] = None) -> None:
        """Добавление текста"""
        try:
            if not text.strip():
                print("Ошибка: Текст не может быть пустым")
                return

            n_docs = self.rag.add_documents([text], [metadata] if metadata else None)
            print(f"Текст успешно добавлен")

        except Exception as e:
            print(f"Ошибка при добавлении текста: {str(e)}")

    def ask_question(self, query: str) -> None:
        """Задать вопрос системе"""
        try:
            if not query.strip():
                print("Ошибка: Вопрос не может быть пустым")
                return

            result = self.rag.generate_response(
                query,
                top_k=3,
                min_similarity=0.1,
                max_tokens=512,
                temperature=0.7
            )

            print("\nОтвет:", result['response'])
            
            if result['documents']:
                print("\nИспользованные документы:")
                for i, doc in enumerate(result['documents'], 1):
                    print(f"\n{i}. Документ (схожесть: {doc['similarity']:.3f}):")
                    print(f"   Текст: {doc['document']['text'][:200]}...")
                    if doc['document']['metadata']:
                        print(f"   Метаданные: {doc['document']['metadata']}")

        except Exception as e:
            print(f"Ошибка при генерации ответа: {str(e)}")

def main():
    """Основная функция программы"""
    # Путь к модели
    model_path = "/Users/zarinamacbook/Desktop/rag_system/mistral-7b-v0.1.Q4_K_M.gguf"  # Измените на ваш путь
    
    if not os.path.exists(model_path):
        print(f"Ошибка: Модель не найдена по пути {model_path}")
        return

    try:
        rag_interface = RAGInterface(model_path)

        while True:
            print("\n=== RAG Система (Локальная модель) ===")
            print("1. Добавить текст")
            print("2. Задать вопрос")
            print("3. Выйти")
            
            choice = input("\nВыберите действие (1-3): ")

            if choice == '1':
                text = input("Введите текст:\n")
                metadata_str = input("Введите метаданные в JSON формате (или Enter для пропуска): ")
                
                metadata = None
                if metadata_str:
                    try:
                        import json
                        metadata = json.loads(metadata_str)
                    except json.JSONDecodeError:
                        print("Ошибка: Неверный формат JSON")
                        continue

                rag_interface.add_text(text, metadata)

            elif choice == '2':
                query = input("Введите ваш вопрос:\n")
                rag_interface.ask_question(query)

            elif choice == '3':
                print("До свидания!")
                break

            else:
                print("Неверный выбор. Пожалуйста, выберите 1, 2 или 3")

    except Exception as e:
        print(f"Произошла ошибка: {str(e)}")
        print("Программа будет закрыта")

if __name__ == "__main__":
    main()