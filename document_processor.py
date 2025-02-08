# document_processor.py

import PyPDF2
import docx
import json
import pandas as pd
from bs4 import BeautifulSoup
import markdown
import yaml
from typing import List, Dict

class DocumentProcessor:
    """Базовый класс для обработки документов различных форматов"""
    
    @staticmethod
    def read_file(file_path: str, encoding: str = 'utf-8') -> str:
        """Базовый метод для чтения файла"""
        try:
            with open(file_path, 'r', encoding=encoding) as file:
                return file.read()
        except UnicodeDecodeError:
            # Если не удалось прочитать в UTF-8, пробуем другие кодировки
            encodings = ['latin1', 'cp1251', 'iso-8859-1']
            for enc in encodings:
                try:
                    with open(file_path, 'r', encoding=enc) as file:
                        return file.read()
                except UnicodeDecodeError:
                    continue
            raise ValueError(f"Не удалось прочитать файл {file_path} ни в одной из известных кодировок")

    @staticmethod
    def process_txt(file_path: str) -> List[str]:
        """Обработка текстового файла"""
        text = DocumentProcessor.read_file(file_path)
        # Разделяем на параграфы по двойному переносу строки
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        return paragraphs

    @staticmethod
    def process_pdf(file_path: str) -> List[str]:
        """Обработка PDF файла"""
        texts = []
        try:
            with open(file_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text = page.extract_text().strip()
                    if text:  # Добавляем только непустые страницы
                        texts.append(text)
            return texts
        except Exception as e:
            raise ValueError(f"Ошибка при обработке PDF файла {file_path}: {str(e)}")

    @staticmethod
    def process_docx(file_path: str) -> List[str]:
        """Обработка DOCX файла"""
        try:
            doc = docx.Document(file_path)
            return [paragraph.text for paragraph in doc.paragraphs if paragraph.text.strip()]
        except Exception as e:
            raise ValueError(f"Ошибка при обработке DOCX файла {file_path}: {str(e)}")

    @staticmethod
    def process_json(file_path: str, text_fields: List[str]) -> List[str]:
        """Обработка JSON файла"""
        try:
            text = DocumentProcessor.read_file(file_path)
            data = json.loads(text)
            texts = []
            
            def extract_text_from_dict(item: Dict) -> str:
                return ' '.join(str(item.get(field, '')) for field in text_fields if field in item)

            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        text = extract_text_from_dict(item)
                        if text.strip():
                            texts.append(text)
            elif isinstance(data, dict):
                text = extract_text_from_dict(data)
                if text.strip():
                    texts.append(text)
                    
            return texts
        except Exception as e:
            raise ValueError(f"Ошибка при обработке JSON файла {file_path}: {str(e)}")

    @staticmethod
    def validate_file(file_path: str, expected_type: str) -> bool:
        """Проверка соответствия файла ожидаемому типу"""
        file_extension = file_path.lower().split('.')[-1]
        type_extensions = {
            'txt': ['txt'],
            'pdf': ['pdf'],
            'docx': ['docx', 'doc'],
            'json': ['json'],
            'csv': ['csv'],
            'html': ['html', 'htm'],
            'md': ['md', 'markdown'],
            'yaml': ['yaml', 'yml']
        }
        
        if expected_type not in type_extensions:
            raise ValueError(f"Неподдерживаемый тип файла: {expected_type}")
            
        return file_extension in type_extensions[expected_type]

# Пример использования
if __name__ == "__main__":
    processor = DocumentProcessor()
    
    # Пример обработки текстового файла
    try:
        texts = processor.process_txt("example.txt")
        print("Обработан текстовый файл:", len(texts), "параграфов")
    except Exception as e:
        print(f"Ошибка при обработке текстового файла: {str(e)}")
    
    # Пример обработки PDF
    try:
        texts = processor.process_pdf("example.pdf")
        print("Обработан PDF файл:", len(texts), "страниц")
    except Exception as e:
        print(f"Ошибка при обработке PDF файла: {str(e)}")