import PyPDF2
import docx
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class TextExtractionService:
    @staticmethod
    def extract_text_from_file(file_path: str) -> str:
        """Extract text from various file formats"""
        try:
            file_path = Path(file_path)
            
            if file_path.suffix.lower() == '.pdf':
                return TextExtractionService._extract_from_pdf(file_path)
            elif file_path.suffix.lower() == '.docx':
                return TextExtractionService._extract_from_docx(file_path)
            elif file_path.suffix.lower() == '.txt':
                return TextExtractionService._extract_from_txt(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path.suffix}")
                
        except Exception as e:
            logger.error(f"Error extracting text from {file_path}: {e}")
            raise

    @staticmethod
    def _extract_from_pdf(file_path: Path) -> str:
        """Extract text from PDF"""
        text = ""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            for page_num, page in enumerate(pdf_reader.pages):
                page_text = page.extract_text()
                if page_text.strip():
                    text += f"\n[Page {page_num + 1}]\n{page_text}\n"
        return text

    @staticmethod
    def _extract_from_docx(file_path: Path) -> str:
        """Extract text from Word document"""
        doc = docx.Document(file_path)
        text = ""
        for paragraph in doc.paragraphs:
            if paragraph.text.strip():
                text += f"{paragraph.text}\n"
        return text

    @staticmethod
    def _extract_from_txt(file_path: Path) -> str:
        """Extract text from TXT file"""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()