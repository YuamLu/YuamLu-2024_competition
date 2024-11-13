import os
import nest_asyncio
from llama_parse import LlamaParse
from typing import List
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Apply nest_asyncio for async operations
nest_asyncio.apply()

class PDFProcessor:
    """Class to handle PDF to Markdown conversion using LlamaParse"""

    def __init__(self, api_key: str, output_base_dir: str = "reference_processed"):
        """
        Initialize PDFProcessor

        Args:
            api_key: LlamaParse API key
            output_base_dir: Base directory for processed files
        """
        self.api_key = api_key
        self.output_base_dir = Path(output_base_dir)

    def _get_file_type(self, pdf_path: str) -> str:
        """
        Determine file type from path

        Args:
            pdf_path: Path to PDF file

        Returns:
            String indicating file type ('finance' or 'insurance')
        """
        if "finance" in pdf_path.lower():
            return "finance"
        elif "insurance" in pdf_path.lower():
            return "insurance"
        raise ValueError(f"Unknown file type for path: {pdf_path}")

    def _create_parser(self, premium_mode: bool = False) -> LlamaParse:
        """
        Create LlamaParse instance

        Args:
            premium_mode: Whether to use GPT-4 mode

        Returns:
            LlamaParse instance
        """
        return LlamaParse(
            api_key=self.api_key,
            result_type="markdown",
            language="ch_tra",
            gpt4o_mode=premium_mode,
            do_not_cache=True
        )

    def _extract_text(self, documents: List) -> str:
        """
        Extract text from parsed documents

        Args:
            documents: List of parsed documents

        Returns:
            Combined text from all documents
        """
        return "".join(doc.get_text() for doc in documents)

    def parse_single_pdf(self, pdf_path: str) -> None:
        """
        Parse a single PDF file to Markdown

        Args:
            pdf_path: Path to PDF file
        """
        try:
            file_type = self._get_file_type(pdf_path)
            filename = Path(pdf_path).stem

            # First attempt with standard mode
            parser = self._create_parser(premium_mode=False)
            documents = parser.load_data(pdf_path)
            text_content = self._extract_text(documents)

            # Check if content is valid, if not try premium mode
            if (len(text_content.replace(" ","").replace("\n","")) <= 40) or \
                    ("NO_CONTENT_HERE" in text_content):
                logger.info(f"Retrying {filename} with premium mode...")
                parser = self._create_parser(premium_mode=True)
                documents = parser.load_data(pdf_path)
                text_content = self._extract_text(documents)

            # Prepare output directory and save
            output_dir = self.output_base_dir / file_type
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f"{filename}.md"

            if not text_content:
                raise ValueError(f"Empty content in {pdf_path}")

            output_path.write_text(text_content)
            logger.info(f"Successfully processed {pdf_path}")

        except Exception as e:
            logger.error(f"Error processing {pdf_path}: {str(e)}")
            raise

    def process_folder(self, folder_path: str) -> None:
        """
        Process all PDFs in a folder

        Args:
            folder_path: Path to folder containing PDFs
        """
        folder = Path(folder_path)
        if not folder.exists():
            raise ValueError(f"Folder not found: {folder_path}")

        for pdf_file in folder.rglob("*.pdf"):
            output_md = self.output_base_dir / self._get_file_type(str(pdf_file)) / f"{pdf_file.stem}.md"

            if output_md.exists():
                logger.info(f"Skipping {pdf_file.name} - already processed")
                continue

            self.parse_single_pdf(str(pdf_file))

def main():
    """Main execution function"""
    # Configuration
    API_KEY = os.environ["llamaparse"]
    FOLDERS_TO_PROCESS = ['reference/finance', 'reference/insurance']

    # Initialize processor
    processor = PDFProcessor(api_key=API_KEY)

    # Process each folder
    for folder in FOLDERS_TO_PROCESS:
        try:
            logger.info(f"Processing folder: {folder}")
            processor.process_folder(folder)
        except Exception as e:
            logger.error(f"Error processing folder {folder}: {str(e)}")

if __name__ == "__main__":
    main()