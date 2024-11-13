import os
from typing import List, Dict, Optional
import json
import cohere
import voyageai
from tqdm import tqdm
from nltk.tokenize import word_tokenize
import re

# Constants
CHUNK_SIZE = 256
OVERLAP_LINE = 2
COHERE_API_KEY = os.environ["cohere"]
VOYAGE_API_KEY = os.environ["voyage"]

class ChunkProcessor:
    def __init__(self):
        self.co = cohere.Client(api_key=COHERE_API_KEY)
        self.vo = voyageai.Client(api_key=VOYAGE_API_KEY)

    def nltk_tokenizer(self, text: str) -> int:
        """Count tokens in text using NLTK."""
        return len(word_tokenize(text))

    def is_table_content(self, line: str) -> bool:
        """Check if line contains table content."""
        return line.strip().startswith('|') and line.strip().endswith('|')

    def should_replace_value(self, value: str) -> bool:
        """Determine if a value should be replaced with '-'."""
        value = value.strip()
        if not value:
            return False

        numeric_pattern = r'^[\d\.\,\-\$\%\+\/\*\=]*$'
        basic_pattern = r'^[\w\s\.\,\-\(\)\[\]\$\%\+\/\*\=\&\;\:\#\@\!]*$'

        if re.match(numeric_pattern, value):
            return True

        if re.match(basic_pattern, value):
            cleaned_value = re.sub(r'[\W_]', '', value)
            if len(cleaned_value) > 0 and not cleaned_value.isdigit():
                return False
            return True

        return False

    def process_table(self, table: str) -> Optional[str]:
        """Process table by replacing numeric values with '-'."""
        lines = table.strip().split('\n')
        if len(lines) < 3:
            return None

        processed_lines = [lines[0], lines[1]]

        for line in lines[2:]:
            cells = line.split('|')
            processed_cells = []

            for cell in cells:
                cell = cell.strip()
                if not cell or cell.isspace():
                    processed_cells.append(cell)
                elif self.should_replace_value(cell):
                    processed_cells.append('-')
                else:
                    processed_cells.append(cell)

            processed_lines.append('|' + '|'.join(processed_cells) + '|')

        return '\n'.join(processed_lines)

    def extract_tables(self, content: str) -> List[str]:
        """Extract tables from markdown content."""
        tables = []
        current_table = []
        in_table = False

        for line in content.split('\n'):
            line = line.strip()
            if self.is_table_content(line):
                if not in_table:
                    in_table = True
                current_table.append(line)
            elif in_table:
                if current_table:
                    tables.append('\n'.join(current_table))
                    current_table = []
                in_table = False

        if current_table:
            tables.append('\n'.join(current_table))

        return tables

    def process_regular_content(self, content: str) -> List[str]:
        """Process regular markdown content into chunks."""
        lines = [line.strip().replace(" ", "") for line in content.split('\n') if line.strip()]
        chunks = []
        current_chunk = []
        current_size = 0

        for i in range(len(lines)):
            line = lines[i]
            line_size = len(line)

            if current_size + line_size > CHUNK_SIZE and current_chunk:
                chunks.append('\n'.join(current_chunk))
                current_chunk = current_chunk[-OVERLAP_LINE:] if len(current_chunk) >= OVERLAP_LINE else current_chunk
                current_size = sum(self.nltk_tokenizer(line) for line in current_chunk)

            current_chunk.append(line)
            current_size += line_size

        if current_chunk:
            chunks.append('\n'.join(current_chunk))

        return chunks

    def get_embeddings(self, texts: List[str]) -> Dict[str, List[List[float]]]:
        """Get embeddings using both Cohere and Voyage."""
        embeddings = {
            'cohere': self.co.embed(
                texts=texts,
                model='embed-multilingual-light-v3.0',
                input_type='search_document'
            ).embeddings,
            'voyage': self.vo.embed(
                texts=texts,
                model='voyage-3',
                input_type='document'
            ).embeddings
        }
        return embeddings

    def process_directory(self, directory_path: str) -> List[Dict]:
        """Process all files in directory and create chunks with embeddings."""
        processed_chunks = []
        chunk_id = 0

        for root, _, files in os.walk(directory_path):
            if not any(category in root for category in ['finance', 'insurance', 'faq']):
                continue

            category = next((cat for cat in ['finance', 'insurance', 'faq'] if cat in root), None)

            for file in files:
                file_path = os.path.join(root, file)

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read()

                    # Process tables if markdown file
                    if file.endswith('.md'):
                        tables = self.extract_tables(content)
                        for table in tables:
                            processed_table = self.process_table(table)
                            if processed_table:
                                chunk_id += 1
                                chunk_data = {
                                    'id': f'{category}_table_{chunk_id}',
                                    'text': processed_table,
                                    'category': category,
                                    'source_file': file_path
                                }
                                processed_chunks.append(chunk_data)

                    # Process regular content
                    if file.endswith('.json'):
                        qa_data = json.loads(content)
                        for qa_id, qa_list in qa_data.items():
                            for qa_pair in qa_list:
                                chunk_id += 1
                                chunk_data = {
                                    'id': f'{category}_qa_{chunk_id}',
                                    'text': f"Question: {qa_pair['question']}\nAnswer: {qa_pair['answers'][0]}",
                                    'category': category,
                                    'source_file': file_path
                                }
                                processed_chunks.append(chunk_data)
                    else:
                        regular_chunks = self.process_regular_content(content)
                        for chunk in regular_chunks:
                            chunk_id += 1
                            chunk_data = {
                                'id': f'{category}_{chunk_id}',
                                'text': chunk,
                                'category': category,
                                'source_file': file_path
                            }
                            processed_chunks.append(chunk_data)

                except Exception as e:
                    print(f"Error processing file {file_path}: {str(e)}")

        return processed_chunks

    def save_chunks_with_embeddings(self, chunks: List[Dict]) -> None:
        """Save chunks with both Cohere and Voyage embeddings."""
        # Process chunks in batches to avoid API limits
        batch_size = 100
        total_batches = (len(chunks) + batch_size - 1) // batch_size

        # Clear existing files
        open('cohere_chunks_retrieval.jsonl', 'w').close()
        open('voyage_chunks_retrieval.jsonl', 'w').close()

        with tqdm(total=len(chunks), desc="Processing chunks") as pbar:
            for i in range(0, len(chunks), batch_size):
                batch = chunks[i:i + batch_size]
                texts = [chunk['text'] for chunk in batch]

                # Get embeddings for batch
                cohere_embeddings = self.co.embed(
                    texts=texts,
                    model='embed-multilingual-light-v3.0',
                    input_type='search_document'
                ).embeddings

                voyage_embeddings = self.vo.embed(
                    texts=texts,
                    model='voyage-3',
                    input_type='document'
                ).embeddings

                # Save to separate files for each embedding type
                for j, chunk in enumerate(batch):
                    # Following exact format
                    cohere_chunk = {
                        'text': chunk['text'],
                        'embeddings': {
                            'float': cohere_embeddings[j],
                            'int8': None,
                            'uint8': None,
                            'binary': None,
                            'ubinary': None
                        },
                        'category': chunk['category'],
                        'id': chunk['id'],
                        'source_file': chunk['source_file']
                    }

                    voyage_chunk = {
                        'text': chunk['text'],
                        'embeddings': {
                            'float': voyage_embeddings[j],
                            'int8': None,
                            'uint8': None,
                            'binary': None,
                            'ubinary': None
                        },
                        'category': chunk['category'],
                        'id': chunk['id'],
                        'source_file': chunk['source_file']
                    }

                    # Append to respective files
                    with open('cohere_chunks_retrieval.jsonl', 'a', encoding='utf-8') as f:
                        json.dump(cohere_chunk, f, ensure_ascii=False)
                        f.write('\n')

                    with open('voyage_chunks_retrieval.jsonl', 'a', encoding='utf-8') as f:
                        json.dump(voyage_chunk, f, ensure_ascii=False)
                        f.write('\n')

                pbar.update(len(batch))

if __name__ == '__main__':
    # Initialize processor
    processor = ChunkProcessor()

    # Process directories
    directories = [
        "../reference_processed/finance",
        "../reference_processed/insurance",
        "../reference_processed/faq"
    ]

    all_chunks = []
    for directory in directories:
        chunks = processor.process_directory(directory)
        all_chunks.extend(chunks)

    # Save chunks with embeddings
    processor.save_chunks_with_embeddings(all_chunks)