import json
import numpy as np
from collections import defaultdict
import voyageai
import cohere
from tqdm import tqdm
from typing import List, Dict, Any
import re

class RAGSystem:
    def __init__(self,
                 cohere_chunks_path: str,
                 voyage_chunks_path: str,
                 cohere_api_key: str,
                 voyage_api_key: str):
        """
        Initialize RAG system with both Cohere and Voyage chunks

        Args:
            cohere_chunks_path: Path to Cohere chunks jsonl file
            voyage_chunks_path: Path to Voyage chunks jsonl file
            cohere_api_key: Cohere API key
            voyage_api_key: Voyage API key
        """
        # Initialize API clients
        self.co = cohere.Client(api_key=cohere_api_key)
        self.vo = voyageai.Client(api_key=voyage_api_key)

        # Load chunks
        self.cohere_chunks = self.load_jsonl(cohere_chunks_path)
        self.voyage_chunks = self.load_jsonl(voyage_chunks_path)

        # Create chunk embeddings dictionaries
        self.cohere_chunk_embeddings = {}
        self.voyage_chunk_embeddings = {}

        for chunk in self.cohere_chunks:
            doc_id = self.extract_doc_id(chunk['source_file'], chunk['text'])
            self.cohere_chunk_embeddings[chunk['id']] = {
                'embedding': np.array(chunk['embeddings']['float']),
                'text': chunk['text'],
                'doc_id': doc_id
            }

        for chunk in self.voyage_chunks:
            doc_id = self.extract_doc_id(chunk['source_file'], chunk['text'])
            self.voyage_chunk_embeddings[chunk['id']] = {
                'embedding': np.array(chunk['embeddings']['float']),
                'text': chunk['text'],
                'doc_id': doc_id
            }

        print(f"Loaded {len(self.cohere_chunk_embeddings)} Cohere chunks")
        print(f"Loaded {len(self.voyage_chunk_embeddings)} Voyage chunks")

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict]:
        """Load data from jsonl file"""
        data = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data.append(json.loads(line))
        return data

    @staticmethod
    def extract_doc_id(source_file: str, text: str) -> int:
        """Extract doc_id from source_file path or QA text"""
        if 'QA ID:' in text:
            match = re.search(r'QA ID: (\d+)', text)
            if match:
                return int(match.group(1))
            raise ValueError(f"Could not extract QA ID from text: {text[:50]}...")
        else:
            return int(source_file.split('/')[-1].replace('.md', ''))

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        """Calculate cosine similarity between two vectors"""
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

    def get_query_embeddings(self, query_text: str) -> Dict[str, np.ndarray]:
        """
        Get embeddings for query using both Cohere and Voyage

        Args:
            query_text: Query text to embed

        Returns:
            Dictionary with both embeddings
        """
        # Get Cohere embedding
        cohere_response = self.co.embed(
            texts=[query_text],
            model="embed-multilingual-light-v3.0",
            input_type="search_query"
        )

        # Get Voyage embedding
        voyage_response = self.vo.embed(
            texts=[query_text],
            model="voyage-3",
            input_type="query"
        )

        return {
            'cohere': np.array(cohere_response.embeddings[0]),
            'voyage': np.array(voyage_response.embeddings[0])
        }

    def retrieve(self, query_text: str, source_ids: List[int], category: str, k: int = 100) -> Dict[str, Any]:
        """
        Retrieve relevant documents using both embeddings and reranking
        Returns separate predictions for both models
        """
        # Get query embeddings
        query_embeddings = self.get_query_embeddings(query_text)

        # Get similarities using both models
        cohere_similarities = []
        voyage_similarities = []

        # Get Cohere similarities
        for chunk_id, chunk_data in self.cohere_chunk_embeddings.items():
            if chunk_data['doc_id'] in source_ids:
                if category == 'faq' and 'qa' not in chunk_id:
                    continue
                sim = self.cosine_similarity(query_embeddings['cohere'], chunk_data['embedding'])
                cohere_similarities.append({
                    'chunk_id': chunk_id,
                    'similarity': float(sim),
                    'text': chunk_data['text'],
                    'doc_id': chunk_data['doc_id']
                })

        # Get Voyage similarities
        for chunk_id, chunk_data in self.voyage_chunk_embeddings.items():
            if chunk_data['doc_id'] in source_ids:
                if category == 'faq' and 'qa' not in chunk_id:
                    continue
                sim = self.cosine_similarity(query_embeddings['voyage'], chunk_data['embedding'])
                voyage_similarities.append({
                    'chunk_id': chunk_id,
                    'similarity': float(sim),
                    'text': chunk_data['text'],
                    'doc_id': chunk_data['doc_id']
                })

        # Sort by similarity
        cohere_similarities.sort(key=lambda x: x['similarity'], reverse=True)
        voyage_similarities.sort(key=lambda x: x['similarity'], reverse=True)

        # Process Cohere results
        cohere_results = self.process_model_results(cohere_similarities, query_text)

        # Process Voyage results
        voyage_results = self.process_model_results(voyage_similarities, query_text)

        return {
            'cohere': cohere_results[:k],
            'voyage': voyage_results[:k]
        }

    def process_model_results(self, similarities: List[Dict], query_text: str) -> List[Dict]:
        """Helper function to process results for each model"""
        # Group chunks by doc_id
        doc_chunks = defaultdict(list)
        for chunk in similarities[:100]:
            doc_chunks[chunk['doc_id']].append(chunk)

        # Sort chunks within each doc and get top 4
        for doc_id in doc_chunks:
            doc_chunks[doc_id].sort(key=lambda x: x['similarity'], reverse=True)
            doc_chunks[doc_id] = doc_chunks[doc_id][:4]

        # Prepare documents for reranking
        docs_to_rerank = []
        doc_ids_order = []
        for doc_id, chunks in doc_chunks.items():
            combined_text = " ".join([chunk['text'] for chunk in chunks])
            docs_to_rerank.append(combined_text)
            doc_ids_order.append(doc_id)

        try:
            # Rerank with Voyage
            rerank_results = self.vo.rerank(
                query=query_text,
                documents=docs_to_rerank,
                model="rerank-2",
                top_k=len(docs_to_rerank)
            )

            # Get best doc_id from reranking
            top_doc_id = doc_ids_order[rerank_results.results[0].index]

            # Get all chunks from top doc
            top_doc_chunks = []
            for chunk in similarities:
                if chunk['doc_id'] == top_doc_id:
                    top_doc_chunks.append(chunk)

            # Sort top doc chunks by similarity
            top_doc_chunks.sort(key=lambda x: x['similarity'], reverse=True)

            # Combine results
            final_results = []

            # Add top doc chunks first
            for chunk in top_doc_chunks:
                final_results.append(chunk)
                break

            # Add remaining chunks
            for chunk in similarities:
                if chunk not in final_results:
                    final_results.append(chunk)

        except Exception as e:
            print(f"Reranking failed: {str(e)}. Using similarity only.")
            final_results = similarities

        return final_results

    def process_queries(self, questions_path: str):
        """
        Process all queries and save results to separate files
        Updates files after each prediction
        """
        # Load questions
        with open(questions_path, 'r', encoding='utf-8') as f:
            questions = json.load(f)['questions']

        # Initialize results files with empty answer lists
        for model in ['cohere', 'voyage']:
            with open(f'{model}_pred.json', 'w', encoding='utf-8') as f:
                json.dump({'answers': []}, f, indent=2, ensure_ascii=False)

        # Process each question
        for question in tqdm(questions, desc='Processing queries'):
            try:
                # Get predictions
                retrieval_results = self.retrieve(
                    query_text=question['query'],
                    source_ids=question['source'],
                    category=question['category']
                )

                # Update both model files
                for model in ['cohere', 'voyage']:
                    # Read current results
                    with open(f'{model}_pred.json', 'r', encoding='utf-8') as f:
                        current_results = json.load(f)

                    # Add new prediction if available
                    if retrieval_results[model]:
                        top_doc_id = retrieval_results[model][0]['doc_id']
                        current_results['answers'].append({
                            'qid': question['qid'],
                            'retrieve': top_doc_id
                        })

                    # Write updated results
                    with open(f'{model}_pred.json', 'w', encoding='utf-8') as f:
                        json.dump(current_results, f, indent=2, ensure_ascii=False)

            except Exception as e:
                print(f"Error processing question {question['qid']}: {e}")
                continue