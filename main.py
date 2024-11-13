import json
import os
import anthropic
from pathlib import Path
import time
from Model.retrieval import RAGSystem
from tqdm import tqdm

class RAGEnsemble:
    def __init__(self):
        self.client = anthropic.Anthropic(
            api_key=os.environ["claude"]
        )
        self.cache = self.load_cache()

    def load_cache(self):
        """Load or create cache file"""
        if os.path.exists('cache.jsonl'):
            with open('cache.jsonl', 'r', encoding='utf-8') as f:
                return {int(json.loads(line)['qid']): json.loads(line)
                        for line in f if line.strip()}
        return {}

    def update_cache(self, qid, response_data):
        """Update cache"""
        self.cache[qid] = response_data
        with open('cache.jsonl', 'a', encoding='utf-8') as f:
            f.write(json.dumps(response_data, ensure_ascii=False) + '\n')

    def get_doc_content(self, doc_id, category):
        """Get document content based on category"""
        if category == 'faq':
            try:
                with open('reference_processed/faq/pid_map_content.json', 'r', encoding='utf-8') as f:
                    faq_data = json.load(f)
                    return str(faq_data.get(str(doc_id)))
            except:
                return None
        else:
            try:
                file_path = f"reference_processed/{category}/{doc_id}.md"
                with open(file_path, 'r', encoding='utf-8') as f:
                    return f.read().strip()
            except:
                return None

    def create_prompt(self, query, doc1_content, doc2_content):
        """Create prompt for Claude"""
        return f"""Analyze the following query and two documents, and determine which document better answers the query.

Query:
{query}

Document 1:
{doc1_content}

Document 2:
{doc2_content}

Please analyze carefully and answer in the following format:
xxx<better_doc>1 or 2</better_doc>xxx
<reason>Reason for choice</reason>xxx

Note:
1. Only answer with number 1 or 2
2. If neither document answers the query well, choose the more relevant one
3. Must use the specified XML tag format"""

    def query_claude(self, prompt):
        """Query Claude API"""
        try:
            message = self.client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                messages=[{"role": "user", "content": prompt}]
            )
            return message.content[0].text
        except Exception as e:
            print(f"API Error: {e}")
            return None

    def parse_claude_response(self, response):
        """Parse Claude's response"""
        try:
            import re
            doc_match = re.search(r'<better_doc>(\d)</better_doc>', response)
            if doc_match:
                return int(doc_match.group(1))
            return None
        except:
            return None

    def process_predictions(self, cohere_preds, voyage_preds, questions):
        """Process predictions from both models"""
        # Create dictionaries for easier lookup
        question_dict = {q['qid']: q for q in questions['questions']}
        cohere_dict = {item['qid']: item['retrieve'] for item in cohere_preds['answers']}
        voyage_dict = {item['qid']: item['retrieve'] for item in voyage_preds['answers']}

        new_predictions = []

        for qid in tqdm(cohere_dict.keys(), desc="Processing predictions"):
            if qid in self.cache:
                # Use cached result
                better_doc = self.parse_claude_response(self.cache[qid]['response'])
            elif cohere_dict[qid] != voyage_dict[qid]:
                # Need to query Claude
                question = question_dict[qid]
                doc1_content = self.get_doc_content(cohere_dict[qid], question['category'])
                doc2_content = self.get_doc_content(voyage_dict[qid], question['category'])

                if doc1_content and doc2_content:
                    prompt = self.create_prompt(question['query'], doc1_content, doc2_content)
                    response = self.query_claude(prompt)

                    if response:
                        self.update_cache(qid, {
                            'qid': qid,
                            'prompt': prompt,
                            'response': response
                        })
                        better_doc = self.parse_claude_response(response)
                    else:
                        better_doc = None
                else:
                    better_doc = None
            else:
                better_doc = None

            # Decide final prediction
            if better_doc == 1:
                retrieve = cohere_dict[qid]
            elif better_doc == 2:
                retrieve = voyage_dict[qid]
            else:
                retrieve = voyage_dict[qid]  # Default to Voyage prediction

            new_predictions.append({
                'qid': qid,
                'retrieve': retrieve
            })

        # Save final results
        with open('pred_retrieve_ensemble.json', 'w', encoding='utf-8') as f:
            json.dump({'answers': new_predictions}, f, indent=4, ensure_ascii=False)

def main():
    # API keys
    COHERE_API_KEY = "mVgLGio8Vtt6AF6xPiAMVWt8Ilh3uS3R3EFbF4QS"
    VOYAGE_API_KEY = "pa-ba2YDLXqM9GeWR1ZTMyun0885G9qiA9oifbZ5tbzZJY"

    # File paths
    cohere_chunks_path = "cohere_chunks_retrieval.jsonl"
    voyage_chunks_path = "voyage_chunks_retrieval.jsonl"
    questions_path = "questions_preliminary.json"

    print("Step 1: Initializing RAG System...")
    rag = RAGSystem(
        cohere_chunks_path=cohere_chunks_path,
        voyage_chunks_path=voyage_chunks_path,
        cohere_api_key=COHERE_API_KEY,
        voyage_api_key=VOYAGE_API_KEY
    )

    print("Step 2: Processing queries and generating predictions...")
    if not os.path.exists('cohere_pred.json') or not os.path.exists('voyage_pred.json'):
        rag.process_queries(questions_path=questions_path)

    print("Step 3: Loading predictions for ensemble...")
    with open('cohere_pred.json', 'r') as f1, \
            open('voyage_pred.json', 'r') as f2, \
            open(questions_path, 'r') as fq:
        cohere_preds = json.load(f1)
        voyage_preds = json.load(f2)
        questions = json.load(fq)

    print("Step 4: Running ensemble stacking with Claude...")
    ensemble = RAGEnsemble()
    ensemble.process_predictions(cohere_preds, voyage_preds, questions)

    print("Pipeline completed! Final predictions saved in 'pred_retrieve_ensemble.json'")

if __name__ == "__main__":
    main()