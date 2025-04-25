import os
import json
import csv
import re
import time
from typing import List, Dict, Any, Tuple
import nltk
from pathlib import Path

# For document processing
import PyPDF2

# For Gemini API
import google.generativeai as genai

# Fixed API key - replace with your actual key
API_KEY = "AIzaSyB04qQY08JsMO5wFUodErkLCxNNkb359mA"
genai.configure(api_key=API_KEY)

# Download required NLTK data
try:
    nltk.download('punkt')
except:
    print("Could not download NLTK data automatically. Using regex-based sentence splitting instead.")

class DocumentProcessor:
    """Agent responsible for processing the document and creating chunks"""
    
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.chunks = []
    
    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Extract text from a PDF file with page markers"""
        print(f"Opening PDF file: {pdf_path}")
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            full_text = ""
            print(f"Total pages in PDF: {len(reader.pages)}")
            for page_num in range(len(reader.pages)):
                if page_num % 10 == 0:
                    print(f"Processing page {page_num+1}/{len(reader.pages)}")
                page = reader.pages[page_num]
                text = page.extract_text()
                if text:  # Some pages might be empty or contain only images
                    full_text += text + f" [PAGE:{page_num+1}]"
            print(f"Extracted {len(full_text)} characters from PDF")
            return full_text
    
    def custom_sentence_tokenize(self, text: str) -> List[str]:
        """Custom sentence tokenizer as a fallback if NLTK fails"""
        # Use regex to split on sentence boundaries (period, exclamation, question mark)
        # followed by space and uppercase letter or end of string
        sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z]|$)', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def identify_sections(self, text: str) -> List[Dict[str, Any]]:
        """Identify section titles and their positions in text"""
        # Pattern to identify section headers (can be adjusted based on textbook format)
        section_patterns = [
            r'(\d+\.\d+\s+[A-Za-z\s]+)',  # e.g., "1.2 The Industrial Revolution"
            r'(Chapter\s+\d+\s*:\s*[A-Za-z\s]+)',  # e.g., "Chapter 3: Colonial Period"
            r'(UNIT\s+\d+\s*:\s*[A-Za-z\s]+)'  # e.g., "UNIT 2: COLONIAL PERIOD"
        ]
        
        sections = []
        
        for pattern in section_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                sections.append({
                    "title": match.group(1).strip(),
                    "start": match.start(),
                    "end": match.end()
                })
        
        # Sort sections by their position in text
        sections.sort(key=lambda x: x["start"])
        
        # Set the end of each section to the start of the next one
        for i in range(len(sections) - 1):
            sections[i]["end"] = sections[i + 1]["start"]
        
        print(f"Identified {len(sections)} sections in the document")
        return sections
    
    def create_chunks(self, text: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks from text with metadata"""
        # First, identify sections in the document
        sections = self.identify_sections(text)
        
        # Now create chunks
        chunks = []
        
        # Try to use NLTK, fall back to custom tokenizer if needed
        try:
            from nltk.tokenize import sent_tokenize
            sentences = sent_tokenize(text)
            print(f"Tokenized text into {len(sentences)} sentences using NLTK")
        except:
            sentences = self.custom_sentence_tokenize(text)
            print(f"Tokenized text into {len(sentences)} sentences using custom tokenizer")
        
        current_chunk = ""
        current_chunk_sections = set()
        current_chunk_pages = set()
        current_position = 0
        
        for sentence in sentences:
            # Extract page information if present
            page_match = re.search(r'\[PAGE:(\d+)\]', sentence)
            if page_match:
                page_num = page_match.group(1)
                # Remove the page marker from the text
                sentence = re.sub(r'\[PAGE:\d+\]', '', sentence)
                current_chunk_pages.add(page_num)
            
            # Update current position in text
            sentence_position = text.find(sentence, current_position)
            if sentence_position != -1:
                current_position = sentence_position
                
                # Check which section this sentence belongs to
                for section in sections:
                    if section["start"] <= sentence_position <= section["end"]:
                        current_chunk_sections.add(section["title"])
            
            # Add sentence to current chunk
            if len(current_chunk) + len(sentence) <= self.chunk_size:
                current_chunk += sentence + " "
            else:
                # Store current chunk and start a new one with overlap
                if current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "sections": list(current_chunk_sections),
                        "pages": list(current_chunk_pages)
                    })
                
                # Start new chunk with overlap from the previous one
                words = current_chunk.split()
                overlap_text = " ".join(words[-self.chunk_overlap:]) if len(words) > self.chunk_overlap else ""
                current_chunk = overlap_text + " " + sentence + " "
                # Keep section and page info
        
        # Add the last chunk if not empty
        if current_chunk:
            chunks.append({
                "text": current_chunk.strip(),
                "sections": list(current_chunk_sections),
                "pages": list(current_chunk_pages)
            })
        
        print(f"Created {len(chunks)} chunks from the document")
        self.chunks = chunks
        return chunks

    def process_document(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Process document end-to-end and return chunks"""
        text = self.extract_text_from_pdf(pdf_path)
        chunks = self.create_chunks(text)
        return chunks

class SimpleRetrievalAgent:
    """Simple agent for retrieving relevant text chunks without vector DB"""
    
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Initialized Gemini model for semantic search")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini model: {e}")
            self.model = None
        self.chunks = []
    
    def add_chunks(self, chunks: List[Dict[str, Any]]):
        """Store document chunks"""
        self.chunks = chunks
    
    def semantic_search(self, query: str, chunks: List[Dict[str, Any]], n_results=3) -> List[Dict[str, Any]]:
        """Use Gemini to rank chunks by relevance to query"""
        if not self.model or not chunks:
            return self.keyword_search(query, chunks, n_results)
        
        try:
            # First, do keyword filtering to get a manageable number of candidates
            candidate_chunks = self.keyword_search(query, chunks, n_results=10)
            if not candidate_chunks:
                print("No chunks matched keywords, using basic retrieval")
                return chunks[:n_results] if chunks else []
            
            # Prepare chunks for evaluation
            chunk_texts = [chunk["text"][:500] for chunk in candidate_chunks]  # First 500 chars
            
            prompt = f"""
            QUERY: {query}
            
            I will show you {len(chunk_texts)} text passages. For each passage, rate its relevance to the query on a scale from 0-10.
            Return only a JSON array of scores, like [7, 2, 9, ...]. No other text.
            
            PASSAGES:
            """
            
            for i, text in enumerate(chunk_texts):
                prompt += f"\n{i+1}. {text}\n"
            
            response = self.model.generate_content(prompt)
            
            # Parse the response to get relevance scores
            try:
                scores = json.loads(response.text)
                if not isinstance(scores, list) or len(scores) != len(chunk_texts):
                    raise ValueError("Invalid response format")
                
                # Create list of (index, score) pairs and sort by score
                chunk_scores = [(i, score) for i, score in enumerate(scores)]
                chunk_scores.sort(key=lambda x: x[1], reverse=True)
                
                # Return the top n chunks
                return [candidate_chunks[idx] for idx, _ in chunk_scores[:n_results]]
            except:
                # Fall back to keyword search result if response parsing fails
                print("Failed to parse semantic search results, using keyword search instead")
                return candidate_chunks[:n_results]
                
        except Exception as e:
            print(f"Error in semantic search: {str(e)}")
            # Fall back to keyword search
            return self.keyword_search(query, chunks, n_results)
    
    def keyword_search(self, query: str, chunks: List[Dict[str, Any]], n_results=3) -> List[Dict[str, Any]]:
        """Simple keyword-based retrieval"""
        if not chunks:
            return []
            
        # Extract keywords from the query (words with 4+ characters)
        keywords = [word.lower() for word in re.findall(r'\b\w{4,}\b', query)]
        
        # If no significant keywords, return first chunks
        if not keywords:
            return chunks[:n_results]
        
        # Score chunks based on keyword occurrence
        chunk_scores = []
        for i, chunk in enumerate(chunks):
            score = 0
            text = chunk["text"].lower()
            
            # Count occurrences of each keyword
            for keyword in keywords:
                score += text.count(keyword)
            
            # Bonus for exact phrase matches
            for phrase in re.findall(r'\b(\w+\s+\w+\s+\w+)\b', query):
                if phrase.lower() in text:
                    score += 5
            
            chunk_scores.append((i, score))
        
        # Sort by score (descending)
        chunk_scores.sort(key=lambda x: x[1], reverse=True)
        
        # Return top n chunks with scores greater than 0
        results = [chunks[idx] for idx, score in chunk_scores[:n_results*2] if score > 0][:n_results]
        
        # If no chunks matched keywords, return first chunks
        if not results:
            return chunks[:n_results]
            
        return results
    
    def retrieve_relevant_chunks(self, query: str, n_results=3) -> List[Dict[str, Any]]:
        """Retrieve the most relevant chunks for a query"""
        print(f"Retrieving relevant chunks for query: {query[:50]}...")
        if self.model:
            return self.semantic_search(query, self.chunks, n_results)
        else:
            return self.keyword_search(query, self.chunks, n_results)

class QueryAnalyzerAgent:
    """Agent responsible for analyzing queries and planning the response strategy"""
    
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Initialized Gemini model for query analysis")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini model: {e}")
            self.model = None
    
    def analyze_query(self, query: str) -> Dict[str, Any]:
        """Analyze the query to determine the type and keywords"""
        print(f"Analyzing query: {query[:50]}...")
        if not self.model:
            # Simple fallback analysis
            keywords = [w for w in query.lower().split() if len(w) > 3]
            return {
                "topic": "history",
                "keywords": keywords[:5],
                "answer_type": "factual" if "?" in query else "explanation"
            }
        
        prompt = f"""
        Analyze the following history question:
        
        "{query}"
        
        As an expert historian and educator, identify:
        1. The main topic
        2. 3-5 key words or phrases that should be in the retrieved context
        3. What type of answer is required (factual, explanation, comparison, cause-effect)
        
        Return your analysis in JSON format with fields: topic, keywords, answer_type.
        """
        
        try:
            response = self.model.generate_content(prompt)
            response_text = response.text
            
            # Try to parse as JSON
            try:
                # Clean up the response to ensure valid JSON
                json_str = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_str:
                    analysis = json.loads(json_str.group(0))
                    return analysis
                else:
                    raise ValueError("No JSON object found in response")
            except:
                # Extract information from the text response
                topic = "unknown"
                keywords = []
                answer_type = "factual"
                
                # Try to extract info line by line
                lines = response_text.split('\n')
                for line in lines:
                    if "topic" in line.lower():
                        topic_match = re.search(r'topic.*?:\s*(.*)', line, re.IGNORECASE)
                        if topic_match:
                            topic = topic_match.group(1).strip()
                    elif "keywords" in line.lower() or "key words" in line.lower():
                        keywords_str = re.search(r'(keywords|key words).*?:\s*(.*)', line, re.IGNORECASE)
                        if keywords_str:
                            keywords = [k.strip() for k in re.split(r'[,"]', keywords_str.group(2)) if k.strip()]
                    elif "answer type" in line.lower() or "type of answer" in line.lower():
                        answer_match = re.search(r'(answer type|type of answer).*?:\s*(.*)', line, re.IGNORECASE)
                        if answer_match:
                            answer_type = answer_match.group(2).strip().lower()
                
                return {
                    "topic": topic,
                    "keywords": keywords[:5],
                    "answer_type": answer_type
                }
        except Exception as e:
            print(f"Error in query analysis: {str(e)}")
            # Fallback to simple keyword extraction
            keywords = [w for w in query.lower().split() if len(w) > 3]
            return {
                "topic": "history",
                "keywords": keywords[:5],
                "answer_type": "factual" if "?" in query else "explanation"
            }

class ContextRefinerAgent:
    """Agent responsible for refining and combining retrieved contexts"""
    
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Initialized Gemini model for context refinement")
        except Exception as e:
            print(f"Warning: Could not initialize Gemini model: {e}")
            self.model = None
    
    def refine_context(self, query: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Refine and combine retrieved contexts"""
        print("Refining context from retrieved chunks...")
        # Simple concatenation if no model available or no chunks
        if not self.model or not chunks:
            combined_text = "\n\n".join([chunk["text"] for chunk in chunks]) if chunks else "No relevant context found."
            
            # Collect sections and pages
            all_sections = set()
            all_pages = set()
            
            for chunk in chunks:
                sections = chunk.get("sections", [])
                pages = chunk.get("pages", [])
                
                all_sections.update(sections)
                all_pages.update(pages)
                
            return {
                "refined_context": combined_text[:4000],  # Limit to avoid token issues
                "sections": list(all_sections),
                "pages": list(all_pages)
            }
        
        try:
            # Combine texts with markers to identify sources
            marked_texts = []
            for i, chunk in enumerate(chunks):
                marked_texts.append(f"CHUNK {i+1}:\n{chunk['text'][:1500]}")  # Limit each chunk
            
            combined_text = "\n\n".join(marked_texts)
            
            prompt = f"""
            Given this question and retrieved context chunks, extract only the most relevant information 
            needed to answer the question accurately. Remove any irrelevant information.
            
            QUESTION: {query}
            
            RETRIEVED CONTEXTS:
            {combined_text}
            
            Provide ONLY the refined context with the essential information needed to answer the question.
            Do not label or number the information. Create a cohesive, continuous text.
            """
            
            response = self.model.generate_content(prompt)
            refined_text = response.text
            
            # Collect sections and pages
            all_sections = set()
            all_pages = set()
            
            for chunk in chunks:
                sections = chunk.get("sections", [])
                pages = chunk.get("pages", [])
                
                all_sections.update(sections)
                all_pages.update(pages)
            
            return {
                "refined_context": refined_text,
                "sections": list(all_sections),
                "pages": list(all_pages)
            }
        except Exception as e:
            print(f"Error in context refinement: {str(e)}")
            # Fallback to simple combination
            combined_text = "\n\n".join([chunk["text"] for chunk in chunks]) if chunks else "No relevant context found."
            
            # Collect sections and pages
            all_sections = set()
            all_pages = set()
            
            for chunk in chunks:
                sections = chunk.get("sections", [])
                pages = chunk.get("pages", [])
                
                all_sections.update(sections)
                all_pages.update(pages)
                
            return {
                "refined_context": combined_text[:4000],  # Limit to avoid token issues
                "sections": list(all_sections),
                "pages": list(all_pages)
            }

class AnswerGenerationAgent:
    """Agent responsible for generating the final answer based on context"""
    
    def __init__(self):
        try:
            self.model = genai.GenerativeModel('gemini-1.5-flash')
            print("Initialized Gemini model for answer generation")
        except Exception as e:
            print(f"Error initializing Gemini model: {str(e)}")
            self.model = None
    
    def generate_answer(self, query: str, refined_context: Dict[str, Any], query_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a concise, accurate answer based on the context"""
        print("Generating answer...")
        context_text = refined_context["refined_context"]
        
        # If there's no context, provide a default answer
        if not context_text or context_text == "No relevant context found.":
            return {
                "answer": "The textbook doesn't contain information directly addressing this question.",
                "sections": refined_context["sections"],
                "pages": refined_context["pages"]
            }
            
        # Check if model is available
        if not self.model:
            # Extract sentences from context as a simple answer
            sentences = context_text.split('.')[:3]
            simple_answer = '. '.join(sentences) + '.'
            return {
                "answer": simple_answer,
                "sections": refined_context["sections"],
                "pages": refined_context["pages"]
            }
            
        answer_type = query_analysis.get("answer_type", "factual")
        
        prompt_template = {
            "factual": """
            You are an expert historian answering a student's factual question.
            
            QUESTION: {query}
            
            CONTEXT: {context}
            
            Based ONLY on the provided context, provide a precise, factual answer (max 150 words).
            Include specific dates, names, and events mentioned in the context.
            DO NOT include information not present in the context.
            DO NOT include phrases like "According to the context" or "Based on the information provided".
            DO NOT apologize for limitations in the answer.
            
            Just provide the direct answer.
            """,
            
            "explanation": """
            You are an expert historian answering a student's question that requires explanation.
            
            QUESTION: {query}
            
            CONTEXT: {context}
            
            Based ONLY on the provided context, provide a clear explanation (max 150 words).
            Organize your answer logically and focus on causes, processes, and effects as appropriate.
            DO NOT include information not present in the context.
            DO NOT include phrases like "According to the context" or "Based on the information provided".
            DO NOT apologize for limitations in the answer.
            
            Just provide the direct explanation.
            """,
            
            "comparison": """
            You are an expert historian answering a student's question that requires comparison.
            
            QUESTION: {query}
            
            CONTEXT: {context}
            
            Based ONLY on the provided context, provide a comparative analysis (max 150 words).
            Highlight similarities and differences between the elements being compared.
            DO NOT include information not present in the context.
            DO NOT include phrases like "According to the context" or "Based on the information provided".
            DO NOT apologize for limitations in the answer.
            
            Just provide the direct comparison.
            """,
            
            "cause-effect": """
            You are an expert historian answering a student's question about cause and effect.
            
            QUESTION: {query}
            
            CONTEXT: {context}
            
            Based ONLY on the provided context, explain the causes and effects (max 150 words).
            Clearly distinguish between causes and their resulting effects.
            DO NOT include information not present in the context.
            DO NOT include phrases like "According to the context" or "Based on the information provided".
            DO NOT apologize for limitations in the answer.
            
            Just provide the direct cause-effect explanation.
            """
        }
        
        # Use the appropriate prompt template based on answer_type, default to factual
        prompt_text = prompt_template.get(answer_type, prompt_template["factual"])
        prompt = prompt_text.format(query=query, context=context_text[:4000])
        
        try:
            response = self.model.generate_content(prompt)
            answer = response.text.strip()
            
            # Check if answer is empty
            if not answer:
                answer = "Based on the available information, a specific answer to this question cannot be provided."
            
            # Ensure answer is concise (150 words max)
            words = answer.split()
            if len(words) > 150:
                answer = " ".join(words[:150]) + "..."
                
            return {
                "answer": answer,
                "sections": refined_context["sections"],
                "pages": refined_context["pages"]
            }
        except Exception as e:
            print(f"Error in answer generation: {str(e)}")
            # Extract sentences from context as a fallback answer
            sentences = context_text.split('.')[:3]
            fallback_answer = '. '.join(sentences) + '.'
            return {
                "answer": fallback_answer,
                "sections": refined_context["sections"],
                "pages": refined_context["pages"]
            }

class HistoryChatbot:
    """Main class that orchestrates the multi-agent collaboration"""
    
    def __init__(self):
        print("Initializing History Chatbot with multiple agents...")
        
        # Initialize agents
        self.document_processor = DocumentProcessor()
        self.retrieval_agent = SimpleRetrievalAgent()
        self.query_analyzer = QueryAnalyzerAgent()
        self.context_refiner = ContextRefinerAgent()
        self.answer_generator = AnswerGenerationAgent()
        
        self.chunks = []
    
    def process_document(self, pdf_path: str):
        """Process a document and prepare it for retrieval"""
        print(f"Processing document: {pdf_path}")
        
        # Process document and get chunks
        self.chunks = self.document_processor.process_document(pdf_path)
        
        # Add chunks to retrieval agent
        self.retrieval_agent.add_chunks(self.chunks)
        
        print(f"Processed document and created {len(self.chunks)} chunks")
    
    def answer_query(self, query_id: str, question: str) -> Dict[str, Any]:
        """Process a query and generate an answer"""
        print(f"Processing query {query_id}: {question[:50]}...")
        
        try:
            # Step 1: Analyze the query
            query_analysis = self.query_analyzer.analyze_query(question)
            print(f"Query analysis: {query_analysis}")
            
            # Step 2: Retrieve relevant chunks
            relevant_chunks = self.retrieval_agent.retrieve_relevant_chunks(question, n_results=3)
            print(f"Retrieved {len(relevant_chunks)} relevant chunks")
            
            # Step 3: Refine context
            refined_context = self.context_refiner.refine_context(question, relevant_chunks)
            
            # Step 4: Generate answer
            result = self.answer_generator.generate_answer(question, refined_context, query_analysis)
            
            # Step 5: Format and return response
            response = {
                "ID": query_id,
                "Question": question,
                "Context": refined_context["refined_context"],
                "Answer": result["answer"],
                "Sections": ",".join(result["sections"]),
                "Pages": ",".join(result["pages"])
            }
            
            return response
        except Exception as e:
            print(f"Error processing query {query_id}: {str(e)}")
            
            # Provide a meaningful fallback response with available information
            if hasattr(self, 'chunks') and self.chunks:
                # Use first chunk as emergency context
                emergency_chunk = self.chunks[0]
                emergency_text = emergency_chunk.get("text", "")[:500]
                emergency_sections = emergency_chunk.get("sections", [])
                emergency_pages = emergency_chunk.get("pages", [])
                
                return {
                    "ID": query_id,
                    "Question": question,
                    "Context": emergency_text,
                    "Answer": "This question requires information not clearly identified in the textbook.",
                    "Sections": ",".join(emergency_sections),
                    "Pages": ",".join(emergency_pages)
                }
            else:
                # Complete fallback with no chunks
                return {
                    "ID": query_id,
                    "Question": question,
                    "Context": "Unable to retrieve relevant context.",
                    "Answer": "This question requires information not found in the textbook.",
                    "Sections": "N/A",
                    "Pages": "N/A"
                }
    
    def process_query_file(self, query_file_path: str, output_file_path: str):
        """Process all queries in a file and save results to CSV"""
        print(f"Processing queries from: {query_file_path}")
        
        # Load queries
        with open(query_file_path, 'r', encoding='utf-8') as f:
            queries = json.load(f)
        
        results = []
        
        # Process each query
        for query in queries:
            query_id = query.get("query_id", "")
            question = query.get("question", "")
            
            try:
                result = self.answer_query(query_id, question)
                results.append(result)
                
                # Add a small delay to avoid rate limits
                time.sleep(1)
            except Exception as e:
                print(f"Error processing query {query_id}: {str(e)}")
                # Add a basic response to maintain completeness
                results.append({
                    "ID": query_id,
                    "Question": question,
                    "Context": query.get("context", "Unable to retrieve relevant context."),
                    "Answer": query.get("answer", "This question requires information not found in the textbook."),
                    "Sections": "N/A",
                    "Pages": "N/A"
                })
        
        # Save results to CSV
        fieldnames = ["ID", "Question", "Context", "Answer", "Sections", "Pages"]
        
        with open(output_file_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(results)
        
        print(f"Results saved to {output_file_path}")
        return results

# Example usage
if __name__ == "__main__":
    # Initialize the chatbot
    chatbot = HistoryChatbot()
    
    # Process the textbook PDF - try different potential paths
    pdf_paths = [
        "refer/grade-11-history-text-book.pdf",
        "grade-11-history-text-book.pdf",
        "./grade-11-history-text-book.pdf"
    ]
    
    pdf_loaded = False
    for path in pdf_paths:
        if os.path.exists(path):
            print(f"Found PDF at: {path}")
            try:
                chatbot.process_document(path)
                pdf_loaded = True
                break
            except Exception as e:
                print(f"Error processing PDF at {path}: {str(e)}")
    
    if not pdf_loaded:
        print("Could not find or process the PDF file. Please ensure it exists and is accessible.")
        print("Available files in current directory:")
        for file in os.listdir("."):
            if file.endswith(".pdf"):
                print(f"  - {file}")
        exit(1)
    
    # Process queries and generate answers - try different potential paths
    query_paths = [
        "refer/queries.json",
        "queries.json",
        "./queries.json"
    ]
    
    output_path = "future_minds_submission.csv"
    
    query_processed = False
    for path in query_paths:
        if os.path.exists(path):
            print(f"Found queries at: {path}")
            try:
                chatbot.process_query_file(path, output_path)
                query_processed = True
                break
            except Exception as e:
                print(f"Error processing queries at {path}: {str(e)}")
    
    if not query_processed:
        print("Could not find or process the queries file. Please ensure it exists and is accessible.")
        print("Available JSON files in current directory:")
        for file in os.listdir("."):
            if file.endswith(".json"):
                print(f"  - {file}")
    else:
        print(f"Successfully processed all queries and saved results to {output_path}")