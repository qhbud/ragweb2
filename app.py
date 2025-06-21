from flask import Flask, request, jsonify, render_template, session
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import logging

# Import your existing RAG components
from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import torch
from sentence_transformers import SentenceTransformer

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - these should be environment variables in production
CHROMA_PATH = os.environ.get('CHROMA_PATH', '/opt/chroma')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')

class GPUHuggingFaceEmbeddings:
    """Optimized embeddings class for serverless deployment"""
    def __init__(self, model_name="all-MiniLM-L6-v2", batch_size=32):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Using device: {self.device}")
        
        # Use smaller batch size for serverless environment
        self.model = SentenceTransformer(model_name, device=self.device)
        self.batch_size = batch_size
        
    def embed_documents(self, texts):
        all_embeddings = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i+self.batch_size]
            with torch.no_grad():
                batch_embeddings = self.model.encode(
                    batch, 
                    batch_size=self.batch_size,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=True
                )
            all_embeddings.extend(batch_embeddings.tolist())
        return all_embeddings
    
    def embed_query(self, text):
        with torch.no_grad():
            embedding = self.model.encode([text], convert_to_numpy=True, normalize_embeddings=True)
        return embedding[0].tolist()

class WebRAGSystem:
    """Web-optimized RAG system"""
    
    def __init__(self):
        self.embeddings = None
        self.vector_store = None
        self.llm = None
        self.qa_chain = None
        self.initialized = False
        
    def initialize(self):
        """Lazy initialization for better startup time"""
        if self.initialized:
            return True
            
        try:
            logger.info("Initializing RAG System...")
            
            # Check if OpenAI API key is available
            if not OPENAI_API_KEY:
                logger.error("OpenAI API key not found")
                return False
            
            # Initialize embeddings
            self.embeddings = GPUHuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            
            # Load vector store
            if not os.path.exists(CHROMA_PATH):
                logger.error(f"Chroma database not found at {CHROMA_PATH}")
                return False
                
            self.vector_store = Chroma(
                persist_directory=CHROMA_PATH,
                embedding_function=self.embeddings
            )
            
            # Initialize LLM
            os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
            self.llm = ChatOpenAI(
                model_name="gpt-3.5-turbo",
                temperature=0.1,
                max_tokens=1000,
                request_timeout=30
            )
            
            # Setup prompt template
            template = """You are an expert legal assistant specializing in USC (United States Code) documents. 
Use the following pieces of context to answer the question at the end. 

Instructions:
- Provide accurate, detailed answers based on the context
- If you don't know the answer based on the context, say so clearly
- Include relevant USC section numbers when applicable
- Explain legal concepts in clear, understandable language
- Use direct quotes from sources whenever possible

Context:
{context}

Question: {question}

Answer: """

            prompt_template = PromptTemplate(
                template=template,
                input_variables=["context", "question"]
            )
            
            # Create QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=self.llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": 5}
                ),
                return_source_documents=True,
                chain_type_kwargs={"prompt": prompt_template}
            )
            
            self.initialized = True
            logger.info("RAG System initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize RAG system: {e}")
            return False
    
    def query(self, question: str):
        """Process a query and return formatted response"""
        if not self.initialize():
            return {
                "error": "System not properly initialized",
                "answer": "Sorry, the system is currently unavailable. Please try again later.",
                "sources": []
            }
        
        try:
            # Process query
            result = self.qa_chain({"query": question})
            
            # Format sources for web display
            sources = []
            for i, doc in enumerate(result["source_documents"], 1):
                source_info = {
                    "id": i,
                    "title": doc.metadata.get("source", "Unknown Source"),
                    "content": doc.page_content[:500] + "..." if len(doc.page_content) > 500 else doc.page_content,
                    "metadata": doc.metadata
                }
                sources.append(source_info)
            
            return {
                "answer": result["result"],
                "sources": sources,
                "num_sources": len(sources),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing query: {e}")
            return {
                "error": str(e),
                "answer": "Sorry, I encountered an error processing your question. Please try again.",
                "sources": []
            }

# Global RAG system instance
rag_system = WebRAGSystem()

@app.route('/')
def index():
    """Serve the main page"""
    return render_template('index.html')

@app.route('/api/query', methods=['POST'])
def api_query():
    """API endpoint for processing queries"""
    try:
        data = request.get_json()
        
        if not data or 'question' not in data:
            return jsonify({
                "error": "Missing question in request",
                "answer": "Please provide a question.",
                "sources": []
            }), 400
        
        question = data['question'].strip()
        
        if not question:
            return jsonify({
                "error": "Empty question",
                "answer": "Please provide a valid question.",
                "sources": []
            }), 400
        
        # Log the query (be careful with sensitive data in production)
        logger.info(f"Processing query: {question[:100]}...")
        
        # Process the query
        result = rag_system.query(question)
        
        # Add session tracking (optional)
        if 'session_id' not in session:
            session['session_id'] = str(uuid.uuid4())
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            "error": "Internal server error",
            "answer": "Sorry, something went wrong. Please try again.",
            "sources": []
        }), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_initialized": rag_system.initialized
        }
        
        # Check if vector store is accessible
        if rag_system.initialized and rag_system.vector_store:
            try:
                # Simple test query
                test_docs = rag_system.vector_store.similarity_search("test", k=1)
                status["vector_store"] = "accessible"
                status["documents_count"] = len(test_docs)
            except Exception as e:
                status["vector_store"] = f"error: {str(e)}"
        
        return jsonify(status)
        
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }), 500

@app.route('/api/stats', methods=['GET'])
def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "system_initialized": rag_system.initialized,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
            "timestamp": datetime.now().isoformat()
        }
        
        if rag_system.initialized and rag_system.vector_store:
            # Get collection info if available
            try:
                collection = rag_system.vector_store._collection
                stats["collection_count"] = collection.count()
            except:
                stats["collection_count"] = "unknown"
        
        return jsonify(stats)
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.errorhandler(404)
def not_found(error):
    return jsonify({"error": "Endpoint not found"}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({"error": "Internal server error"}), 500

if __name__ == '__main__':
    # For local development
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)