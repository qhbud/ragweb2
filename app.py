from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import logging
import tarfile
import shutil

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - these should be environment variables in production
CHROMA_PATH = os.environ.get('CHROMA_PATH', '/tmp/chroma')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
S3_CHROMA_KEY = os.environ.get('S3_CHROMA_KEY', 'chroma-db.tar.gz')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Global variables for lazy loading
rag_system = None
system_initialized = False

# Add this after your app configuration
@app.before_first_request
def check_environment():
    required_vars = ['OPENAI_API_KEY', 'S3_BUCKET_NAME']
    missing_vars = [var for var in required_vars if not os.environ.get(var)]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        # You might want to set a flag here instead of failing completely

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
from datetime import datetime
import boto3
from botocore.exceptions import ClientError
import logging
import tarfile
import shutil

app = Flask(__name__)
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this')
CORS(app)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration - these should be environment variables in production
CHROMA_PATH = os.environ.get('CHROMA_PATH', '/tmp/chroma')
OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
S3_CHROMA_KEY = os.environ.get('S3_CHROMA_KEY', 'chroma-db.tar.gz')
AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')

# Global variables for lazy loading
rag_system = None
system_initialized = False

# ADD THIS ENVIRONMENT CHECK HERE - RIGHT AFTER CONFIGURATION
def check_required_environment():
    """Check if all required environment variables are set"""
    required_vars = {
        'OPENAI_API_KEY': OPENAI_API_KEY,
        'S3_BUCKET_NAME': S3_BUCKET
    }
    
    missing_vars = [var for var, value in required_vars.items() if not value]
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        logger.error("App will start but RAG functionality will be disabled until these are set")
        return False
    else:
        logger.info("All required environment variables are set")
        return True

# Call the check at startup
env_vars_available = check_required_environment()

def download_chroma_from_s3():
    """Download Chroma database files from S3"""
    if not S3_BUCKET:
        logger.error("S3_BUCKET_NAME environment variable not set")
        return False
    
    try:
        logger.info(f"Downloading Chroma database from S3 bucket: {S3_BUCKET}")
        
        # Initialize S3 client
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        
        # Create chroma directory
        if os.path.exists(CHROMA_PATH):
            shutil.rmtree(CHROMA_PATH)
        os.makedirs(CHROMA_PATH, exist_ok=True)
        
        # List all objects in the bucket
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET)
        
        downloaded_files = 0
        for page in pages:
            if 'Contents' in page:
                for obj in page['Contents']:
                    key = obj['Key']
                    
                    # Create local file path
                    local_file_path = os.path.join(CHROMA_PATH, key)
                    
                    # Create directory if needed
                    local_dir = os.path.dirname(local_file_path)
                    if local_dir != CHROMA_PATH:
                        os.makedirs(local_dir, exist_ok=True)
                    
                    # Skip if it's a directory marker
                    if key.endswith('/'):
                        continue
                    
                    # Download the file
                    logger.info(f"Downloading: {key}")
                    s3_client.download_file(S3_BUCKET, key, local_file_path)
                    downloaded_files += 1
        
        logger.info(f"Downloaded {downloaded_files} files to {CHROMA_PATH}")
        return downloaded_files > 0
        
    except ClientError as e:
        logger.error(f"AWS S3 error: {e}")
        return False
    except Exception as e:
        logger.error(f"Error downloading Chroma database: {e}")
        return False

def initialize_rag_system():
    """Initialize the RAG system with proper error handling"""
    global rag_system, system_initialized
    
    if system_initialized and rag_system:
        return True
    
    try:
        # Import heavy dependencies only when needed
        from langchain_chroma import Chroma
        from langchain_community.embeddings import HuggingFaceEmbeddings
        from langchain_openai import ChatOpenAI
        from langchain.chains import RetrievalQA
        from langchain.prompts import PromptTemplate
        
        logger.info("Initializing RAG System...")
        
        # Check if OpenAI API key is available
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found")
            return False
        
        # Download Chroma database from S3 if it doesn't exist locally
        if not os.path.exists(CHROMA_PATH):
            logger.info("Chroma database not found locally, downloading from S3...")
            if not download_chroma_from_s3():
                logger.error("Failed to download Chroma database from S3")
                return False
        
        # Initialize embeddings with CPU-only for serverless
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        
        # Load vector store
        if not os.path.exists(CHROMA_PATH):
            logger.error(f"Chroma database not found at {CHROMA_PATH}")
            return False
            
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        
        # Initialize LLM
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000,
            timeout=30
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
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt_template}
        )
        
        # Store in global variables
        rag_system = {
            'qa_chain': qa_chain,
            'vector_store': vector_store,
            'embeddings': embeddings
        }
        
        system_initialized = True
        logger.info("RAG System initialized successfully!")
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        return False

def process_query(question: str):
    """Process a query and return formatted response"""
    if not initialize_rag_system():
        return {
            "error": "System not properly initialized",
            "answer": "Sorry, the system is currently unavailable. Please try again later.",
            "sources": []
        }
    
    try:
        # Process query
        result = rag_system['qa_chain']({"query": question})
        
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

@app.route('/')
def index():
    """Simple homepage"""
    return jsonify({
        "message": "USC RAG System API",
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "endpoints": [
            "/",
            "/api/health",
            "/api/query",
            "/api/stats"
        ]
    })

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
        result = process_query(question)
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"API error: {e}")
        return jsonify({
            "error": "Internal server error",
            "answer": "Sorry, something went wrong. Please try again.",
            "sources": []
        }), 500

@app.route('/api/health')
def health_check():
    """Health check endpoint"""
    try:
        # Basic health check
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_initialized": system_initialized
        }
        
        # Check if vector store is accessible
        if system_initialized and rag_system:
            try:
                # Simple test query
                test_docs = rag_system['vector_store'].similarity_search("test", k=1)
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

@app.route('/api/stats')
def get_stats():
    """Get system statistics"""
    try:
        stats = {
            "system_initialized": system_initialized,
            "timestamp": datetime.now().isoformat()
        }
        
        if system_initialized and rag_system:
            # Get collection info if available
            try:
                collection = rag_system['vector_store']._collection
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
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    app.run(host='0.0.0.0', port=port, debug=debug)
