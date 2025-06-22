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
import sys
import traceback

# Configure logging FIRST - before anything else
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.StreamHandler(sys.stderr)
    ]
)
logger = logging.getLogger(__name__)

# Log the very start of the application
logger.info("="*50)
logger.info("STARTING APPLICATION INITIALIZATION")
logger.info("="*50)

try:
    logger.info("Creating Flask app instance...")
    app = Flask(__name__)
    logger.info("Flask app created successfully")
    
    logger.info("Setting Flask secret key...")
    app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'your-secret-key-change-this')
    logger.info("Flask secret key set")
    
    logger.info("Initializing CORS...")
    CORS(app)
    logger.info("CORS initialized successfully")
    
except Exception as e:
    logger.error(f"CRITICAL ERROR during Flask initialization: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Configuration - these should be environment variables in production
logger.info("Loading environment variables...")
try:
    CHROMA_PATH = os.environ.get('CHROMA_PATH', '/tmp/chroma')
    OPENAI_API_KEY = os.environ.get('OPENAI_API_KEY')
    S3_BUCKET = os.environ.get('S3_BUCKET_NAME')
    S3_CHROMA_KEY = os.environ.get('S3_CHROMA_KEY', 'chroma-db.tar.gz')
    AWS_REGION = os.environ.get('AWS_REGION', 'us-east-1')
    
    logger.info("Environment variables loaded:")
    logger.info(f"  CHROMA_PATH: {CHROMA_PATH}")
    logger.info(f"  OPENAI_API_KEY: {'SET' if OPENAI_API_KEY else 'NOT SET'}")
    logger.info(f"  S3_BUCKET_NAME: {'SET' if S3_BUCKET else 'NOT SET'}")
    logger.info(f"  S3_CHROMA_KEY: {S3_CHROMA_KEY}")
    logger.info(f"  AWS_REGION: {AWS_REGION}")
    
except Exception as e:
    logger.error(f"ERROR loading environment variables: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    raise

# Global variables for lazy loading
logger.info("Initializing global variables...")
rag_system = None
system_initialized = False
initialization_attempted = False
logger.info("Global variables initialized")

def check_required_environment():
    """Check if all required environment variables are set"""
    logger.info("Checking required environment variables...")
    try:
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
    except Exception as e:
        logger.error(f"Error checking environment variables: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

# Call the check at startup
logger.info("Running environment check...")
env_vars_available = check_required_environment()
logger.info(f"Environment check result: {env_vars_available}")

def download_chroma_from_s3():
    """Download Chroma database files from S3"""
    logger.info("Starting download_chroma_from_s3 function...")
    
    if not S3_BUCKET:
        logger.error("S3_BUCKET_NAME environment variable not set")
        return False
    
    try:
        logger.info(f"Downloading Chroma database from S3 bucket: {S3_BUCKET}")
        
        # Initialize S3 client
        logger.info("Creating S3 client...")
        s3_client = boto3.client('s3', region_name=AWS_REGION)
        logger.info("S3 client created successfully")
        
        # Create chroma directory
        logger.info(f"Setting up directory: {CHROMA_PATH}")
        if os.path.exists(CHROMA_PATH):
            logger.info(f"Directory {CHROMA_PATH} exists, removing...")
            shutil.rmtree(CHROMA_PATH)
        os.makedirs(CHROMA_PATH, exist_ok=True)
        logger.info(f"Directory {CHROMA_PATH} created successfully")
        
        # List all objects in the bucket
        logger.info("Listing objects in S3 bucket...")
        paginator = s3_client.get_paginator('list_objects_v2')
        pages = paginator.paginate(Bucket=S3_BUCKET)
        
        downloaded_files = 0
        for page_num, page in enumerate(pages):
            logger.info(f"Processing page {page_num + 1}")
            if 'Contents' in page:
                for obj_num, obj in enumerate(page['Contents']):
                    key = obj['Key']
                    logger.info(f"Processing object {obj_num + 1}: {key}")
                    
                    # Create local file path
                    local_file_path = os.path.join(CHROMA_PATH, key)
                    
                    # Create directory if needed
                    local_dir = os.path.dirname(local_file_path)
                    if local_dir != CHROMA_PATH:
                        os.makedirs(local_dir, exist_ok=True)
                        logger.info(f"Created directory: {local_dir}")
                    
                    # Skip if it's a directory marker
                    if key.endswith('/'):
                        logger.info(f"Skipping directory marker: {key}")
                        continue
                    
                    # Download the file
                    logger.info(f"Downloading: {key}")
                    s3_client.download_file(S3_BUCKET, key, local_file_path)
                    downloaded_files += 1
                    logger.info(f"Downloaded {key} successfully")
        
        logger.info(f"Downloaded {downloaded_files} files to {CHROMA_PATH}")
        return downloaded_files > 0
        
    except ClientError as e:
        logger.error(f"AWS S3 error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False
    except Exception as e:
        logger.error(f"Error downloading Chroma database: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return False

def initialize_rag_system():
    """Initialize the RAG system with proper error handling"""
    global rag_system, system_initialized, initialization_attempted
    
    logger.info("Starting RAG system initialization...")
    
    if system_initialized and rag_system:
        logger.info("RAG system already initialized, returning True")
        return True
    
    if initialization_attempted:
        logger.warning("RAG system initialization already attempted and failed")
        return False
    
    initialization_attempted = True
    logger.info("Setting initialization_attempted flag to True")
    
    try:
        # Import heavy dependencies only when needed
        logger.info("Importing heavy dependencies...")
        logger.info("  - Importing langchain_chroma.Chroma...")
        from langchain_chroma import Chroma
        logger.info("  - Importing langchain_community.embeddings.HuggingFaceEmbeddings...")
        from langchain_community.embeddings import HuggingFaceEmbeddings
        logger.info("  - Importing langchain_openai.ChatOpenAI...")
        from langchain_openai import ChatOpenAI
        logger.info("  - Importing langchain.chains.RetrievalQA...")
        from langchain.chains import RetrievalQA
        logger.info("  - Importing langchain.prompts.PromptTemplate...")
        from langchain.prompts import PromptTemplate
        logger.info("All dependencies imported successfully")
        
        logger.info("Initializing RAG System...")
        
        # Check if OpenAI API key is available
        logger.info("Checking OpenAI API key...")
        if not OPENAI_API_KEY:
            logger.error("OpenAI API key not found")
            return False
        logger.info("OpenAI API key is available")
        
        # Download Chroma database from S3 if it doesn't exist locally
        logger.info(f"Checking if Chroma database exists at {CHROMA_PATH}...")
        if not os.path.exists(CHROMA_PATH):
            logger.info("Chroma database not found locally, downloading from S3...")
            if not download_chroma_from_s3():
                logger.error("Failed to download Chroma database from S3")
                return False
        else:
            logger.info("Chroma database found locally")
        
        # Initialize embeddings with CPU-only for serverless
        logger.info("Initializing HuggingFace embeddings...")
        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True}
        )
        logger.info("HuggingFace embeddings initialized successfully")
        
        # Load vector store
        logger.info(f"Double-checking Chroma database path: {CHROMA_PATH}")
        if not os.path.exists(CHROMA_PATH):
            logger.error(f"Chroma database not found at {CHROMA_PATH}")
            return False
            
        logger.info("Loading Chroma vector store...")
        vector_store = Chroma(
            persist_directory=CHROMA_PATH,
            embedding_function=embeddings
        )
        logger.info("Chroma vector store loaded successfully")
        
        # Initialize LLM
        logger.info("Setting OpenAI API key environment variable...")
        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
        logger.info("Initializing ChatOpenAI LLM...")
        llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.1,
            max_tokens=1000,
            timeout=30
        )
        logger.info("ChatOpenAI LLM initialized successfully")
        
        # Setup prompt template
        logger.info("Setting up prompt template...")
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
        logger.info("Prompt template created successfully")
        
        # Create QA chain
        logger.info("Creating RetrievalQA chain...")
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
        logger.info("RetrievalQA chain created successfully")
        
        # Store in global variables
        logger.info("Storing RAG system components in global variables...")
        rag_system = {
            'qa_chain': qa_chain,
            'vector_store': vector_store,
            'embeddings': embeddings
        }
        
        system_initialized = True
        logger.info("RAG System initialized successfully!")
        logger.info("="*50)
        return True
        
    except Exception as e:
        logger.error(f"Failed to initialize RAG system: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    logger.info("="*30)
    logger.info("INDEX ROUTE ACCESSED")
    logger.info("="*30)
    
    try:
        response_data = {
            "message": "USC RAG System API",
            "status": "running",
            "timestamp": datetime.now().isoformat(),
            "system_initialized": system_initialized,
            "endpoints": [
                "/",
                "/api/health",
                "/api/query",
                "/api/stats"
            ]
        }
        logger.info(f"Returning response: {response_data}")
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error in index route: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
        return jsonify({"error": "Internal server error in index route"}), 500

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
    logger.info("="*30)
    logger.info("HEALTH CHECK ACCESSED")
    logger.info("="*30)
    
    try:
        # Basic health check
        status = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "system_initialized": system_initialized,
            "env_vars_available": env_vars_available
        }
        
        logger.info(f"Basic health status: {status}")
        
        # Check if vector store is accessible
        if system_initialized and rag_system:
            logger.info("System is initialized, checking vector store...")
            try:
                # Simple test query
                test_docs = rag_system['vector_store'].similarity_search("test", k=1)
                status["vector_store"] = "accessible"
                status["documents_count"] = len(test_docs)
                logger.info(f"Vector store check successful: {len(test_docs)} documents found")
            except Exception as e:
                logger.error(f"Vector store check failed: {e}")
                status["vector_store"] = f"error: {str(e)}"
        else:
            logger.info("System not initialized, skipping vector store check")
        
        logger.info(f"Final health status: {status}")
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Health check error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")
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
    logger.error(f"Internal server error: {error}")
    return jsonify({"error": "Internal server error"}), 500

# Add startup logging
logger.info("="*60)
logger.info("FLASK APPLICATION STARTUP SEQUENCE")
logger.info("="*60)
logger.info("Flask application starting...")
logger.info(f"Environment variables check: {env_vars_available}")
logger.info(f"Python version: {sys.version}")
logger.info(f"Current working directory: {os.getcwd()}")
logger.info(f"Available environment variables: {list(os.environ.keys())}")

# Test basic functionality
try:
    logger.info("Testing basic Flask functionality...")
    with app.test_client() as client:
        logger.info("Flask test client created successfully")
except Exception as e:
    logger.error(f"Error creating Flask test client: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")

logger.info("="*60)
logger.info("STARTUP SEQUENCE COMPLETE")
logger.info("="*60)

if __name__ == '__main__':
    # For local development
    logger.info("Running in __main__ mode (development)")
    port = int(os.environ.get('PORT', 8000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    logger.info(f"Debug mode: {debug}")
    app.run(host='0.0.0.0', port=port, debug=debug)
else:
    # For production deployment
    logger.info("Flask app loaded for production deployment")
    logger.info("App is ready to receive requests")
