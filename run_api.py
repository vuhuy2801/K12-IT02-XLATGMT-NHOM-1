import uvicorn
import argparse

def main():
    parser = argparse.ArgumentParser(description='Run Animal Classification API')
    parser.add_argument('--host', type=str, default='0.0.0.0',
                      help='Host to run the API on')
    parser.add_argument('--port', type=int, default=8000,
                      help='Port to run the API on')
    parser.add_argument('--reload', action='store_true',
                      help='Enable auto-reload for development')
    
    args = parser.parse_args()
    
    uvicorn.run(
        "api:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=1
    )

if __name__ == "__main__":
    main() 