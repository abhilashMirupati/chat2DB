#!/usr/bin/env python
"""
Minimal debug runner - Entry point to test queries locally.

Usage:
    python run_debug.py
"""

import sys
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root (same pattern as Streamlit app)
project_root = Path(__file__).parent
load_dotenv(project_root / ".env")

# Add src to path
sys.path.insert(0, str(project_root / "src"))

from sqlai.services.analytics_service import AnalyticsService
from sqlai.utils.logging import configure_logging


def main():
    """Minimal entry point."""
    # Configure logging (same as Streamlit app)
    configure_logging()
    
    print("Initializing...")
    service = AnalyticsService()
    print("Ready! Type 'exit' to quit.\n")
    
    while True:
        question = input("Question: ").strip()
        if not question or question.lower() in ("exit", "quit", "q"):
            break
        
        try:
            result = service.ask(question)
            
            # Handle similar question confirmation
            if result.get("similar_question_found") and result.get("confirmation_required"):
                print(f"\n{result['answer']['text']}\n")
                response = input("Reuse previous query? (y/n): ").strip().lower()
                if response == "y":
                    result = service.execute_saved_query(result["entry_id"])
                else:
                    result = service.ask(question, skip_similarity_check=True)
            
            # Display answer
            answer = result.get("answer", "")
            if isinstance(answer, dict):
                answer = answer.get("text", "")
            print(f"\n{answer}\n")
            
            # Show SQL if available
            sql = result.get("final_sql") or (result.get("executions", [{}])[0].get("sql") if result.get("executions") else None)
            if sql:
                print(f"SQL: {sql}\n")
            
        except Exception as e:
            print(f"Error: {e}\n")


if __name__ == "__main__":
    main()
