from pathlib import Path
from client import Gemini

# Initialize model
gemini = Gemini()

# Set data path
data_dir = Path(__file__).parent / 'data'


def test_google_notebook():
    # set up prompt (can be given via input)
    query = """Questions:
     - What are the critical difference between various graphs for Class A Share?
     - Which index best matches Class A share performance closely where Google is not already a part? Explain the reasoning.
     - Identify key chart patterns for Google Class A shares.
     - What is cost of revenues, operating expenses and net income for 2020. Do mention the percentage change
     - What was the effect of Covid in the 2020 financial year?
     - What are the total revenues for APAC and USA for 2021?
     - What is deferred income taxes?
     - How do you compute net income per share?
     - What drove percentage change in the consolidated revenue and cost of revenue for the year 2021 and was there any effect of Covid?
     - What is the cause of 41% increase in revenue from 2020 to 2021 and how much is dollar change?
     """
    # set path to PDF files
    pdfs_dir = data_dir / 'google_test'
    # call RAG with Gemini
    return gemini(query, pdfs_dir, rag_top_n=10, verbose=True)


if __name__ == '__main__':
    response = test_google_notebook()
    print(response)
