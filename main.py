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

def kth_dd221():
    # set up prompt (can be given via input)
    # VT2021 / Subject 2
    query = """Questions:
     - Before I took a COVID test, the doctor said 99% of the people in the area have COVID, and 90% of them are testing positive. A few days later the doctor called and said my test was positive,
and that the probability I have COVID given this positive test is p% — I can’t remember because I was in shock. Find the minimum value of p such that I can compute the probability I got a positive
test but don’t have COVID, and then compute the maximum probability I don’t have COVID given my positive test. Show your work.
    - Consider the data in the following markdown table:
| walk | poopoos | doggo     |
|------|---------|-----------|
| 1    | 2       | Shoogee   |
| 2    | 3       | Shoogee   |
| 3    | 1       | Shoogee   |
| 4    | 2       | Shoogee   |
| 5    | 0       | Shoogee   |
| 1    | 2       | MaxTheTax |
| 2    | 3       | MaxTheTax |
| 3    | 4       | MaxTheTax |
| 4    | 3       | MaxTheTax |
| 5    | 4       | MaxTheTax |
Assume all observations are independent. For each doggo, fit the number of its poopoos with a Poisson distribution using maximum likelihood estimation of the
parameters, and compute those parameters. Show your work.
    - Consider the data in the table above. Assume all observations are independent. For each doggo, fit its number of poopoos with a Poisson distribution using maximum a posteriori estimation of
the parameters, and compute those parameters. Assume the prior distribution of the parameters is exponential with parameter gamma=3. Show your work.
    - Consider the data in the table above. Assume all observations are independent. For each doggo, find the Bayes’ predictive posterior for the number of its poopoos. Model the number of poopoos of
each dog with a Poisson distribution and assume the prior distribution of the parameters is exponential with parameter gamma=3. For convenience, assume P (Dk) = 1. Express the Bayes’
predictive posterior in closed-form, i.e., not as an integral. Show your work.
     """
    # set path to PDF files
    pdfs_dir = data_dir / 'kth_dd2421'
    # call RAG with Gemini
    return gemini(query, pdfs_dir, rag_top_n=10, verbose=False)

if __name__ == '__main__':
    response = kth_dd221()
    print(response)
