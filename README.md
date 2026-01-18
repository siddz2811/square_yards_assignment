# square_yards_assignment

## Goal
The task is to build a Retrieval-Augmented Generation (RAG) pipeline that can answer employee questions accurately by finding the correct version of the truth, citing its sources, and ignoring irrelevant noise.

## Setup & How to run (Windows)
1. Clone:
   - git clone https://github.com/siddz2811/square_yards_assignment.git
   - cd square_yards_assignment
2. Create venv:
   - python -m venv myenv
3. Activate myenv:
   myenv\Scripts\Activate
4. Install dependencies:
   - pip install -r requirements.txt
5. Set API key for pinecone and groq and store them in an .env file:
   - GROQ_API_KEY="your_api_key"
   - Pinecone_api_key="your_pinecone_api_key"
6. Run:
   - python rag.py
   
## NOISE_FILTERING_RESULT.png
Specifically shows that cafeteria docs is not included in company policies
- Although it shows that the Q&A Chatbot responds to questions related to cafeteria menu. But when questions regarding company working days,policies is asked it is explicitly ignored.

## OTHER RESULT IMAGES
Has multiple queries related to company policies,includes noise filtering as well.
