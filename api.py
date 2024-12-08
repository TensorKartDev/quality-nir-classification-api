from fastapi import FastAPI,HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel,Field
from typing import List,Dict,Any,Optional
import ollama
import lancedb
import requests
import json
import os
import pyarrow as pa
import uuid
from datetime import datetime
from fastapi import Query
from sentence_transformers import SentenceTransformer

def generate_unique_id():
    """
    Generate a unique identifier for each record.
    """
    return str(uuid.uuid4())

# Connect to LanceDB and open the table
uri = "data/ncs-lancedb"
db = lancedb.connect(uri)
vectorizedncs_tbl = db.open_table("vectorizedncs")
model = SentenceTransformer("all-MiniLM-L6-v2")
HITLschema = pa.schema([
    ("id", pa.string(), False),         # Unique identifier (non-nullable)
    ("content", pa.string(), True),     # Nullable content field
    ("rating", pa.int32(), False),      # Non-nullable rating
    ("comment", pa.string(), True),     # Nullable comment
    ("timestamp", pa.string(), False),  # Non-nullable timestamp
    ("vector", pa.list_(pa.float32()), True),  # Nullable vector field
])
hitlfeedback_table = db.open_table("HITL") if "HITL" in db.table_names() else db.create_table("HITL",schema=HITLschema)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins; restrict to specific domains in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    topics: List[str]  # List of topics
    numrows: int       # Number of rows to return per topic


# Schema for the feedback
class HITLFeedback(BaseModel):
    rating: int = Field(..., ge=1, le=5, description="Rating must be between 1 and 5")
    content: str  # String for NC number or LLM insight
    comment: Optional[str] = None  # Optional for non-conformance
    topic: Optional[str] = None  # For LLM insights
    cluster_name: Optional[str] = None  # For LLM insights

# Initialize LanceDB or file-based storage
def generate_vector(content):
    _embedding = ollama.embed(model='wizardlm2', input=[content])
    print("Generated vector successfully")
    return _embedding

@app.post("/add_hitl_feedback/")
def add_hitl_feedback(feedback: HITLFeedback):
    """
    Add Human-in-the-Loop feedback to the HITL table in LanceDB.
    """
    try:
        # Generate unique ID and timestamp
        unique_id = generate_unique_id()
        timestamp = datetime.now().isoformat()

        # Prepare the row
        row = {
            "id": unique_id,
            "content": feedback.content.strip(),
            "rating": feedback.rating,
            "timestamp": timestamp,
            "comment": feedback.comment.strip() if feedback.comment else "",
        }

        # Generate vector if comment is provided, else use a default vector
        row["vector"] = [0.0]  # Default vector for empty comments
        if feedback.comment:
            try:
                row["vector"] = generate_vector(feedback.comment).tolist()
            except Exception as vector_error:
                print(f"Error generating vector: {vector_error}")
                row["vector"] = [0.0]  # Fallback to default vector

        # Validate row format
        print("Row to be added to LanceDB:", row)

        # Add the row to LanceDB
        try:
            hitlfeedback_table.add([row])
        except Exception as db_error:
            print(f"Error adding row: {db_error}")
            raise HTTPException(status_code=500, detail=f"Database Error: {str(db_error)}")

        return {"message": "Feedback added successfully.", "data": row}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error adding feedback: {str(e)}")

    
@app.post("/find_non_conformances/")
async def find_non_conformances(request: QueryRequest):
    print(request)
    results = []

    for topic in request.topics:
        query_embedding = model.encode(topic)  # Generate embedding for the topic

        # Perform semantic search in LanceDB
        numrows = request.numrows
        search_results = vectorizedncs_tbl.search(query_embedding, vector_column_name="vector").limit(numrows).to_pandas()

        # Columns to include in the response
        columns_to_show = [
            "NonconformanceNumber", "SupplierName", "PartNumber", 
            "PartName", "NonconformanceDescription", "SiteName","_distance"
        ]
        context = search_results[columns_to_show]

        # Prepare rows as dictionaries
        rows = []
        for _, row in context.iterrows():
            row_dict = {col: str(row[col]) for col in context.columns}
            rows.append(row_dict)

        # Append topic with its rows to the results
        results.append({"topic": topic, "rows": rows})
    print(results)
    return {"results": results}
def RA(messages, model):
    r = requests.post(
        "http://127.0.0.1:11434/api/chat",
        json={"model": model, "messages": messages, "stream": True},
        stream=True
    )
    r.raise_for_status()
    output = ""

    for line in r.iter_lines():
        body = json.loads(line)
        if "error" in body:
            raise Exception(body["error"])
        if body.get("done") is False:
            message = body.get("message", "")
            content = message.get("content", "")
            output += content
        if body.get("done", False):
            message["content"] = output
            return message
# Request model for API
class InsightsRequest(BaseModel):
    context: str  # List of rows with non-conformance data
    question: str               # User's question
    model: str = "wizardlm2"        # Default model to use

# Route to get insights
@app.post("/get_insights")
async def get_insights(request: InsightsRequest):
    """
    Fetch insights from the RA function based on provided non-conformance data and user question.

    Args:
        request (InsightsRequest): Contains the 20 rows of data, user question, and model.

    Returns:
        dict: Insights generated by the RA function.
    """
    try:
        
        # Prepare the prompt
        # rows_context = "\n".join(
        #     [f"{idx+1}. {json.dumps(row)}" for idx, row in enumerate(request.rows)]
        # )
        rows_context = request.context
        messages = [
            {
                "role": "system",
                "content": (
                    "You are an intelligent assistant specializing in analyzing non-conformance data. "
                    "Use the data provided as context to answer the question as accurately as possible.Never ever give code as response, Do not include information that is not found in the provided context, Ensure the response is precise, concise and directly answers the question."
                )
            },
            {
                "role": "user",
                "content": (
                    f"### Non-Conformance Data:\n{rows_context}\n\n"
                    f"### User Question:\n{request.question}\n\n"
                    "Provide a detailed answer based on the provided data."
                )
            }
        ]

        # Call the RA function
        response = RA(messages=messages, model=request.model)
        
        # Return the insights
        return {"insights": response["content"]}

    except requests.exceptions.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Error communicating with RA API: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An error occurred: {str(e)}")
    
@app.get("/get_last_n_feedback/")
def get_last_n_feedback(n: int = Query(10, ge=1)):
    try:
        # Load all rows into a PyArrow Table
        arrow_table = hitlfeedback_table.to_arrow()

        # Check if the table is empty
        if arrow_table.num_rows == 0:
            return {"message": "No feedback entries found.", "data": []}

        # Sort rows by the `timestamp` column in descending order
        sorted_table = arrow_table.sort_by([("timestamp", "descending")])

        # Limit to the last `n` rows
        limited_table = sorted_table.slice(0, n)

        # Convert PyArrow Table to a list of dictionaries
        feedback_list = []
        for row_idx in range(limited_table.num_rows):
            feedback_dict = {
                column_name: limited_table[column_name][row_idx].as_py()
                if hasattr(limited_table[column_name][row_idx], "as_py")
                else limited_table[column_name][row_idx]
                for column_name in limited_table.column_names
            }
            feedback_list.append(feedback_dict)

        return {"message": f"Last {n} feedback retrieved successfully.", "data": feedback_list}
    except Exception as e:
        print(f"Error occurred: {e}")
        raise HTTPException(status_code=500, detail=f"Error fetching feedback: {str(e)}")
