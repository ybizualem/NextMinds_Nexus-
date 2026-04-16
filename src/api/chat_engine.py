"""
Gemini conversation engine with function calling.

The LLM can call `search_curriculum` to query the pgvector database,
then formats results into a teacher-friendly response.
"""

from __future__ import annotations

import json
import logging
import uuid
from collections import defaultdict

from google import genai
from google.genai import types

from src.config import GEMINI_API_KEY
from src.embeddings.embedder import embed_text
from src.db.operations import search_activities

logger = logging.getLogger(__name__)

# In-memory session store (swap for Redis in production)
_sessions: dict[str, list[types.Content]] = defaultdict(list)

SYSTEM_INSTRUCTION = """You are the NextMinds Curriculum Assistant, helping Connecticut teachers discover \
invention-based curriculum activities for their classrooms.

You have access to a search tool that queries a database of NextMinds curriculum activities. \
Each activity belongs to a grade band (K-2, 3-5, 6-8, 9-12) and an invention stage \
(Introduction to Inventing, Identifying & Ideating, Understanding, Engineering Design, \
Communication, Entrepreneurship).

IMPORTANT — Conversation context:
- Remember the grade band, stage, and other preferences the teacher mentioned earlier in the conversation.
- When a follow-up question references an activity from previous results (e.g. "what is in What Is An Invention?"), \
apply the same grade_band filter from earlier unless the teacher explicitly asks for a different grade.
- If multiple versions of the same activity exist across grade bands, only show the one matching the \
teacher's previously stated grade. If no grade was stated, show all versions but note the difference.

When a teacher asks for activities:
1. Use the search_curriculum tool with an appropriate query and any filters they mention or implied from context.
2. Present results clearly: activity name, grade band, stage, a brief description of what the activity contains, and a clickable link.
3. When a description is available, summarize what students will actually do in the activity (materials, exercises, learning goals). Don't just restate the activity name.
4. If keywords are available, use them to highlight key topics covered.
5. If no results match, suggest broadening the search (different stage).
6. Be concise, warm, and teacher-focused.

Never invent activities that aren't in the search results.
Always include the complete resource URL as a clickable markdown link — never truncate or omit it.
Format links as: [Open Resource](url)"""

# Define the function tool for Gemini
SEARCH_TOOL = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="search_curriculum",
            description="Search the NextMinds curriculum database for activities matching a query. "
                        "Returns ranked results with activity names, grade bands, stages, and resource URLs.",
            parameters=types.Schema(
                type="OBJECT",
                properties={
                    "query": types.Schema(
                        type="STRING",
                        description="Natural language search query describing what the teacher needs",
                    ),
                    "grade_band": types.Schema(
                        type="STRING",
                        description="Optional grade band filter: K-2, 3-5, 6-8, or 9-12",
                        enum=["K-2", "3-5", "6-8", "9-12"],
                    ),
                    "stage": types.Schema(
                        type="STRING",
                        description="Optional stage name filter (partial match)",
                    ),
                    "limit": types.Schema(
                        type="INTEGER",
                        description="Number of results to return (default 5, max 10)",
                    ),
                },
                required=["query"],
            ),
        )
    ]
)


def _get_client() -> genai.Client:
    """Create a Gemini client using the API key."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set in environment. Get one at https://aistudio.google.com/apikey")
    return genai.Client(api_key=GEMINI_API_KEY)


def _execute_search(args: dict) -> list[dict]:
    """Execute the search_curriculum function call against our database."""
    query = args.get("query", "")
    grade_band = args.get("grade_band")
    stage = args.get("stage")
    limit = min(args.get("limit", 5), 10)

    query_embedding = embed_text(query)
    results = search_activities(
        query_embedding=query_embedding,
        grade_band=grade_band,
        stage=stage,
        limit=limit,
    )
    return results


def chat(message: str, session_id: str | None = None) -> tuple[str, str, list[dict]]:
    """
    Send a message to Gemini and get a response, possibly with search results.

    Returns: (reply_text, session_id, activities_found)
    """
    if not session_id:
        session_id = str(uuid.uuid4())

    client = _get_client()
    history = _sessions[session_id]

    # Add user message
    history.append(types.Content(role="user", parts=[types.Part(text=message)]))

    all_activities: list[dict] = []

    # Call Gemini with tools
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[SEARCH_TOOL],
            temperature=0.7,
            max_output_tokens=4096,
        ),
    )

    # Handle function calls (may need multiple rounds)
    max_rounds = 3
    for _ in range(max_rounds):
        # Check if the model wants to call a function
        function_calls = [
            part for part in (response.candidates[0].content.parts or [])
            if part.function_call
        ]

        if not function_calls:
            break

        # Add assistant's function call to history
        history.append(response.candidates[0].content)

        # Execute each function call and collect results
        function_responses = []
        for fc_part in function_calls:
            fc = fc_part.function_call
            if fc.name == "search_curriculum":
                results = _execute_search(fc.args)
                all_activities.extend(results)
                # Build a JSON-serializable response
                response_data = {
                    "results": [
                        {
                            "activity_name": r["activity_name"],
                            "grade_band": r["grade_band"],
                            "stage": r["stage"],
                            "description": r.get("description") or "",
                            "keywords": r.get("keywords") or [],
                            "resource_url": r["resource_url"],
                            "resource_type": r["resource_type"],
                            "similarity": round(r.get("similarity", 0), 3),
                        }
                        for r in results
                    ],
                    "count": len(results),
                }
                function_responses.append(
                    types.Part(function_response=types.FunctionResponse(
                        name="search_curriculum",
                        response=response_data,
                    ))
                )

        # Send function results back to Gemini
        history.append(types.Content(role="user", parts=function_responses))

        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=history,
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM_INSTRUCTION,
                tools=[SEARCH_TOOL],
                temperature=0.7,
                max_output_tokens=4096,
            ),
        )

    # Extract final text reply
    reply = response.candidates[0].content.parts[0].text if response.candidates else "I couldn't generate a response."

    # Add assistant reply to history
    history.append(types.Content(role="model", parts=[types.Part(text=reply)]))

    # Keep history bounded (last 20 turns)
    if len(history) > 40:
        _sessions[session_id] = history[-40:]

    return reply, session_id, all_activities


def clear_session(session_id: str):
    """Clear conversation history for a session."""
    _sessions.pop(session_id, None)
