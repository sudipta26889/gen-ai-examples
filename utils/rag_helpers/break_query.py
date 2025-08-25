import json
from utils.llm_client import lm_studio_client

def break_query(search_query, model):
    subquery_prompt = f'''
    Given below the user request for queries, break down the query into multiple subqueries.
    user query: {search_query}
    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to search the database for recipes.

    > respond with a valid json array of strings, do not add any other information or context.
    '''
    llm_response = lm_studio_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": subquery_prompt}
        ]
    )
    resp = llm_response.choices[0].message.content
    
    # Handle cases where the response might contain markdown code blocks
    try:
        if '```json' in resp:
            # Extract JSON from markdown code block
            subqueries = json.loads(resp.split('```json')[1].split('```')[0].strip())
        elif '```' in resp:
            # Extract JSON from generic code block  
            subqueries = json.loads(resp.split('```')[1].split('```')[0].strip())
        else:
            # Try to parse directly as JSON
            subqueries = json.loads(resp.strip())
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {resp}")
        # Return empty list as fallback
        subqueries = []
        
    print('subqueries:', subqueries)
    return subqueries
