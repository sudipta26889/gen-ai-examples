import json
from utils.llm_client import lm_studio_client

def generate_metadata(search_query, model):
    meta_prompt = f'''
    Given below the user request for queries, create metadata filter dictionary for the search.

    user query: {search_query}

    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to filter the recipes.

    available metadata:
    - 'Cuisine': string: ['Indian', 'Kerala Recipes', 'Oriya Recipes', 'Continental',
        'Chinese', 'Konkan', 'Chettinad', 'Mexican', 'Kashmiri',
        'South Indian Recipes', 'North Indian Recipes', 'Andhra',
        'Gujarati Recipes', 'Bengali Recipes']
    - 'Diet': string: ['Vegetarian', 'High Protein Vegetarian', 'Non Vegeterian',
        'Eggetarian', 'Diabetic Friendly', 'High Protein Non Vegetarian',
        'Gluten Free', 'Sugar Free Diet', 'No Onion No Garlic (Sattvic)',
        'Vegan']
    - 'ComplexityLevel': string: ['Medium', 'Hard']

    We can do exact match only. Choose the most appropriate values from the lists above.

    For queries about:
    - "Bengali" dishes -> use "Bengali Recipes" for Cuisine
    - "North Indian" dishes -> use "North Indian Recipes" for Cuisine
    - "South Indian" dishes -> use "South Indian Recipes" for Cuisine
    - "non veg" or "meat" -> use "Non Vegeterian" for Diet
    - "vegetarian" -> use "Vegetarian" for Diet

    respond with a valid json dictionary, do not add any other information or context.
    '''
    
    llm_response = lm_studio_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": meta_prompt}
        ]
    )
    
    # Extract the actual content string from the response
    resp = llm_response.choices[0].message.content
    
    # Handle cases where the response might contain markdown code blocks
    try:
        if '```json' in resp:
            # Extract JSON from markdown code block
            metadata = json.loads(resp.split('```json')[1].split('```')[0].strip())
        elif '```' in resp:
            # Extract JSON from generic code block
            metadata = json.loads(resp.split('```')[1].split('```')[0].strip())
        else:
            # Try to parse directly as JSON
            metadata = json.loads(resp.strip())
    except (json.JSONDecodeError, IndexError) as e:
        print(f"Error parsing JSON response: {e}")
        print(f"Raw response: {resp}")
        # Return empty metadata as fallback
        metadata = {}

    print('metadata:', metadata)
    return metadata
