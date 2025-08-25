from utils.llm_client import lm_studio_client

def rewrite_query(search_query, model):
    prompt = f'''
    Given below the user request for queries regarding Indian food recipes, rephrase and expand the query to a more search friendly term.

    user query: {search_query}

    > provide only and only a simple phrase for the user query, do not add any other information or context.
    > this output will be used to search the database for recipes.
    '''
    llm_response = lm_studio_client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ]
    )
    
    resp = llm_response.choices[0].message.content
    print('rewritting query: ', resp)
    return resp
