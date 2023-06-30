import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

def cleanify(response: str):
    response = response.lower()
    if len(response) < 2:
        return response
    
    if response.startswith("df.query"):
        response = response[8:] 
    
    if (
        response[0] == response[-1] == "'"
        or response[0] == response[-1] == '"'
    ):
        response = response[1:-1]
    
    return response

def get_pandas_query(user_input: str) -> str:
    system_prompt = (
        "You are a natural language processing tool from natural language to a single Python string. "
        "The user will input natural language, you will output a pandas query string. "
        "The output must be a single string that can be directly passed to the pandas DataFrame query method (pd.DataFrame.query). "
        "Do not answer anything that does not belong to the output string, no explanations, no comments, no variable definition, no usage examples. "
        "If there is an error in the input, you will return 'ERROR'. "
        "'ERROR' is the only string you can return that is not a valid pandas query string. "
        "The target pandas DataFrame has the following columns: "
        "city (string lowercase), "
        "price (float indicating the house price; ranging from 300000 the cheapest to 1000000 the most expensive), "
        "nr_bedrooms (integer; number of bedrooms), "
        "living_area_size (integer; number of square meters, ranging from 10 the smallest to 1000 the biggest), "
        "energy_label (float real number categorical value; ranging from 4 for the best to 10 for the worst). "
        "The output must only contain keys corresponding to the columns of the target DataFrame. "
        "The output logical expression contains brackets to group subexpressions. "
        "Any kind of leading and ending quotation marks must be stripped from the output string. "
    )
    print('GPT INPUT:', user_input)

    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
    )
    answer = out["choices"][0]["message"]["content"]
    answer = cleanify(answer)
    print('GPT OUTPUT:', answer)
    return answer


def get_pretty_prompt(filter_str: str) -> str:
    """
    This function takes a pandas query string and returns a pretty expression.
    """
    system_prompt = (
        "You are a natural language processing tool from a pandas query string to a brief pretty expression. "
        "The user will input a pandas query string logical expression, "
        "The output is a brief keywords summary representation of the input. "
        "The user must be able to understand and interpret the output in a single glance. "
        "The output must be minimalistic, and must only contain key values (values of the expression) and logical operators (using words, e.g. OR). "
        "The output must include symbols like '<' and '[' for value range limits and list attributes whenever possible, to enhance compactness in the output expression. "
    )

    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": filter_str},
        ],
        temperature=0.1,
    )
    answer = out["choices"][0]["message"]["content"]
    return answer


if __name__ == "__main__":
    user_input = "I want a house in Eindhoven with 3 bedrooms"
    response = get_pandas_query(user_input)
    print(response)
