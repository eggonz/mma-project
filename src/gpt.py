import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = (
    "You are a natural language processing tool from natural language to a single Python string. "
    "The user will input natural language, you will output a pandas query string. "
    "The output must be a single string that can be directly passed to the pandas DataFrame query method (pd.DataFrame.query). "
    "Do not answer anything that does not belong to the output string, no explanations, no comments, no variable definition, no usage examples. "
    "If there is an error in the input, you will return 'ERROR'. "
    "'ERROR' is the only string you can return that is not a valid pandas query string. "
    "The target pandas DataFrame has the following columns: "
    "city (string lowercase), "
    "price (float indicating the house price; ranging from 100000 the cheapest to 2000000 the most expensive), "
    "nr_bedrooms (integer; number of bedrooms), "
    "living_area_size (integer; number of square meters, ranging from 10 the smallest to 2000 the biggest), "
    "energy_label (float real number categorical value; ranging from 1 for the best to 13 for the worst). "
)


def get_pandas_query(user_input: str) -> str:
    out = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
    )
    answer = out["choices"][0]["message"]["content"]
    return answer


if __name__ == "__main__":
    user_input = "I want a house in Eindhoven with 3 bedrooms"
    response = get_pandas_query(user_input)
    print(response)
