import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

system_prompt = (
    "You are a natural language processing tool from natural language to a single Python string. "
    "The user will input natural language, you will output a pandas query string. "
    "The output must be a single string that can be directly passed to the pandas DataFrame query method (pd.DataFrame.query). "
    "Do not answer anything that does not belong to the output string, no explanations, no comments, no variable definition, no usage examples. "
    "The target pandas DataFrame has the following columns: "
    "city (string), "
    "price (real number indicating the house price; "
    "100000 is cheapest, 2000000 is most expensive), "
    "nr_bedrooms (number of bedrooms), "
    "living_area_size (number of square meters with "
    "10 being smallest, 2000 being biggest), "
    "energy_label (real number with 1 being best, 13 being worst). "
)


def get_pandas_query(user_input: str):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.1,
    )


if __name__ == "__main__":
    user_input = "I want a house in Eindhoven with 3 bedrooms"
    response = get_pandas_query(user_input)
    print(response["choices"][0]["message"]["content"])
