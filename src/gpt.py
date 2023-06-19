import os

import openai

openai.api_key = os.getenv("OPENAI_API_KEY")

prompt = (
    "In the next messages, I will send you a natural language query, "
    "you have to transform it to a pandas query string. "
    "My dataframe has the following columns: "
    "city (string), "
    "price (real number indicating the house price; "
    "100000 is cheapest, 2000000 is most expensive), "
    "nr_bedrooms (number of bedrooms), "
    "living_area_size (number of square meters with "
    "10 being smallest, 2000 being biggest), "
    "energy_label (real number with 1 being best, 13 being worst). "
    "Please format your output in such a way that it can be directly "
    "passed to the pandas DataFrame query method."
)

def get_pandas_query(user_input: str):
    return openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
            {"role": "user", "content": user_input},
        ],
        temperature=0.9,
    )

if __name__ == "__main__":
    user_input = "I want a house in Eindhoven with 3 bedrooms"
    response = get_pandas_query(user_input)
    print(response["choices"][0]["message"]["content"])
