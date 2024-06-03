# Tagging and Extraction Using Langchain

## Overview

This project demonstrates how to use Langchain for tagging and extracting information from text using OpenAI's models, also to show how you can implement and use tagging and extraction chains using Langchain's features.

## Requirements

- Python 3.8+
- OpenAI API Key
- Langchain
- Pydantic
- Python-dotenv

## Setup

1. Clone the repository:

    ```bash
    git clone https://github.com/da-ros/TaggingExtractionLangchain.git
    cd TaggingExtractionLangchain
    ```

2. Create and activate a virtual environment:

    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. Create a `.env` file in the root directory and add your OpenAI API key:

    ```plaintext
    OPENAI_API_KEY=your-openai-api-key
    ```

## Usage

### Tagging

The `Tagging` class is used to tag text with sentiment and language information.

```python
from langchain.prompts import ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.utils.openai_functions import convert_pydantic_to_openai_function
from pydantic import BaseModel, Field

class Tagging(BaseModel):
    sentiment: str = Field(description="sentiment of text, should be `pos`, `neg`, or `neutral`")
    language: str = Field(description="language of text (should be ISO 639-1 code)")

model = ChatOpenAI(temperature=0)
tagging_functions = [convert_pydantic_to_openai_function(Tagging)]
prompt = ChatPromptTemplate.from_messages([("system", "Think carefully, and then tag the text as instructed"), ("user", "{input}")])

tagging_chain = prompt | model.bind(functions=tagging_functions, function_call={"name": "Tagging"})
result = tagging_chain.invoke({"input": "I love langchain"})
print(result)
```

### Extraction

The `Extraction` class is used to extract specific information from text, such as people's names and ages.

```python
from pydantic import BaseModel, Field
from typing import List, Optional

class Person(BaseModel):
    name: str = Field(description="person's name")
    age: Optional[int] = Field(description="person's age")

class Information(BaseModel):
    people: List[Person] = Field(description="List of info about people")

extraction_functions = [convert_pydantic_to_openai_function(Information)]
extraction_model = model.bind(functions=extraction_functions, function_call={"name": "Information"})

prompt = ChatPromptTemplate.from_messages([("system", "Extract the relevant information, if not explicitly provided do not guess. Extract partial info"), ("human", "{input}")])
extraction_chain = prompt | extraction_model

result = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
print(result)
```

## Example

### Tagging Example

```python
result = tagging_chain.invoke({"input": "I love langchain"})
print(result)  # Output: {"sentiment": "pos", "language": "en"}
```

### Extraction Example

```python
result = extraction_chain.invoke({"input": "Joe is 30, his mom is Martha"})
print(result)  # Output: {"people": [{"name": "Joe", "age": 30}, {"name": "Martha", "age": None}]}
```

## Advanced Usage

The project also includes examples of more advanced use cases, such as tagging and extracting information from large bodies of text, using external document loaders, and chaining multiple models and functions.

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any features, bug fixes, or enhancements.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more information.
