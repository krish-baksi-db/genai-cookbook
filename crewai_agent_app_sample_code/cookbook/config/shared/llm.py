from pydantic import BaseModel

class LLMParametersConfig(BaseModel):
    """
    Configuration for LLM response parameters.

    Attributes:
        temperature (float): Controls randomness in the response.
        max_tokens (int): Maximum number of tokens in the response.
        timeout (int): Maximum time in seconds for the LLM to generate a response.
    """

    # Parameters that control how the LLM responds.
    temperature: float = None
    max_tokens: int = None
    timeout: int = 300        # Longer timeout for complex tasks

class LLMConfig(BaseModel):
    """
    Configuration for the function-calling LLM.

    Attributes:
        llm_endpoint_name (str): Databricks Model Serving endpoint name.
            This is the generator LLM where your LLM queries are sent.
            Databricks foundational model endpoints can be found here:
            https://docs.databricks.com/en/machine-learning/foundation-models/index.html
        llm_model_name (str): Model name for the LLM. For e.g. "openai/gpt-35-turbo".
            P.S. If the model is an openai model served using Databricks Model Serving AI Gateway, then append the "openai/" prefix to the model name.
            For e.g. "gpt-35-turbo_db_gateway" will become "openai/gpt-35-turbo_db_gateway".
        llm_parameters (LLMParametersConfig): Parameters that control how the LLM responds.
    """

    # Databricks Model Serving endpoint name
    # This is the generator LLM where your LLM queries are sent.
    # Databricks foundational model endpoints can be found here: https://docs.databricks.com/en/machine-learning/foundation-models/index.html
    llm_endpoint_name: str

    # Name of the model -- For e.g. "openai/gpt-35-turbo". More details provided in docstrings.
    llm_model_name: str

    # Parameters that control how the LLM responds.
    llm_parameters: LLMParametersConfig
