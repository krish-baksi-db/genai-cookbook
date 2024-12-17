# Databricks notebook source
# MAGIC %md
# MAGIC ## 👉 START HERE: How to use this notebook
# MAGIC
# MAGIC # Step 3: Build, evaluate, & deploy your Agent
# MAGIC
# MAGIC Use this notebook to iterate on the code and configuration of your Agent.
# MAGIC
# MAGIC By the end of this notebook, you will have 1+ registered versions of your Agent, each coupled with a detailed quality evaluation.
# MAGIC
# MAGIC Optionally, you can deploy a version of your Agent that you can interact with in the [Mosiac AI Playground](https://docs.databricks.com/en/large-language-models/ai-playground.html) and let your business stakeholders who don't have Databricks accounts interact with it & provide feedback in the [Review App](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html#review-app-ui).
# MAGIC
# MAGIC
# MAGIC For each version of your agent, you will have an MLflow run inside your MLflow experiment that contains:
# MAGIC - Your Agent's code & config
# MAGIC - Evaluation metrics for cost, quality, and latency

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Important note:** Throughout this notebook, we indicate which cell's code you:
# MAGIC - ✅✏️ should customize - these cells contain code & config with business logic that you should edit to meet your requirements & tune quality.
# MAGIC - 🚫✏️ should not customize - these cells contain boilerplate code required to load/save/execute your Agent
# MAGIC
# MAGIC *Cells that don't require customization still need to be run!  You CAN change these cells, but if this is the first time using this notebook, we suggest not doing so.*

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Install Python libraries
# MAGIC
# MAGIC You do not need to modify this cell unless you need additional Python packages in your Agent.

# COMMAND ----------

# MAGIC %pip install -qqqq -U -r requirements.txt
# MAGIC # Restart to load the packages into the Python environment
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Connect to Databricks
# MAGIC
# MAGIC If running locally in an IDE using Databricks Connect, connect the Spark client & configure MLflow to use Databricks Managed MLflow.  If this running in a Databricks Notebook, these values are already set.

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 3

# COMMAND ----------

# add token here
# token = ####

# COMMAND ----------

from mlflow.utils import databricks_utils as du
import os

if not du.is_in_databricks_notebook():
    from databricks.connect import DatabricksSession
    import os

    spark = DatabricksSession.builder.getOrCreate()
    os.environ["MLFLOW_TRACKING_URI"] = "databricks"

# COMMAND ----------



# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Load the Agent's UC storage locations; set up MLflow experiment
# MAGIC
# MAGIC This notebook uses the UC model, MLflow Experiment, and Evaluation Set that you specified in the [Agent setup](02_agent_setup.ipynb) notebook.

# COMMAND ----------

from cookbook.config.shared.agent_storage_location import AgentStorageConfig
from cookbook.databricks_utils import get_mlflow_experiment_url
from cookbook.config import load_serializable_config_from_yaml_file
import mlflow 

# Load the Agent's storage locations
agent_storage_config: AgentStorageConfig= load_serializable_config_from_yaml_file("./configs/agent_storage_config.yaml")

# Show the Agent's storage locations
agent_storage_config.pretty_print()

# set the MLflow experiment
experiment_info = mlflow.set_experiment(agent_storage_config.mlflow_experiment_name)
# If running in a local IDE, set the MLflow experiment name as an environment variable
os.environ["MLFLOW_EXPERIMENT_NAME"] = agent_storage_config.mlflow_experiment_name

print(f"View the MLflow Experiment `{agent_storage_config.mlflow_experiment_name}` at {get_mlflow_experiment_url(experiment_info.experiment_id)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🚫✏️ Helper method to log the Agent's code & config to MLflow
# MAGIC
# MAGIC Before we start, let's define a helper method to log the Agent's code & config to MLflow.  We will use this to log the agent's code & config to MLflow & the Unity Catalog.  It is used in evaluation & for deploying to Agent Evaluation's [Review App](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html#review-app-ui) (a chat UI for your stakeholders to test this agent) and later, deplying the Agent to production.

# COMMAND ----------

from cookbook.tools.vector_search import (
    VectorSearchRetrievalTool,
    VectorSearchSchema
)
import json
from cookbook.tools.uc_tool import UCTool
from cookbook.config.data_pipeline import (
    DataPipelineConfig,
)
from cookbook.config import load_serializable_config_from_yaml_file


########################
# #### 🚫✏️ Load the Vector Index Unity Cataloglocation from the data pipeline configuration
# Usage:
# - If you used `01_data_pipeline` to create your Vector Index, run this cell.
# - If your Vector Index was created elsewhere, comment out this logic and set the UC location in the Retriever config.
########################

data_pipeline_config: DataPipelineConfig = load_serializable_config_from_yaml_file(
    "./configs/data_pipeline_config.yaml"
)

########################
# #### ✅✏️ Retriever tool that connects to the Vector Search index
########################

retriever_tool = VectorSearchRetrievalTool(
    name="search_product_docs",
    description="Use this tool to search for product documentation about Databricks and Spark based products.",
    vector_search_index = "krish_db.cookbook_local_test.test_product_docs_docs_chunked_index__v3",
    vector_search_schema=VectorSearchSchema(
        # These columns are the default values used in the `01_data_pipeline` notebook
        # If you used a different column names in that notebook OR you are using a pre-built vector index, update the column names here.
        chunk_text="content_chunked",  # Contains the text of each document chunk
        document_uri="doc_uri",  # The document URI of the chunk e.g., "/Volumes/catalog/schema/volume/file.pdf" - displayed as the document ID in the Review App
        # additional_metadata_columns=[],  # Additional columns to return from the vector database and present to the LLM
    )
    # Optional parameters, see VectorSearchRetrieverTool.__doc__ for details.  The default values are shown below.
    # doc_similarity_threshold=0.0,
    # vector_search_parameters=VectorSearchParameters(
    #     num_results=5,
    #     query_type="ann"
    # ),
    # Adding columns here will allow the Agent's LLM to dynamically apply filters based on the user's query.
    # filterable_columns=[]
)

########################
# #### ✅✏️ Add Unity Catalog tools to the Agent
########################

translate_sku_tool = UCTool(uc_function_name="krish_db.cookbook_local_test.sku_sample_translator")



# COMMAND ----------

type(translate_sku_tool)

# COMMAND ----------

########################
#### CrewAI Agent Configuration
########################

from crewai import Agent, Crew, Task, LLM

## Role
role = """You are a helpful assistant that answers questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request."""

## Goal
goal = """Your goal is to provide accurate, relevant, and helpful response based solely on the outputs from these tools. You are concise and direct in your responses.

Details of the goal:

1. **Understand the Query**: Think step by step to analyze the user's question and determine the core need or problem. 

2. **Understand the past conversation***: You will also be provided the past conversation. You can think step by step to analyse the past conversation to understand the context of the question, and if necessary rephrase or complete the question.

2. **Assess available tools**: Think step by step to consider each available tool and understand their capabilities in the context of the user's query. Do not try to answer the questions on your own. Use the tools to get more content for creating a detailed answer.

3. **Select the appropriate tool(s) OR ask follow up questions**: Based on your understanding of the query, the past conversation and the tool descriptions, decide which tool(s) should be used to generate a response. If you do not have enough information to use the available tools to answer the question, ask the user follow up questions to refine their request.  If you do not have a relevant tool for a question or the outputs of the tools are not helpful, respond with: "I'm sorry, I can't help you with that."""


## Backstory
backstory = """" With over 10 years experience as a customer service agent, you excel at answering customer's questions on a variety of topics. You have learned that the best way to answer a customer's question is to exploit different specialized tools. You use the tools to execute specialized tasks whenever needed and use the task results to produce a coherent and complete response to the customer's questions"""


## Task
task_description = """
You will be provided a question and previous conversation with a user. Your task is to answer the user's question. If the question is not complete, you can rely on the past conversation to provide the missing pieces. 

Here is the past conversation: {conversation}

Here is the question: {question}

"""


## LLM Class Initialization (needed to work with custom models)

model = "openai/agents-demo-gpt4o" # this may look a little weird, but this is only way to work with custom OpenAI endpoints

api_base = "https://e2-demo-field-eng.cloud.databricks.com/serving-endpoints"

# COMMAND ----------

from cookbook.agents.function_calling_agent import CrewAIFunctionCallingAgent

agent = CrewAIFunctionCallingAgent(llm_model=model, 
                             llm_api_base=api_base, 
                             llm_api_key=token, 
                             agent_role=role, 
                             agent_goal=goal, 
                             agent_backstory = backstory,
                             agent_tools=[retriever_tool, translate_sku_tool],
                             task_description=task_description,
                             crew_planning=True)

agent_input_example = {"messages": [{"role": "user", "content": "What is Spark?"}]}

# COMMAND ----------

agent.predict(model_input = agent_input_example)

# COMMAND ----------

?mlflow.pyfunc.log_model

# COMMAND ----------

from typing import List
import mlflow

class MyModel(mlflow.pyfunc.PythonModel):
    def predict(self, context, model_input: List[str], params=None) -> List[str]:
        return [i.upper() for i in model_input]

mlflow.end_run()
with mlflow.start_run():
    model_info = mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=agent,
    )

# COMMAND ----------

from mlflow.models.rag_signatures import StringResponse

with open("requirements.txt", "r") as file:
        pip_requirements = [line.strip() for line in file.readlines()] + ["pyspark"] # manually add pyspark


logged_agent_info = mlflow.pyfunc.log_model(
  artifact_path="agent",
  python_model=agent,
  # input_example=agent_input_example,
  # code_paths=[os.path.join(os.getcwd(), "cookbook")],
  # pip_requirements=pip_requirements
)

# COMMAND ----------

import mlflow
from mlflow.types.llm import CHAT_MODEL_INPUT_SCHEMA
from mlflow.models.rag_signatures import StringResponse
from cookbook.agents.utils.signatures import STRING_RESPONSE_WITH_MESSAGES
from mlflow.models.signature import ModelSignature

# This helper will log the Agent's code & config to an MLflow run and return the logged model's URI
# If run from inside a mlfow.start_run() block, it will log to that run, otherwise it will log to a new run.
# This logged Agent is ready for deployment, so if you are happy with your evaluation, it is ready to deploy!
def log_function_calling_agent_to_mlflow(agent):
    # Get the agent's code path from the imported Agent class
    agent_code_path = f"{os.getcwd()}/{FunctionCallingAgent.__module__.replace('.', '/')}.py"

    # Get the pip requirements from the requirements.txt file
    with open("requirements.txt", "r") as file:
        pip_requirements = [line.strip() for line in file.readlines()] + ["pyspark"] # manually add pyspark

    logged_agent_info = mlflow.pyfunc.log_model(
        artifact_path="agent",
        python_model=agent_code_path,
        input_example=agent_config.input_example,
        model_config=agent_config.model_dump(),
        resources=agent_config.get_resource_dependencies(), # This allows the agents.deploy() command to securely provision credentials for the Agent's databricks resources e.g., vector index, model serving endpoints, etc
        signature=ModelSignature(
            inputs=CHAT_MODEL_INPUT_SCHEMA,
            # outputs=STRING_RESPONSE_WITH_MESSAGES #TODO: replace with MLflow signature
            outputs=StringResponse()
        ),
        code_paths=[os.path.join(os.getcwd(), "cookbook")],
        pip_requirements=pip_requirements,
    )

    return logged_agent_info

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## 1️⃣ Iterate on the Agent's code & config to improve quality
# MAGIC
# MAGIC The below cells are used to execute your inner dev loop to improve the Agent's quality.
# MAGIC
# MAGIC We suggest the following process:
# MAGIC 1. Vibe check the Agent for 5 - 10 queries to verify it works
# MAGIC 2. Make any necessary changes to the code/config
# MAGIC 3. Use Agent Evaluation to evaluate the Agent using your evaluation set, which will provide a quality assessment & identify the root causes of any quality issues
# MAGIC 4. Based on that evaluation, make & test changes to the code/config to improve quality
# MAGIC 5. 🔁 Repeat steps 3 and 4 until you are satisified with the Agent's quality
# MAGIC 6. Deploy the Agent to Agent Evaluation's [Review App](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html#review-app-ui) for pre-production testing
# MAGIC 7. Use the following notebooks to review that feedback (optionally adding new records to your evaluation set) & identify any further quality issues
# MAGIC 8. 🔁 Repeat steps 3 and 4 to fix any issues identified in step 7
# MAGIC 9. Deploy the Agent to a production-ready REST API endpoint (using the same cells in this notebook as step 6)
# MAGIC

# COMMAND ----------

# Import Cookbook Agent configurations, which are Pydantic models
from cookbook.config import serializable_config_to_yaml_file
from cookbook.config.agents.function_calling_agent import (
    FunctionCallingAgentConfig,
)
from cookbook.config.data_pipeline import (
    DataPipelineConfig,
)
from cookbook.config.shared.llm import LLMConfig, LLMParametersConfig
from cookbook.config import load_serializable_config_from_yaml_file
from cookbook.tools.vector_search import (
    VectorSearchRetrieverTool,
    VectorSearchSchema,
)
import json
from cookbook.tools.uc_tool import UCTool


########################
# #### 🚫✏️ Load the Vector Index Unity Cataloglocation from the data pipeline configuration
# Usage:
# - If you used `01_data_pipeline` to create your Vector Index, run this cell.
# - If your Vector Index was created elsewhere, comment out this logic and set the UC location in the Retriever config.
########################

data_pipeline_config: DataPipelineConfig = load_serializable_config_from_yaml_file(
    "./configs/data_pipeline_config.yaml"
)

########################
# #### ✅✏️ Retriever tool that connects to the Vector Search index
########################

retriever_tool = VectorSearchRetrieverTool(
    name="search_product_docs",
    description="Use this tool to search for product documentation.",
    vector_search_index="ep.cookbook_local_test.product_docs_docs_chunked_index__v1",
    vector_search_schema=VectorSearchSchema(
        # These columns are the default values used in the `01_data_pipeline` notebook
        # If you used a different column names in that notebook OR you are using a pre-built vector index, update the column names here.
        chunk_text="content_chunked",  # Contains the text of each document chunk
        document_uri="doc_uri",  # The document URI of the chunk e.g., "/Volumes/catalog/schema/volume/file.pdf" - displayed as the document ID in the Review App
        # additional_metadata_columns=[],  # Additional columns to return from the vector database and present to the LLM
    ),
    # Optional parameters, see VectorSearchRetrieverTool.__doc__ for details.  The default values are shown below.
    # doc_similarity_threshold=0.0,
    # vector_search_parameters=VectorSearchParameters(
    #     num_results=5,
    #     query_type="ann"
    # ),
    # Adding columns here will allow the Agent's LLM to dynamically apply filters based on the user's query.
    # filterable_columns=[]
)

########################
# #### ✅✏️ Add Unity Catalog tools to the Agent
########################

translate_sku_tool = UCTool(uc_function_name="ep.cookbook_local_test.sku_sample_translator")


########################
# #### ✅✏️ Add a local Python function as a tool in the Agent
########################

from cookbook.tools.local_function import LocalFunctionTool
from tools.sample_tool import sku_sample_translator

# translate_sku_tool = LocalFunctionTool(func=translate_sku, description="Translates a pre-2024 SKU formatted as 'OLD-XXX-YYYY' to the new SKU format 'NEW-YYYY-XXX'.")

########################
#### ✅✏️ Agent's LLM configuration
########################

system_prompt = """
## Role
You are a helpful assistant that answers questions using a set of tools. If needed, you ask the user follow-up questions to clarify their request.

## Objective
Your goal is to provide accurate, relevant, and helpful response based solely on the outputs from these tools. You are concise and direct in your responses.

## Instructions
1. **Understand the Query**: Think step by step to analyze the user's question and determine the core need or problem. 

2. **Assess available tools**: Think step by step to consider each available tool and understand their capabilities in the context of the user's query.

3. **Select the appropriate tool(s) OR ask follow up questions**: Based on your understanding of the query and the tool descriptions, decide which tool(s) should be used to generate a response. If you do not have enough information to use the available tools to answer the question, ask the user follow up questions to refine their request.  If you do not have a relevant tool for a question or the outputs of the tools are not helpful, respond with: "I'm sorry, I can't help you with that."
""".strip()

fc_agent_config = FunctionCallingAgentConfig(
    llm_config=LLMConfig(
        llm_endpoint_name="ep-gpt4o-new",  # Model serving endpoint w/ a Chat Completions API
        llm_system_prompt_template=system_prompt,  # System prompt template
        llm_parameters=LLMParametersConfig(
            temperature=0.01, max_tokens=1500
        ),  # LLM parameters
    ),
    # Add one or more tools that comply with the CookbookTool interface
    tools=[retriever_tool, translate_sku_tool],
    # tools=[retriever_tool],
)

# Print the configuration as a JSON string to see it all together
# print(json.dumps(fc_agent_config.model_dump(), indent=4))

########################
##### Dump the configuration to a YAML
# Optional step, this allows the Agent's code file to be run by itself (e.g., outside of this notebook) using the above configuration.
########################
# Import the default YAML config file name from the Agent's code file
from cookbook.agents.function_calling_agent import FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME

# Dump the configuration to a YAML file
serializable_config_to_yaml_file(fc_agent_config, "./configs/"+FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME)

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ Optionally, adjust the Agent's code
# MAGIC
# MAGIC Here, we import the Agent's code so we can run the Agent locally within the notebook.  To modify the code, open the Agent's code file in a separate window, enable reload, make your changes, and re-run this cell.
# MAGIC
# MAGIC **Typically, when building the first version of your agent, we suggest first trying to tune the configuration (prompts, etc) to improve quality.  If you need more control to fix quality issues, you can then modify the Agent's code.**

# COMMAND ----------

from cookbook.agents.function_calling_agent import FunctionCallingAgent
import inspect

# Print the Agent code for inspection
print(inspect.getsource(FunctionCallingAgent))

# COMMAND ----------

# MAGIC %load_ext autoreload
# MAGIC %autoreload 3

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ 🅰 Vibe check the Agent for a single query
# MAGIC
# MAGIC Running this cell will produce an MLflow Trace that you can use to see the Agent's outputs and understand the steps it took to produce that output.
# MAGIC
# MAGIC If you are running in a local IDE, browse to the MLflow Experiment page to view the Trace (link to the Experiment UI is at the top of this notebook).  If running in a Databricks Notebook, your trace will appear inline below.

# COMMAND ----------

from cookbook.databricks_utils import get_mlflow_experiment_traces_url
from cookbook.agents.function_calling_agent import FunctionCallingAgent

# Load the Agent's code with the above configuration
agent = FunctionCallingAgent(agent_config=fc_agent_config)

# Vibe check the Agent for a single query
output = agent.predict(model_input={"messages": [{"role": "user", "content": "How does the blender work?"}]})
# output = agent.predict(model_input={"messages": [{"role": "user", "content": "Translate the sku `OLD-abs-1234` to the new format"}]})

print(f"View the MLflow Traces at {get_mlflow_experiment_traces_url(experiment_info.experiment_id)}")
print(f"Agent's final response:\n----\n{output['content']}\n----")
print()
# print(f"Agent's full message history (useful for debugging):\n----\n{json.dumps(output['messages'], indent=2)}\n----")


# COMMAND ----------

# MAGIC %md
# MAGIC Now, let's test a multi-turn conversation with the Agent.

# COMMAND ----------

second_turn = {'messages': output['messages'] + [{"role": "user", "content": "How do I turn it on?"}]}

# Run the Agent again with the same input to continue the conversation
second_turn_output = agent.predict(model_input=second_turn)

print(f"View the MLflow Traces at {get_mlflow_experiment_traces_url(experiment_info.experiment_id)}")
print(f"Agent's final response:\n----\n{second_turn_output['content']}\n----")
print()
print(f"Agent's full message history (useful for debugging):\n----\n{json.dumps(second_turn_output['messages'], indent=2)}\n----")

# COMMAND ----------

# MAGIC %md
# MAGIC #### ✅✏️ 🅱 Evaluate the Agent using your evaluation set
# MAGIC
# MAGIC Note: If you do not have an evaluation set, you can create a synthetic evaluation set by using the 03_synthetic_evaluation notebook.

# COMMAND ----------

evaluation_set = spark.table(agent_storage_config.evaluation_set_uc_table)

with mlflow.start_run():
    logged_agent_info = log_function_calling_agent_to_mlflow(fc_agent_config)

    # Run the agent for these queries, using Agent evaluation to parallelize the calls
    eval_results = mlflow.evaluate(
        model=logged_agent_info.model_uri,  # use the MLflow logged Agent
        data=evaluation_set,  # Evaluate the Agent for every row of the evaluation set
        model_type="databricks-agent",  # use Agent Evaluation
    )

    # Show all outputs.  Click on a row in this table to display the MLflow Trace.
    display(eval_results.tables["eval_results"])

    # Click 'View Evaluation Results' to see the Agent's inputs/outputs + quality evaluation displayed in a UI

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2️⃣ Deploy a version of your Agent - either to the Review App or Production
# MAGIC
# MAGIC Once you have a version of your Agent that has sufficient quality, you will register the Agent's model from the MLflow Experiment into the Unity Catalog & use Agent Framework's `agents.deploy(...)` command to deploy it.  Note these steps are the same for deploying to pre-production (e.g., the [Review App](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html#review-app-ui) or production.
# MAGIC
# MAGIC By the end of this step, you will have deployed a version of your Agent that you can interact with and share with your business stakeholders for feedback, even if they don't have access to your Databricks workspace:
# MAGIC
# MAGIC 1. A production-ready scalable REST API deployed as a Model Serving endpoint that logged every request/request/MLflow Trace to a Delta Table.
# MAGIC     - REST API for querying the Agent
# MAGIC     - REST API for sending user feedback from your UI to the Agent
# MAGIC 2. Agent Evaluation's [Review App](https://docs.databricks.com/en/generative-ai/agent-evaluation/human-evaluation.html#review-app-ui) connected to these endpoints.
# MAGIC 3. [Mosiac AI Playground](https://docs.databricks.com/en/large-language-models/ai-playground.html) connected to these endpoints.

# COMMAND ----------

# MAGIC %md
# MAGIC Option 1: Deploy the last agent you logged above

# COMMAND ----------

from databricks import agents

# Use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

# Register the Agent's model to the Unity Catalog
uc_registered_model_info = mlflow.register_model(
    model_uri=logged_agent_info.model_uri, name=agent_storage_config.uc_model_name
)

# Deploy the model to the review app and a model serving endpoint
agents.deploy(agent_storage_config.uc_model_name, uc_registered_model_info.version)

# COMMAND ----------

# MAGIC %md
# MAGIC Option 2: Log the latest copy of the Agent's code/config and deploy it

# COMMAND ----------

from databricks import agents

# Use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    logged_agent_info = log_function_calling_agent_to_mlflow(fc_agent_config)

    # Register the Agent's model to the Unity Catalog
    uc_registered_model_info = mlflow.register_model(
        model_uri=logged_agent_info.model_uri, name=agent_storage_config.uc_model_name
    )

# Deploy the model to the review app and a model serving endpoint
# agents.deploy(agent_storage_config.uc_model_name, uc_registered_model_info.version)

# COMMAND ----------

# MAGIC %md
# MAGIC Load the logged model to test it locally

# COMMAND ----------

import mlflow

loaded_model = mlflow.pyfunc.load_model(logged_agent_info.model_uri)

loaded_model.predict({"messages": [{"role": "user", "content": "A test question?"}]})

# COMMAND ----------

from databricks import agents

# Use Unity Catalog as the model registry
mlflow.set_registry_uri("databricks-uc")

with mlflow.start_run():
    logged_agent_info = log_agent_to_mlflow(fc_agent_config)