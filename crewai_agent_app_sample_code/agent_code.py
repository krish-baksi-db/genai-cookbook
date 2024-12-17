# Databricks notebook source


from cookbook.tools.vector_search import (
    VectorSearchRetrievalTool,
    VectorSearchSchema
)
import json
from cookbook.tools.uc_tool import UCTool
from crewai import Agent, Crew, Task, LLM
from cookbook.agents.function_calling_agent import CrewAIFunctionCallingAgent
import mlflow
from mlflow.models import set_model

# add a db endpoint t here


mlflow.litellm.autolog(log_traces= False, disable = True, silent = False)


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



########################
#### CrewAI Agent Configuration
########################


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

agent = CrewAIFunctionCallingAgent(llm_model=model, 
                             llm_api_base=api_base, 
                             llm_api_key=token, 
                             agent_role=role, 
                             agent_goal=goal, 
                             agent_backstory = backstory,
                             agent_tools=[retriever_tool, translate_sku_tool],
                             task_description=task_description,
                             crew_planning=True)

set_model(agent)
