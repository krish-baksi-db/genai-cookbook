# In this file, we construct a function-calling Agent with a Retriever tool using MLflow + the OpenAI SDK connected to Databricks Model Serving. This Agent is encapsulated in a MLflow PyFunc class called `FunctionCallingAgent()`.

# Add the parent directory to the path so we can import the `cookbook` modules
# import sys
# sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


import json
from typing import Any, Dict, List, Optional, Union
import mlflow
import pandas as pd
from mlflow.models import set_model
from mlflow.models.rag_signatures import StringResponse, ChatCompletionRequest
from databricks.sdk import WorkspaceClient
from cookbook.agents.utils.execute_function import execute_function

from cookbook.agents.utils.chat import (
    get_messages_array,
    extract_user_query_string,
    extract_chat_history,
)

from cookbook.agents.utils.execute_function import execute_function
from cookbook.agents.utils.load_config import load_config
import logging
import numpy as np
from crewai import Agent, Crew, Task, LLM

FC_AGENT_DEFAULT_YAML_CONFIG_FILE_NAME = "function_calling_agent_config.yaml"




import mlflow


class CrewAIFunctionCallingAgent(mlflow.pyfunc.PythonModel):
    """
    A class to represent a function calling agent using CrewAI framework.
    """

    def __init__(self, 
                 llm_model: str, 
                 llm_api_base: str, 
                 llm_api_key: str,
                 agent_role: str,
                 agent_goal: str,
                 agent_backstory: str,
                 agent_tools: list,
                 task_description: str,
                 crew_planning: bool):
        """
        Initializes the FunctionCallingAgent with the provided parameters.

        Parameters:
        ----------
        llm_model : str
            The model name of the large language model.
        llm_api_base : str
            The base URL for the LLM API.
        llm_api_key : str
            The API key for accessing the LLM.
        agent_role : str
            The role of the agent.
        agent_goal : str
            The goal of the agent.
        agent_backstory : str
            The backstory of the agent.
        agent_tools : list
            The tools available to the agent.
        task_description: str
            The task to be executed by the agent.
        crew_planning : str
            The planning strategy for the crew.

        Returns:
        -------
        None
        """
        super().__init__()
        self.llm = LLM(model = llm_model,
                       api_key= llm_api_key,
                       api_base= llm_api_base)
        
        self.agent = Agent(llm = self.llm,
                           role = agent_role,
                           goal = agent_goal,
                           backstory=agent_backstory,
                           tools=agent_tools,
                           step_callback = self.agent_step_callback, 
                        #    callbacks = [self.finegrained_callbacks]
                           )
        
        self.task = Task(description=task_description, 
                         expected_output="string", 
                         agent=self.agent,
                         callback = self.task_callback)
        
        self.crew = Crew(agents=[self.agent], 
                         tasks=[self.task],
                         planning = crew_planning,
                         planning_llm = self.llm)
    
    @mlflow.trace(span_type="CHAT_MODEL", name = "Chatbot")
    def predict(
        self,
        context: Any = None,
        model_input: Union[ChatCompletionRequest, Dict, pd.DataFrame] = None,
        params: Any = None,
    ) -> str:
        
        print("Inputs:\n")
        print(model_input)
        print("\n\n")
        ##############################################################################
        # Extract `messages` key from the `model_input`
        messages = get_messages_array(model_input)

        ##############################################################################
        # Parse `messages` array into the user's query & the chat history
        with mlflow.start_span(name="parse_input", span_type="PARSER") as span:
            span.set_inputs({"messages": messages})
            # in a multi-agent setting, the last message can be from another assistant, not the user
            last_message = extract_user_query_string(messages)
            last_message_role = messages[-1]["role"]
            # Save the history inside the Agent's internal state
            chat_history = extract_chat_history(messages)
            span.set_outputs(
                {
                    "last_message": last_message,
                    "chat_history": chat_history,
                    "last_message_role": last_message_role,
                }
            )

        # print("##### TYPE ####")
        # print(type(self.chat_history))
        if type(chat_history) == np.ndarray:
            chat_history = chat_history.tolist()
        ##############################################################################
        response = self.kickoff(last_message, chat_history)
        return response
    
    @mlflow.trace(span_type="AGENT", name = "Agent")
    def kickoff(self, question, chat_history):
        print(question)
        if type(chat_history) == np.array:
            chat_history = chat_history.tolist()
        response = self.crew.kickoff(inputs = {"question": question, "conversation": json.dumps(chat_history)})
        print(response.raw)
        chat_history.append({"role": "user", "content": question})
        chat_history.append({"role": "assistant", "content": response.raw})
        print(chat_history)
        print("######\n\n######")
        return {"user": "assistant", "content": response.raw, "messages": chat_history}


    def agent_step_callback(self, output):
        self.agent_step_output = output
        with mlflow.start_span(name="agent_step", span_type="UNKNOWN") as span:
            if hasattr(output, "text"):
                span.set_inputs({"step description": output.text})
            else:
                span.set_inputs({"step description": output.text})
            
            if hasattr(output, "output"):
                span.set_outputs({"step output": output.output})
            else:
                span.set_outputs({"step output": ""})

    def task_callback(self, output):
        with mlflow.start_span(name="task", span_type="UNKNOWN") as span:
            span.set_inputs({"task description": output.description})
            span.set_outputs({"task output": output.raw})


