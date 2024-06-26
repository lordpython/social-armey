import streamlit as st
import sys
from crewai import Agent, Task, Crew, Process
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from langchain_community.chat_models import ChatOpenAI
from stream import StreamToStreamlit
from textwrap import dedent
import os
from dotenv import load_dotenv
import json
import logging
import pandas as pd
from datetime import datetime

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(filename='app.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

def initialize_llm(api, api_key, model, temp):
    """
    Initialize and return a language model based on the selected API.
    
    Args:
    api (str): The selected API ('Groq', 'OpenAI', or 'Anthropic')
    api_key (str): The API key for the selected service
    model (str): The name of the model to use
    temp (float): The temperature setting for the model
    
    Returns:
    object: An instance of the selected language model
    """
    try:
        if api == 'Groq':
            return ChatGroq(
                temperature=temp,
                model_name=model,
                groq_api_key=api_key
            )
        elif api == 'OpenAI':
            return ChatOpenAI(
                temperature=temp,
                openai_api_key=api_key,
                model_name=model
            )
        elif api == 'Anthropic':
            return ChatAnthropic(
                temperature=temp,
                anthropic_api_key=api_key,
                model_name=model
            )
    except Exception as e:
        st.error(f"Error initializing LLM: {str(e)}")
        return None

def create_agent(role, backstory, goal, llm):
    """
    Create and return an Agent instance.
    
    Args:
    role (str): The role of the agent
    backstory (str): The backstory of the agent
    goal (str): The goal of the agent
    llm (object): The language model to use for the agent
    
    Returns:
    Agent: An instance of the Agent class
    """
    return Agent(
        role=role,
        backstory=backstory,
        goal=goal,
        allow_delegation=True,
        verbose=True,
        max_iter=3,
        max_rpm=3,
        llm=llm
    )

def create_task(description, expected_output, agent, context=None):
    """
    Create and return a Task instance.
    
    Args:
    description (str): The description of the task
    expected_output (str): The expected output of the task
    agent (Agent): The agent assigned to the task
    context (list, optional): A list of related tasks for context
    
    Returns:
    Task: An instance of the Task class
    """
    task = Task(
        description=description,
        expected_output=expected_output,
        agent=agent
    )
    if context:
        task.context = context
    return task

def save_config(config):
    """
    Save the current configuration to a JSON file.
    
    Args:
    config (dict): The configuration to save
    """
    with open('config.json', 'w') as f:
        json.dump(config, f)

def load_config():
    """
    Load a configuration from a JSON file.
    
    Returns:
    dict: The loaded configuration, or an empty dict if the file doesn't exist
    """
    try:
        with open('config.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return {}

def configuration_tab():
    """
    Render the Configuration tab in the Streamlit app.
    
    Returns:
    tuple: Configuration parameters
    """
    st.header("Configuration")
    
    config = load_config()
    
    api = st.selectbox(
        'Choose an API',
        ['Groq', 'OpenAI', 'Anthropic'],
        index=['Groq', 'OpenAI', 'Anthropic'].index(config.get('api', 'Groq'))
    )

    api_key = st.text_input('Enter API Key', value=config.get('api_key', os.getenv(f"{api.upper()}_API_KEY", "")), type="password")

    temp = st.slider("Model Temperature", min_value=0.0, max_value=1.0, value=config.get('temp', 0.7), step=0.1)

    model_options = {
        'Groq': ['llama3-70b-8192', 'mixtral-8x7b-32768', 'gemma-7b-it'],
        'OpenAI': ['gpt-4-turbo', 'gpt-4-1106-preview', 'gpt-3.5-turbo-0125', 'gpt-4o'],
        'Anthropic': ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-haiku-20240307']
    }

    model = st.selectbox('Choose a model', model_options[api], index=model_options[api].index(config.get('model', model_options[api][0])))

    with st.expander("Agent Definitions", expanded=False):
        agent_1_role = st.text_input("Agent 1 Role", value=config.get('agent_1_role', "Mr. White"))
        agent_1_backstory = st.text_area("Agent 1 Backstory", value=config.get('agent_1_backstory', "Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics."))
        agent_1_goal = st.text_area("Agent 1 Goal", value=config.get('agent_1_goal', "The individual objective that the agent aims to achieve. It guides the agent's decision-making process."))
        
        agent_2_role = st.text_input("Agent 2 Role", value=config.get('agent_2_role', "Mr. Orange"))
        agent_2_backstory = st.text_area("Agent 2 Backstory", value=config.get('agent_2_backstory', "Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics."))
        agent_2_goal = st.text_area("Agent 2 Goal", value=config.get('agent_2_goal', "The individual objective that the agent aims to achieve. It guides the agent's decision-making process."))
        
        agent_3_role = st.text_input("Agent 3 Role", value=config.get('agent_3_role', "Mr. Pink"))
        agent_3_backstory = st.text_area("Agent 3 Backstory", value=config.get('agent_3_backstory', "Provides context to the agent's role and goal, enriching the interaction and collaboration dynamics."))
        agent_3_goal = st.text_area("Agent 3 Goal", value=config.get('agent_3_goal', "The individual objective that the agent aims to achieve. It guides the agent's decision-making process."))

    if st.button("Save Configuration"):
        config = {
            'api': api,
            'api_key': api_key,
            'temp': temp,
            'model': model,
            'agent_1_role': agent_1_role,
            'agent_1_backstory': agent_1_backstory,
            'agent_1_goal': agent_1_goal,
            'agent_2_role': agent_2_role,
            'agent_2_backstory': agent_2_backstory,
            'agent_2_goal': agent_2_goal,
            'agent_3_role': agent_3_role,
            'agent_3_backstory': agent_3_backstory,
            'agent_3_goal': agent_3_goal,
        }
        save_config(config)
        st.success("Configuration saved successfully!")

    return api, api_key, temp, model, agent_1_role, agent_1_backstory, agent_1_goal, agent_2_role, agent_2_backstory, agent_2_goal, agent_3_role, agent_3_backstory, agent_3_goal

def execution_tab(api, api_key, temp, model, agent_1_role, agent_1_backstory, agent_1_goal, agent_2_role, agent_2_backstory, agent_2_goal, agent_3_role, agent_3_backstory, agent_3_goal):
    st.header("LinkedIn Post Generator")
    
    recent_project = st.text_input("Recent Project:", help="Enter a brief description of your recent project or achievement")
    target_audience = st.text_input("Target Audience:", help="Describe your target audience on LinkedIn")
    key_skills = st.text_input("Key Skills:", help="List your key skills related to Python and AI agents")

    if st.button("Generate LinkedIn Post", disabled=not (recent_project and target_audience and key_skills and api_key)):
        with st.spinner("Generating your LinkedIn post..."):
            try:
                llm = initialize_llm(api, api_key, model, temp)
                if not llm:
                    st.warning("Failed to initialize LLM. Please check your API key and selected model.")
                    return

                content_strategist = create_agent("Content Strategist", "Expert in creating engaging content strategies for social media", "Develop a content strategy for a LinkedIn post", llm)
                python_ai_expert = create_agent("Python/AI Expert", "Experienced Python programmer specializing in AI agents", "Provide technical insights and validate content accuracy", llm)
                linkedin_optimizer = create_agent("LinkedIn Post Optimizer", "Specialist in optimizing content for LinkedIn's algorithm and user engagement", "Refine and optimize the post for maximum impact on LinkedIn", llm)

                task_1 = create_task(
                    description=f"Develop a content strategy for a LinkedIn post about a freelance Python programmer working on AI agents. Recent project: {recent_project}. Target audience: {target_audience}. Key skills: {key_skills}.",
                    expected_output="A content strategy outlining key points to cover in the LinkedIn post.",
                    agent=content_strategist
                )

                task_2 = create_task(
                    description="Based on the content strategy, draft a LinkedIn post that showcases expertise in Python programming and AI agents. Include technical insights and highlight the recent project.",
                    expected_output="A draft LinkedIn post with technical details and project highlights.",
                    agent=python_ai_expert,
                    context=[task_1]
                )

                task_3 = create_task(
                    description="Optimize the draft LinkedIn post for maximum engagement. Ensure it follows LinkedIn best practices, includes relevant hashtags, and has a compelling call-to-action.",
                    expected_output="A final, optimized LinkedIn post ready for publishing.",
                    agent=linkedin_optimizer,
                    context=[task_1, task_2]
                )

                crew = Crew(
                    agents=[content_strategist, python_ai_expert, linkedin_optimizer],
                    tasks=[task_1, task_2, task_3],
                    verbose=2,
                    process=Process.sequential,
                    manager_llm=llm
                )

                output_expander = st.expander("Generated LinkedIn Post", expanded=True)
                original_stdout = sys.stdout
                sys.stdout = StreamToStreamlit(output_expander)

                result = ""
                result_container = output_expander.empty()
                for delta in crew.kickoff():
                    result += delta
                    result_container.markdown(result)
                
                # Save results
                results_df = pd.DataFrame({
                    'Timestamp': [datetime.now()],
                    'API': [api],
                    'Model': [model],
                    'Temperature': [temp],
                    'Recent Project': [recent_project],
                    'Target Audience': [target_audience],
                    'Key Skills': [key_skills],
                    'Generated Post': [result]
                })
                results_df.to_csv('linkedin_posts.csv', mode='a', header=not os.path.exists('linkedin_posts.csv'), index=False)
                
                logging.info("LinkedIn post generated successfully")
                st.success("LinkedIn post generated successfully!")

            except Exception as e:
                logging.error(f"An error occurred during execution: {str(e)}")
                st.error(f"An error occurred during execution: {str(e)}")
            finally:
                sys.stdout = original_stdout

    """
    Render the Execution tab in the Streamlit app and run the Agents process.
    
    Args:
    api (str): The selected API
    api_key (str): The API key
    temp (float): The temperature setting
    model (str): The selected model
    agent_*_role (str): The role of each agent
    agent_*_backstory (str): The backstory of each agent
    agent_*_goal (str): The goal of each agent
    """
    st.header("Execution")
    
    var_1 = st.text_input("Variable 1:", help="Enter the first variable for the task")
    var_2 = st.text_input("Variable 2:", help="Enter the second variable for the task")
    var_3 = st.text_input("Variable 3:", help="Enter the third variable for the task")

    if st.button("Start", disabled=not (var_1 and var_2 and var_3 and api_key)):
        with st.spinner("Generating..."):
            try:
                llm = initialize_llm(api, api_key, model, temp)
                if not llm:
                    st.warning("Failed to initialize LLM. Please check your API key and selected model.")
                    return

                agent_1 = create_agent(agent_1_role, agent_1_backstory, agent_1_goal, llm)
                agent_2 = create_agent(agent_2_role, agent_2_backstory, agent_2_goal, llm)
                agent_3 = create_agent(agent_3_role, agent_3_backstory, agent_3_goal, llm)

                task_1 = create_task(
                    description=f"A clear, concise statement of what the task entails.\n---\nVARIABLE 1: {var_1}\nVARIABLE 2: {var_2}\nVARIABLE 3: {var_3}",
                    expected_output="A detailed description of what the task's completion looks like.",
                    agent=agent_1
                )

                task_2 = create_task(
                    description=f"A clear, concise statement of what the task entails.\n---\nVARIABLE 1: {var_1}\nVARIABLE 2: {var_2}\nVARIABLE 3: {var_3}",
                    expected_output="A detailed description of what the task's completion looks like.",
                    agent=agent_2,
                    context=[task_1]
                )

                task_3 = create_task(
                    description=f"A clear, concise statement of what the task entails.\n---\nVARIABLE 1: {var_1}\nVARIABLE 2: {var_2}\nVARIABLE 3: {var_3}",
                    expected_output="A detailed description of what the task's completion looks like.",
                    agent=agent_3,
                    context=[task_1, task_2]
                )

                crew = Crew(
                    agents=[agent_1, agent_2, agent_3],
                    tasks=[task_1, task_2, task_3],
                    verbose=2,
                    process=Process.hierarchical,
                    manager_llm=llm,
                    output_log_file="./output.log"
                )

                output_expander = st.expander("Output", expanded=True)
                original_stdout = sys.stdout
                sys.stdout = StreamToStreamlit(output_expander)

                result = ""
                result_container = output_expander.empty()
                for delta in crew.kickoff():
                    result += delta
                    result_container.markdown(result)
                
                # Save results to a CSV file
                results_df = pd.DataFrame({
                    'Timestamp': [datetime.now()],
                    'API': [api],
                    'Model': [model],
                    'Temperature': [temp],
                    'Variable 1': [var_1],
                    'Variable 2': [var_2],
                    'Variable 3': [var_3],
                    'Result': [result]
                })
                results_df.to_csv('results.csv', mode='a', header=not os.path.exists('results.csv'), index=False)
                
                logging.info("Agents process completed successfully")
                st.success("Process completed successfully!")

            except Exception as e:
                logging.error(f"An error occurred during execution: {str(e)}")
                st.error(f"An error occurred during execution: {str(e)}")
            finally:
                sys.stdout = original_stdout

def results_tab():
    """
    Render the Results tab in the Streamlit app.
    """
    st.header("Results")
    
    try:
        results_df = pd.read_csv('results.csv')
        st.write("Here are the latest results from your agents executions:")
        st.dataframe(results_df)
        
        if st.button("Download Results CSV"):
            csv = results_df.to_csv(index=False)
            st.download_button(
                label="Click here to download",
                data=csv,
                file_name="Agents_results.csv",
                mime="text/csv",
            )
    except FileNotFoundError:
        st.info("No results found. Run an execution to generate results.")
    
    st.subheader("Log Output")
    try:
        with open("output.log", "r") as log_file:
            st.code(log_file.read())
    except FileNotFoundError:
        st.info("No output log found. Run an execution to generate a log.")

def main():
    """
    Main function to run the Streamlit app.
    """
    st.set_page_config(page_title="KWT GPT", page_icon="🤖", layout="wide")
    
    st.title('KWT GPT')
    st.markdown("This app demonstrates the power of AI agents working together to accomplish complex tasks.")

    col1, col2 = st.columns([5, 1])
    with col2:
            st.image('logo.png')
            st.audio('robot.wav', start_time=0, format='audio/wav', autoplay=True)
        
    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Configuration", "Execution", "Results"])

    with tab1:
        config_params = configuration_tab()

    with tab2:
        execution_tab(*config_params)

    with tab3:
        results_tab()

if __name__ == "__main__":
    main()