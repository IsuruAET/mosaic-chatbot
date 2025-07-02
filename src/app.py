from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_community.utilities import SQLDatabase
from sqlalchemy import create_engine
from langchain_openai import ChatOpenAI
# from langchain_groq import ChatGroq
import streamlit as st

# Initialize the database
def init_database(user: str, password: str, host: str, port: str, database: str) -> SQLDatabase:
  db_uri = f"mysql+mysqlconnector://{user}:{password}@{host}:{port}/{database}"
  engine = create_engine(db_uri)
  return SQLDatabase(engine)

# Get the SQL chain
def get_sql_chain(db: SQLDatabase):
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, write a SQL query that would answer the user's question. Take the conversation history into account.
    
    <SCHEMA>{schema}</SCHEMA>
    
    Conversation History: {chat_history}
    
    Write only the SQL query and nothing else. Do not wrap the SQL query in any other text, not even backticks.
    
    For example:
    Question: which 3 artists have the most tracks?
    SQL Query: SELECT ArtistId, COUNT(*) as track_count FROM Track GROUP BY ArtistId ORDER BY track_count DESC LIMIT 3;
    Question: Name 10 artists
    SQL Query: SELECT Name FROM Artist LIMIT 10;
    
    Your turn:
    
    Question: {question}
    SQL Query:
  """
  prompt = ChatPromptTemplate.from_template(template)
  
  llm = ChatOpenAI(model="gpt-4-0125-preview")
  # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  def get_schema(_):
    return db.get_table_info()
  
  return (
    RunnablePassthrough.assign(schema=get_schema)
    | prompt
    | llm
    | StrOutputParser()
  )

# Get response from the database
def get_response(db: SQLDatabase, user_query: str, chat_history: list):
  sql_chain = get_sql_chain(db)
  
  template = """
    You are a data analyst at a company. You are interacting with a user who is asking you questions about the company's database.
    Based on the table schema below, question, sql query, and sql response, write a natural language response.
    <SCHEMA>{schema}</SCHEMA>

    Conversation History: {chat_history}
    SQL Query: <SQL>{query}</SQL>
    User question: {question}
    SQL Response: {response}
  """

  prompt = ChatPromptTemplate.from_template(template)

  llm = ChatOpenAI(model="gpt-4-0125-preview")
  # llm = ChatGroq(model="mixtral-8x7b-32768", temperature=0)
  
  chain = (
    RunnablePassthrough.assign(query=sql_chain).assign(
      schema=lambda _: db.get_table_info(),
      response=lambda vars: db.run(vars["query"]),
    )
    | prompt
    | llm
    | StrOutputParser()
  )
  
  return chain.stream({
    "question": user_query,
    "chat_history": chat_history,
  })

# Initialize the chat history
if "chat_history" not in st.session_state:
  st.session_state.chat_history = [
    AIMessage(content="Hello! I'm the Mosaic Chatbot. How can I help you today?"),
  ]

# Load the environment variables
load_dotenv()

# Set the page config
st.set_page_config(page_title="Mosaic Chatbot", page_icon=":robot_face:")

st.title("Mosaic Chatbot")

with st.sidebar:
  st.subheader("Settings")
  st.write("This is a chatbot that can answer questions about the Mosaic dataset.")

  st.text_input("Host", value="localhost", key="Host")
  st.text_input("Port", value="3306", key="Port")
  st.text_input("Database", value="chinook", key="Database")
  st.text_input("User", value="root", key="User")
  st.text_input("Password", type="password", value="root", key="Password")

  if st.button("Connect"):
    with st.spinner("Connecting to database..."):
      db = init_database(st.session_state["User"], st.session_state["Password"], st.session_state["Host"], st.session_state["Port"], st.session_state["Database"])
      st.session_state.db = db
      st.success("Connected to database")

# Display chat history
for msg in st.session_state.chat_history:
  if isinstance(msg, HumanMessage):
    with st.chat_message("user"): 
      st.write(msg.content)
  elif isinstance(msg, AIMessage):
    with st.chat_message("assistant"):
      st.write(msg.content)

# User input
user_query = st.chat_input("Type a question...")
if user_query is not None and user_query.strip() != "":
  st.session_state.chat_history.append(HumanMessage(content=user_query))

  with st.chat_message("user"):
    st.markdown(user_query)

  with st.chat_message("assistant"):
    if "db" not in st.session_state:
      st.error("Please connect to a database first using the sidebar.")
    else:
      ai_response = st.write_stream(get_response(st.session_state.db, user_query, st.session_state.chat_history))
      st.session_state.chat_history.append(AIMessage(content=ai_response))
