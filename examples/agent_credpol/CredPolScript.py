# Credpol

import fire
import uuid
import textwrap
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger, RAGDocument
from llama_stack_client.types import Document
from termcolor import colored
from utils import check_model_is_available, get_any_available_model


# Crear el liente

llama_stack_endpoint=f"http://172.206.50.74:5001"

client = LlamaStackClient(
    base_url=llama_stack_endpoint,
    )


# Create a vector database instance

vector_db_id = f"v{uuid.uuid4().hex}"

embed_lm = next(m for m in client.models.list() if m.model_type == "embedding")
embedding_model = embed_lm.identifier
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=embedding_model,
    embedding_dimension=384,
    provider_id="faiss",
)


# Cargar información al RAG. Se pasa una lista de documentos

urls = [
    "Politicas.txt",
]

# Los documetos estan en GIT
documents = [
    RAGDocument(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pmontiveros/llama-stack-demo/refs/heads/main/data/Politicas/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
    ##print(documents)
]


# Insert documents
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=32,
)


## Custom Tool para identificar la intención del usuario
import random
def identificar_intencion(user_prompt:str) -> int:
    #Versión Mock
    """
        Identifica la intención del usuario.
        :param user_prompt: Pregunta del usuario
        :return: La intención del usuario que puede ser  una Consulta, una Solicitud o un Reclamao]
    """
    opciones = ["Reclamo", "Consulta", "Solicitud"]
    return random.choice(opciones)
    
   

## Verificación de escudos y Seleccción del modelo.
available_shields = [shield.identifier for shield in client.shields.list()]
if not available_shields:
    print(colored("No available shields. Disabling safety.", "yellow"))
else:
    print(f"Available shields found: {available_shields}")

model_id= None

if model_id is None:
    model_id = get_any_available_model(client)
    if model_id is None:
        sys.exit("No hay un modelo")
else:
    if not check_model_is_available(client, model_id):
        sys.exit("El modelo no esta disponible")

print(f"Using model: {model_id}")


# Creación  del Agente
agent = Agent(
    client,
    model=model_id,
    instructions=textwrap.dedent(
            """
                Eres un asistente crediticio que ayuda al usuario con sus consultas, solicitudes o reclamos. Tú Debes utilizar la herramienta identificar_intencion en cada pregunta que haga el usuario para identificar que necesita. 
                Si el usuario hace un reclamo, Debe indicarle que debe dirigirse al sector de reclamos. Si hace una consulta, simplemente responde su consulta buscando infomración en el RAG. Si esta haciendo una solicitud, entonces ofrecele un Credito de Cariño.
                Responde de manera breve y consisa.
            """
        ),
    #tools=["builtin::websearch"],
        tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        },identificar_intencion
    ],
    input_shields=available_shields,
    output_shields=available_shields,
    enable_session_persistence=False,
)


# Creación de la sesion
session_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")



# Mock user_inputs...
user_inputs = ["¿Que es Kuak?","¿Que Son las Unidades de cariño?"]

# Runing the turns
for t in user_inputs:
    print("user>", t)
    turn_response = agent.create_turn(
        messages=[{"role": "user", "content": t}], session_id=session_id, stream=True
    )
    for event in AgentEventLogger().log(turn_response):
        event.print()
    #if turn_response.output_message.tool_calls:
    #    for tool_call in turn_response.output_message.tool_calls:
    #        if tool_call["tool_name"]=="identificar_intencion":
    #            aux=tool_call


# Evaluación de resultados:
## Obtención de todas las respuestas de la sesion

from rich.pretty import pprint

session_response = client.agents.session.retrieve(
    session_id=session_id,
    agent_id=agent.agent_id,
)
#pprint(session_response)

# Evaluación:
eval_rows = []

expected_answers = [
    "Kuak S.A. es una firma multinacional de servicios de auditoría, consultoría y Advisory con 80000 empleados alrededor del Mundo",
    "En En Kuak S.A las unidades de cariño, son un sistema complementario de recompensas y bonificaciones",
]

for i, turn in enumerate(session_response.turns):
    eval_rows.append(
        {
            "input_query": turn.input_messages[0].content,
            "generated_answer": turn.output_message.content,
            "expected_answer": expected_answers[i],
        }
    )

pprint(eval_rows)


#Scoring:
scoring_params = {
    "basic::subset_of": None,
}
scoring_response = client.scoring.score(
    input_rows=eval_rows, scoring_functions=scoring_params
)
pprint(scoring_response)