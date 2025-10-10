# CredPolScriptv2.py

import fire
import uuid
import textwrap
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger, RAGDocument
from llama_stack_client.types import Document
from termcolor import colored
from utils import check_model_is_available, get_any_available_model

# Crear el cliente
llama_stack_endpoint = "http://172.206.50.74:5001"
client = LlamaStackClient(base_url=llama_stack_endpoint)

# Crear una instancia de base de datos vectorial
vector_db_id = f"v{uuid.uuid4().hex}"
embed_lm = next(m for m in client.models.list() if m.model_type == "embedding")
embedding_model = embed_lm.identifier
client.vector_dbs.register(
    vector_db_id=vector_db_id,
    embedding_model=embedding_model,
    embedding_dimension=384,
    provider_id="faiss",
)

# Cargar información al RAG
urls = ["Politicas.txt"]
documents = [
    RAGDocument(
        document_id=f"num-{i}",
        content=f"https://raw.githubusercontent.com/pmontiveros/llama-stack-demo/refs/heads/main/data/Politicas/{url}",
        mime_type="text/plain",
        metadata={},
    )
    for i, url in enumerate(urls)
]

# Insertar documentos
client.tool_runtime.rag_tool.insert(
    documents=documents,
    vector_db_id=vector_db_id,
    chunk_size_in_tokens=32,
)

# Custom Tool para identificar la intención del usuario
def identificar_intencion(user_prompt: str) -> str:
    """
    Identifies the intent behind a user's prompt for a credit assistant.

    :param user_prompt: The user's input text.
    :return: A string representing the identified intent, one of 'Consulta', 'Solicitud', or 'Reclamo'.
    """
    prompt_lower = user_prompt.lower().strip()
    consulta_keywords = [
        "qué es", "qué son", "cómo funciona", "cómo es", "explica",
        "dime", "cuál es", "cuáles son", "por qué", "información sobre"
    ]
    solicitud_keywords = [
        "quiero", "necesito", "dame", "pido", "solicito",
        "crédito", "préstamo", "ayuda con", "puedes darme"
    ]
    reclamo_keywords = [
        "reclamo", "queja", "no estoy conforme", "problema",
        "error", "no funciona", "mal servicio", "quejarme"
    ]
    for keyword in reclamo_keywords:
        if keyword in prompt_lower:
            return "Reclamo"
    for keyword in solicitud_keywords:
        if keyword in prompt_lower:
            return "Solicitud"
    for keyword in consulta_keywords:
        if keyword in prompt_lower:
            return "Consulta"
    return "Consulta"

# Verificación de escudos y selección del modelo
available_shields = [shield.identifier for shield in client.shields.list()]
if not available_shields:
    print(colored("No available shields. Disabling safety.", "yellow"))
else:
    print(f"Available shields found: {available_shields}")

model_id = None
if model_id is None:
    model_id = get_any_available_model(client)
    if model_id is None:
        sys.exit("No hay un modelo")
else:
    if not check_model_is_available(client, model_id):
        sys.exit("El modelo no está disponible")

print(f"Using model: {model_id}")

# Creación del Agente
agent = Agent(
    client,
    model=model_id,
    instructions=textwrap.dedent(
        """
        Eres un asistente crediticio que ayuda al usuario con sus consultas, solicitudes o reclamos. Debes utilizar la herramienta identificar_intencion en cada pregunta que haga el usuario para identificar qué necesita.
        Si el usuario hace un reclamo, indícale que debe dirigirse al sector de reclamos. Si hace una consulta, responde su consulta buscando información en el RAG. Si está haciendo una solicitud, ofrécele un Crédito de Cariño.
        Responde de manera breve y concisa.
        """
    ),
    tools=[
        {
            "name": "builtin::rag/knowledge_search",
            "args": {"vector_db_ids": [vector_db_id]},
        },
        identificar_intencion
    ],
    input_shields=available_shields,
    output_shields=available_shields,
    enable_session_persistence=False,
)

# Creación de la sesión
session_id = agent.create_session(session_name=f"s{uuid.uuid4().hex}")

# Mock user_inputs
user_inputs = ["¿Qué es Kuak?", "¿Qué son las Unidades de cariño?", "Quiero un crédito", "Hubo un error en mi factura"]

# Running the turns
for t in user_inputs:
    print("user>", t)
    turn_response = agent.create_turn(
        messages=[{"role": "user", "content": t}], session_id=session_id, stream=True
    )
    for event in AgentEventLogger().log(turn_response):
        event.print()

# Evaluación de resultados
from rich.pretty import pprint

session_response = client.agents.session.retrieve(
    session_id=session_id,
    agent_id=agent.agent_id,
)

# Evaluación
eval_rows = []
expected_answers = [
    "Kuak S.A. es una firma multinacional de servicios de auditoría, consultoría y Advisory con 80000 empleados alrededor del Mundo",
    "En Kuak S.A las unidades de cariño, son un sistema complementario de recompensas y bonificaciones",
    "Te ofrecemos un Crédito de Cariño. ¿Deseas más detalles?",
    "Por favor, dirígete al sector de reclamos para resolver tu problema.",
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

# Scoring
scoring_params = {
    "basic::subset_of": None,
}
scoring_response = client.scoring.score(
    input_rows=eval_rows, scoring_functions=scoring_params
)
pprint(scoring_response)