# CredPolScript v3.py

import fire
import uuid
import textwrap
import sys
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger, RAGDocument
from llama_stack_client.types import Document,UserMessage
from termcolor import colored
from utils import check_model_is_available, get_any_available_model

# Crear el cliente
# Tesla T4
T4_llama_stack_endpoint = "http://172.206.50.74:5001"

# H100
H100_llama_stack_endpoint = "http://20.246.72.227:5001"

llama_stack_endpoint=T4_llama_stack_endpoint
model_id = None
#model_id="meta-llama/Llama-4-Maverick-17B-128E-Instruct"


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

# Verificación de escudos y selección del modelo
available_shields = [shield.identifier for shield in client.shields.list()]
if not available_shields:
    print(colored("No available shields. Disabling safety.", "yellow"))
else:
    print(f"Available shields found: {available_shields}")


if model_id is None:
    model_id = get_any_available_model(client)
    if model_id is None:
        sys.exit("No hay un modelo")
else:
    if not check_model_is_available(client, model_id):
        sys.exit("El modelo no está disponible")

print(f"Using model: {model_id}")


# Custom Tool para identificar la intención del usuario
def identificar_intencion(user_prompt: str) -> str:
    """
    Identifies the intent behind a user's prompt for a credit assistant by querying the model.

    :param user_prompt: The user's input text.
    :return: A string representing the identified intent, one of 'Consulta', 'Solicitud', or 'Reclamo'.
    """
    global client
    global model_id
    prompt = textwrap.dedent(
        """
        Eres un clasificador de intenciones para un asistente crediticio. Tu tarea es analizar el siguiente texto del usuario y determinar si expresa una Consulta, Solicitud o Reclamo.

        - Una Consulta es una pregunta o solicitud de información (por ejemplo, "¿Qué es un crédito?").
        - Una Solicitud es una petición de algo, como un crédito o ayuda (por ejemplo, "Quiero un préstamo").
        - Un Reclamo es una queja o problema (por ejemplo, "Hubo un error en mi factura").

        Devuelve SOLO una de las siguientes palabras: Consulta, Solicitud, Reclamo.

        Ejemplos:
        - Texto: "¿Qué son las Unidades de cariño?" → Consulta
        - Texto: "Quiero un crédito de cariño" → Solicitud
        - Texto: "No estoy conforme con el servicio" → Reclamo

        Texto del usuario: "{}"

        Respuesta:
        """
    ).format(user_prompt)

    try:
        response = client.inference.chat_completion(
            messages=[
                UserMessage(
                    content=prompt,
                    role="user",
                ),
            ],
            model_id=model_id,
            stream=False,
        )
        
        intent = response.completion_message.content
        valid_intents = ["Consulta", "Solicitud", "Reclamo"]
        if intent not in valid_intents:
            return "Consulta"
        return intent
    except Exception as e:
        print(f"Error al clasificar la intención: {str(e)}")
        return "Consulta"





# Creación del Agente
agent = Agent(
    client,
    model=model_id,
    instructions=textwrap.dedent(
        """
        Eres un asistente crediticio que responde a consultas, solicitudes o reclamos. 
        Para cada mensaje del usuario, SIEMPRE debes usar la herramienta `identificar_intencion` para determinar la intención del mensaje, sin excepción. 
        Basándote únicamente en el resultado de `identificar_intencion`, actúa de la siguiente manera:
        - Si la intención es "Reclamo", El tono del mensaje debe ser serio. Indica que el usuario debe dirigirse al sector de reclamos y pregunta si hay algo mas en lo que lo puedas ayudar.
        - Si la intención es "Consulta", debes usar la información en memoria o en el RAG. El Tono debe ser distentido.
        - Si la intención es "Solicitud", ofrece un Crédito de Cariño. El Tono de la charla debe ser jobial. 
        Mantén tus respuestas breves y concisas.
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
user_inputs = [
    "¿Qué es Kuak?",
    "¿Qué son las Unidades de cariño?",
    "Quiero un crédito",
    "Hubo un error en mi factura"
]

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
    "Se ve que huvo un error en tu Factura. Por favor, dirígete al sector de reclamos para resolver tu problema.",
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


JUDGE_PROMPT = """
Given a QUESTION and GENERATED_RESPONSE and EXPECTED_RESPONSE.

Compare the factual content of the GENERATED_RESPONSE with the EXPECTED_RESPONSE. Ignore any differences in style, grammar, or punctuation.
  The GENERATED_RESPONSE may either be a subset or superset of the EXPECTED_RESPONSE, or it may conflict with it. Determine which case applies. Answer the question by selecting one of the following options:
  (A) The GENERATED_RESPONSE is a subset of the EXPECTED_RESPONSE and is fully consistent with it.
  (B) The GENERATED_RESPONSE is a superset of the EXPECTED_RESPONSE and is fully consistent with it.
  (C) The GENERATED_RESPONSE contains all the same details as the EXPECTED_RESPONSE.
  (D) There is a disagreement between the GENERATED_RESPONSE and the EXPECTED_RESPONSE.
  (E) The answers differ, but these differences don't matter from the perspective of factuality.

Give your answer in the format "Answer: One of ABCDE, Explanation: ".

Your actual task:

QUESTION: {input_query}
GENERATED_RESPONSE: {generated_answer}
EXPECTED_RESPONSE: {expected_answer}
"""



# Scoring
scoring_params = {
    "llm-as-judge::base": {
        "judge_model": model_id,
        "prompt_template": JUDGE_PROMPT,
        "type": "llm_as_judge",
        "judge_score_regexes": ["Answer: (A|B|C|D|E)"],
    },
    "basic::subset_of": None,
}
scoring_response = client.scoring.score(
    input_rows=eval_rows, scoring_functions=scoring_params
)
pprint(scoring_response)