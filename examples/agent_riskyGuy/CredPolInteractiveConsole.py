# CredPolScriptInteractive.py

import fire
import uuid
import textwrap
import sys
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger, RAGDocument
from llama_stack_client.types import Document, UserMessage
from termcolor import colored
from utils import check_model_is_available, get_any_available_model

# Crear el cliente
# Tesla T4
T4_llama_stack_endpoint = "http://20.96.106.121:5001"       

# H100
H100_llama_stack_endpoint = "http://20.246.72.227:5001"

llama_stack_endpoint=T4_llama_stack_endpoint
model_id = None

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

# Función para manejar la interacción
def run_interactive_chat():
    print(colored("Asistente Crediticio: ¡Hola! ¿En qué puedo ayudarte hoy? (Escribe 'salir' para terminar)", "cyan"))
    
    while True:
        # Capturar entrada del usuario
        user_input = input(colored("Tú: ", "green")).strip()
        
        # Salir si el usuario escribe "salir"
        if user_input.lower() in ["salir", "exit", "quit"]:
            print(colored("Asistente: ¡Gracias por usar el asistente crediticio! Hasta pronto.", "cyan"))
            break
        
        # Ignorar entradas vacías
        if not user_input:
            print(colored("Asistente: Por favor, escribe un mensaje.", "yellow"))
            continue
        
        # Procesar el turno
        try:
            print(colored("Asistente: ", "cyan"), end="")
            turn_response = agent.create_turn(
                messages=[{"role": "user", "content": user_input}],
                session_id=session_id,
                stream=True
            )
            for event in AgentEventLogger().log(turn_response):
                event.print()
        except Exception as e:
            print(colored(f"Error: {str(e)}", "red"))

# Iniciar la interacción
if __name__ == "__main__":
    run_interactive_chat()