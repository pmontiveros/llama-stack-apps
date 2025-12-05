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


## simple rag query

import re
import textwrap
from termcolor import colored
from pprint import pprint


def _clean_text(text: str) -> str:
    if not text:
        return ""
    # Normalizar saltos de línea
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Reemplazar tabs por espacios
    text = text.replace("\t", " ")
    # Colapsar espacios repetidos
    text = re.sub(r"[ ]{2,}", " ", text)
    # Colapsar más de 2 saltos de línea seguidos
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _pretty_print_chunk(content: str, width: int = 100, indent: str = "  ") -> None:
    content = _clean_text(content)
    if not content:
        print(indent + "[Sin contenido]")
        return

    wrapped = textwrap.fill(content, width=width)
    for line in wrapped.splitlines():
        print(indent + line)

# rag query function
def simple_rag_query(question: str, debug: bool = False):
    """
    Hace una consulta directa al vector DB y muestra los chunks recuperados
    de forma ordenada y legible.
    Soporta distintas versiones del schema de respuesta de rag_tool.query.
    """
    print(colored("\n=== PREGUNTA ===", "green", attrs=["bold"]))
    print(question)

    result = client.tool_runtime.rag_tool.query(
        vector_db_ids=[vector_db_id],
        content=question,
    )

    if debug:
        print(colored("\n=== RESULTADO CRUDO (DEBUG) ===", "yellow", attrs=["bold"]))
        pprint(result)

    # ---- Interpretar respuesta de forma robusta ----
    hits = None

    # 1) Caso Red Hat / LlamaStack actual: result.content (lista de items con .text)
    if hasattr(result, "content"):
        hits = result.content

    # 2) Caso anterior: result.results
    if hits is None and hasattr(result, "results"):
        hits = result.results

    # 3) Caso dict
    if hits is None and isinstance(result, dict):
        hits = result.get("content") or result.get("results")

    # 4) Caso lista directamente
    if hits is None and isinstance(result, list):
        hits = result

    if not hits:
        print(colored("\n[WARN] No se encontraron resultados RAG]", "yellow"))
        return

    # ---- Extraer contenido y score (si existe) ----
    processed_hits = []

    for idx, hit in enumerate(hits):
        content = None
        score = None

        # Objeto con atributos (lo más probable)
        if hasattr(hit, "text"):          # TextContentItem(text=...)
            content = hit.text
        elif hasattr(hit, "content"):     # Algunos schemas usan .content
            content = hit.content

        if hasattr(hit, "score"):
            score = hit.score

        # Dict
        if content is None and isinstance(hit, dict):
            content = hit.get("text") or hit.get("content")
            score = hit.get("score", score)

        if content is None:
            continue

        processed_hits.append(
            {
                "content": str(content),
                # Si no hay score, usamos el índice como “score” (mantiene el orden)
                "score": float(score) if score is not None else float(len(hits) - idx),
            }
        )

    if not processed_hits:
        print(colored("\n[WARN] No se pudo extraer contenido de los resultados]", "yellow"))
        return

    # Ordenar por score desc (si no hay score real, mantiene el orden original)
    processed_hits.sort(key=lambda x: x["score"], reverse=True)

    print(colored("\n=== CHUNKS RECUPERADOS (ordenados por score) ===", "cyan", attrs=["bold"]))

    for i, hit in enumerate(processed_hits, start=1):
        print(colored(f"\n--- Chunk #{i} (score={hit['score']:.4f}) ---", "cyan"))
        _pretty_print_chunk(hit["content"])


# Ejemplos de uso (adaptalos al contenido real de Politicas.txt)
simple_rag_query("¿Qué restricciones existen para solicitar unidades de cariño?", debug=False)
#simple_rag_query("¿Qué pasa si el empleado se atrasa en los pagos?", debug=False)

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
    
   
## Listar modelos

from rich.pretty import pprint

print("Available models:")
for m in client.models.list():
    print(f"- {m.identifier}")


## Seleccción del modelo.
model_id= "accounts/fireworks/models/llama-v3p3-70b-instruct"

if model_id is None:
    model_id = get_any_available_model(client)
    if model_id is None:
        sys.exit("No hay un modelo")
else:
    if not check_model_is_available(client, model_id):
        sys.exit("El modelo no esta disponible")

print(f"Using model: {model_id}")



## Verificación de escudos y Seleccción del modelo.
available_shields = [shield.identifier for shield in client.shields.list()]
if not available_shields:
    print(colored("No available shields. Disabling safety.", "yellow"))
else:
    print(f"Available shields found: {available_shields}")





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
