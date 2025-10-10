import gradio as gr
import fire
from llama_stack_client import LlamaStackClient, Agent, AgentEventLogger
from termcolor import colored
import textwrap
from utils import check_model_is_available, get_any_available_model
import uuid

# Inicializar el cliente de Llama Stack
# Asume que el servidor de Llama Stack está corriendo en localhost:8321

llama_stack_endpoint=f"http://172.206.50.74:5001"

client = LlamaStackClient(
    base_url=llama_stack_endpoint,
    provider_data={"tavily_search_api_key": "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"},
)

# Crear un agente con configuración básica
def create_agent():
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

    agent = Agent(
        client,
        model=model_id,
        instructions=textwrap.dedent(
                """
                    Eres un asistente útil que responde a las preguntas del usuario con precisión.
                    Siempre utiliza la herramienta de búsqueda web para obtener resultados relevantes y cita las fuentes.
                    Responde de manera concisa y clara.
                """
            ),
        tools=["builtin::websearch"],
        input_shields=available_shields,
        output_shields=available_shields,
        enable_session_persistence=False,
    )
    
    return agent

# Función para procesar consultas del usuario
def process_query(user_input, chat_history, agent_id, session_id):
    if not agent_id or not session_id:
        return "Error: Agente o sesión no inicializados.", chat_history
    
    # Enviar la consulta al agente
    response = client.agent.chat(
        agent_id=agent_id,
        session_id=session_id,
        messages=[{"role": "user", "content": user_input}]
    )
    
    # Obtener la respuesta del agente
    assistant_response = response["message"]["content"]
    
    # Actualizar el historial de chat
    chat_history.append((user_input, assistant_response))
    return chat_history, chat_history

# Función para inicializar el agente y la sesión
def initialize_chat():
    try:
        # Crear un nuevo agente
        agent = create_agent()
        
        # Crear una nueva sesión
        session_id = agent.create_session(uuid.uuid4())
        agent_id=f"{(agent.agent_id).uuid4().hex}"
        
        return agent_id, session_id, [], "Agente y sesión inicializados correctamente."
    except Exception as e:
        return None, None, [], f"Error al inicializar: {str(e)}"

# Configurar la interfaz de Gradio
def main():
    with gr.Blocks() as demo:
        gr.Markdown("# Chatbot con Llama Stack")
        
        # Estado para almacenar el ID del agente, sesión y historial
        agent_id = gr.State()
        session_id = gr.State()
        chat_history = gr.State([])
        
        # Botón para inicializar el agente
        init_button = gr.Button("Inicializar Agente")
        init_message = gr.Textbox(label="Estado de Inicialización", interactive=False)
        
        # Interfaz de chat
        chatbot = gr.Chatbot(label="Conversación")
        user_input = gr.Textbox(label="Tu mensaje", placeholder="Escribe tu consulta aquí...")
        send_button = gr.Button("Enviar")
        
        # Acción para inicializar el agente
        init_button.click(
            fn=initialize_chat,
            outputs=[agent_id, session_id, chat_history, init_message]
        )
        
        # Acción para enviar mensajes
        send_button.click(
            fn=process_query,
            inputs=[user_input, chat_history, agent_id, session_id],
            outputs=[chatbot, chat_history]
        )
    
    # Lanzar la aplicación
    demo.launch()

if __name__ == "__main__":
    main()