import os
import time
from time import sleep
import tempfile
import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain_groq import ChatGroq
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import WebBaseLoader, YoutubeLoader, CSVLoader, PyPDFLoader, TextLoader
from langchain.prompts import ChatPromptTemplate
from fake_useragent import UserAgent
from openai import RateLimitError
import random
#
TIPOS_ARQUIVOS_VALIDOS = [
    'Site', 'Youtube', 'PDF', 'CSV', 'TXT'
]
#
CONFIG_MODELOS = {
    'OpenAI': {
        'modelos': ['gpt-4o-mini', 'gpt-4o',
                    'gpt-5-mini', 'gpt-5'],
        'chat': ChatOpenAI,
        # 👇 coloque sua chave padrão aqui#
        'default_key': "SUA_CHAVE_AQUI"
    },
    'Groq': {
        'modelos': ['llama-3.3-70b-versatile', 'llama-3.1-8b-instant'],
        'chat': ChatGroq,
        'default_key': None  # não define nada
    }
}

MEMORIA = ConversationBufferMemory()

def stream_com_retry(chain, payload, max_retries=5, base=1.2):
    tentativa = 0
    while True:
        try:
            for chunk in chain.stream(payload):
                yield chunk
            return
        except RateLimitError as e:
            if tentativa >= max_retries:
                raise
            espera = (base ** tentativa) + random.uniform(0, 0.6)
            st.info(f"Calma aí… limite da API atingido. Tentando de novo em {espera:.1f}s (tentativa {tentativa+1}/{max_retries})")
            time.sleep(espera)
            tentativa += 1
        except Exception:
            # Re-levanta para o handler externo mostrar stack
            raise

def carrega_arquivos(tipo_arquivo, arquivo):
    if tipo_arquivo == 'Site':
        if not arquivo or not isinstance(arquivo, str):
            st.error("Informe uma URL válida.")
            st.stop()
        return carrega_site(arquivo)

    if tipo_arquivo == 'Youtube':
        if not arquivo:
            st.error("Informe a URL do vídeo do YouTube.")
            st.stop()
        return carrega_youtube(arquivo)

    # Para uploads
    if arquivo is None:
        st.error(f"Faça o upload de um arquivo {tipo_arquivo}.")
        st.stop()

    if tipo_arquivo == 'PDF':
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as temp:
            temp.write(arquivo.read())
            return carrega_pdf(temp.name)

    if tipo_arquivo == 'CSV':
        with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp:
            temp.write(arquivo.read())
            return carrega_csv(temp.name)

    if tipo_arquivo == 'TXT':
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as temp:
            temp.write(arquivo.read())
            return carrega_txt(temp.name)


def carrega_site(url):
    documento = ''
    for i in range(5):
        try:
            os.environ['USER_AGENT'] = UserAgent().random
            loader = WebBaseLoader(url, raise_for_status=True)
            lista_documentos = loader.load()
            documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
            break
        except:
            print(f'Erro ao carregar o site {i+1} vez. Tentando novamente...')
            sleep(3)
    if documento == '':
        st.error("Não foi possível carregar o site.")     
        st.stop()
    return documento

def carrega_youtube(video_id):
    loader = YoutubeLoader(video_id, add_video_info=False, language='pt')
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_csv(caminho):
    try:
        loader = CSVLoader(caminho, encoding="utf-8")
        lista_documentos = loader.load()
        return '\n\n'.join([doc.page_content for doc in lista_documentos])
    except Exception:
        loader = CSVLoader(caminho, encoding="cp1252")
        lista_documentos = loader.load()
        return '\n\n'.join([doc.page_content for doc in lista_documentos])


def carrega_pdf(caminho):
    loader = PyPDFLoader(caminho)
    lista_documentos = loader.load()
    documento = '\n\n'.join([doc.page_content for doc in lista_documentos])
    return documento

def carrega_txt(caminho):
    # 1) tenta detecção automática do TextLoader
    try:
        loader = TextLoader(caminho, autodetect_encoding=True)
        lista_documentos = loader.load()
        return '\n\n'.join([doc.page_content for doc in lista_documentos])
    except Exception as e:
        # 2) fallback: tenta decodificações comuns (utf-8, cp1252, latin-1)
        try:
            with open(caminho, 'rb') as f:
                raw = f.read()
            for enc in ('utf-8', 'cp1252', 'latin-1'):
                try:
                    texto = raw.decode(enc)
                    return texto
                except UnicodeDecodeError:
                    continue
            # 3) se nada funcionar, mostra o erro
            st.error(f"Não consegui decodificar o TXT. Tente salvar o arquivo em UTF-8. Erro original: {type(e).__name__}: {e}")
            st.stop()
        except Exception as e2:
            st.error(f"Falha ao ler o TXT: {type(e2).__name__}: {e2}")
            st.stop()

#
def carrega_modelo(provedor, modelo, api_key, tipo_arquivo, arquivo):

    try:
        documento = carrega_arquivos(tipo_arquivo, arquivo)
    except Exception as e:
        st.error(f"Erro ao carregar o {tipo_arquivo}: {type(e).__name__}: {e}")
        st.stop()
    
    system_message = '''Você é um assistente amigável chamado Oráculo.
        Você possui acesso às seguintes informações vindas 
        de um documento {}: 

        ####
        {}
        ####

        Utilize as informações fornecidas para basear as suas respostas.

        Sempre que houver $ na sua saída, substita por S.

        Se a informação do documento for algo como "Just a moment...Enable JavaScript and cookies to continue" 
        sugira ao usuário carregar novamente o Oráculo!'''.format(tipo_arquivo, documento)
    
    

    template = ChatPromptTemplate.from_messages([
        ('system', system_message),
        ('placeholder', '{chat_history}'),
        ('user', '{input}')
    ])

    api_key = api_key or CONFIG_MODELOS[provedor]['default_key']

    if provedor == "OpenAI":
        chat = CONFIG_MODELOS[provedor]['chat'](
            model=modelo,
            openai_api_key=api_key
        )
    else:
        chat = CONFIG_MODELOS[provedor]['chat'](
            model=modelo,
            api_key=api_key
        )

    chain = template | chat
    st.session_state['chain'] = chain
    

def pagina_chat():
    st.header("🤖 Bem-vindo ao Assistente do DCA!", divider=True)

    chain = st.session_state.get('chain')
    if chain is None:
        st.error("Por favor, carregue um assistente na barra lateral.")
        st.stop()

    memoria = st.session_state.get("memoria", MEMORIA)

    # re-render do histórico
    for mensagem in memoria.buffer_as_messages:
        chat = st.chat_message(mensagem.type)  # 'human'/'ai' já funcionam com Streamlit
        chat.markdown(mensagem.content)

    input_usuario = st.chat_input("Digite sua mensagem aqui...")
    if not input_usuario:
        return

    # anti-duplicado rápido (evita flood por duplo Enter)
    agora = time.time()
    if (st.session_state.get("last_input_text") == input_usuario and
        agora - st.session_state.get("last_input_at", 0) < 2.0):
        st.warning("Mensagem duplicada ignorada.")
        return
    st.session_state["last_input_text"] = input_usuario
    st.session_state["last_input_at"] = agora

    # adiciona e mostra a mensagem do usuário
    memoria.chat_memory.add_user_message(input_usuario)
    st.chat_message("human").markdown(input_usuario)

    # responde com streaming (com retry e tratamento de erros)
    ai_box = st.chat_message("ai")
    resposta_final = ""
    try:
        resposta_final = ai_box.write_stream(
            stream_com_retry(
                chain,
                {
                    'input': input_usuario,
                    'chat_history': memoria.buffer_as_messages
                }
            )
        )
    except RateLimitError as e:
        st.error("A OpenAI limitou suas requisições agora. Tente novamente em alguns segundos "
                 "ou altere modelo/chave na aba lateral.")
        st.caption("Dicas: use gpt-4o-mini quando possível; reduza o tamanho do contexto; evite mandar muitas mensagens em sequência.")
        st.stop()
    except Exception as e:
        st.error(f"Falha ao chamar o modelo: {type(e).__name__}: {e}")
        # mostra só o final do stack para não poluir
        st.code("".join(traceback.format_exc())[-2000:])
        st.stop()

    # salva resposta na memória e persiste estado
    memoria.chat_memory.add_ai_message(resposta_final or "")
    st.session_state["memoria"] = memoria



def sidebar():
    tabs = st.tabs(["Upload de Arquivos", "Seleção de Modelos"])
    with tabs[0]:
        tipo_arquivo = st.selectbox('Selecione o tipo de arquivo:', TIPOS_ARQUIVOS_VALIDOS)
        if tipo_arquivo == 'Site':
            # valor padrão + persistência no estado
            if 'url_site' not in st.session_state:
                st.session_state['url_site'] = 'https://veigaepostal.com.br/'
            arquivo = st.text_input(
                'Digite a URL do site:',
                value=st.session_state['url_site'],
                key='input_url_site'
            )
            st.session_state['url_site'] = arquivo
        if tipo_arquivo == 'Youtube':
            arquivo = st.text_input('Digite a URL do vídeo do YouTube:')
        if tipo_arquivo == 'PDF':
            arquivo = st.file_uploader('Faça o upload do arquivo PDF:', type=['pdf'])
        if tipo_arquivo == 'CSV':
            arquivo = st.file_uploader('Faça o upload do arquivo CSV:', type=['csv'])
        if tipo_arquivo == 'TXT':
            arquivo = st.file_uploader('Faça o upload do arquivo TXT:', type=['txt'])

    with tabs[1]:
        provedor = st.selectbox('Selecione o provedor dos modelos:', CONFIG_MODELOS.keys())
        modelo = st.selectbox('Selecione o modelo:', CONFIG_MODELOS[provedor]['modelos'])

        # --- inicializa o estado com a default key do provedor selecionado
        state_key = f'api_key_{provedor}'
        if state_key not in st.session_state:
            st.session_state[state_key] = CONFIG_MODELOS[provedor]['default_key']

        # --- key único por provedor para o widget
        api_key = st.text_input(
            f'Digite sua chave de API {provedor}:',
            type='password',
            value=st.session_state[state_key],      # aparece já preenchido
            key=f'input_{state_key}'               # widget separado por provedor
        )

        # mantém o estado sincronizado com o que o usuário digitar
        st.session_state[state_key] = api_key

    if st.button('Carregar Assistente', use_container_width=True):
        carrega_modelo(provedor, modelo, st.session_state[state_key], tipo_arquivo, arquivo)
        st.success('Assistente carregado com sucesso!')

    if st.button('Apagar Histórico de Conversa', use_container_width=True):
        st.session_state["memoria"] = MEMORIA
        st.success('Histórico apagado com sucesso!')



def main():
    with st.sidebar:
        sidebar()
    pagina_chat()
    
if __name__ == "__main__":
    main()  