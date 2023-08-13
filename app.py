import json
import streamlit as st


st.set_page_config(page_title="ChatXu-Int4")
st.title("ChatXu-Int4")


@st.cache_resource
def init_model():
    model_id = 'xuqinyang/chatxu-ggml'
    from huggingface_hub import snapshot_download,hf_hub_download
    #旧
    #snapshot_download(model_id, local_dir="./",revision="7f71a8abefa7b2eede3f74ce0564abe5fbe6874a")
    snapshot_download(model_id, local_dir="./")
    from llama_cpp import Llama
    llm = Llama(model_path="./ggml-model-q4_0.bin", n_ctx=4096,seed=-1)

    return llm


def clear_chat_history():
    del st.session_state.messages


def init_chat_history():
    with st.chat_message("assistant", avatar='🤖'):
        st.markdown("你好，我是ChatXu。")

    if "messages" in st.session_state:
        for message in st.session_state.messages:
            avatar = '🧑‍💻' if message["role"] == "user" else '🤖'
            with st.chat_message(message["role"], avatar=avatar):
                st.markdown(message["content"])
    else:
        st.session_state.messages = []

    return st.session_state.messages


def main():
    llm = init_model()
    messages = init_chat_history()

    if prompt := st.chat_input("Shift + Enter 换行, Enter 发送"):
        with st.chat_message("user", avatar='🧑‍💻'):
            st.markdown(prompt)
        messages.append({"role": "user", "content": prompt})
        print(f"[user] {prompt}", flush=True)
        with st.chat_message("assistant", avatar='🤖'):
            placeholder = st.empty()
            response = ''
            for responses in llm.create_chat_completion(messages,stop=["</s>"],stream=True,max_tokens=-1,temperature=0.3,top_k=5,top_p=0.85,repeat_penalty=1.1):
                if "content" in responses["choices"][0]["delta"]:
                    response+=responses["choices"][0]["delta"]["content"]
                    placeholder.markdown(response + "▌")
            placeholder.markdown(response)
        messages.append({"role": "assistant", "content": response})
        print(json.dumps(messages, ensure_ascii=False), flush=True)

        st.button("清空对话", on_click=clear_chat_history)


if __name__ == "__main__":
    main()
