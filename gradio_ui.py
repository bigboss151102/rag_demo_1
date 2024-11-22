import gradio as gr
from rag_chain import RagInput, final_chain

def generate_answer(question: str):
    input_data = {"question": question}
    result = final_chain.invoke(input_data)
    answer = result['answer'].content.strip()
    return answer

iface = gr.Interface(
    fn=generate_answer,
    inputs=gr.Textbox(label="Ask a Question", placeholder="Type your question here..."),
    outputs=gr.Textbox(label="Answer")
)

iface.launch()