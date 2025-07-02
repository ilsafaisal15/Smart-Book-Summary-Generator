import gradio as gr
from transformers import pipeline

# Load the summarization model
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

# Summary length map
length_map = {
    "Short": (30, 80),
    "Medium": (80, 150),
    "Long": (150, 300)
}

def generate_summary(text, length_choice):
    if not text.strip():
        return "❗ Please enter some text to summarize."

    min_len, max_len = length_map[length_choice]

    try:
        summary = summarizer(text, max_length=max_len, min_length=min_len, do_sample=False)
        return summary[0]['summary_text']
    except Exception as e:
        return f"❌ Error: {str(e)}"

with gr.Blocks(css=".gradio-container {font-family: 'Segoe UI', sans-serif;}") as demo:
    gr.Markdown(
        """
        # 📚 Smart Book Summary Generator  
        Summarize books, articles, or long paragraphs using Hugging Face's powerful transformer models!
        """
    )

    with gr.Row():
        with gr.Column():
            text_input = gr.Textbox(
                label="📖 Enter your text",
                placeholder="Paste your article or book excerpt here...",
                lines=10
            )
            summary_length = gr.Radio(["Short", "Medium", "Long"], value="Medium", label="📏 Summary Length")
            submit_button = gr.Button("✨ Summarize")

        with gr.Column():
            output_text = gr.Textbox(
                label="📝 Summary Output",
                placeholder="Your summary will appear here...",
                lines=10
            )

    submit_button.click(generate_summary, inputs=[text_input, summary_length], outputs=output_text)

demo.launch()
