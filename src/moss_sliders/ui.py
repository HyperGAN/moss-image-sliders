"""Local inference controls; serialized requests protect mutable adapter scale."""

from .inference import render


def build_ui(pipe, slider):
    import gradio as gr
    with gr.Blocks(title="MOSS-Image Sliders") as demo:
        gr.Markdown("# MOSS-Image Sliders\nSigned concept control. Zero restores the frozen base.")
        prompt = gr.Textbox(label="Prompt", value="A ceramic teapot on a wooden table.")
        scale = gr.Slider(-3, 3, value=0, step=0.1, label="Concept strength")
        seed = gr.Number(value=42, precision=0, label="Seed")
        with gr.Row():
            width = gr.Slider(256, 2048, value=1024, step=16, label="Width")
            height = gr.Slider(256, 2048, value=1024, step=16, label="Height")
        steps = gr.Slider(1, 100, value=50, step=1, label="Sampling steps")
        guidance = gr.Slider(1, 6, value=3, step=0.1, label="CFG")
        output = gr.Image(label="Result")

        def generate(prompt: str, scale: float, seed: int, width: int, height: int,
                     steps: int, guidance: float):
            """Render a MOSS image with a signed concept adapter and reproducible seed."""
            return render(pipe, slider, prompt, scale=scale, seed=int(seed), width=int(width),
                          height=int(height), steps=int(steps), guidance=guidance)

        gr.Examples([["A ceramic teapot on a wooden table."],
                     ["A mountain lake at sunrise, a sign reads WELCOME."]], [prompt])
        gr.Button("Generate", variant="primary").click(
            generate, [prompt, scale, seed, width, height, steps, guidance], output,
            concurrency_limit=1, concurrency_id="moss-model")
    return demo


def launch(pipe, slider):
    build_ui(pipe, slider).queue(default_concurrency_limit=1).launch(
        server_name="127.0.0.1", share=False)
