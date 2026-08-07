"""
Simple client script to test generation against the deployed Render Orchestrator once Colab nodes are registered.
"""

from openai import OpenAI
import time

def main():
    RENDER_URL = "https://shardflow.onrender.com"

    client = OpenAI(base_url=f"{RENDER_URL}/v1", api_key="shardflow")

    prompt = "Explain in 2 bullet points how distributed pipeline parallel inference works."

    print(f"🚀 Sending prompt to Render API Gateway ({RENDER_URL})...")
    print(f"Prompt: '{prompt}'\n")

    start_t = time.time()
    response = client.chat.completions.create(
        model="Qwen/Qwen2.5-7B-Instruct",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=60,
        temperature=0.7,
        stream=True,
    )

    print("Response: ", end="", flush=True)
    for chunk in response:
        if hasattr(chunk, "choices") and chunk.choices:
            content = chunk.choices[0].delta.content
            if content:
                print(content, end="", flush=True)
        else:
            print(f"[raw: {chunk}]", end="", flush=True)

    duration = time.time() - start_t
    print(f"\n\nDone in {duration:.2f}s!")


if __name__ == "__main__":
    main()

