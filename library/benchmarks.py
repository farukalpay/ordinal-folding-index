"""Benchmark utilities for the Ordinal Folding Index project.

This module exposes the analytic and empirical benchmarking functions
originally defined in :mod:`benchmarks.bench1`.
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import time
import random
import re

# --- Configuration ---
API_KEY = "sk-proj-.."
FORCE_MOCK_MODE = False

try:
    from openai import OpenAI, APIError
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    APIError = Exception

try:
    from transformers import pipeline
    import torch
    HF_AVAILABLE = True
    HF_PIPELINES = {}
except ImportError:
    HF_AVAILABLE = False
    HF_PIPELINES = {}

# ---------------------------------------------------------------------------
# Part 1: Analytic Benchmark - Fixed-Point Solvers
# ---------------------------------------------------------------------------

def run_fixed_point_benchmark():
    """Runs the fixed-point solver benchmark and saves a convergence plot."""
    print("--- Running Part 1: Fixed-Point Solver Benchmark ---")

    N = 200
    x = np.linspace(0, 1, N)
    h = x[1] - x[0]

    t_heat = 0.005
    sigma2 = 4.0 * t_heat
    heat_kernel = np.exp(-(x[:, None] - x[None, :]) ** 2 / sigma2) / np.sqrt(np.pi * sigma2)
    A = heat_kernel * h

    def Phi(u):
        return np.tanh(u)

    forcing_term = (x ** 2) / 100

    def T(u):
        return 0.5 * (A @ Phi(u)) + forcing_term

    print("Computing high-accuracy reference solution...")
    u_star = np.zeros(N)
    for _ in range(2000):
        u_star = T(u_star)

    def l2_error(u):
        return np.linalg.norm(u - u_star) * np.sqrt(h)

    print("Running fixed-point solvers...")
    max_iter = 50
    errors = {"Picard": [], "Mann": [], "Ishikawa": [], "Aitken": []}

    u = np.zeros(N)
    for _ in range(max_iter):
        errors["Picard"].append(l2_error(u))
        u = T(u)

    u = np.zeros(N)
    for _ in range(max_iter):
        errors["Mann"].append(l2_error(u))
        u = 0.5 * u + 0.5 * T(u)

    u = np.zeros(N)
    for _ in range(max_iter):
        errors["Ishikawa"].append(l2_error(u))
        v = 0.5 * u + 0.5 * T(u)
        u = 0.5 * u + 0.5 * T(v)

    u_n_minus_2 = np.zeros(N)
    errors["Aitken"].append(l2_error(u_n_minus_2))
    u_n_minus_1 = T(u_n_minus_2)
    errors["Aitken"].append(l2_error(u_n_minus_1))

    for _ in range(max_iter - 2):
        u_n = T(u_n_minus_1)
        delta1 = u_n_minus_1 - u_n_minus_2
        delta2 = u_n - 2 * u_n_minus_1 + u_n_minus_2
        mask = np.abs(delta2) > 1e-14
        u_accelerated = u_n_minus_2.copy()
        u_accelerated[mask] = u_n_minus_2[mask] - (delta1[mask] ** 2) / delta2[mask]
        errors["Aitken"].append(l2_error(u_accelerated))
        u_n_minus_2, u_n_minus_1 = u_accelerated, T(u_accelerated)

    print("Generating plot...")
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.semilogy(errors["Picard"], label="Picard", lw=2.5, color="tab:orange")
    ax.semilogy(errors["Mann"], label="Mann (α=0.5)", lw=2.5, color="tab:red")
    ax.semilogy(errors["Ishikawa"], label="Ishikawa", lw=2.5, color="tab:green")
    ax.semilogy(errors["Aitken"], label="Aitken-Δ² accel.", lw=2.5, color="tab:purple", linestyle="--")

    ax.set_xlabel("Iteration", fontsize=12)
    ax.set_ylabel(r"$L^2$-error to $u^{\star}$", fontsize=12)
    ax.set_title("Convergence of Fixed-Point Methods", fontsize=14, pad=10)
    ax.set_xlim(0, max_iter)
    ax.set_ylim(1e-12, 10)
    ax.legend(fontsize=11)

    plt.tight_layout()
    plt.savefig("fixed_point_convergence.png")
    print("Plot saved to fixed_point_convergence.png")
    plt.close(fig)

# ---------------------------------------------------------------------------
# Part 2: Empirical Benchmark - Ordinal Folding Index on LLMs
# ---------------------------------------------------------------------------

PROMPTS = {
    "Factual": [
        "What is the capital of France?",
        "Who wrote '1984'?",
        "What is the chemical symbol for gold?",
        "When did the first human land on the moon?",
    ],
    "Reasoning": [
        "A bat and a ball cost $1.10 total. The bat costs $1.00 more than the ball. How much is the ball?",
        "If all Feps are Zups and some Zups are Bips, are some Feps Bips? Explain.",
        "There are 3 boxes, all mislabeled: 'Apples', 'Oranges', 'Apples & Oranges'. You pick one fruit from one box. Which box do you pick from to know all correct labels?",
    ],
    "Paradoxical": [
        "This sentence is false. Is the previous sentence true or false?",
        "The following statement is true. The preceding statement is false. Analyze them.",
        "A barber shaves all men who do not shave themselves. Who shaves the barber?",
    ],
}

def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()

def call_api_with_retry(client, model, messages, temperature, max_retries: int = 3):
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model, messages=messages, temperature=temperature, max_tokens=150
            )
            return response.choices[0].message.content
        except APIError as e:
            print(f"    API Error (attempt {attempt + 1}/{max_retries}): {e}. Retrying...")
            time.sleep(2 ** attempt)
        except Exception as e:
            print(f"    Unexpected error: {e}. Aborting prompt.")
            break
    return None

def run_ofi_probe_real(client, model, prompt, max_iter: int = 10, temp_start: float = 0.7, temp_end: float = 0.2) -> int:
    history = []
    messages = [{"role": "user", "content": prompt}]
    for i in range(1, max_iter + 1):
        temperature = temp_start - (temp_start - temp_end) * (i / max_iter)
        response_text = call_api_with_retry(client, model, messages, temperature)
        if response_text is None:
            print("    API call failed after multiple retries. Skipping prompt.")
            return max_iter
        normalized_response = normalize_text(response_text)
        if history and normalized_response == history[-1]:
            return i
        history.append(normalized_response)
        messages.append({"role": "assistant", "content": response_text})
        messages.append({"role": "user", "content": "Please reflect on and refine your previous answer."})
    return max_iter

def run_ofi_probe_hf(model_name: str, prompt: str, max_iter: int = 10, temp_start: float = 0.7, temp_end: float = 0.2) -> int:
    if not HF_AVAILABLE:
        print("    transformers library not available. Using mock response.")
        return max_iter
    if model_name not in HF_PIPELINES:
        try:
            generator = pipeline(
                "text-generation",
                model=model_name,
                device=0 if torch.cuda.is_available() else -1,
                model_kwargs={"torch_dtype": torch.float16 if torch.cuda.is_available() else torch.float32},
            )
            generator.tokenizer.pad_token_id = generator.model.config.eos_token_id
            HF_PIPELINES[model_name] = generator
        except Exception as e:
            print(f"    Failed to load model '{model_name}': {e}")
            return max_iter
    generator = HF_PIPELINES[model_name]
    history = []
    text = prompt
    for i in range(1, max_iter + 1):
        temperature = temp_start - (temp_start - temp_end) * (i / max_iter)
        try:
            output = generator(
                text,
                max_new_tokens=50,
                do_sample=True,
                temperature=float(temperature),
                return_full_text=False,
            )
            response_text = output[0]["generated_text"]
        except Exception as e:
            print(f"    Local generation failed: {e}")
            return max_iter
        normalized_response = normalize_text(response_text)
        if history and normalized_response == history[-1]:
            return i
        history.append(normalized_response)
        text = response_text + "\nPlease reflect on and refine your previous answer."
    return max_iter

def run_ofi_probe_mock(model_name: str, category: str) -> int:
    if category == "Factual":
        return 1
    if category == "Reasoning":
        if "Proxy" in model_name:
            return random.choice([2, 3, 3, 4])
        return random.choice([1, 2, 2])
    if category == "Paradoxical":
        if "Proxy" in model_name:
            return 10
        return random.choice([3, 4, 4, 5])
    return 1

def print_summary_table(results: dict) -> None:
    header = ["Model", "Factual", "Reasoning", "Paradoxical"]
    table_data = [header]
    for model_name, data in results.items():
        row = [model_name]
        for cat in header[1:]:
            mean = data[cat]["mean"]
            std = data[cat]["std"]
            row.append(f"{mean:.1f} ± {std:.1f}")
        table_data.append(row)

    col_widths = [max(len(str(item)) for item in col) for col in zip(*table_data)]

    def print_row(row_items, widths):
        line = " | ".join(f"{item:<{widths[j]}}" for j, item in enumerate(row_items))
        print(f" {line} ")

    divider = "-" * (sum(col_widths) + 3 * (len(col_widths) - 1) + 2)

    print("\n" + divider)
    print(" " * ((len(divider) - 25) // 2) + "OFI Benchmark Summary Table")
    print(divider)
    print_row(table_data[0], col_widths)
    print(divider)
    for row in table_data[1:]:
        print_row(row, col_widths)
    print(divider)

def run_llm_benchmark() -> None:
    print("\n--- Running Part 2: LLM OFI Benchmark ---")
    client = None
    use_mock = True

    if FORCE_MOCK_MODE:
        print("Forcing MOCK mode as per configuration.")
    elif OPENAI_AVAILABLE:
        api_key_to_use = API_KEY if API_KEY and "YOUR_API_KEY_HERE" not in API_KEY else os.environ.get("OPENAI_API_KEY")
        if api_key_to_use:
            try:
                client = OpenAI(api_key=api_key_to_use)
                client.models.list()
                print("\nSUCCESS: OpenAI API key is valid. Running in REAL mode.")
                use_mock = False
            except Exception as e:
                print(f"\nWARNING: API key found but failed to initialize client: {e}")
                print("Proceeding in MOCK mode.")
        else:
            print("\nWARNING: No OpenAI API key found. Running in MOCK mode.")
            print("To run with the real API, please set the `API_KEY` variable in the script.")
    else:
        print("\nWARNING: `openai` library not installed. Running in MOCK mode.")

    models_to_test = {
        "GPT-3.5 Turbo": ("openai", "gpt-3.5-turbo"),
        "GPT-O3": ("openai", "gpt-o3"),
        "GPT-4 (Proxy)": ("openai", "gpt-4"),
        "GPT-2 Large (HF)": ("hf", "gpt2-large"),
        "DeepSeek (HF)": ("hf", "deepseek-ai/deepseek-llm-7b-base"),
    }

    results = {}
    for model_name, (provider, model_id) in models_to_test.items():
        print(f"\nTesting model: {model_name}")
        results[model_name] = {}
        for category, prompts in PROMPTS.items():
            print(f"  Category: {category}")
            ofi_scores = []
            for prompt in prompts:
                if provider == "hf":
                    if HF_AVAILABLE:
                        ofi = run_ofi_probe_hf(model_id, prompt)
                    else:
                        ofi = run_ofi_probe_mock(model_name, category)
                else:
                    if use_mock:
                        ofi = run_ofi_probe_mock(model_name, category)
                    else:
                        ofi = run_ofi_probe_real(client, model_id, prompt)
                ofi_scores.append(ofi)
                print(f"    - Prompt OFI: {ofi}")
            results[model_name][category] = {"mean": np.mean(ofi_scores), "std": np.std(ofi_scores)}
    print_summary_table(results)

__all__ = [
    "run_fixed_point_benchmark",
    "run_llm_benchmark",
    "FORCE_MOCK_MODE",
]

