"""
app.py
------
A small Streamlit front-end over the existing pipeline (src/pipeline.py).
This is a thin UI layer only -- all the actual retrieval/scoring logic
still lives in src/, exactly as it does for the CLI (main.py --score). The
app just gives a paste-a-conversation-and-see-results experience instead of
a terminal flag.

Run with:
    streamlit run app.py
"""

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Make sure src/ is importable regardless of the working directory Streamlit
# was launched from (mirrors the same pattern used in main.py / the notebook).
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Ahoum Facet Scoring Pipeline", page_icon="🧭", layout="wide")


def check_ollama_status() -> tuple[bool, str]:
    """
    Lightweight Ollama reachability check, used to show a status badge and
    to fail fast with a clear message instead of letting the pipeline hang
    or throw a confusing low-level connection error mid-run.
    Returns (is_ready, message).
    """
    try:
        import ollama
        # Same OLLAMA_HOST override as src/scorer.py -- needed if this app
        # is ever run inside Docker (see docs/README.md "Docker" section).
        ollama_host = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
        client = ollama.Client(host=ollama_host)
        models_response = client.list()
        models = models_response.get("models", []) if isinstance(models_response, dict) else models_response.models
        model_names = []
        for m in models:
            name = m.get("model") if isinstance(m, dict) else getattr(m, "model", None)
            if name:
                model_names.append(name)
        if any("llama3.1" in name for name in model_names):
            return True, "Ollama is running and llama3.1 is available."
        return False, "Ollama is running, but llama3.1 isn't pulled yet. Run: ollama pull llama3.1"
    except Exception as e:
        return False, (
            f"Can't reach Ollama at {ollama_host} -- is it running? "
            f"(details: {e})"
        )


def check_index_status() -> tuple[bool, str]:
    """Confirm the FAISS index / enriched facets exist before letting the
    user try to score anything -- otherwise retrieval fails with a much
    less friendly FileNotFoundError deep inside embeddings.py."""
    faiss_path = PROJECT_ROOT / "outputs" / "faiss_index.bin"
    facets_path = PROJECT_ROOT / "outputs" / "observable_facets.json"
    if faiss_path.exists() and facets_path.exists():
        return True, "Facet index found."
    return False, (
        "No FAISS index found in outputs/. Run `python main.py --audit` "
        "then `python main.py --embed` first."
    )


def check_gpu_vram_status() -> tuple[str, str]:
    """
    Checks free GPU VRAM via torch. This exists because of a real crash
    found during testing (docs/DEBUGGING.md #3): Ollama's llama-server
    inference worker crashed mid-batch when the Jupyter kernel, browser,
    and Streamlit were all competing for the RTX 4050's 6GB of VRAM at
    once. Surfacing this *before* a run starts is a lot more useful than a
    wall of parse_error rows after the fact.

    Returns (level, message) where level is one of
    "green" / "yellow" / "red" / "unknown".

    Note: this uses torch.cuda.mem_get_info(), not
    torch.cuda.get_device_properties(0).total_memory. total_memory is the
    card's fixed capacity (always ~6GB on this laptop, for example) and
    doesn't change no matter what else is using the GPU -- it would never
    actually detect the contention this badge is meant to warn about.
    mem_get_info() returns the CURRENTLY FREE bytes on the device, which is
    what the green/yellow/red thresholds below are actually about.
    """
    try:
        import torch
    except ImportError:
        return "unknown", "GPU status unknown -- torch isn't installed."

    try:
        # A CPU-only torch build (pip installs this by default on Windows
        # unless you point at the CUDA wheel index) always reports
        # is_available()=False -- even when a real NVIDIA GPU is present
        # and Ollama itself is actively using it via its own CUDA runtime.
        # Reporting "red / no GPU" in that case would be a false alarm, so
        # this is treated as "can't check" rather than "no GPU" -- run
        # `nvidia-smi` yourself to check VRAM directly if you see this.
        if getattr(torch.version, "cuda", None) is None:
            return "unknown", (
                "GPU status unknown -- this torch install has no CUDA support "
                "(pip installed the CPU-only build), so it can't see the GPU "
                "even if one is present and in use by Ollama. Run `nvidia-smi` "
                "to check VRAM directly."
            )
        if not torch.cuda.is_available():
            return "red", "No CUDA GPU detected."
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        free_gb = free_bytes / (1024 ** 3)
        total_gb = total_bytes / (1024 ** 3)
        if free_gb > 3:
            return "green", f"{free_gb:.1f} GB free / {total_gb:.1f} GB total."
        if free_gb >= 1:
            return "yellow", (
                f"Only {free_gb:.1f} GB free / {total_gb:.1f} GB total -- close other "
                "GPU workloads (Jupyter kernels, extra browser tabs) for the most "
                "reliable results."
            )
        return "red", (
            f"Only {free_gb:.1f} GB free / {total_gb:.1f} GB total -- Ollama may crash "
            "mid-batch under this little headroom (see docs/DEBUGGING.md #3)."
        )
    except Exception as e:
        return "unknown", f"GPU status unknown ({e})."


def check_backend_status(force_refresh: bool = False) -> tuple[str, str, str]:
    """
    Determines which LLM backend src/scorer.py will actually use: local
    Ollama (preferred) or the Groq API fallback. Returns
    (level, backend_label, message) where level is "green" | "red".

    force_refresh=True bypasses scorer.py's cached detection result -- used
    right after the user types a Groq key into the sidebar input, so the
    badge updates immediately instead of showing stale "red" until the next
    natural cache expiry (there isn't one; detect_backend() only re-checks
    on an explicit force_refresh).
    """
    try:
        from src.scorer import GROQ_MODEL_NAME, MODEL_NAME, detect_backend
        backend = detect_backend(force_refresh=force_refresh)
        if backend == "ollama":
            return "green", "Ollama (local)", f"Using {MODEL_NAME} locally -- private, no API key needed."
        return "green", "Groq (cloud fallback)", f"Ollama not detected -- using Groq's {GROQ_MODEL_NAME} instead."
    except RuntimeError as e:
        return "red", "None", str(e)
    except Exception as e:
        return "red", "None", f"Backend check failed: {e}"


# ---------------------------------------------------------------------------
# Header + status badges
# ---------------------------------------------------------------------------
st.title("🧭 Ahoum Facet Scoring Pipeline")
st.caption(
    "Paste a short conversation below. The system retrieves the most relevant "
    "personality facets (out of 399) via FAISS, then scores them in small "
    "batches with a local llama3.1 model -- abstaining explicitly wherever "
    "the conversation doesn't actually support a judgment."
)

def _apply_groq_key():
    """on_change callback for the Groq API key input: sets it into this
    process's environment (so scorer.py's os.environ.get("GROQ_API_KEY")
    picks it up immediately) and forces an immediate backend re-check so
    the badge above doesn't keep showing red until some later rerun."""
    key = st.session_state.get("groq_api_key_input", "").strip()
    if key:
        os.environ["GROQ_API_KEY"] = key
    else:
        os.environ.pop("GROQ_API_KEY", None)
    check_backend_status(force_refresh=True)


index_ok, index_msg = check_index_status()
ollama_ok, ollama_msg = check_ollama_status()
backend_level, backend_label, backend_msg = check_backend_status()
gpu_level, gpu_msg = check_gpu_vram_status()

status_col1, status_col2, status_col3 = st.columns(3)
with status_col1:
    if index_ok:
        st.success(f"✅ Facet index: {index_msg}")
    else:
        st.error(f"❌ Facet index: {index_msg}")
with status_col2:
    if backend_level == "green":
        st.success(f"✅ LLM Backend: {backend_label} -- {backend_msg}")
    else:
        st.error(f"❌ LLM Backend: {backend_msg}")
with status_col3:
    if gpu_level == "green":
        st.success(f"✅ GPU VRAM: {gpu_msg}")
    elif gpu_level == "yellow":
        st.warning(f"⚠️ GPU VRAM: {gpu_msg}")
    elif gpu_level == "red":
        st.error(f"❌ GPU VRAM: {gpu_msg}")
    else:
        st.info(f"❔ GPU VRAM: {gpu_msg}")

# Only show the Groq key input when Ollama isn't detected -- no point
# asking for a cloud fallback key when the preferred local backend is
# already working fine.
if not ollama_ok:
    st.text_input(
        "GROQ_API_KEY (Ollama not detected -- paste a Groq API key to use the cloud fallback)",
        type="password",
        key="groq_api_key_input",
        value=os.environ.get("GROQ_API_KEY", ""),
        on_change=_apply_groq_key,
        help=(
            "Get a free key at https://console.groq.com/keys. Only kept in this "
            "process's memory for this session -- never written to disk or logged."
        ),
    )

with st.expander("How this works"):
    st.markdown(
        "- **Retrieval, not a full prompt dump**: out of 399 raw facets, only the "
        "facets FAISS ranks as semantically relevant to your conversation (top-40) "
        "ever get sent to the LLM -- never all 399 at once.\n"
        "- **Two-layer safety filter**: medical/biological, spiritual/esoteric, "
        "social/demographic, and malformed facets are excluded from the retrieval "
        "index entirely at audit time (layer 1), *and* the scorer independently "
        "hard-blocks any medical facet that somehow reaches it (layer 2) -- so a "
        "facet like `Basophil count` or `Depression Symptoms` can never be scored "
        "from a casual conversation, by construction, not by asking the model "
        "nicely not to.\n"
        "- **Abstain by default**: for every facet that does reach the LLM, the "
        "prompt explicitly instructs it to return `insufficient_evidence` rather "
        "than guess whenever the conversation doesn't clearly support a score -- "
        "confirmed by benchmark testing, 0 hallucinations across the medical and "
        "spiritual \"trap\" conversations designed to bait it."
    )

st.divider()

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
if "pipeline_result" not in st.session_state:
    st.session_state.pipeline_result = None

conversation = st.text_area(
    "Conversation",
    height=180,
    placeholder=(
        "e.g. \"I quit my stable job last week to backpack solo through South "
        "America with no itinerary and barely any savings left...\""
    ),
)

run_clicked = st.button(
    "Run Pipeline", type="primary", disabled=not index_ok or backend_level != "green"
)

if not index_ok:
    st.info("Build the facet index first (see the error above) before running the pipeline.")
if index_ok and backend_level != "green":
    st.info("No LLM backend available yet -- start Ollama, or paste a Groq API key above.")

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if run_clicked:
    if not conversation or not conversation.strip():
        st.warning("Please paste a conversation first.")
    else:
        with st.spinner(f"Retrieving relevant facets and scoring with {backend_label}..."):
            try:
                from src.pipeline import run_pipeline
                result = run_pipeline(conversation, save_output=True)
                st.session_state.pipeline_result = result
            except FileNotFoundError as e:
                st.session_state.pipeline_result = None
                st.error(
                    "Missing a required file. Make sure you've run "
                    "`python main.py --audit` and `python main.py --embed` first.\n\n"
                    f"Details: {e}"
                )
            except Exception as e:
                st.session_state.pipeline_result = None
                # Covers Ollama connection refused, llama3.1 not pulled, a
                # Groq API error, or any other unexpected failure -- we
                # never want the app itself to crash on a bad LLM call.
                st.error(
                    "Something went wrong while running the pipeline. This "
                    "usually means neither backend is currently reachable -- "
                    "Ollama isn't running/llama3.1 isn't pulled, and no valid "
                    "GROQ_API_KEY is set.\n\n"
                    f"Details: {e}"
                )

# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
result = st.session_state.pipeline_result

if result is not None:
    if result.get("error"):
        st.error(f"Pipeline reported an error: {result['error']}")
    else:
        st.subheader("Summary")

        def _stat_card(label: str, value, bg: str, fg: str = "#ffffff") -> str:
            """Bigger, colored stat tile than the default st.metric gives you.
            Self-contained inline colors (not theme tokens) so this reads
            correctly in both light and dark Streamlit themes without extra
            handling -- it's always a solid colored box with white text."""
            return (
                f'<div style="background-color:{bg};border-radius:10px;padding:16px 12px;'
                f'text-align:center;">'
                f'<div style="font-size:2.2rem;font-weight:700;color:{fg};line-height:1.1;">{value}</div>'
                f'<div style="font-size:0.85rem;color:{fg};opacity:0.9;margin-top:4px;">{label}</div>'
                f'</div>'
            )

        m1, m2, m3, m4 = st.columns(4)
        with m1:
            st.markdown(_stat_card("Facets retrieved", result.get("total_facets_retrieved", 0), "#3a6ea5"), unsafe_allow_html=True)
        with m2:
            st.markdown(_stat_card("Scored", result.get("scored", 0), "#2e7d32"), unsafe_allow_html=True)
        with m3:
            st.markdown(_stat_card("Abstained", result.get("abstained", 0), "#b8860b"), unsafe_allow_html=True)
        with m4:
            st.markdown(_stat_card("Parse errors", result.get("parse_errors", 0), "#c0392b"), unsafe_allow_html=True)

        st.write("")  # small spacer between the stat row and what follows

        results_list = result.get("results", [])

        if not results_list:
            st.info("No facets were retrieved for this conversation -- try a longer or more descriptive one.")
        else:
            df = pd.DataFrame(results_list)
            column_order = [c for c in ["facet", "score", "status", "confidence", "evidence"] if c in df.columns]
            df = df[column_order]

            # Scored facets first, then insufficient_evidence, then
            # not_observable (a structural safety abstain, distinct from
            # "conversation didn't say enough"), then parse_error last --
            # the reader wants the actual answers before the abstentions.
            status_order = {"scored": 0, "insufficient_evidence": 1, "not_observable": 2, "parse_error": 3}
            df["_sort_key"] = df["status"].map(status_order).fillna(4)
            df = df.sort_values("_sort_key", kind="stable").drop(columns="_sort_key").reset_index(drop=True)

            row_colors = {
                "scored": "#e8f5e9",           # light green
                "insufficient_evidence": "#fff8e1",  # light yellow
                "not_observable": "#eceff1",   # light gray-blue
                "parse_error": "#ffebee",      # light red
            }
            status_emoji = {
                "scored": "🟢",
                "insufficient_evidence": "🟡",
                "not_observable": "⚪",
                "parse_error": "🔴",
            }
            df.insert(0, "", df["status"].map(status_emoji).fillna("⚫"))

            def _highlight_row(row):
                color = row_colors.get(row["status"], "")
                return [f"background-color: {color}; color: #1a1a1a;" if color else "" for _ in row]

            styled = df.style.apply(_highlight_row, axis=1)

            st.subheader("Facet scores")
            st.dataframe(styled, use_container_width=True, hide_index=True)

            st.subheader("Outcome distribution")
            status_counts = df["status"].value_counts()
            st.bar_chart(status_counts)

            with st.expander("Raw JSON output"):
                st.json(result)
else:
    st.caption("Results will appear here after you run the pipeline.")

st.divider()
st.markdown(
    '<div style="text-align:center;color:#888;font-size:0.85rem;">'
    "Running llama3.1 8B locally via Ollama | RTX 4050 | "
    "0 hallucinations on medical/clinical facets"
    "</div>",
    unsafe_allow_html=True,
)
