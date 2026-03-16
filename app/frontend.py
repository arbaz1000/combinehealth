"""
Streamlit chat UI for the insurance policy chatbot.

Simple mental model: This is the user-facing chat interface.
It sends questions to the FastAPI backend via SSE streaming and displays
tokens as they arrive, with sources shown after the stream completes.

Run: streamlit run app/frontend.py
"""

import json
import os
import streamlit as st
import requests

from app.config import INSURER_NAME, INSURER_SHORT_NAME, INSURER_PORTAL

# ── Config ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"CombineHealth — {INSURER_SHORT_NAME} Policy Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 CombineHealth")
st.caption(f"Ask questions about {INSURER_NAME} commercial medical policies")

# ── Example questions ──────────────────────────────────────────────────
EXAMPLES = [
    f"Is spinal ablation covered under {INSURER_SHORT_NAME} commercial plans?",
    "What are the prior authorization requirements for knee arthroscopy?",
    "What CPT codes are associated with cardiac catheterization?",
    "Is cosmetic rhinoplasty covered?",
]

# Show examples in sidebar (disabled while a response is being generated)
with st.sidebar:
    st.header("Example Questions")
    for example in EXAMPLES:
        if st.button(example, key=example, use_container_width=True, disabled=st.session_state.get("processing", False)):
            st.session_state["prefill_question"] = example

    st.divider()
    st.markdown(
        f"**About:** This chatbot queries {INSURER_SHORT_NAME} commercial medical & drug policies "
        "to help doctors and clinic staff check coverage before billing."
    )
    st.markdown(
        "**Disclaimer:** This tool is for informational purposes only. "
        f"Always verify with {INSURER_PORTAL}."
    )

# ── Chat history ───────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "processing" not in st.session_state:
    st.session_state.processing = False

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📄 Sources"):
                for src in msg["sources"]:
                    url = src.get("source_url", "")
                    name = src.get("policy_name", "Unknown Policy")
                    number = src.get("policy_number", "")
                    if url:
                        st.markdown(f"- [{name} ({number})]({url})")
                    else:
                        st.markdown(f"- {name} ({number})")
        # Show retry button for error messages
        if msg.get("is_error") and msg.get("retry_question"):
            if st.button("🔄 Retry", key=f"retry_{id(msg)}"):
                st.session_state["prefill_question"] = msg["retry_question"]
                # Remove the error message and the user message that caused it
                st.session_state.messages = [
                    m for m in st.session_state.messages
                    if m is not msg
                ]
                st.rerun()


# ── SSE helpers ────────────────────────────────────────────────────────

def parse_sse_events(response: requests.Response):
    """
    Parse SSE events from a streaming requests Response.

    Yields parsed dicts from 'data: {...}' lines.
    Skips empty lines, comments, and non-data lines per SSE spec.
    """
    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: "):]
        try:
            yield json.loads(payload)
        except json.JSONDecodeError:
            continue


# ── Chat input ─────────────────────────────────────────────────────────
# Check if there's a prefilled question from sidebar or retry
prefill = st.session_state.pop("prefill_question", None)
question = prefill or st.chat_input(f"Ask about a {INSURER_SHORT_NAME} policy, procedure, or CPT code...")

if question and question.strip():
    # Display user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Build chat_history for the API (role + content only, no sources/metadata)
    chat_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
        if msg["role"] in ("user", "assistant")
    ]
    # Exclude the message we just appended (it's the current question, not history)
    chat_history = chat_history[:-1]

    # Call streaming API and display response
    with st.chat_message("assistant"):
        st.session_state.processing = True
        try:
            resp = requests.post(
                f"{API_URL}/ask/stream",
                json={"question": question, "chat_history": chat_history},
                stream=True,
                timeout=30,
            )

            # Handle input guardrail rejections (422) — empty/too-long
            if resp.status_code == 422:
                detail = resp.json().get("detail", "Invalid input.")
                st.warning(detail)
                st.session_state.messages.append({"role": "assistant", "content": detail})
                st.session_state.processing = False
                st.rerun()

            resp.raise_for_status()

            # Stream tokens into the UI
            status_placeholder = st.empty()
            answer_placeholder = st.empty()
            accumulated_answer = ""
            sources = []
            first_token = True

            status_placeholder.markdown("*Thinking...*")

            for event in parse_sse_events(resp):
                event_type = event.get("type")

                if event_type == "intent":
                    intent = event.get("intent", "")
                    if intent in ("policy_query", "follow_up"):
                        status_placeholder.markdown("*Searching policies...*")
                    else:
                        status_placeholder.empty()

                elif event_type == "token":
                    if first_token:
                        first_token = False
                        status_placeholder.empty()
                    accumulated_answer += event.get("content", "")
                    answer_placeholder.markdown(accumulated_answer + "▌")

                elif event_type == "sources":
                    sources = event.get("sources", [])

                elif event_type == "done":
                    break

            status_placeholder.empty()

            # Final render without cursor
            answer_placeholder.markdown(accumulated_answer)

            # Display sources in expander
            if sources:
                with st.expander("📄 Sources"):
                    for src in sources:
                        url = src.get("source_url", "")
                        name = src.get("policy_name", "Unknown Policy")
                        number = src.get("policy_number", "")
                        if url:
                            st.markdown(f"- [{name} ({number})]({url})")
                        else:
                            st.markdown(f"- {name} ({number})")

            st.session_state.messages.append({
                "role": "assistant",
                "content": accumulated_answer,
                "sources": sources,
            })

        except requests.exceptions.ConnectionError:
            err = "Something went wrong. Please try again."
            st.error(err)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err,
                "is_error": True,
                "retry_question": question,
            })
        except Exception as e:
            err = "Something went wrong. Please try again."
            st.error(err)
            st.session_state.messages.append({
                "role": "assistant",
                "content": err,
                "is_error": True,
                "retry_question": question,
            })
        finally:
            st.session_state.processing = False