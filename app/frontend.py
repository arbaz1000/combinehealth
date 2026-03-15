"""
Streamlit chat UI for the insurance policy chatbot.

Simple mental model: This is the user-facing chat interface.
It sends questions to the FastAPI backend and displays answers with sources.

Run: streamlit run app/frontend.py
"""

import os
import streamlit as st
import requests

# ── Config ─────────────────────────────────────────────────────────────
API_URL = os.getenv("API_URL", "http://localhost:8000")

# ── Page setup ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="CombineHealth — UHC Policy Assistant",
    page_icon="🏥",
    layout="centered",
)

st.title("🏥 CombineHealth")
st.caption("Ask questions about UnitedHealthcare commercial medical policies")

# ── Example questions ──────────────────────────────────────────────────
EXAMPLES = [
    "Is spinal ablation covered under UHC commercial plans?",
    "What are the prior authorization requirements for knee arthroscopy?",
    "What CPT codes are associated with cardiac catheterization?",
    "Is cosmetic rhinoplasty covered?",
]

# Show examples in sidebar (disabled while a response is being generated)
with st.sidebar:
    st.header("Example Questions")
    for example in EXAMPLES:
        if st.button(example, key=example, use_container_width=True, disabled=st.session_state.processing):
            st.session_state["prefill_question"] = example

    st.divider()
    st.markdown(
        "**About:** This chatbot queries UHC commercial medical & drug policies "
        "to help doctors and clinic staff check coverage before billing."
    )
    st.markdown(
        "**Disclaimer:** This tool is for informational purposes only. "
        "Always verify with the official UHC provider portal."
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

# ── Chat input ─────────────────────────────────────────────────────────
# Check if there's a prefilled question from sidebar
prefill = st.session_state.pop("prefill_question", None)
question = prefill or st.chat_input("Ask about a UHC policy, procedure, or CPT code...")

if question:
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

    # Call API and display response
    with st.chat_message("assistant"):
        st.session_state.processing = True
        with st.spinner("Searching policies..."):
            try:
                resp = requests.post(
                    f"{API_URL}/ask",
                    json={"question": question, "chat_history": chat_history},
                    timeout=30,
                )

                # Handle input guardrail rejections (422)
                if resp.status_code == 422:
                    detail = resp.json().get("detail", "Invalid input.")
                    st.warning(detail)
                    st.session_state.messages.append({"role": "assistant", "content": detail})
                    st.session_state.processing = False
                    st.rerun()

                resp.raise_for_status()
                data = resp.json()

                answer = data["answer"]
                sources = data.get("sources", [])

                st.markdown(answer)

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
                    "content": answer,
                    "sources": sources,
                })

            except requests.exceptions.ConnectionError:
                err = "⚠️ Cannot connect to the API server. Make sure the backend is running: `uvicorn app.api:app`"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            except Exception as e:
                err = f"⚠️ Error: {str(e)}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err})
            finally:
                st.session_state.processing = False
