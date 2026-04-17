"""
KnowBot — Streamlit UI
========================
Launch with:
    streamlit run capstone_streamlit.py

Requirements:
    - GROQ_API_KEY set as environment variable
    - All packages from requirements.txt installed

UI flow:
    1. User uploads one or more PDFs in the sidebar
    2. App chunks + embeds them into ChromaDB (shown with a progress bar)
    3. User asks questions in the main chat window
    4. Agent answers with inline citations [Source: X | Page: N]
    5. "New Conversation" button resets thread_id (clears memory)
"""

import streamlit as st
import uuid
from Agent import (
    chunk_pdf,
    build_knowledge_base,
    load_embedder,
    load_llm,
    build_graph,
    ask,
    FAITHFULNESS_THRESHOLD,
)

# ──────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title = "KnowBot — Research Paper Q&A",
    page_icon  = "📚",
    layout     = "wide",
)

# ──────────────────────────────────────────────────────────────────────────────
# EXPENSIVE RESOURCES — loaded once, cached for the entire session
# @st.cache_resource prevents reloading on every Streamlit rerun
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_resource
def get_embedder():
    return load_embedder()

@st.cache_resource
def get_llm():
    return load_llm()

# ──────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALISATION
# Streamlit reruns the entire script on every interaction.
# st.session_state persists values across reruns within one browser session.
# ──────────────────────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []          # chat display history

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())  # unique memory key

if "app" not in st.session_state:
    st.session_state.app = None             # compiled LangGraph app

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False       # True once KB is built

if "uploaded_files" not in st.session_state:
    st.session_state.uploaded_files = []    # names of currently loaded PDFs

# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — PDF UPLOAD + CONTROLS
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📚 KnowBot")
    st.caption("Research Paper Q&A Agent")
    st.divider()

    # ── About section ─────────────────────────────────────────────────────────
    with st.expander("ℹ️ About KnowBot", expanded=False):
        st.markdown("""
**KnowBot** answers questions grounded strictly in your uploaded papers.

**What it can do:**
- Answer questions about paper content and findings
- Extract metadata: authors, year, title, abstract
- Remember context within a conversation session
- Cite exact sources with page numbers
- Admit when information is not in the uploaded papers

**What it won't do:**
- Fabricate information not in your papers
- Answer questions outside the uploaded documents
        """)

    st.divider()

    # ── PDF uploader ──────────────────────────────────────────────────────────
    st.subheader("📄 Upload Papers")
    uploaded = st.file_uploader(
        "Upload one or more PDF files",
        type        = ["pdf"],
        accept_multiple_files = True,
        help        = "Papers are chunked and embedded into a local vector store.",
    )

    if uploaded:
        new_names = sorted([f.name for f in uploaded])

        # only rebuild KB if the set of uploaded files changed
        if new_names != st.session_state.uploaded_files:
            with st.spinner("Building knowledge base..."):
                embedder   = get_embedder()
                all_chunks = []

                progress = st.progress(0)
                for idx, file in enumerate(uploaded):
                    file_bytes = file.read()
                    chunks     = chunk_pdf(file_bytes, file.name)
                    all_chunks.extend(chunks)
                    progress.progress((idx + 1) / len(uploaded))

                if all_chunks:
                    collection = build_knowledge_base(all_chunks, embedder)
                    llm        = get_llm()
                    st.session_state.app            = build_graph(llm, embedder, collection)
                    st.session_state.kb_ready       = True
                    st.session_state.uploaded_files = new_names
                    # reset conversation when new papers are loaded
                    st.session_state.messages  = []
                    st.session_state.thread_id = str(uuid.uuid4())
                    st.success(f"Ready! {len(all_chunks)} chunks from {len(uploaded)} file(s).")
                else:
                    st.error("Could not extract text from the uploaded PDFs.")

    # show currently loaded papers
    if st.session_state.kb_ready:
        st.divider()
        st.subheader("📋 Loaded Papers")
        for name in st.session_state.uploaded_files:
            st.markdown(f"- {name}")

    st.divider()

    # ── New Conversation button ────────────────────────────────────────────────
    # Resets thread_id so MemorySaver starts a fresh memory context.
    # Does NOT reload the KB — papers stay loaded.
    if st.button("🔄 New Conversation", use_container_width=True):
        st.session_state.messages  = []
        st.session_state.thread_id = str(uuid.uuid4())
        st.success("New conversation started!")

    st.caption(f"Session ID: `{st.session_state.thread_id[:8]}...`")

# ──────────────────────────────────────────────────────────────────────────────
# MAIN CHAT AREA
# ──────────────────────────────────────────────────────────────────────────────

st.title("📚 KnowBot — Research Paper Q&A")

if not st.session_state.kb_ready:
    # ── No papers loaded yet — show welcome screen ─────────────────────────────
    st.info("👈 Upload one or more PDF papers in the sidebar to get started.")
    st.markdown("""
### How to use KnowBot

1. **Upload PDFs** using the sidebar uploader
2. Wait for the knowledge base to build (progress bar)
3. **Ask questions** about the paper content
4. KnowBot will answer with **source citations** (filename + page number)

### Example questions to try
- *"What is the main contribution of this paper?"*
- *"What method was used for evaluation?"*
- *"Who are the authors and when was this published?"*
- *"What limitations did the authors mention?"*
- *"How does this compare to previous work?"*
    """)

else:
    # ── Render existing chat history ───────────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            # show metadata (route, faithfulness, sources) for assistant messages
            if msg["role"] == "assistant" and "meta" in msg:
                meta = msg["meta"]
                cols = st.columns([1, 1, 3])
                with cols[0]:
                    route_color = {
                        "retrieve"   : "🟢",
                        "tool"       : "🔵",
                        "memory_only": "🟡",
                    }.get(meta.get("route", ""), "⚪")
                    st.caption(f"{route_color} Route: **{meta.get('route', '—')}**")
                with cols[1]:
                    faith = meta.get("faithfulness", 0.0)
                    faith_color = "🟢" if faith >= FAITHFULNESS_THRESHOLD else "🔴"
                    st.caption(f"{faith_color} Faithfulness: **{faith:.2f}**")
                with cols[2]:
                    sources = meta.get("sources", [])
                    if sources:
                        st.caption(f"📄 Sources: {' · '.join(sources)}")

    # ── Chat input ─────────────────────────────────────────────────────────────
    if prompt := st.chat_input("Ask a question about your uploaded papers..."):

        # display user message immediately
        with st.chat_message("user"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        # run the agent
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                result = ask(
                    st.session_state.app,
                    question  = prompt,
                    thread_id = st.session_state.thread_id,
                )

            answer = result["answer"]
            st.markdown(answer)

            # metadata row below the answer
            cols = st.columns([1, 1, 3])
            with cols[0]:
                route_color = {
                    "retrieve"   : "🟢",
                    "tool"       : "🔵",
                    "memory_only": "🟡",
                }.get(result["route"], "⚪")
                st.caption(f"{route_color} Route: **{result['route']}**")
            with cols[1]:
                faith = result["faithfulness"]
                faith_color = "🟢" if faith >= FAITHFULNESS_THRESHOLD else "🔴"
                st.caption(f"{faith_color} Faithfulness: **{faith:.2f}**")
            with cols[2]:
                if result["sources"]:
                    st.caption(f"📄 Sources: {' · '.join(result['sources'])}")

        # save to display history with metadata for re-rendering
        st.session_state.messages.append({
            "role"   : "assistant",
            "content": answer,
            "meta"   : {
                "route"       : result["route"],
                "faithfulness": result["faithfulness"],
                "sources"     : result["sources"],
            },
        })