from dotenv import load_dotenv
import os
import streamlit as st
from PIL import Image

from qa import build_knowledge_base, answer_question

load_dotenv()
_icon_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Artifacts", "Image Resources", "images.jpeg")
img = Image.open(_icon_path) if os.path.exists(_icon_path) else "📄"
st.set_page_config(page_title="DocGenius: Document Generation AI", page_icon= img)
st.header("Ask Your PDF📄")
pdf = st.file_uploader("Upload your PDF", type="pdf")

if pdf is not None:
    try:
        knowledge_base = build_knowledge_base(pdf)
    except Exception as e:
        st.error(f"Could not process the PDF: {e}")
        st.stop()

    query = st.text_input("Ask your Question about your PDF")
    if query:
        try:
            response, sources = answer_question(knowledge_base, query)
            st.success(response)
            # Show the chunks the answer was based on so the user can check grounding
            with st.expander(f"Sources ({len(sources)} passages)"):
                for doc in sources:
                    st.markdown(f"**Page {doc.metadata.get('page', '?')}**")
                    st.text(doc.page_content)
        except Exception as e:
            st.error(f"Model request failed: {e}")
