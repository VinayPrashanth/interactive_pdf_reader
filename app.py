import streamlit as st
from dotenv import load_dotenv
from langchain_community.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFLoader
from PyPDF2 import PdfReader, PdfWriter
from tempfile import NamedTemporaryFile
import base64, os, io
from htmlTemplates import expander_css, css, bot_template, user_template

# Process the Input PDF
def process_file(doc):

    model_name = "thenlper/gte-small"
    model_kwargs = {'device': 'cpu'}
    encode_kwargs = {'normalize_embeddings': False}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs, encode_kwargs=encode_kwargs)  
    
    pdfsearch = Chroma.from_documents(doc, embeddings)

    chain = ConversationalRetrievalChain.from_llm(ChatOpenAI(api_key = st.secrets["OPENAI_API_KEY"],temperature=0.3), 
        retriever=pdfsearch.as_retriever(search_kwargs={"k": 2}),
        return_source_documents=True)
    return chain



# Method for Handling User Input
def handle_userinput(query):

    if st.session_state.conversation is not None:

        response = st.session_state.conversation({"question": query, 'chat_history':st.session_state.chat_history}, return_only_outputs=True)
        st.session_state.chat_history += [(query, response['answer'])]

        st.session_state.N = list(response['source_documents'][0])[1][1]['page']
        
        for i, message in enumerate(st.session_state.chat_history):
            st.session_state.expander1.write(user_template.replace("{{MSG}}", message[0]), unsafe_allow_html=True)
            st.session_state.expander1.write(bot_template.replace("{{MSG}}", message[1]), unsafe_allow_html=True)

    else:
        st.warning("Please upload and process a PDF before asking questions.")




def main():
    pass
    
    # Create Web-page Layout
    load_dotenv()
    st.set_page_config(layout="wide", 
                       page_title="Interactive Reader",
                       page_icon=":books:")
    
    st.write(css, unsafe_allow_html=True)
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "N" not in st.session_state:
        st.session_state.N = 0
   
    st.session_state.col1, st.session_state.col2 = st.columns([1, 1])
    st.session_state.col1.header("Interactive Reader :books:")
    user_question = st.session_state.col1.text_input("Ask a question on the contents of the uploaded PDF:")
    st.session_state.expander1 = st.session_state.col1.expander('Your Chat', expanded=True)
    st.session_state.col1.markdown(expander_css, unsafe_allow_html=True) 


    # Load and Process the PDF 
    st.session_state.col1.subheader("Your documents")
    st.session_state.pdf_doc = st.session_state.col1.file_uploader("Upload your PDF here and click on 'Process'")

    if st.session_state.col1.button("Process", key='a'):
        with st.spinner("Processing"):
            if st.session_state.pdf_doc is not None:
                with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                # Write the uploaded bytes to the temp file
                    tmp_file.write(st.session_state.pdf_doc.getvalue())
                    tmp_file_path = tmp_file.name
                try:
                    loader = PyPDFLoader(tmp_file_path)
                    pdf = loader.load()
                    st.session_state.conversation = process_file(pdf)
                    st.session_state.col1.markdown("Done processing. You may now ask a question.")

                finally:
                    if 'loader' in locals():
                        del loader  
        
                    if os.path.exists(tmp_file_path):
                        try:
                            os.remove(tmp_file_path)
                        except PermissionError:
                            # If Windows still locks it, just ignore it. 
                            # The OS will eventually clean up temp files.
                            pass

    
    # Handle Query and Display Pages
    if user_question:
        handle_userinput(user_question)

        if st.session_state.pdf_doc is not None:
        
            # Use BytesIO to read the original PDF from memory
            # We don't need a temp file here.
            reader = PdfReader(io.BytesIO(st.session_state.pdf_doc.getvalue()))
            
            pdf_writer = PdfWriter()
            start = max(st.session_state.N-2, 0)
            end = min(st.session_state.N+2, len(reader.pages)-1) 
            
            while start <= end:
                pdf_writer.add_page(reader.pages[start])
                start+=1
                
            # Use BytesIO for the output PDF chunk as well
            # This prevents the second PermissionError you would encounter later with temp2
            with io.BytesIO() as output_pdf_stream:
                pdf_writer.write(output_pdf_stream)
                base64_pdf = base64.b64encode(output_pdf_stream.getvalue()).decode('utf-8')

                # Embed the PDF
                pdf_display = F'<iframe src="data:application/pdf;base64,{base64_pdf}#page={3}" width="100%" height="900" type="application/pdf" frameborder="0"></iframe>'
            
                st.session_state.col2.markdown(pdf_display, unsafe_allow_html=True)

        else:
            st.info("Please upload a PDF file to begin.")

    
       



if __name__ == '__main__':
    main()