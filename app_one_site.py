import streamlit as st
import nbformat
from nbconvert import PythonExporter
import subprocess

# Load and convert the notebook to Python code
with open('/mnt/data/one_site_analysis.ipynb', 'r') as f:
    nb = nbformat.read(f, as_version=4)
    exporter = PythonExporter()
    (script, _) = exporter.from_notebook_node(nb)

exec(script)

def main():
    st.title("One Site Analysis")
    
    # Input field for the user to manually enter the link
    link = st.text_input("Enter the link to analyze:")

    if link:
        # Assuming the notebook script defines a function named analyze(link)
        try:
            # Replace "analyze(link)" with the specific function and variable name used in your notebook.
            result = analyze(link)
            st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()