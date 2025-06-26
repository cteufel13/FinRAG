

# RAG for Document Analysis and Credit Risk Assessment

This repository provides a practical introduction to Retrieval-Augmented Generation (RAG) and demonstrates how RAG techniques can be applied to real-world use cases, including credit risk assessment.

## 📁 Repository Structure

```

├── data/                       # Collection of input documents
│   ├── credit reports/         # Credit risk-related text files
│   ├── embedding papers/       # Research papers used for embeddings
│   ├── EPA\_text/              # Environmental Protection Agency text
│   ├── unstructured\_samples/  # Samples for PDF Loaders
│   └── world\_bank\_doc/       # World Bank PDF documents
│       └── BOSIB13...pdf
├── docs/                     # Contains presentation
├── notebooks/               
│   ├── RAG_Simplified.ipynb      # Intro to RAG, tools and examples
│   └── Credit_Risk_RAG.ipynb    # RAG applied to credit risk
├── .gitignore
├── LICENSE
└── README.md                 

````

## 📓 Notebooks

### 1. `RAG_Simplified.ipynb`
A beginner-friendly introduction to RAG. This notebook covers:
- Core concepts behind RAG
- Tools used for chunking, embedding, vector storage, and retrieval
- Hands-on examples demonstrating RAG functionality

### 2. `Credit_Risk_RAG.ipynb`
An application-driven notebook showing how RAG can be used for credit risk analysis. This builds upon the tools from the introductory notebook and from the previous tasks

## 📊 Documentation

### `docs/`
Contains the introductory presentation into RAG



## 🧠 What is RAG?

RAG (Retrieval-Augmented Generation) is a hybrid approach that combines information retrieval with text generation. It enhances language models by retrieving relevant documents and incorporating them into the response generation process—ideal for tasks involving large corpora or domain-specific knowledge.

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

Let me know if you'd like me to include images, a usage demo, or installation instructions with `requirements.txt`.
