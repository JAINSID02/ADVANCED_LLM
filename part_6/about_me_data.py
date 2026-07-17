"""
about_me_data.py (v2, expanded) — a large Q&A dataset built from Sidharth's
resume (6.x style, matches chitchat_data.py). Compared to v1, this version:
  1. Asks each fact 3-5 different ways, including casual/lowercase/no-punctuation
     phrasing (real chat input is messy — the model needs to see that at train time).
  2. Adds compound questions that span multiple projects/skills.
  3. Adds an "out of scope" set teaching the model to gracefully decline
     questions that have nothing to do with Sidharth, instead of hallucinating
     (this is the fix for the earlier "what is dna" -> nonsense failure mode).

Source: SIDHARTH_RESUME_LATEST_VERSION.pdf (as of July 2026).
If any resume detail changes, update the constants / pairs below and retrain.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

NAME = "Sidharth"


@dataclass
class SFTItem:
    prompt: str
    response: str


def _pairs() -> List[Tuple[str, str]]:
    pairs = [
        # ================= WHO / IDENTITY / SUMMARY =================
        ("Who is Sidharth?",
         "Sidharth is a B.Tech Computer Science student at Jaypee University of Information Technology and an IBM Certified Generative AI Engineer, with hands-on experience building end-to-end Generative AI systems including agentic AI pipelines, Self-RAG architectures, and multi-agent research automation."),
        ("who is sidharth", "Sidharth is a B.Tech Computer Science student at Jaypee University of Information Technology and an IBM Certified Generative AI Engineer, experienced in agentic AI, Self-RAG, and multi-agent systems."),
        ("tell me about sidharth", "Sidharth is a Computer Science student at JUIT with an IBM Generative AI Engineering Professional Certificate. He builds Generative AI systems such as agentic pipelines, Self-RAG architectures, and multi-agent research tools, and is proficient in Python, LangChain, LangGraph, Mistral AI, and RAG pipelines."),
        ("Tell me about Sidharth.", "Sidharth is a Computer Science student at JUIT with an IBM Generative AI Engineering Professional Certificate. He builds Generative AI systems such as agentic pipelines, Self-RAG architectures, and multi-agent research tools, and is proficient in Python, LangChain, LangGraph, Mistral AI, and RAG pipelines."),
        ("whats sidharth about", "Sidharth is a Computer Science student and IBM Certified Generative AI Engineer who builds end-to-end Generative AI systems like agentic pipelines and Self-RAG architectures."),
        ("give me a quick intro to sidharth", "Sidharth is a B.Tech CS student at JUIT and an IBM Certified Generative AI Engineer, focused on building agentic AI, Self-RAG, and multi-agent research systems."),
        ("What does Sidharth do?",
         "Sidharth designs and builds end-to-end Generative AI systems, including agentic AI pipelines, Self-RAG document Q&A systems, and multi-agent research automation, using tools like LangChain, LangGraph, and Mistral AI."),
        ("what does sidharth do", "Sidharth designs and builds end-to-end Generative AI systems, including agentic AI pipelines, Self-RAG document Q&A, and multi-agent research automation."),
        ("Give me a summary of Sidharth's background.",
         "Sidharth is a Computer Science undergraduate at Jaypee University of Information Technology and an IBM Certified Generative AI Engineer. He has hands-on experience in agentic AI, Self-RAG, RAG pipelines, and transformer fine-tuning with LoRA and PEFT, and is currently building production-grade agentic AI systems."),
        ("give me sidharths background", "Sidharth is a Computer Science undergraduate at JUIT and an IBM Certified Generative AI Engineer, with hands-on experience in agentic AI, Self-RAG, RAG pipelines, and transformer fine-tuning."),
        ("What is Sidharth currently working on?",
         "Sidharth is currently building production-grade agentic AI systems and deepening his expertise in LLM fine-tuning and scalable AI deployment."),
        ("what is sidharth working on right now", "Sidharth is currently building production-grade agentic AI systems and deepening his expertise in LLM fine-tuning and scalable AI deployment."),
        ("Is Sidharth a student?",
         "Yes, Sidharth is a B.Tech Computer Science student at Jaypee University of Information Technology, expected to graduate in 2027."),
        ("is sidharth still in college", "Yes, Sidharth is a B.Tech Computer Science student at Jaypee University of Information Technology, expected to graduate in 2027."),

        # ================= CONTACT / LOCATION =================
        ("What is Sidharth's email address?", "Sidharth's email is jisidharthjain@gmail.com."),
        ("whats sidharths email", "Sidharth's email is jisidharthjain@gmail.com."),
        ("how do i email sidharth", "You can email Sidharth at jisidharthjain@gmail.com."),
        ("How can I contact Sidharth?",
         "You can reach Sidharth by email at jisidharthjain@gmail.com, by phone at +91 9468168946, or via his LinkedIn at linkedin.com/in/jisidharthjain."),
        ("how do i reach out to sidharth", "You can reach Sidharth by email at jisidharthjain@gmail.com, by phone at +91 9468168946, or via LinkedIn at linkedin.com/in/jisidharthjain."),
        ("What is Sidharth's phone number?", "Sidharth's phone number is +91 9468168946."),
        ("whats sidharths number", "Sidharth's phone number is +91 9468168946."),
        ("What is Sidharth's LinkedIn?", "Sidharth's LinkedIn profile is linkedin.com/in/jisidharthjain."),
        ("sidharths linkedin?", "Sidharth's LinkedIn profile is linkedin.com/in/jisidharthjain."),
        ("does sidharth have linkedin", "Yes, Sidharth's LinkedIn is linkedin.com/in/jisidharthjain."),
        ("What is Sidharth's GitHub?", "Sidharth's GitHub handle is github.com/JAINSID02."),
        ("sidharths github username", "Sidharth's GitHub handle is JAINSID02, at github.com/JAINSID02."),
        ("does sidharth have a github", "Yes, Sidharth's GitHub is github.com/JAINSID02."),
        ("Where is Sidharth located?",
         "Sidharth is based in Paonta Sahib, Sirmaur, Himachal Pradesh, India, and is open to relocation."),
        ("where does sidharth live", "Sidharth is based in Paonta Sahib, Sirmaur, Himachal Pradesh, India."),
        ("Is Sidharth open to relocating?", "Yes, Sidharth is open to relocation."),
        ("will sidharth relocate for a job", "Yes, Sidharth is open to relocation."),
        ("Where is Sidharth from?", "Sidharth is from Paonta Sahib, Sirmaur, in Himachal Pradesh, India."),
        ("what city is sidharth from", "Sidharth is from Paonta Sahib, Sirmaur, in Himachal Pradesh, India."),

        # ================= EDUCATION =================
        ("Where does Sidharth study?", "Sidharth studies at Jaypee University of Information Technology (JUIT) in Solan, Himachal Pradesh."),
        ("what college does sidharth go to", "Sidharth studies at Jaypee University of Information Technology (JUIT) in Solan, Himachal Pradesh."),
        ("which university is sidharth in", "Sidharth attends Jaypee University of Information Technology, located in Solan, Himachal Pradesh."),
        ("What is Sidharth's degree?", "Sidharth is pursuing a Bachelor of Technology in Computer Science and Engineering."),
        ("what is sidharth studying", "Sidharth is pursuing a Bachelor of Technology in Computer Science and Engineering."),
        ("is sidharth a cs major", "Yes, Sidharth is pursuing a Bachelor of Technology in Computer Science and Engineering."),
        ("When will Sidharth graduate?", "Sidharth is expected to graduate in July 2027."),
        ("when does sidharth graduate", "Sidharth is expected to graduate in July 2027."),
        ("what year will sidharth finish college", "Sidharth is expected to graduate in July 2027."),
        ("What is Sidharth's CGPA?", "Sidharth's CGPA is 7.28 out of 10."),
        ("whats sidharths cgpa", "Sidharth's CGPA is 7.28 out of 10."),
        ("What is Sidharth's GPA?", "Sidharth's GPA (CGPA) is 7.28 out of 10."),
        ("how good are sidharths grades", "Sidharth's CGPA is 7.28 out of 10."),
        ("sidharths marks in college", "Sidharth's CGPA is 7.28 out of 10."),
        ("What year is Sidharth in college?", "Sidharth started his B.Tech in August 2023 and is expected to finish in July 2027."),
        ("when did sidharth start college", "Sidharth started his B.Tech in August 2023."),
        ("What subjects has Sidharth studied?",
         "Sidharth's relevant coursework includes Machine Learning, Deep Learning, Data Structures & Algorithms, Database Management Systems, Operating Systems, and Computer Networks."),
        ("what courses has sidharth taken", "Sidharth's relevant coursework includes Machine Learning, Deep Learning, Data Structures & Algorithms, Database Management Systems, Operating Systems, and Computer Networks."),
        ("has sidharth studied operating systems", "Yes, Operating Systems is part of Sidharth's relevant coursework, along with Computer Networks, DBMS, and Data Structures & Algorithms."),
        ("What did Sidharth study in school before college?",
         "Sidharth completed his Senior Secondary (Class XII) in the Science stream and his Secondary (Class X) education at Guru Nanak Mission Public School in Sirmaur, Himachal Pradesh."),
        ("where did sidharth go to school", "Sidharth attended Guru Nanak Mission Public School in Sirmaur, Himachal Pradesh, for both Class X and Class XII."),
        ("When did Sidharth finish high school?", "Sidharth completed his Senior Secondary (Class XII) in May 2023."),
        ("what stream did sidharth take in school", "Sidharth studied the Science stream in his Senior Secondary (Class XII)."),

        # ================= CERTIFICATION =================
        ("Does Sidharth have any certifications?",
         "Yes, Sidharth holds the IBM Generative AI Engineering Professional Certificate, a 16-course program from IBM via Coursera, completed in July 2026."),
        ("does sidharth have any certs", "Yes, Sidharth holds the IBM Generative AI Engineering Professional Certificate, completed in July 2026."),
        ("is sidharth certified in anything", "Yes, Sidharth is an IBM Certified Generative AI Engineer, having completed IBM's 16-course Generative AI Engineering Professional Certificate."),
        ("What is Sidharth's IBM certification?",
         "Sidharth completed the IBM Generative AI Engineering Professional Certificate, covering LLM architecture, transformer fine-tuning, PEFT/LoRA, AI agents with RAG and LangChain, and building Generative AI applications with Python."),
        ("tell me about sidharths ibm cert", "Sidharth completed the IBM Generative AI Engineering Professional Certificate, a 16-course program covering LLM architecture, fine-tuning, PEFT/LoRA, and AI agents with RAG and LangChain."),
        ("Is Sidharth IBM certified?", "Yes, Sidharth is an IBM Certified Generative AI Engineer, having completed IBM's 16-course Generative AI Engineering Professional Certificate in July 2026."),
        ("when did sidharth get ibm certified", "Sidharth completed his IBM Generative AI Engineering Professional Certificate in July 2026."),
        ("What topics does Sidharth's certification cover?",
         "The certificate covers LLM architecture and data preparation, language modeling with transformers, fine-tuning and PEFT/LoRA, advanced LLM fine-tuning, AI agents with RAG and LangChain, deep learning with Keras, and building Generative AI applications with Python."),
        ("how many courses in sidharths ibm certificate", "Sidharth's IBM Generative AI Engineering Professional Certificate is a 16-course program."),

        # ================= SKILLS: LANGUAGES / FRAMEWORKS =================
        ("What programming languages does Sidharth know?", "Sidharth is proficient in Python and SQL."),
        ("what languages does sidharth code in", "Sidharth is proficient in Python and SQL."),
        ("does sidharth know python", "Yes, Python is one of Sidharth's core programming languages."),
        ("does sidharth know sql", "Yes, SQL is one of Sidharth's programming languages."),
        ("What frameworks does Sidharth use?", "Sidharth works with PyTorch, TensorFlow, Keras, and Scikit-learn."),
        ("what ml frameworks does sidharth use", "Sidharth works with PyTorch, TensorFlow, Keras, and Scikit-learn."),
        ("Does Sidharth know PyTorch?", "Yes, Sidharth is experienced with PyTorch as one of his core deep learning frameworks."),
        ("Does Sidharth know TensorFlow?", "Yes, Sidharth has experience with TensorFlow and Keras alongside PyTorch."),
        ("does sidharth use scikit-learn", "Yes, Scikit-learn is one of Sidharth's ML frameworks."),

        # ================= SKILLS: ML/AI CONCEPTS =================
        ("What machine learning concepts does Sidharth know?",
         "Sidharth has experience with Machine Learning, Deep Learning, Transfer Learning, and Neural Networks."),
        ("what ml concepts does sidharth know", "Sidharth has experience with Machine Learning, Deep Learning, Transfer Learning, and Neural Networks."),
        ("Does Sidharth understand deep learning?", "Yes, Deep Learning is one of Sidharth's core ML & AI skill areas, alongside Transfer Learning and Neural Networks."),
        ("does sidharth know about neural networks", "Yes, Neural Networks is one of Sidharth's core ML & AI skill areas."),
        ("does sidharth know transfer learning", "Yes, Transfer Learning is one of Sidharth's ML & AI skills."),

        # ================= SKILLS: LLMs / TRANSFORMERS =================
        ("What does Sidharth know about transformers?",
         "Sidharth has experience with Transformers, attention mechanisms, and encoder-decoder models, and works with the Hugging Face Transformers and Datasets libraries along with prompt engineering."),
        ("what does sidharth know about llms", "Sidharth has experience with Transformers, attention mechanisms, encoder-decoder models, and prompt engineering, and works with the Hugging Face Transformers and Datasets libraries."),
        ("Does Sidharth know about attention mechanisms?", "Yes, attention mechanisms are one of Sidharth's core areas of LLM and transformer knowledge."),
        ("Does Sidharth use Hugging Face?", "Yes, Sidharth works with the Hugging Face Transformers and Datasets libraries."),
        ("does sidharth use huggingface", "Yes, Sidharth works with the Hugging Face Transformers and Datasets libraries."),
        ("Does Sidharth do prompt engineering?", "Yes, prompt engineering is part of Sidharth's LLM and transformer skill set."),
        ("is sidharth good at prompt engineering", "Yes, prompt engineering is part of Sidharth's LLM and transformer skill set."),

        # ================= SKILLS: FINE-TUNING / ALIGNMENT =================
        ("What fine-tuning techniques does Sidharth know?",
         "Sidharth has experience with LoRA, QLoRA, PEFT, RLHF, instruction tuning, and supervised fine-tuning (SFT)."),
        ("what fine tuning methods does sidharth know", "Sidharth has experience with LoRA, QLoRA, PEFT, RLHF, instruction tuning, and supervised fine-tuning (SFT)."),
        ("Does Sidharth know LoRA?", "Yes, Sidharth has hands-on experience with LoRA, QLoRA, and PEFT for parameter-efficient fine-tuning."),
        ("does sidharth know qlora", "Yes, Sidharth has hands-on experience with QLoRA, along with LoRA and PEFT."),
        ("Has Sidharth worked with RLHF?", "Yes, RLHF (Reinforcement Learning from Human Feedback) is one of Sidharth's fine-tuning and alignment skills."),
        ("does sidharth know rlhf", "Yes, RLHF is one of Sidharth's fine-tuning and alignment skills."),
        ("What is Sidharth's experience with supervised fine-tuning?",
         "Sidharth has hands-on experience with supervised fine-tuning (SFT) as part of his LLM fine-tuning and alignment skill set."),
        ("has sidharth done instruction tuning", "Yes, instruction tuning is part of Sidharth's fine-tuning and alignment skill set."),

        # ================= SKILLS: GENERATIVE / AGENTIC AI =================
        ("What agentic AI tools does Sidharth use?",
         "Sidharth works with LangChain, LangGraph, and LangSmith to build agentic AI systems, along with Retrieval-Augmented Generation (RAG)."),
        ("what agentic ai tools does sidharth know", "Sidharth works with LangChain, LangGraph, and LangSmith to build agentic AI systems, along with RAG."),
        ("Does Sidharth know LangChain?", "Yes, LangChain is one of Sidharth's core tools for building Generative AI and agentic AI systems."),
        ("does sidharth use langchain", "Yes, LangChain is one of Sidharth's core tools for building Generative AI and agentic AI systems."),
        ("Does Sidharth know LangGraph?", "Yes, Sidharth uses LangGraph for building stateful, conditional agentic workflows."),
        ("does sidharth use langgraph", "Yes, Sidharth uses LangGraph for building stateful, conditional agentic workflows."),
        ("does sidharth use langsmith", "Yes, LangSmith is one of Sidharth's Generative AI and agentic AI tools."),
        ("What is RAG in the context of Sidharth's skills?",
         "Retrieval-Augmented Generation (RAG) is one of Sidharth's core Generative AI skills, which he has applied in projects like his Self-RAG AI Agent and AI Video Assistant."),
        ("does sidharth know rag", "Yes, Retrieval-Augmented Generation (RAG) is one of Sidharth's core Generative AI skills, applied in his Self-RAG AI Agent and AI Video Assistant projects."),

        # ================= SKILLS: DATA SCIENCE / TOOLS =================
        ("What data science tools does Sidharth use?", "Sidharth uses NumPy, Pandas, Matplotlib, Seaborn, and Optuna for data science work."),
        ("what data tools does sidharth know", "Sidharth uses NumPy, Pandas, Matplotlib, Seaborn, and Optuna for data science work."),
        ("does sidharth know pandas", "Yes, Pandas is one of Sidharth's data science tools."),
        ("does sidharth use optuna", "Yes, Optuna is one of Sidharth's data science tools, used for hyperparameter optimization."),
        ("What DevOps or tooling does Sidharth use?", "Sidharth works with Docker, Git, GitHub, Streamlit, FastAPI, and Pydantic."),
        ("what tools does sidharth use for dev work", "Sidharth works with Docker, Git, GitHub, Streamlit, FastAPI, and Pydantic."),
        ("Does Sidharth know Docker?", "Yes, Docker is part of Sidharth's tools and DevOps skill set."),
        ("does sidharth know fastapi", "Yes, FastAPI is part of Sidharth's tools and DevOps skill set."),
        ("does sidharth use streamlit", "Yes, Streamlit is one of Sidharth's tools, used to build frontends for his AI projects."),
        ("What databases does Sidharth work with?", "Sidharth works with vector databases, specifically ChromaDB and FAISS."),
        ("what dbs does sidharth know", "Sidharth works with vector databases, specifically ChromaDB and FAISS."),
        ("Does Sidharth know FAISS?", "Yes, Sidharth uses FAISS as a vector database, notably in his Self-RAG AI Agent project."),
        ("does sidharth use chromadb", "Yes, Sidharth uses ChromaDB as a vector database, notably in his AI Video Assistant project."),

        # ================= PROJECTS: OVERVIEW =================
        ("What projects has Sidharth built?",
         "Sidharth has built three main projects: a Self-RAG AI Agent for document Q&A, MAARS (a multi-agent research system), and an AI Video Assistant for transcription and video Q&A."),
        ("what projects has sidharth made", "Sidharth has built three main projects: a Self-RAG AI Agent, MAARS (a multi-agent research system), and an AI Video Assistant."),
        ("list sidharths projects", "Sidharth has built three main projects: a Self-RAG AI Agent for document Q&A, MAARS (a multi-agent research system), and an AI Video Assistant for transcription and video Q&A."),
        ("What is Sidharth's most recent project?",
         "Sidharth's recent projects include the Self-RAG AI Agent, MAARS (Multi-Agent AI Research System), and the AI Video Assistant, all built in 2026."),
        ("how many projects does sidharth have", "Sidharth has three main projects listed: the Self-RAG AI Agent, MAARS, and the AI Video Assistant."),
        ("which of sidharths projects use langchain", "All three of Sidharth's main projects use LangChain: the Self-RAG AI Agent, MAARS, and the AI Video Assistant."),
        ("which projects does sidharth use mistral ai in", "Sidharth uses Mistral AI in all three of his main projects: the Self-RAG AI Agent, MAARS, and the AI Video Assistant."),

        # ================= SELF-RAG AI AGENT =================
        ("What is Sidharth's Self-RAG AI Agent project?",
         "It's a Self-RAG inspired document Q&A system that dynamically chooses between the LLM's internal reasoning and semantic PDF retrieval, using conditional LangGraph stateful workflows."),
        ("tell me about the self rag project", "It's a Self-RAG inspired document Q&A system that dynamically chooses between the LLM's internal reasoning and semantic PDF retrieval, using conditional LangGraph stateful workflows."),
        ("what does the self rag agent do", "It dynamically chooses between the LLM's internal reasoning and semantic PDF retrieval to answer questions, using conditional LangGraph stateful workflows, and self-corrects to reduce hallucinations."),
        ("What technologies did Sidharth use for the Self-RAG project?",
         "Sidharth used Python, LangGraph, LangChain, Mistral AI, and FAISS to build the Self-RAG AI Agent."),
        ("what tech stack is the self rag agent built with", "Sidharth used Python, LangGraph, LangChain, Mistral AI, and FAISS to build the Self-RAG AI Agent."),
        ("How does Sidharth's Self-RAG agent reduce hallucinations?",
         "It uses LLM-as-a-Judge grading nodes for retrieval relevance, hallucination detection, and usefulness scoring, plus an iterative self-correction loop that revises and re-retrieves until a verified, context-grounded answer is produced."),
        ("how does the self rag agent avoid hallucinating", "It uses LLM-as-a-Judge grading nodes for retrieval relevance, hallucination detection, and usefulness scoring, plus an iterative self-correction loop until a verified answer is produced."),
        ("Does Sidharth's Self-RAG project use a vector database?",
         "Yes, it uses a FAISS vector store along with Mistral Embeddings and recursive text chunking for semantic search across multi-document PDF knowledge bases."),
        ("what vector db does the self rag agent use", "It uses a FAISS vector store along with Mistral Embeddings and recursive text chunking for semantic search across multi-document PDF knowledge bases."),
        ("Where can I find Sidharth's Self-RAG project on GitHub?", "It's available at github.com/JAINSID02/self_rag."),
        ("github link for self rag project", "It's available at github.com/JAINSID02/self_rag."),

        # ================= MAARS =================
        ("What is MAARS?",
         "MAARS (Multi-Agent AI Research System) is a fully autonomous multi-agent research pipeline built by Sidharth, orchestrated via LangChain, that conducts end-to-end research on any topic from a single user query."),
        ("what is maars", "MAARS (Multi-Agent AI Research System) is a fully autonomous multi-agent research pipeline built by Sidharth that conducts end-to-end research on any topic from a single user query."),
        ("tell me about maars", "MAARS is a fully autonomous multi-agent research pipeline orchestrated via LangChain that conducts end-to-end research on any topic from a single user query, ending in a polished, source-cited report."),
        ("What agents does MAARS use?",
         "MAARS includes a Web Search Agent using the Tavily API for real-time information retrieval, a Web Reader Agent using BeautifulSoup to parse article content, a Research Writer module to synthesize findings into structured reports, and a Critic Agent that scores output quality and triggers refinement."),
        ("what agents are in maars", "MAARS includes a Web Search Agent (Tavily API), a Web Reader Agent (BeautifulSoup), a Research Writer module, and a Critic Agent that scores quality and triggers refinement."),
        ("What technologies power MAARS?", "MAARS is built with Python, LangChain, Mistral AI, Tavily, and BeautifulSoup."),
        ("what is maars built with", "MAARS is built with Python, LangChain, Mistral AI, Tavily, and BeautifulSoup."),
        ("What does MAARS output?",
         "MAARS produces a polished, source-cited structured research report generated autonomously, compressing hours of manual research into minutes."),
        ("what kind of report does maars generate", "MAARS produces a polished, source-cited structured research report generated autonomously, compressing hours of manual research into minutes."),
        ("Where can I find MAARS on GitHub?", "It's available at github.com/JAINSID02/MULTI-AGENT-AI-RESEARCH-SYSTEM-MAARS."),
        ("github link for maars", "It's available at github.com/JAINSID02/MULTI-AGENT-AI-RESEARCH-SYSTEM-MAARS."),
        ("Does MAARS have a frontend?", "Yes, Sidharth built a professional Streamlit frontend for MAARS with a black-and-white editorial aesthetic."),
        ("does maars have a ui", "Yes, Sidharth built a professional Streamlit frontend for MAARS with a black-and-white editorial aesthetic."),

        # ================= AI VIDEO ASSISTANT =================
        ("What is Sidharth's AI Video Assistant?",
         "It's an end-to-end AI assistant that processes uploaded video files or YouTube URLs, transcribes audio using OpenAI Whisper, and lets users query the video content conversationally through a RAG-based Q&A system."),
        ("what is the ai video assistant", "It's an end-to-end AI assistant that processes uploaded video files or YouTube URLs, transcribes audio using OpenAI Whisper, and lets users query the video content conversationally."),
        ("tell me about the video assistant project", "It's an AI assistant that transcribes videos with Whisper and lets users query the content conversationally through a transcript-aware RAG Q&A system."),
        ("What technologies did Sidharth use for the AI Video Assistant?",
         "Sidharth used Python, Whisper, LangChain, Mistral AI, ChromaDB, and Streamlit."),
        ("what tech is the video assistant built with", "Sidharth used Python, Whisper, LangChain, Mistral AI, ChromaDB, and Streamlit to build the AI Video Assistant."),
        ("How does the AI Video Assistant handle long videos?", "It uses automatic audio segmentation for long-form content during Whisper transcription."),
        ("how does it transcribe long videos", "It uses automatic audio segmentation for long-form content during Whisper transcription."),
        ("What does the AI Video Assistant do after transcription?",
         "It uses Mistral AI to automatically generate a video title, summary, and key thematic aspects, delivering instant structured content insights."),
        ("what happens after the video gets transcribed", "It uses Mistral AI to automatically generate a video title, summary, and key thematic aspects, delivering instant structured content insights."),
        ("How does the Q&A system in the AI Video Assistant work?",
         "It's a transcript-aware RAG Q&A system built with LangChain, a ChromaDB vector store, and MiniLM embeddings with an overlapping chunk strategy for accurate context retrieval across extended videos."),
        ("how does the video qa system work", "It's a transcript-aware RAG Q&A system built with LangChain, a ChromaDB vector store, and MiniLM embeddings with an overlapping chunk strategy."),
        ("Where can I find the AI Video Assistant on GitHub?", "It's available at github.com/JAINSID02/AI-VIDEO-ASSISTANT."),
        ("github link for the video assistant", "It's available at github.com/JAINSID02/AI-VIDEO-ASSISTANT."),
        ("Does the AI Video Assistant support YouTube links?",
         "Yes, users can either upload a video file or paste a YouTube URL, and query the content conversationally through an interactive Streamlit frontend."),
        ("can it process youtube videos", "Yes, users can either upload a video file or paste a YouTube URL, and query the content conversationally through an interactive Streamlit frontend."),

        # ================= INTERESTS / GOALS =================
        ("What is Sidharth interested in?",
         "Sidharth's areas of interest include Large Language Models, Agentic AI & Multi-Agent Systems, Retrieval-Augmented Generation, LLM Fine-Tuning & Alignment, Generative AI & Foundation Models, and MLOps & AI System Deployment."),
        ("what is sidharth interested in", "Sidharth's areas of interest include Large Language Models, Agentic AI & Multi-Agent Systems, RAG, LLM Fine-Tuning & Alignment, Generative AI & Foundation Models, and MLOps & AI System Deployment."),
        ("What kind of roles is Sidharth looking for?", "Sidharth is targeting Generative AI Engineer and AI/ML roles as a fresher."),
        ("what job is sidharth looking for", "Sidharth is targeting Generative AI Engineer and AI/ML roles as a fresher."),
        ("Is Sidharth looking for a job?", "Yes, Sidharth is actively seeking Generative AI Engineer and AI/ML roles as he approaches graduation in 2027."),
        ("is sidharth looking for work", "Yes, Sidharth is actively seeking Generative AI Engineer and AI/ML roles as he approaches graduation in 2027."),
        ("What is Sidharth's career goal?",
         "Sidharth is focused on building production-grade Generative AI and agentic AI systems, and is deepening his expertise in LLM fine-tuning and scalable AI deployment as he pursues Generative AI Engineer roles."),
        ("what are sidharths career goals", "Sidharth is focused on building production-grade Generative AI and agentic AI systems, and is deepening his expertise in LLM fine-tuning and scalable AI deployment as he pursues Generative AI Engineer roles."),

        # ================= OUT OF SCOPE / FALLBACK =================
        # These teach the model to gracefully decline instead of hallucinating
        # when asked something that has nothing to do with Sidharth.
        ("What is DNA?", "I'm built to answer questions about Sidharth — his background, education, skills, and projects. I don't have general knowledge outside of that."),
        ("What is the capital of France?", "I'm built to answer questions about Sidharth — his background, education, skills, and projects. I don't have general world knowledge like that."),
        ("Write a poem about the ocean.", "I'm designed to answer questions about Sidharth rather than write creative content. Ask me about his projects, skills, or education instead!"),
        ("What's 2 plus 2?", "I'm focused on answering questions about Sidharth rather than general knowledge or math. Feel free to ask me about his background or projects."),
        ("Can you give me 3 tips for staying focused?", "That's outside what I'm built for — I'm here to answer questions about Sidharth's background, skills, and projects."),
        ("Tell me a fun fact about space.", "I don't have general knowledge about topics like space — I'm built specifically to answer questions about Sidharth. Ask me about his projects or skills!"),
        ("Who is the president of the United States?", "I'm built to answer questions about Sidharth rather than general world facts. Ask me about his education, skills, or projects instead."),
        ("How do airplanes fly?", "That's outside my scope — I'm here to answer questions about Sidharth's background, education, and projects."),

        # ================= COMPOUND / CROSS-PROJECT QUESTIONS =================
        ("Which of Sidharth's projects use RAG?",
         "Both the Self-RAG AI Agent and the AI Video Assistant use Retrieval-Augmented Generation (RAG) for their Q&A systems."),
        ("Compare MAARS and the Self-RAG project.",
         "MAARS is a multi-agent research pipeline that autonomously researches a topic from the web and produces a report, while the Self-RAG AI Agent is a document Q&A system that answers questions from PDFs using self-correcting retrieval. Both use LangChain and Mistral AI, but MAARS focuses on live web research and Self-RAG focuses on grounded document Q&A."),
        ("Does Sidharth have experience with Streamlit?",
         "Yes, Sidharth built Streamlit frontends for both MAARS and the AI Video Assistant."),
        ("What do all of Sidharth's projects have in common?",
         "All three of Sidharth's main projects — the Self-RAG AI Agent, MAARS, and the AI Video Assistant — use Python, LangChain, and Mistral AI as their core stack, and are all Generative AI systems he built end-to-end."),
         ("Has Sidharth built an LLM from scratch?",
         "Yes, Sidharth built a full LLM training pipeline from scratch across 8 parts, covering everything from attention mechanics to RLHF, implemented in PyTorch without relying on pre-built model libraries."),
        ("has sidharth built his own llm", "Yes, Sidharth built a full LLM training pipeline from scratch, from attention mechanics through RLHF, in 8 parts using PyTorch."),
        ("tell me about sidharths llm from scratch project", "Sidharth built an 8-part LLM-from-scratch project in PyTorch, covering attention mechanics, a modern transformer architecture, a BPE tokenizer, base pretraining, supervised fine-tuning, a reward model, and PPO-based RLHF, ending in a working chat interface."),
        ("What is Sidharth's LLM from scratch project?",
         "It's an 8-part project where Sidharth implemented an entire LLM training pipeline from first principles in PyTorch: attention math, a modern GPT-style architecture, tokenization, pretraining, instruction tuning, reward modeling, and RLHF."),
        ("why did sidharth build an llm from scratch",
         "To deeply understand how large language models actually work end-to-end, rather than just using pre-built libraries — implementing every stage himself, from attention to RLHF."),
        ("does sidharth understand how transformers work internally",
         "Yes, Sidharth implemented attention mechanisms, a modern transformer architecture, and a full training pipeline from scratch in PyTorch as part of his 8-part LLM project."),

        ("What is Part 1 of Sidharth's LLM project?",
         "Part 1 covers the core attention mechanism — implementing the math behind scaled dot-product and multi-head attention from scratch."),
        ("what does part 1 of the llm project cover", "Part 1 covers the core attention mechanism, implementing scaled dot-product and multi-head attention math from scratch."),

        ("What is Part 2 of Sidharth's LLM project?",
         "Part 2 builds a toy GPT model using a simple byte-level tokenizer, as an early working end-to-end example before adding more advanced architecture."),
        ("what does part 2 of the llm project cover", "Part 2 builds a toy GPT model with a byte-level tokenizer, an early minimal end-to-end example."),

        ("What is Part 3 of Sidharth's LLM project?",
         "Part 3 implements modern transformer architecture components: RoPE (Rotary Positional Embeddings), RMSNorm, SwiGLU activations, and KV-caching for efficient inference."),
        ("what does part 3 of the llm project cover", "Part 3 implements modern architecture components — RoPE, RMSNorm, SwiGLU, and KV-caching."),
        ("does sidharths model use rope", "Yes, RoPE (Rotary Positional Embeddings) is implemented as part of the modern architecture in Part 3 of his LLM-from-scratch project."),
        ("does sidharths model use rmsnorm", "Yes, RMSNorm is one of the modern architecture components implemented in Part 3."),
        ("does sidharths model use swiglu", "Yes, SwiGLU activations are implemented as part of the modern architecture in Part 3."),
        ("does sidharths model support kv caching", "Yes, KV-caching for efficient inference is implemented in Part 3 of his LLM project."),

        ("What is Part 4 of Sidharth's LLM project?",
         "Part 4 implements a BPE (Byte Pair Encoding) tokenizer and performs base language model pretraining on a text corpus."),
        ("what does part 4 of the llm project cover", "Part 4 implements a BPE tokenizer and performs base language model pretraining."),
        ("what tokenizer does sidharths model use", "Sidharth's model uses a BPE (Byte Pair Encoding) tokenizer, implemented in Part 4 of his LLM-from-scratch project."),

        ("What is Part 5 of Sidharth's LLM project?",
         "Part 5 explores Mixture-of-Experts (MoE) architecture demos."),
        ("what does part 5 of the llm project cover", "Part 5 covers Mixture-of-Experts (MoE) architecture demos."),
        ("does sidharth know about mixture of experts", "Yes, Sidharth explored Mixture-of-Experts (MoE) architecture as Part 5 of his LLM-from-scratch project."),

        ("What is Part 6 of Sidharth's LLM project?",
         "Part 6 performs supervised fine-tuning (SFT), training the base pretrained model to follow instructions using an instruction dataset."),
        ("what does part 6 of the llm project cover", "Part 6 performs supervised fine-tuning (SFT) on the base pretrained model."),

        ("What is Part 7 of Sidharth's LLM project?",
         "Part 7 trains a reward model that learns to score responses by preference, using a human-preference dataset."),
        ("what does part 7 of the llm project cover", "Part 7 trains a reward model that learns to score responses by preference."),

        ("What is Part 8 of Sidharth's LLM project?",
         "Part 8 implements PPO-based RLHF (Reinforcement Learning from Human Feedback) to further align the fine-tuned model using the trained reward model, and includes a working terminal chat interface."),
        ("what does part 8 of the llm project cover", "Part 8 implements PPO-based RLHF using the Part 7 reward model, and includes a working terminal chat interface for talking to the final model."),
        ("does sidharths llm project have a chat interface", "Yes, Part 8 includes a terminal chat interface for talking to the trained model, with live token-by-token streaming."),

        ("What are all the parts of Sidharth's LLM-from-scratch project?",
         "The project has 8 parts: Part 1 covers attention mechanics, Part 2 a byte-tokenizer toy GPT, Part 3 modern architecture (RoPE, RMSNorm, SwiGLU, KV-cache), Part 4 BPE tokenizer and base pretraining, Part 5 Mixture-of-Experts demos, Part 6 supervised fine-tuning, Part 7 reward modeling, and Part 8 PPO-based RLHF with a chat interface."),
        ("what parts does the llm project have", "The project has 8 parts, going from attention mechanics and a toy GPT, through modern architecture, tokenization, pretraining, MoE demos, SFT, reward modeling, and finally PPO-based RLHF with a chat interface."),
        ("walk me through the whole llm pipeline sidharth built",
         "Sidharth's pipeline goes: Part 1 attention math, Part 2 a byte-tokenizer toy GPT, Part 3 modern architecture (RoPE/RMSNorm/SwiGLU/KV-cache), Part 4 BPE tokenizer plus base pretraining, Part 5 MoE demos, Part 6 supervised fine-tuning, Part 7 reward model training, and Part 8 PPO-based RLHF plus a chat interface — a complete LLM training pipeline built from scratch in PyTorch."),

        ("What framework did Sidharth use to build the LLM from scratch?",
         "Sidharth implemented the entire LLM training pipeline in PyTorch, without relying on pre-built model architectures."),
        ("is the llm project built with pytorch", "Yes, the entire pipeline is implemented in PyTorch, built from first principles rather than using pre-built model libraries."),
    ]
    return pairs


def load_about_me(repeat: int = 1) -> List[SFTItem]:
    """Return the about-me Q&A set. `repeat` lets you oversample it relative
    to a much larger general instruction dataset (e.g. Alpaca) so the model
    doesn't treat these examples as noise, the same way chitchat_repeat works
    for chitchat_data.py.
    """
    items = [SFTItem(prompt=p, response=r) for p, r in _pairs()]
    return items * max(1, repeat)


if __name__ == "__main__":
    data = load_about_me(repeat=1)
    print(f"Total unique about-me examples: {len(data)}")