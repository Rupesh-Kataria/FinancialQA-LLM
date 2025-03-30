# ConvFinQA LLM  

## Introduction  

This project is focused on developing a Large Language Model (LLM) driven prototype to answer questions based on financial documents (texts, tables) using the ConvFinQA dataset. The challenge was to extract structured information from the dataset, retrieve relevant content, and generate step-by-step solutions to complex financial questions.  

## Understanding the Dataset  

Initially, I explored the dataset provided in `train.json`. The dataset consists of:  

- **Pretext** (introductory text before a table)  
- **Posttext** (text following the table)  
- **Table** (structured financial data)  
- **Questions and their corresponding answers**  
- **Dialogue breaks** (step-by-step breakdowns of how the solution is derived)  
- **Turn programs** (code-like representations of intermediate steps in solving the problem)  

## Data Formatting  

To properly format the dataset for training, I wrote preprocessing scripts in `formatting_1.ipynb`. This notebook extracts:  

- The **question**  
- The **context** (pretext, table, posttext)  
- The **dialogue turn** (step-by-step intermediate breakdown)  
- The **corresponding turn program** (which acts as the model output)  

This ensures that the model learns to generate solutions step-by-step rather than directly predicting the final numerical answer.  

### Challenge in Formatting  

One major challenge was dealing with multiple questions derived from the same context. In some cases, two questions were given for the same context, but their dialogue turns were combined rather than separated. Since I couldn't determine where one dialogue turn ended and the next began, I had to discard those data points, leaving out several multi-question examples.  

## Retrieval Model  

The ConvFinQA repository uses a GPT-2 based encoder retriever, which was trained by generating positive and negative examples:  

- **Positive example:** The question with its relevant context.  
- **Negative example:** The question with an unrelated context.  

Instead of training a retriever from scratch, I leveraged modern text embedding models to perform retrieval efficiently.  

### Retrieval Approach  

1. **Vector Embeddings:**  
   - Used Gemini embedding model to generate vector representations of the context.  
   - Stored the embeddings in **FAISS**, a vector database optimized for fast retrieval.  

2. **Chunk Selection:**  
   - Experimented with how many chunks to retrieve for each question before passing them to reranking.  
   - Extracted **9 chunks per question** and checked whether the first retrieved chunk matched the ground-truth question chunk.  

3. **Hybrid Retrieval:**  
   - Used **TF-IDF** for word-based matching.  
   - Used **semantic similarity** to capture meaning-based relationships.  
   - Combined **keyword matching** (extracting important words from the question while ignoring stopwords) with **semantic similarity** to improve retrieval accuracy.  

This process was implemented in `retrieval.ipynb`.  

## Generation Model  

For generating answers, I experimented with multiple LLMs:  

### DeepSeek 7B + LoRA Fine-tuning  

- Initially, I trained **DeepSeek 7B** using **LoRA** (low-rank adaptation) while loading the model in a quantized manner.  
- However, the model struggled to generate correct answers, often producing responses like _"Let me think"_ rather than solving the problem correctly.  
- I suspected that a **low-rank LoRA adapter** and **high dropout** weakened the training effect.  
- I then tried **fine-tuning only the last few layers** instead of using LoRA, but training with `trainer.train()` exceeded RAM capacity.  

### Microsoft Phi-2 Fine-tuning  

Since **Phi-2** is a **3B parameter model**, I attempted fine-tuning it using **LoRA**.  

#### Changes I made:  

- Increased **LoRA rank (r=128)** to train a larger adapter matrix.  
- Reduced **dropout** to 0.01.  
- Increased **learning rate** from `2e-5` to `1e-3` to speed up convergence.  

❌ **Issue:** Loss initially decreased but then started increasing, likely due to a high learning rate causing instability.  

### Mistral 7B (Instruction-tuned Model)  

- I tried **Mistral 7B**, which required **5GB RAM**.  
- Directly prompting it with `question + context` produced **decent answers**.  
- However, training it required **30+ GB RAM**, which exceeded my system’s limit.  
- I attempted loading it using **Unsloth**, which optimizes memory usage and inference speed.  

❌ **Problem:** While inference worked, the model started **repeating tokens** instead of generating proper answers.  

#### Hypothesis:  
Unsloth optimizations likely altered model behavior, preventing it from reasoning correctly.  

Final attempt:  
I trained **Mistral 7B** with **Unsloth**, but even though **loss reduced**, **generation quality did not improve significantly**.  

---

## Challenges & Learnings  

### Handling Long Contexts  

- The dataset has **long contexts** (pretext, table, posttext).  
- Smaller models struggled with **long-context understanding**.  
- A **larger dataset** and **more computation** would be needed for better fine-tuning.  

### Training Large Models  

- Running `trainer.train()` on large models like **Mistral 7B** without quantization **exceeds RAM limits**.  
- **LoRA fine-tuning** helped but wasn’t enough.  
- **Unsloth reduced RAM usage** but **negatively impacted answer quality**.  

### Retriever Improvements  

- **Hybrid retrieval (TF-IDF + embeddings) improved accuracy**.  
- **Chunk selection strategies helped optimize retrieval effectiveness**.  
- **Tuning retrieval parameters** (number of retrieved chunks, weighting of keyword matching vs. semantic similarity) was crucial.  

---
