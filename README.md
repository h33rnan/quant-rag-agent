# Quant RAG Agent: Autonomous Trading Copilot

## Overview
This repository contains a proof-of-concept for an Autonomous Quantitative Agent using Retrieval-Augmented Generation (RAG). The system bridges the gap between natural language queries, private algorithmic trading rules, and real-time market data execution.

## Architecture Pipeline
The agent operates through a strict production pipeline designed for semantic resilience and execution safety:

1. **Phase A: Semantic Firewall (Guardrails)**
   - Intercepts prompts to prevent prompt-injection and ensures the query strictly adheres to the financial domain.
2. **Phase B: Vector Retrieval (RAG & Query Expansion)**
   - Expands the user's query into technical parameters before querying **ChromaDB**. This guarantees the retrieval of the correct asset-specific quantitative rule.
3. **Phase C: Context Distillation**
   - Employs an LLM to compress the retrieved context, extracting only the core technical rule to optimize token usage.
4. **Phase D & E: Autonomous Execution & Parsing**
   - The Agent evaluates the distilled rule and autonomously triggers external APIs (e.g., fetching real-time Volume) via **Function Calling**.
   - Output is strictly parsed into a structured JSON schema.

## Tech Stack
* **Language:** Python
* **Orchestration & LLM:** Google Gemini 2.5 Flash
* **Vector Database:** ChromaDB
* **Architecture:** RAG, Tool Calling, Query Expansion.

> **Note:** This is a technical study of LLM orchestration. Not financial advice.
