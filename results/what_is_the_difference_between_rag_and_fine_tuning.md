# Research Task: What is the difference between RAG and fine-tuning?

## Key Differences Between RAG and Fine-tuning

**RAG (Retrieval-Augmented Generation)**
- **How it works**: Retrieves relevant external knowledge from a knowledge base *at inference time* without modifying the model itself
- **Model changes**: None—the model weights remain unchanged
- **Knowledge source**: Pulls information from external databases or documents in real-time
- **Best for**: Tasks requiring current information, multiple knowledge domains, or when you need to update knowledge without retraining
- **Flexibility**: Easy to update knowledge by modifying the external knowledge base

**Fine-tuning**
- **How it works**: Updates the model's weights and parameters through additional training on domain-specific data
- **Model changes**: Significant—the model is retrained and its internal parameters are adjusted
- **Knowledge source**: Embeds task-specific knowledge directly into the model during training
- **Best for**: Domain-specific specialization, when you want the model to deeply understand a particular subject
- **Flexibility**: Requires retraining to update or change knowledge

**In short**: RAG is about *retrieving* knowledge at query time, while fine-tuning is about *embedding* knowledge into the model through training. Choose RAG when you need flexible, up-to-date external information; choose fine-tuning when you want deep domain expertise built into the model itself.