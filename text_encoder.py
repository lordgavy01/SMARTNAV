import torch
from transformers import RobertaTokenizer, RobertaModel

roberta_tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
roberta_model = RobertaModel.from_pretrained("roberta-base")

def get_roberta_embeddings(text):
    if isinstance(text, str):
        text = [text]  # Convert to list if it's a single string

    tokenized_texts = roberta_tokenizer.batch_encode_plus(text, padding=True, return_tensors="pt")
    input_ids = tokenized_texts['input_ids']
    attention_mask = tokenized_texts['attention_mask']

    with torch.no_grad():
        embeddings = roberta_model(input_ids, attention_mask=attention_mask)[0]

    embeddings = torch.mean(embeddings, 1)
    return embeddings.detach().numpy()


# texts = ["This is a sample text.", "Here's another example."]
# embeddings = get_roberta_embeddings(texts, roberta_model, roberta_tokenizer)

