using Transformers

# _bert_model = hgf"bert-base-uncased:forsequenceclassification"

# model_cons = Transformers.HuggingFace.get_model_type(Val{:bert}(), Val{:forsequenceclassification}())
model_name = "bert-base-uncased"
config = Transformers.load_config(model_name)
# model = Transformers.load_model(model_cons, model_name; config=cfg)
# print(model)


# Transformers.HuggingFace.HGFBertModel(config)
embeddings = Transformers.HuggingFace.HGFBertEmbeddings(config)
Transformers.load_model(model_cons, model_name; config=cfg)