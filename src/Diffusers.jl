module Diffusers

# Write your package code here.
using Transformers

using Transformers.HuggingFace

# model = hgf"openai/clip-vit-large-patch14"
# text_config = Transformers.HuggingFace.HGFCLIPTextConfig()

# Transformers.HuggingFace.slow_tkr_files((Val ∘ Symbol)("bert"))

tokenizer, vocab = Transformers.HuggingFace.load_slow_tokenizer((Val ∘ Symbol)("clip"), 
"/home/haranand/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff/vocab.json", 
"/home/haranand/.cache/huggingface/hub/models--openai--clip-vit-large-patch14/snapshots/8d052a0f05efbaefbc9e8786ba291cfdf93e5bff/merges.txt")

using TextEncodeBase: Sentence

# jl_tokens = TextEncodeBase.getvalue.(TextEncodeBase.tokenize(tokenizer, "test"))

t = tokenizer(Sentence("a photo of an astronaut riding a horse on mars"))

t[1]


enc = TextEncodeBase.getvalue.(t)

enc


TextEncoder(t, v)
text = "a photo of an astronaut riding a horse on mars"

textenc = Transformers.Basic.TransformerTextEncoder(tokenizer, vocab; trunc = nothing, startsym = "<s>", endsym = "</s>", unksym = "<unk>", padsym = "<pad>")

e = encode(textenc, [text])
e.tok[1, 1, 1]

end
