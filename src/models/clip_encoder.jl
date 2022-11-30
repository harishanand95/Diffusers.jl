using Transformers
using Transformers.HuggingFace

"""
    get_tokenizer(model)

Get tokenizer and vocab file from HuggingFace API.

# Example
```julia-repl
julia> get_tokenizer("openai/clip-vit-large-patch14")

```
"""
function get_tokenizer(model)
    vocab_file = HuggingFace.hgf_vocab_json(model)
    merges_file = Transformers.HuggingFace.hgf_merges(model)
    tokenizer, vocab = Transformers.HuggingFace.load_slow_tokenizer((Val ∘ Symbol)("clip"), vocab_file, merges_file)
    return (tokenizer, vocab)
end

