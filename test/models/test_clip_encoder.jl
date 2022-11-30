using Diffusers
using Transformers
using TextEncodeBase
using Test

@testset "Verify tokenizer" begin
    text = "a photo of an astronaut riding a horse on mars"
    tokenizer, vocab = Diffusers.get_tokenizer("openai/clip-vit-large-patch14")
    # one hot encoded values
    text_encoder = Transformers.Basic.TransformerTextEncoder(tokenizer, vocab; trunc=77, startsym="<|startoftext|>", endsym="<|endoftext|>", unksym="<|endoftext|>", padsym="<|endoftext|>")

    # tokens start from 1, so all values have +1 compared to pytorch tokenizer
    println(Transformers.Basic.onehot2indices(encode(text_encoder, [text]).tok)) # TODO: check with pytorch
end


# using Transformers.HuggingFace
# model = "openai/clip-vit-large-patch14"
# vocab_file = HuggingFace.hgf_vocab_json(model)
# merges_file = HuggingFace.hgf_merges(model)
# tokenizer, vocab = HuggingFace.load_slow_tokenizer((Val ∘ Symbol)("clip"), vocab_file, merges_file)