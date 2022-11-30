using Diffusers
using Test

@testset "Diffusers.jl" begin
    # Write your tests here.
    include("models/test_clip_encoder.jl")
end
