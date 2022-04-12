using DSP, FFTW, FastTransforms, LinearAlgebra, Test

@testset "BigFloat FFT and DCT" begin

    c = collect(range(-big(3.0),stop=2,length=20))
    @test norm(fft(c) - fft(Float64.(c))) < 4Float64(norm(c))*eps()
    @test norm(ifft(c) - ifft(Float64.(c))) < 4Float64(norm(c))*eps()

    c = collect(range(-big(3.0),stop=2.0,length=402))
    @test norm(ifft(fft(c))-c) < 300norm(c)eps(BigFloat)

    s = big(1) ./ (1:20)
    s64 = Float64.(s)
    @test Float64.(conv(s, s)) ≈ conv(s64, s64)
    @test s == big(1) ./ (1:30) #67, ensure conv doesn't overwrite input
    @test all(s64 .=== Float64.(big(1) ./ (1:20)))

    p = plan_dct(c)
    @test norm(FastTransforms.generic_dct(c) - p*c) == 0

    pi = plan_idct!(c)
    @test norm(pi*dct(c) - c) < 10000norm(c)*eps(BigFloat)

    @test norm(dct(c)-dct(map(Float64,c)),Inf) < 100eps()

    cc = cis.(c)
    @test norm(dct(cc)-dct(map(Complex{Float64},cc)),Inf) < 100eps()

    c = big.(rand(1000)) + im*big.(rand(1000))
    @test norm(dct(c)-dct(map(ComplexF64,c)),Inf) < 100eps()
    @test norm(idct(c)-idct(map(ComplexF64,c)),Inf) < 100eps()
    @test norm(idct(dct(c))-c,Inf) < 10000eps(BigFloat)
    @test norm(dct(idct(c))-c,Inf) < 10000eps(BigFloat)

    c = randn(ComplexF16, 200)
    p = plan_fft(c)
    @test inv(p) * (p * c) ≈ c

    c = randn(ComplexF16, 200)
    pinpl = plan_fft!(c)
    @test inv(pinpl) * (pinpl * c) ≈ c

    # Make sure we don't accidentally hijack any FFTW plans
    for T in (Float32, Float64)
        @test plan_fft(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_fft(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_fft(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_fft(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_fft!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_fft!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_fft(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_fft(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_fft(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_fft(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_fft!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_fft!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_ifft(rand(T,10)) isa FFTW.ScaledPlan
        @test plan_ifft(rand(T,10), 1:1) isa FFTW.ScaledPlan
        @test plan_ifft(rand(Complex{T},10)) isa FFTW.ScaledPlan
        @test plan_ifft(rand(Complex{T},10), 1:1) isa FFTW.ScaledPlan
        @test plan_ifft!(rand(Complex{T},10)) isa FFTW.ScaledPlan
        @test plan_ifft!(rand(Complex{T},10), 1:1) isa FFTW.ScaledPlan

        @test plan_bfft(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_bfft(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_bfft(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_bfft(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_bfft!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_bfft!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_bfft(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_bfft(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_bfft(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_bfft(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_bfft!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_bfft!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_dct(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_dct(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_dct(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_dct(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_dct!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_dct!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_dct(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_dct(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_dct(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_dct(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_dct!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_dct!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_idct(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_idct(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_idct(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_idct(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test plan_idct!(rand(Complex{BigFloat},10)) isa FastTransforms.DummyPlan
        @test plan_idct!(rand(Complex{BigFloat},10), 1:1) isa FastTransforms.DummyPlan
        @test !( plan_idct(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_idct(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_idct(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_idct(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_idct!(rand(Complex{T},10)) isa FastTransforms.DummyPlan )
        @test !( plan_idct!(rand(Complex{T},10), 1:1) isa FastTransforms.DummyPlan )

        @test plan_rfft(rand(BigFloat,10)) isa FastTransforms.DummyPlan
        @test plan_rfft(rand(BigFloat,10), 1:1) isa FastTransforms.DummyPlan
        @test plan_brfft(rand(Complex{BigFloat},10), 19) isa FastTransforms.DummyPlan
        @test plan_brfft(rand(Complex{BigFloat},10), 19, 1:1) isa FastTransforms.DummyPlan
        @test !( plan_rfft(rand(T,10)) isa FastTransforms.DummyPlan )
        @test !( plan_rfft(rand(T,10), 1:1) isa FastTransforms.DummyPlan )
        @test !( plan_brfft(rand(Complex{T},10), 19) isa FastTransforms.DummyPlan )
        @test !( plan_brfft(rand(Complex{T},10), 19, 1:1) isa FastTransforms.DummyPlan )

    end

end
