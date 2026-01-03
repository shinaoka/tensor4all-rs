using Test
using Tensor4all
using ITensors

@testset "ITensors Extension" begin
    @testset "Tensor4all.Index → ITensors.Index" begin
        # Basic conversion
        t4a_idx = Tensor4all.Index(5; tags="Site,n=1")
        it_idx = ITensors.Index(t4a_idx)

        @test ITensors.dim(it_idx) == 5
        @test ITensors.hastags(it_idx, "Site")
        @test ITensors.hastags(it_idx, "n=1")

        # ID should match (lower 64 bits)
        t4a_id = Tensor4all.id(t4a_idx)
        expected_id = UInt64(t4a_id & 0xFFFFFFFFFFFFFFFF)
        @test ITensors.id(it_idx) == expected_id
    end

    @testset "ITensors.Index → Tensor4all.Index" begin
        # Basic conversion
        it_idx = ITensors.Index(3, "Link,l=2")
        t4a_idx = Tensor4all.Index(it_idx)

        @test Tensor4all.dim(t4a_idx) == 3
        @test Tensor4all.hastag(t4a_idx, "Link")
        @test Tensor4all.hastag(t4a_idx, "l=2")

        # ID should match (ITensors ID becomes lower 64 bits)
        it_id = ITensors.id(it_idx)
        t4a_id = Tensor4all.id(t4a_idx)
        @test UInt64(t4a_id & 0xFFFFFFFFFFFFFFFF) == it_id
        @test UInt64(t4a_id >> 64) == 0  # Upper bits should be 0
    end

    @testset "Roundtrip conversion" begin
        # Tensor4all → ITensors → Tensor4all
        orig = Tensor4all.Index(4; tags="Test")
        it_idx = ITensors.Index(orig)
        back = Tensor4all.Index(it_idx)

        @test Tensor4all.dim(orig) == Tensor4all.dim(back)
        @test Tensor4all.tags(orig) == Tensor4all.tags(back)
        # IDs match in lower 64 bits (upper bits may differ after roundtrip)
        orig_lo = UInt64(Tensor4all.id(orig) & 0xFFFFFFFFFFFFFFFF)
        back_lo = UInt64(Tensor4all.id(back) & 0xFFFFFFFFFFFFFFFF)
        @test orig_lo == back_lo

        # ITensors → Tensor4all → ITensors
        it_orig = ITensors.Index(6, "Bond")
        t4a_idx = Tensor4all.Index(it_orig)
        it_back = ITensors.Index(t4a_idx)

        @test ITensors.dim(it_orig) == ITensors.dim(it_back)
        @test ITensors.id(it_orig) == ITensors.id(it_back)
        @test ITensors.hastags(it_back, "Bond")
    end

    @testset "Convert function" begin
        t4a_idx = Tensor4all.Index(2)
        it_idx = convert(ITensors.Index, t4a_idx)
        @test it_idx isa ITensors.Index
        @test ITensors.dim(it_idx) == 2

        it_idx2 = ITensors.Index(3)
        t4a_idx2 = convert(Tensor4all.Index, it_idx2)
        @test t4a_idx2 isa Tensor4all.Index
        @test Tensor4all.dim(t4a_idx2) == 3
    end

    @testset "Empty tags" begin
        t4a_idx = Tensor4all.Index(4)
        it_idx = ITensors.Index(t4a_idx)
        @test ITensors.dim(it_idx) == 4

        it_idx2 = ITensors.Index(5)
        t4a_idx2 = Tensor4all.Index(it_idx2)
        @test Tensor4all.dim(t4a_idx2) == 5
        @test Tensor4all.tags(t4a_idx2) == ""
    end
end
