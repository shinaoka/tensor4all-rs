using Test
using Tensor4all
using ITensors

# Use qualified names to avoid ambiguity
const T4AIndex = Tensor4all.Index

@testset "Tensor4all.jl" begin
    @testset "Index creation" begin
        # Basic creation
        i = T4AIndex(5)
        @test Tensor4all.dim(i) == 5
        @test Tensor4all.tags(i) == ""
        @test Tensor4all.id(i) != 0

        # With tags
        j = T4AIndex(3; tags="Site,n=1")
        @test Tensor4all.dim(j) == 3
        @test Tensor4all.hastag(j, "Site")
        @test Tensor4all.hastag(j, "n=1")
        @test !Tensor4all.hastag(j, "Missing")

        # Tags string contains both
        t = Tensor4all.tags(j)
        @test occursin("Site", t)
        @test occursin("n=1", t)
    end

    @testset "Index with custom ID" begin
        id_val = UInt128(0x12345678_9ABCDEF0_FEDCBA98_76543210)
        i = T4AIndex(4, id_val; tags="Custom")
        @test Tensor4all.dim(i) == 4
        @test Tensor4all.id(i) == id_val
        @test Tensor4all.hastag(i, "Custom")
    end

    @testset "Index copy" begin
        i = T4AIndex(5; tags="Original")
        j = copy(i)

        @test Tensor4all.dim(i) == Tensor4all.dim(j)
        @test Tensor4all.id(i) == Tensor4all.id(j)
        @test Tensor4all.tags(i) == Tensor4all.tags(j)
        @test i == j  # Equal by ID
    end

    @testset "Index equality and hashing" begin
        i = T4AIndex(5)
        j = copy(i)
        k = T4AIndex(5)  # Different ID

        @test i == j
        @test i != k
        @test hash(i) == hash(j)
        @test hash(i) != hash(k)  # Very likely different
    end

    @testset "Index display" begin
        i = T4AIndex(3; tags="Site")
        s = sprint(show, i)
        @test occursin("dim=3", s)
        @test occursin("Site", s)
    end

    @testset "Error handling" begin
        @test_throws ArgumentError T4AIndex(0)
        @test_throws ArgumentError T4AIndex(-1)
    end

    @testset "ITensors Extension" begin
        @testset "Tensor4all.Index → ITensors.Index" begin
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
            it_idx = ITensors.Index(3, "Link,l=2")
            t4a_idx = Tensor4all.Index(it_idx)

            @test Tensor4all.dim(t4a_idx) == 3
            @test Tensor4all.hastag(t4a_idx, "Link")
            @test Tensor4all.hastag(t4a_idx, "l=2")

            # ID should match
            it_id = ITensors.id(it_idx)
            t4a_id = Tensor4all.id(t4a_idx)
            @test UInt64(t4a_id & 0xFFFFFFFFFFFFFFFF) == it_id
        end

        @testset "Roundtrip conversion" begin
            # Tensor4all → ITensors → Tensor4all
            orig = Tensor4all.Index(4; tags="Test")
            it_idx = ITensors.Index(orig)
            back = Tensor4all.Index(it_idx)

            @test Tensor4all.dim(orig) == Tensor4all.dim(back)
            @test Tensor4all.tags(orig) == Tensor4all.tags(back)

            # ITensors → Tensor4all → ITensors
            it_orig = ITensors.Index(6, "Bond")
            t4a_idx = Tensor4all.Index(it_orig)
            it_back = ITensors.Index(t4a_idx)

            @test ITensors.dim(it_orig) == ITensors.dim(it_back)
            @test ITensors.id(it_orig) == ITensors.id(it_back)
        end
    end
end
