using Test, MaskRCNN, Pkg, ArgParse, FileIO, Knet
Pkg.build("Knet")
Pkg.add("Libdl")
using Libdl
push!(Libdl.DL_LOAD_PATH,"/usr/lib")

@testset "MaskRCNN" begin

    @time include("test_utils.jl")
    @time include("test_resnet.jl")
    @time include("test_rpn.jl")
    @time include("test_fpn.jl")
    @time include("test_roi_align.jl")
    @time include("test_mrcnn.jl")

end
