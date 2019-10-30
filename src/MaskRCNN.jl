#__precompile__()

module mrcnn

using Knet, ArgCheck, ArgParse, Images
using Pkg; Pkg.build("Knet")   

module MaskRCNN

export
    MaskRCNN,
    config,
    predict,
    update

import("mrcnn.jl")

end # module
