#__precompile__()

module MaskRCNN

using Knet, ArgCheck, ArgParse, Images, FileIO, MAT
using Pkg; Pkg.build("Knet")

#export
#    MaskRCNN,
#    config,
#    predict,
#    update

export
    smooth_l1_loss,
    clip_to_window,
    clip_boxes,
    resize_image,
    mold_image,
    compose_image_meta,
    parse_image_meta,
    mold_inputs,
    compute_iou,
    compute_overlaps_bbox,
    extract_bboxes,
    box_refinement_graph,
    matconvnet,
    imgdata,
    resnet50,
    resnet101,
    resnet152,
    get_params

include("utils.jl")
include("mrcnn.jl")
include("resnet.jl")
include("rpn.jl")
include("fpn.jl")
include("roi_align.jl")

end # module
