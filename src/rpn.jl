#=
Julia implementation of a Region Proposal Network (RPN)
=#

struct rpn
    anchors_per_loc
    anchor_stride
    depth
end
