using Knet
using Pkg; Pkg.build("Knet")
# DEPENDECIES JUST FOR ISOLATED TESTING - REMOVE BEFORE PUBLISHING

#=
Julia implementation of a Region Proposal Network (RPN)
=#

struct rpn
    anchors_per_loc
    anchor_stride
    depth
    

end # module
