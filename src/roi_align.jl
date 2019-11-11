#=
Define region of interest alignment 
=#

struct ROI_Align
    crop_height
    crop_width
    extrapolation_value
    transform_fpcoor
end
