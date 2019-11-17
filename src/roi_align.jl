#=
Define region of interest alignment 
=#

struct ROI_Align
    crop_height::Int
    crop_width::Int
    extrapolation_value::Float64
    transform_fpcoor::Bool
end
#=
ROA_Align(crop_height, crop_width, extrapolation_value = 0.0f0, transform_fpcoor = true) = ROI_Align(crop_height, crop_width, extrapolation_value, transform_fpcoor)

function (c::ROA_Align)(feature_map, boxes, box_ind)
    x1 = boxes[:, 1]
    y1 = boxes[:, 2]
    x2 = boxes[:, 3]
    y2 = boxes[:, 4]

    img_height, img_width = Float.(size(feature_map)[1:2])

    if c.transform_fpcoor
        spacing_w = (x2 .- x1) ./ img_width
        spacing_h = (y2 .- y1) ./ img_height

        nx0 = (x1 .+ (spacing_w ./ 2.f0) .- 0.5f0) ./ (img_width .- 1.0f0)
        ny0 = (y1 .+ (spacing_h ./ 2.f0) .- 0.5f0) ./ (img_height .- 1.0f0)
        nw = spacing_w .* (c.crop_width .- 1.0f0) ./ (img_width .- 1.0f0)
        nh = spacing_h .* (c.crop_height .- 1.0f0) ./ (img_height .- 1.0f0)
        boxes = cat(ny0, nx0, ny0 .+ nh, nx0 .+ nw, dims = 2)

    else
        x1 = x1 ./ (img_width .- 1.0f0)
        x2 = x2 ./ (img_width .- 1.0f0)
        y1 = y1 ./ (img_height .- 1.0f0)
        y2 = y2 ./ (img_height .- 1.0f0)
        boxes = cat(y1, x1, y2, x2, dims = 2)
    end

    crop_and_resize(feature_map, boxes, box_ind, c.crop_height, c.crop_width)
end
=#
