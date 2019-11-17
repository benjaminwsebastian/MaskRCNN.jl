# utilities
# commented out are imported functions that haven't been thoroughly tested

# Migrate these to MaskRCNN when done testing
using StatsBase
using StatsBase: shuffle
using Base.Iterators
using Statistics

const _mcnurl = "http://www.vlfeat.org/matconvnet/models"
const _mcndir = joinpath(@__DIR__, "imagenet")

function matconvnet(name, pass = false)
    global _mcncache
    if !@isdefined(_mcncache); _mcncache=Dict(); end
    if !haskey(_mcncache, name)
        matfile = "$name.mat"
        @info("Loading $matfile...")
        path = joinpath(_mcndir,matfile)
        if !isfile(path)
            if !pass
                println("Should I download $matfile? (y if Yes)")
                readline()[1] == 'y' || error(:ok)
                isdir(_mcndir) || mkpath(_mcndir)
                download("$_mcnurl/$matfile",path)
            else
                isdir(_mcndir) || mkpath(_mcndir)
                download("$_mcnurl/$matfile",path)
            end
        end
        _mcncache[name] = matread(path)
    end
    return _mcncache[name]
end

function imgdata(img, averageImage)
    global _imgcache
    if !@isdefined(_imgcache); _imgcache = Dict(); end
    if !haskey(_imgcache,img)
        if occursin("://",img)
            @info("Downloading $img")
            a0 = load(download(img))
        else
            a0 = load(img)
        end
        new_size = ntuple(i->div(size(a0,i)*224,minimum(size(a0))),2)
        a1 = Images.imresize(a0, new_size)
        i1 = div(size(a1,1)-224,2)
        j1 = div(size(a1,2)-224,2)
        b1 = a1[i1+1:i1+224,j1+1:j1+224]
        # ad-hoc solution for Mac-OS image
        macfix = convert(Array{FixedPointNumbers.Normed{UInt8,8},3}, channelview(b1))
        c1 = permutedims(macfix, (3,2,1))
        d1 = convert(Array{Float32}, c1)
        e1 = reshape(d1[:,:,1:3], (224,224,3,1))
        f1 = (255 * e1 .- averageImage)
        g1 = permutedims(f1, [2,1,3,4])
        _imgcache[img] = g1
    end
    return _imgcache[img]
end

# Resize image
function resize_image(img; min_dim = false, max_dim = false, padding = false)
    h, w = size(img)[1:2]
    window = (0.,0.,h,w)
    scale = 1.f0
    
    if min_dim > 0
	scale = max(1, min_dim / min(h, w))
    end

    if max_dim > 0
	image_max = max(h, w)
	if round(image_max * scale) > max_dim
	    scale = max_dim / image_max
	end
    end

    if scale != 1.
	img = imresize(img, (round(Int, h*scale), round(Int, w*scale), 3))
    end

    resized_img = zeros(max_dim, max_dim, 3)
    h, w = size(img)[1:2]
    top_pad = max(1, round(Int, (max_dim - h) / 2))
    bottom_pad = max_dim - h - top_pad
    left_pad = max(1, round(Int, (max_dim - w) / 2))
    right_pad = max_dim - w - left_pad
    resized_img[top_pad:(h+top_pad-1), left_pad:(w+left_pad-1), 1:3] .= img
    padding = [(top_pad, bottom_pad), (left_pad, right_pad), (0, 0)]
    window = (top_pad, left_pad, h + top_pad, w + left_pad)
    return resized_img, window, scale, padding
end

# Computes the IoU of a box
function compute_iou(box, boxes, box_area, boxes_area)
    #=
    Calculates the IoU of the given box with the array of the given boxes
    
    box: 1D vector [y1 x1 y2 x2]
    boxes: [boxes_count, (y1, x1, y2, x2)]
    box_area: float, the area of the box
    boxes_area:  array of the length of boxes_count

    Note: Areas are passed to avoid duplicate calculations
    =#
    y1 = max.(box[1], boxes[:,1])
    y2 = min.(box[3], boxes[:,3])
    x1 = max.(box[2], boxes[:,2])
    x2 = min.(box[4], boxes[:,4])
    intersection = max.(x2 .- x1, 0) .* max.(y2 .- y1, 0)
    union = box_area .+ boxes_area .- intersection
    iou = intersection ./ union
    return iou
end

# Computes overlaps of two sets of bboxes
function compute_overlaps_bbox(boxes1, boxes2)
    #=
    Computes the IoU overlaps between two sets of boxes
    
    boxes1, boxes2: [N, (y1, x1, y2, x2)]
    =#
    areas1 = (boxes1[:,3] .- boxes1[:,1]) .* (boxes1[:, 4] .- boxes1[:, 2]) 
    areas2 = (boxes2[:,3] .- boxes2[:,2]) .* (boxes2[:, 4] .- boxes2[:, 2]) 
    
    overlaps = zeros(size(boxes1,1), size(boxes2,1))
    for i = 1:size(overlaps, 2)
	box2 = boxes2[i,:]
	overlaps[:,i] = compute_iou(box2, boxes1, areas2[i], areas1)
    end
    return overlaps
end

# Computes the refinement needed to transform box to gt_box
function box_refinement_graph(box, gt_box)
    # box = Float32.(box)
    # gt_box = Float32.(gt_box)
    height = box[:, 3] .- box[:, 1]
    width = box[:, 4] .- box[:, 2]
    # width = replace(x -> x < 1f-3 ? 1.f0 : x, width)
    # height = replace(x -> x < 1f-3 ? 1.f0 : x, height)
    width = clamp.(width, 1f-3, 1.f0) # remove
    height = clamp.(height, 1f-3, 1.f0) # remove
    center_y = box[:, 1] .+ 0.5f0 .* height
    center_x = box[:, 2] .+ 0.5f0 .* width
    # return center_x
    gt_height = gt_box[:, 3] .- gt_box[:, 1]
    gt_width = gt_box[:, 4] .- gt_box[:, 2]
    gt_center_y = gt_box[:, 1] .+ 0.5f0 .* gt_height
    gt_center_x = gt_box[:, 2] .+ 0.5f0 .* gt_width
    
    dy = (gt_center_y .- center_y) ./ height
    dx = (gt_center_x .- center_x) ./ width
    # return dy
    dh = log.(abs.(gt_height ./ height))
    dw = log.(abs.(gt_width ./ width))
    
    return hcat(dy, dx, dh, dw)
end

#=
function apply_box_deltas(boxes, deltas)
    @show boxes[1,:]
    @show deltas[1,:]
    @show findall(isnan, boxes)
    @show findall(isnan, deltas)
    @show typeof(deltas)
    # bs = copy(boxes)
    
    heights = boxes[:,3] .- boxes[:,1]
    widths = boxes[:,4] .- boxes[:,2]
    centers_y = boxes[:,1] .+ (0.5f0 .* heights)
    centers_x = boxes[:,2] .+ (0.5f0 .* widths)
    
    
    centers_y = centers_y .+ (deltas[:,1] .* heights)
    centers_x = centers_x .+ (deltas[:,2] .* widths)
    # hs = deepcopy(heights)
    # @show findall(x -> isinf(x) || isnan(x), deltas[:,3])
    # @show findall(x -> isinf(x) || isnan(x), deltas[:,4])
    # @show findall(x -> isinf(x) || isnan(x), heights)
    # @show findall(x -> isinf(x) || isnan(x), widths)
    # @show findall(x -> isinf(x) || isnan(x), centers_x)
    # @show findall(x -> isinf(x) || isnan(x), centers_y)

    # @show maximum(deltas[:,3])
    # @show maximum(deltas[:,4])
    # @show maximum(heights)
    # @show maximum(widths)
    # @show maximum(centers_x)
    # @show maximum(centers_y)
    # @show std(deltas[:,3])
    # @show std(deltas[:,4])
    
    heights = heights .* exp.(deltas[:,3])
    widths = widths .* exp.(deltas[:,4])
    # global gh = heights
    # global gw = widths
    # error()
    
    infinds = findall(x -> isinf(x) || isnan(x), heights)
    @show deltas[:,3][infinds]
    @warn "something"
    # @show hs[findall(x -> isinf(x) || isnan(x), heights)]
    @show exp.(deltas[:,3])[findall(x -> isinf(x) || isnan(x), heights)]
    # @show findall(x -> isinf(x) || isnan(x), widths)

    # t = findall(isinf, heights)
    # v = findall(isinf, deltas[:,3])
    # @show v
    # # push!(t, 1)
    # @show exp.(deltas[:,3])[t]
    # @show hs[t]
    # @show hs[t] .* exp.(deltas[t,3])
    # p = findall(isinf, heights)
    # @show all(v.==p)
    # heights = replace(x -> isinf(x) ? 0.0f0 : x, heights)
    # widths = replace(x -> isinf(x) ? 0.0f0 : x, widths)
    y1s = centers_y .- (0.50f0 .* heights)
    x1s = centers_x .- (0.5f0 .* widths)
    y2s = y1s .+ heights
    x2s = x1s .+ widths
    Flux.stack([y1s, x1s, y2s, x2s], 2)
    # m = hcat(y1s, x1s, y2s, x2s)
    # ff = findall(isinf, heights)
    # if length(ff) > 0
    # 	@show size(y1s)
    # 	@show size(y2s)
    # 	t = findall(isinf, heights)
    # 	@show y1s[t]
    # 	@show heights[t]
    # 	@warn "nans in s"
    # 	@show "!!!!!!!!!!!!!!!"
    # end
end
=#
#=
function crop(boxes, window)
    y1 = clamp.(boxes[:,1], window[1], window[3])
    x1 = clamp.(boxes[:,2], window[2], window[4])
    y2 = clamp.(boxes[:,3], window[1], window[3])
    x2 = clamp.(boxes[:,4], window[2], window[4])
    hcat(y1,x1,y2,x2)
end
=#
function generate_anchors
end

function generate_pyramid_anchors
end

function add_mask
end

function add_bbox
end

function add_class
end

function resize_mask
end

function unmold_mask
end

# mask is the shape of an image wtf
# mask = rand(28,28, 10) => 10 masks of size 28 x 28 (to be applied to every channel)
function extract_bboxes(mask)
    nth = last(size(mask))
    boxes = zeros(Integer, nth, 4)
    for i =1:nth
	m = mask[:,:,i]
	cluster = findall(!iszero, m)
	if length(cluster) > 0	
	    Is = map(x -> [x.I[1], x.I[2]], cluster) |> x -> hcat(x...)'
	    x1, x2 = extrema(Is[:,1])
	    y1, y2 = extrema(Is[:,2])
	else
	    x1 ,x2, y1, y2 = 0, 0, 0, 0
	end
        
	boxes[i,:] = [y1, x1, y2, x2]
    end
    return boxes
end


function upsample(x, scale_factor::Tuple)
    return repeat(x, inner = scale_factor)
end

function clip_boxes(boxes, window)
    y1 = max.(min.(boxes[:,1], window[:,3]), window[:,1])
    x1 = max.(min.(boxes[:,2], window[:,4]), window[:,2])
    y2 = max.(min.(boxes[:,3], window[:,3]), window[:,1])
    x2 = max.(min.(boxes[:,4], window[:,4]), window[:,2])
    return hcat(y1,x1,y2,x2)
end

#=
generate all the boxes that need to be refined and chosen and set
scales = rand(n)
ratios = rand(m)
shape = (28,28)
feature_stride = 0.2 # stride compared to the image to align the features with the image
anchor_stride = 2 # generate an anchor every `2` steps

pytorch expected [... ...
... ...]
the input shape remains this one ^

but we operate on it transposed [. .
. .

. .
. .]


For reference:
	xx, yy = np.meshgrid(l1, l2)
	xx == collect(flatten(l1'))' .* ones(length(l1), length(l2)))
	yy == collect(flatten(l2)) .* ones(length(l1), length(l2)))

	could also use `repeat`, but would allocate off the butt

	use `broadcasting` here

	xx == repeat(l1', length(l2))
=#
#=
function generate_anchors(scales, ratios, shape, feature_stride, anchor_stride)
   ls, lr = length(scales), length(ratios)
   sr = size(ratios)

   scales = collect(flatten(scales'))
   ratios = sqrt.(flatten(ratios'))

   heights = collect(flatten(scales / ratios))
   widths = collect(flatten(scales * ratios'))
   lw, lh = length(widths), length(heights)

   shifts_y = collect((0:anchor_stride:shape[1]) .* feature_stride)[1:end-1]
   shifts_x = collect((0:anchor_stride:shape[2]) .* feature_stride)[1:end-1]
   ly = length(shifts_y)
   lx = length(shifts_x)

   shifts_x = repeat(collect(shifts_x), ly)    #|> x -> reshape(x, ly,ly)
   shifts_y = repeat(collect(shifts_y), lx)  #|> x -> reshape(x, lx,lx)

   box_widths = repeat(widths', length(shifts_x))
   box_centers_x = repeat(shifts_x, 1, lw) # |> x -> reshape(x, length(widths), length(shifts_x))
   # @show lw, lh
   box_heights = repeat(heights', length(shifts_y))
   box_centers_y = repeat(shifts_y[1:ly], inner = (ly, lh)) # |> x -> reshape(x, length(widths), length(shifts_x))

   box_centers = Flux.stack([box_centers_y, box_centers_x], 3)
   global gbc = box_centers
   box_centers = permutedims(box_centers, (2,1,3)) |> x -> reshape(x, :, 2)

   box_sizes = Flux.stack([box_heights, box_widths], 3)
   global gbs = box_sizes
   box_sizes = permutedims(box_sizes, (2,1,3)) |> x -> reshape(x, :, 2)

   cat(box_centers .- (0.5f0 .* box_sizes), box_centers .+ (0.5f0 .* box_sizes), dims = 2)
   # return box_centers
end
=#
#=
function generate_anchors2(scales, ratios, shape, feature_stride, anchor_stride)
    ls = length(scales)
    lr = length(ratios)
    s = repeat(scales, lr)
    r = repeat(ratios, inner = ls)
    h = s ./ sqrt.(r)
    w = s .* sqrt.(r)
    
    shifts_y = collect((0:anchor_stride:shape[1]) * feature_stride)[1:end-1]
    shifts_x = collect((0:anchor_stride:shape[2]) * feature_stride)[1:end-1]
    
    lx = length(shifts_x)
    ly = length(shifts_y)
    shifts_x = repeat(shifts_x', ly)
    shifts_y = repeat(shifts_y, outer = (1,lx))

    lsx = length(shifts_x)
    lsy = length(shifts_y)
    lw = length(w)
    lh = length(h)
    box_widths = repeat(w', lsx)
    box_centers_x = repeat(shifts_x[1,:], outer = (size(shifts_x, 1),lw))
    
    box_heights = repeat(h', lsy)
    box_centers_y = repeat(shifts_y[:,1], inner = (size(shifts_y, 1),lh))
    
    box_centers = Flux.stack([box_centers_y, box_centers_x], 3)

    Flux.stack([box_centers_y, box_centers_x], 3)
end
=#

#=
function generate_pyramid_anchors(scales, ratios, feature_shapes, 
				  feature_strides, anchor_stride)
    anchors = []
    # scales = transpose(scales)
    for i = 1:size(scales, 1)
	push!(anchors, generate_anchors(scales[i,:], ratios, feature_shapes[i,:],
					feature_strides[i], anchor_stride))
    end
    
    gpu(cat(anchors..., dims = 1))
end
=#
#=
inputs => (boxes, feature_maps...)
boxes => rand(10,4) (TODO: => rand(4,10,10)
feature_maps = similar(batch)
batch => rand(299, 299, 3, 10)
pool_size => [7,7]
=#
#=
function bbox_overlaps2(boxes1, boxes2)
    boxes1_repeat = (size(boxes2, 1), 1, 1)
    boxes2_repeat = (size(boxes1, 1), 1, 1)
    boxes1 = repeat(boxes1, outer = boxes1_repeat)
    boxes2 = repeat(boxes2, outer = boxes2_repeat)
    b1_y1, b2_y1 = boxes1[:, 1], boxes2[:, 1]
    b1_x1, b2_x1 = boxes1[:, 2], boxes2[:, 2]
    b1_y2, b2_y2 = boxes1[:, 3], boxes2[:, 3]
    b1_x2, b2_x2 = boxes1[:, 4], boxes2[:, 4]
    y1 = max.(b1_y1, b2_y1)
    x1 = max.(b1_x1, b2_x1)
    y2 = min.(b1_y2, b2_y2)
    x2 = min.(b1_x2, b2_x2)
    zs = zeros(size(y1, 1)) |> gpu
    intersection = max.(x2 .- x1 .+ 1, zs) .* max.(y2 .- y1 .+ 1, zs)
    b1_area = (b1_y2 .- b1_y1 .+ 1) .* (b1_x2 .- b1_x1 .+ 1)
    b2_area = (b2_y2 .- b2_y1 .+ 1) .* (b2_x2 .- b2_x1 .+ 1)
    unions = b1_area .+ b2_area .- intersection
    iou = intersection ./ unions
    iou = reshape(iou, boxes2_repeat[1], boxes1_repeat[1])
    return iou
end
=#

function clip_to_window(window, boxes)
    boxes[:, 1] = clamp.(boxes[:, 1], window[1], window[3])
    boxes[:, 2] = clamp.(boxes[:, 2], window[2], window[4])
    boxes[:, 3] = clamp.(boxes[:, 3], window[1], window[3])
    boxes[:, 4] = clamp.(boxes[:, 4], window[2], window[4])
    return boxes
end

#=
function clip_to_window2(window, boxes)
    @show window
    b1 = clamp.(boxes[:, 1], window[1], window[3])
    b2 = clamp.(boxes[:, 2], window[2], window[4])
    b3 = clamp.(boxes[:, 3], window[1], window[3])
    b4 = clamp.(boxes[:, 4], window[2], window[4])
    
    return hcat(b1, b2, b3, b4)
end
=#
#=
function refine_detections(rois, probs, deltas, window, config = nothing)
    # global grois = rois
    # global gprobs = probs
    # global gdeltas = deltas
    # global gwindow = window
    @show "in refine"
    @show size(probs)
    # gradient dropped here ?
    _, class_ids = findmax(Tracker.data(probs), dims = 1)
    @show size(deltas)
    @show size(probs)
    @show size(class_ids)
    # @show class_ids
    @warn "find max"
    idx = 1:length(class_ids)
    class_scores = vec(probs[class_ids])
    @show "maybe here"
    class_ids = vec(map(x -> x.I[1], class_ids))
    # @show idx
    # @show class_ids
    z = zip(idx, class_ids)
    # @show class_ids
    # @show idx
    # @show size(deltas)
    # @show size(rois)
    # @show size(probs)
    # deltas_specific = [deltas[x[1],:,x[2]] for x in z]
    # @show size(deltas_specific[1])
    # deltas_specific = reduce(hcat, deltas_specific)'
    # global gprobs = probs
    # global gclass_ids = class_ids
    # global gidx = idx
    # global gdeltas = deltas
    # deltas_specific = deltas[class_ids,:,idx]
    std_dev = [0.1 0.1 0.2 0.2] |> gpu
    @show "guess what"
    # deltas_specific .* std_dev
    deltas_specific = []
    # @show size(deltas_specific)
    for (m,k) in zip(idx, class_ids)
        push!(deltas_specific, deltas[k,:,m])
    end
    deltas_specific = cat(deltas_specific..., dims = 2)
    # @show size(deltas_specific' .* std_dev)
    # @show size(transpose(rois))
    refined_rois = apply_box_deltas(transpose(rois), transpose(deltas_specific) .* std_dev)
    @warn "applied deltas"
    height, width = 1024, 1024
    scale = [height width height width] |> gpu
    refined_rois = refined_rois .* scale
    refined_rois = clip_to_window2(window, refined_rois)
    @show typeof(refined_rois)
    refined_rois = round.(refined_rois)
    # @show refined_rois
    
    
    keep_bool = class_ids .> 0
    @show sum(keep_bool)
    DETECTION_MIN_CONFIDENCE = .5f0
    @show minimum(class_scores), mean(class_scores), maximum(class_scores)
    @show sum(class_scores .> DETECTION_MIN_CONFIDENCE)
    cs = copy(class_scores) |> cpu
    keep_bool = keep_bool .& (cs .>= DETECTION_MIN_CONFIDENCE)
    # @show sum(keep_bool), size(keep_bool)
    keep = findall(!iszero, keep_bool)
    @show keep
    pre_nms_class_ids = class_ids[keep]
    pre_nms_scores = class_scores[keep]
    # idxs = map(x -> x.I[1], findall(keep))
    pre_nms_rois = refined_rois[keep, :]
    
    unis = unique(pre_nms_class_ids) |> sort
    @warn "going to loop"
    nms_keep = Int[]
    for (i,class_id) in enumerate(unis)
	ixs = findall(pre_nms_class_ids .== class_id)
        
	ix_rois = pre_nms_rois[ixs, :]
	@show typeof(ix_rois)
	ix_scores = pre_nms_scores[ixs]
	order = sortperm(ix_scores, rev = true)
	ix_scores = ix_scores[order]
	ix_rois = ix_rois[order, :]
	# @show ix_rois
	DETECTION_NMS_THRESHOLD = 0.3f0
	class_keep = nms2(hcat(ix_rois .* 1.0f0, ix_scores), DETECTION_NMS_THRESHOLD)
	class_keep = keep[ixs[order[class_keep]]]
	if i==1
            push!(nms_keep, class_keep...)
        else
            push!(nms_keep, unique(class_keep)...)
        end
    end
    nms_keep = sort(nms_keep)
    keep = sort(intersect(keep, nms_keep))
    DETECTION_MAX_INSTANCES = 100
    roi_count = DETECTION_MAX_INSTANCES
	ends = min(roi_count, length(keep))
    top_ids = sortperm(class_scores[keep], rev = true)[1:ends]
    keep = keep[top_ids]


    @show size(refined_rois)
    @show size(class_ids)
    @show size(class_scores)

    # global grefined_rois = refined_rois
    # global gclass_ids = class_ids
    # global gclass_scores = class_scores
    # global gkeep = keep
    hcat(refined_rois[keep, :] .* 1.0f0, cu(class_ids[keep]) .* 1.0f0, class_scores[keep])

end
=#
#=
Loss Functions
=#

#=
function compute_rpn_class_loss(rpn_match, rpn_class_logits; labels = 1:80)
    anchor_class = Int.(rpn_match .== 1)
    indices = findall(!iszero, rpn_match .!= 0)
	
    rpn_class_logits = rpn_class_logits[:, indices, :]
    rpn_class_logits = dropdims(rpn_class_logits, dims = ndims(rpn_class_logits))
    anchor_class = anchor_class[indices]
    
    anchor_class = Flux.onehotbatch(anchor_class, 0:1) |> gpu
    Flux.logitcrossentropy(rpn_class_logits, anchor_class)
end
=#

function smooth_l1_loss(y, fx; δ = 1)
    α = abs(y - fx)
    abs(α) <= δ && return 0.5f0 * α ^ 2
    δ * α - (0.5f0 * δ ^ 2)
end

#=
huber_loss(y, ŷ; kwargs...) = smooth_l1_loss(y, ŷ, kwargs...)
=#
#=
function compute_rpn_bbox_loss(target_bbox, rpn_match, rpn_bbox)
    inds = findall(rpn_match .== 1)

    rpn_bbox = rpn_bbox[:, inds, :]
    rpn_bbox = dropdims(rpn_bbox, dims = ndims(rpn_bbox))

    target_bbox = target_bbox[1:size(rpn_bbox,2), :]
    target_bbox = transpose(target_bbox)

    # smooth L1 loss
    mean(smooth_l1_loss.(target_bbox, rpn_bbox))
end
=#
#=
function compute_mrcnn_class_loss(target_class_ids, pred_class_logits; labels = 0:80)
    if length(target_class_ids) > 0
	target_class_ids = Int.(target_class_ids)
	y = Flux.onehotbatch(target_class_ids, labels) |> gpu
	return Flux.logitcrossentropy(pred_class_logits, y)
    else
	return param(0.0f0)
    end
end
=#
#=
function compute_mrcnn_bbox_loss(target_deltas, target_class_ids, pred_bbox; labels = 0:80)
    if length(target_class_ids) > 0
	target_class_ids = map(y -> findall(x -> x == y, labels)[1], target_class_ids)
	positive_roi_ix = findall(target_class_ids .> 0)
        
	positive_roi_class_ids = target_class_ids[positive_roi_ix]
	target_deltas = target_deltas[positive_roi_ix, :]
	target_deltas = transpose(target_deltas)
	# pred_bbox = pred_bbox[positive_roi_class_ids, :, positive_roi_ix]
	bb = []
	for i = 1:length(positive_roi_ix)
	    a = pred_bbox[positive_roi_class_ids[i], :, i]
	    push!(bb, a)
	end
	pred_bbox = reduce(hcat, bb)
        
	mean(smooth_l1_loss.(pred_bbox, target_deltas))
    else
	return param(0.0f0)
    end
end
=#
#=
function compute_mrcnn_mask_loss(target_mask, target_class_ids, mrcnn_mask; labels = 0:80)
    if length(target_class_ids) > 0
	target_class_ids = map(y -> findall(x -> x == y, labels)[1], target_class_ids)
	positive_ix = findall(target_class_ids .> 0.f0)
	positive_class_ids = target_class_ids[positive_ix]
        
	y_true = target_mask[:,:,positive_ix]
	bb = []
	for i = 1:length(positive_ix)
	    a = mrcnn_mask[:,:,positive_class_ids[i], i]
	    push!(bb, a)
	end
	y_pred = cat(bb..., dims = 3)
		# y_pred = cpu(y_pred)
	# y_true = cpu(y_true)
        
	mean(bce(y_pred, y_true))
    else
	return param(0.0f0)
    end
end
=#
##################
# Data Generator #
##################

function mold_image(image, config = Nothing)
    MEAN_PIXEL = [123.7f0 / 255.0f0, 116.8f0 / 255.0f0, 103.9f0 / 255.0f0]
    # MEAN_PIXEL = [123.7f0, 116.8f0, 103.9]
    image[:,:,1] .-= MEAN_PIXEL[1]
    image[:,:,2] .-= MEAN_PIXEL[2]
    image[:,:,3] .-= MEAN_PIXEL[3]
    return image
end

function compose_image_meta(image_id, image_shape, window, active_class_ids)
    return ([image_id], image_shape, window, active_class_ids)
end

parse_image_meta(image_meta) = return image_meta

function mold_inputs(images, config = Nothing)
    IMAGE_MIN_DIM = 800 # just for testing
    IMAGE_MAX_DIM = 1024 # 128
    IMAGE_PADDING = true # just for testing
    NUM_CLASSES = 81 # just for testing - coco
    molded_images = []
    image_metas = []
    windows = []
    for image in images
	molded_image, window, scale, padding = resize_image(
	    image,
            min_dim = IMAGE_MIN_DIM,
            max_dim = IMAGE_MAX_DIM,
            padding = IMAGE_PADDING)

	molded_image = mold_image(molded_image)
	image_meta = compose_image_meta(0, size(image), window, zeros(Int32, NUM_CLASSES))
	push!(molded_images, molded_image)
	push!(windows, window)
	push!(image_metas, image_meta)
    end

    molded_images = cat(molded_images..., dims = 4)
    windows = reduce(hcat, windows)
    image_metas = reduce(hcat, image_metas)
    return molded_images, image_metas, windows
end
