#=
Test utils operations:
=#

@testset "Utils" begin
    @testset "Bounding box manipulations" begin
        box = rand(10, 4)
        gt_box = rand(10, 4)
        new = box_refinement_graph(box, gt_box)
        @test (10, 4) == size(new)

        mask = rand(28, 28, 10)
        bboxes = extract_bboxes(mask)
        @test (10, 4) == size(bboxes)

        boxes1 = [0 0 2 2; 1 0 3 2]
        boxes2 = [0 0 1 1; 0 0 2 1; 0 0 4 4]
        
        overlaps = compute_overlaps_bbox(boxes1, boxes2)
        @test overlaps == [.25 .5 .25; 0 0.2 .25]

        windows = [0 0 2 2; 0 0 1 2; 0 0 2 2]

        clipped_boxes = clip_boxes(boxes2, windows)
        @test [0 0 1 1; 0 0 1 1; 0 0 2 2] == clipped_boxes

        boxes = clip_to_window([0 0 1 1], boxes2)
        @test [0 0 1 1; 0 0 1 1; 0 0 1 1] == boxes
    end

    @testset "Image manipulations" begin
        image = rand(1000, 1000, 3)

        resized_image = resize_image(image, min_dim = 800, max_dim = 1024)
        @test size(resized_image[1]) == (1024, 1024, 3)
        
        MEAN_PIXEL = [123.7f0 / 255.0f0, 116.8f0 / 255.0f0, 103.9f0 / 255.0f0]
        molded_image = mold_image(image)

        molded_image[:, :, 1] .+= MEAN_PIXEL[1]
        molded_image[:, :, 2] .+= MEAN_PIXEL[2]
        molded_image[:, :, 3] .+= MEAN_PIXEL[3]
        
        @test all(image[:, :, 1] .== molded_image[:, :, 1])
        @test all(image[:, :, 2] .== molded_image[:, :, 2])
        @test all(image[:, :, 3] .== molded_image[:, :, 3])

        images = [rand(1000, 1000, 3), rand(1000, 1000, 3), rand(1000, 1000, 3)]
        
        info = mold_inputs(images)
        @test (1024, 1024, 3, 3) == size(info[1])
        @test (1, 3) == size(info[2])
        @test (1, 3) == size(info[3])
    end

    @testset "Loss functions" begin
        
    end
    
    @testset "Resnet functions" begin
        for model_name in ("imagenet-resnet-50-dag", "imagenet-resnet-101-dag", "imagenet-resnet-152-dag")
            model = matconvnet(model_name, true)
            @test all(keys(model) .== ["meta", "params", "vars", "layers"])
        end
    end
end
