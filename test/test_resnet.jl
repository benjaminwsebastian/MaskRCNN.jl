#=
Test resnet operations:
=#

@testset "ResNet" begin
    # define vars
    # use for testing
    @testset "Basic example" begin

        # Coming from https://github.com/denizyuret/Knet.jl/blob/master/examples/resnet/ -- license in src/resnet.jl

        function test_resnet(image, model, target)
            # "resnet.jl (c) Ilker Kesen, 2017. Classifying images with Deep Residual Networks."
            
            o = Dict(:model => model, :atype => "KnetArray{Float32}", :top => 5, :image => image)
            
            @test [(k,v) for (k,v) in o][[1,2,4]] == [(:atype, "KnetArray{Float32}"), (:top, 5), (:model, model)]

            atype = eval(Meta.parse(o[:atype]))
            
            model = matconvnet(o[:model])

            avgimg = model["meta"]["normalization"]["averageImage"]
            avgimg = convert(Array{Float32}, avgimg)
            description = model["meta"]["classes"]["description"]
            
            w, ms = get_params(model["params"], atype)
            
            img = imgdata(o[:image], avgimg)
            img = convert(atype, img)
            # get model by length of parameters
            modeldict = Dict(
                162 => (resnet50, "resnet50"),
                314 => (resnet101, "resnet101"),
                467 => (resnet152, "resnet152"))
            
            !haskey(modeldict, length(w)) && error("wrong resnet MAT file")
            resnet, name = modeldict[length(w)]

            @info("Testing classification with $name")
            y1 = resnet(w,img,ms)
            z1 = vec(Array(y1))
            s1 = sortperm(z1,rev=true)
            p1 = exp.(logp(z1))
            @test description[s1[1:o[:top]]][1] == target
            
            println()
        end
        
        elephant = download("http://home.mweb.co.za/pa/pak04857/uniweb/animalimages/elephantthumb.jpg")
        cat = "https://github.com/BVLC/caffe/raw/master/examples/images/cat.jpg"
        
        test_resnet(elephant, "imagenet-resnet-101-dag", "African elephant, Loxodonta africana")
        test_resnet(cat, "imagenet-resnet-50-dag", "gibbon, Hylobates lar") # Should guess incorretly
        test_resnet(cat, "imagenet-resnet-152-dag", "gibbon, Hylobates lar") # Should guess incorrectly
    end

end
