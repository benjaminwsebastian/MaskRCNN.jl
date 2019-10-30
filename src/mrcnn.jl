#__precompile__()

module mrcnn

using ArgCheck
using Knet
using Pkg; Pkg.build("Knet")
# Remove these later

#=
Julia implementation of Mask R-CNN
=#

include("rpn.jl")
include("fpn.jl")
inclide("resnet.jl")


struct MaskRCNN
    rpn
    fpn
    anchors
end

function config(config = nothing)
    # Copying most of the config from matterport/mask_RCNN
    # Define resnet
    # Define FPN

    # Can change name depending on experiement
    NAME = nothing

    # Number of GPUs to use. When using only a CPU, this needs to be set to 1.
    GPU_COUNT = 1

    # Number of images to train with on each GPU. A 12GB GPU can typically handle 2 images of 1024x1024
    # REMOVE ME BEFORE PUBLISHING - currently have IMAGES_PER_GPU set to 1 because my GPU (1070ti) has 8GB memory
    IMAGES_PER_GPU = 1

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 1000

    # Number of validation steps to run at the end of every training epoch
    VALIDATION_STEPS = 50

    # Backbone network architecture
    # This supports resnet50, resnet101, and resnet152
    BACKBONE = "resnet101"

    # The strides of each layer of the FPN. These values are based on the default resnet101 backbone
    BACKBONE_STRIDES = [4, 8, 16, 32, 64]

j    # The shape of the backbone
    BACKBONE_SHAPES = [256 256; 128 128; 64 64; 32 32; 166 16;]
    
    # Size of the fully-connected layers in the classifier
    FPN_CLASSIF_FC_LAYERS_SIZE = 1024

    # Size of the top-down layers used to build the feature pyramid
    TOP_DOWN_PYRAMID_SIZE = 256

    # Number of classification classes (including background) -> overwite this in implementation
    NUM_CLASSES = 1

    # Length of square anchor side in pixels
    RPN_ANCHORS_SCALES = [32, 64, 128, 256, 512]

    # Ratios of anchors at each cell (width/height)
    # A value of 1 represents a square anchor, and 0.5 is a wide anchor
    RPN_ANCHOR_RATIOS = [0.5, 1, 2]

    # Anchor stride
    # If 1, then anchors are created for each cell in the backbone feature map
    # If 2, then anchors are created for every other cell, and so on.
    RPN_ANCHOR_STRIDE = 1
    
    # Non-max suppression threshold tot filter RPN proposals
    # You can increase this during training to generate more proposals
    RPN_NMS_THRESHOLD = 0.7

    # How many anchors per image to use for RPN training
    RPN_TRAIN_ANCHORS_PER_IMAGE = 256

    # Define pool size
    POOL_SIZE = 7
    
    
end # module
