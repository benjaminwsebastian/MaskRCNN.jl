# MaskRCNN
Julia implementation of Mask R-CNN using Knet

## For development:

Once you've installed julia 1.2 and clone the repo you can run the test suite like this:

```
~$ cd MaskRCNN
~/MaskRCNN$ julia
julia> ]
(v1.2) pkg> activate .
(MaskRCNN) pkg> instantiate
(MaskRCNN) pkg> test
```
If this doesn't work, then you might have to instantiate the test suite:

```
~/MaskRCNN$ cd test
~/MaskRCNN/test$ julia
julia> ]
(v1.2) pkg> activate .
(test) pkg> instantiate
(test) pkg> <Delete Key>
julia> exit()
```

## Working on
Utils  
ROI Align

## To do
1. Finish implementing utils  
2. RPN  
3. FPN  
4. ROI Align  
~~5. Resnet~~
6. Design Pipeline  
