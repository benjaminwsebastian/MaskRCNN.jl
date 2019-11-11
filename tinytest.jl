using Test, ArgParse, MAT

#=
The point of this script is to run a component of the test suite while not running the whole thing as the start up overhead is unreasonably long.

Environment requirements: Test, MAT

Example:
julia tinytest.jl --f test/test_utils.jl
=#

###########
## INPUT ##
###########

function parse_commandline()
    s = ArgParseSettings()    
    @add_arg_table s begin
        "--f"
        arg_type = String
        help = "input data file"
        required = true
    end
    
    return parse_args(s)
end    

##########
## MAIN ##
##########

args = parse_commandline()

base = split(args["f"], "_")[end]

@testset "Tiny Test" begin
    include(string(pwd(), "/src/", base)) # source file
    include(string(pwd(), "/", args["f"])) # test file
    
end
