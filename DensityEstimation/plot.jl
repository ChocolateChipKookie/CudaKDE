using Plots

function main()
    cd(@__DIR__)

    result_dir = "./data/out"
    plot_dir = "./plot"

    out_dirs = readdir(result_dir)

    for out_dir in out_dirs
        txt_files = readdir("$result_dir/$out_dir")

        for txt_file in txt_files
            x = Array{Float32, 1}(undef, 0)
            y = Array{Float32, 1}(undef, 0)

            lines = readlines("$result_dir/$out_dir/$txt_file")

            for line in lines
                sub = split(line, " ")
                push!(x, parse(Float32, sub[1]))
                push!(y, parse(Float32, sub[2]))
            end

            plot(x, y)
            new_file_path = "$plot_dir/$out_dir/$(split(txt_file, ".")[1]).png"

            savefig(new_file_path)
        end

        println(out_dir)
    end
end

main()
