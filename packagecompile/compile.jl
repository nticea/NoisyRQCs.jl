# Modeled on ITensors.jl/packagecompile

using PackageCompiler

export compile

default_compile_dir() = joinpath(homedir(), ".julia", "sysimages")

default_compile_filename() = "sys_noisyrqcs.so"

default_compile_path() = joinpath(default_compile_dir(), default_compile_filename())

function compile_note(; dir=default_compile_dir(), filename=default_compile_filename())
    path = joinpath(dir, filename)
    return """
    You will be able to start Julia with a compiled system image for the TJStudies project using:
    ```
    ~ julia --sysimage $path
    ```
    and you should see that the startup times and JIT compilation times are substantially improved.
    In unix, you can create an alias with the Bash command:
    ```
    ~ alias julia_tj="julia --sysimage $path"
    ```
    which you can put in your `~/.bashrc`, `~/.zshrc`, etc. Then you can start Julia with:
    ```
    ~ julia_tj
    ```
    Note that if you update any packages to a new version, for example with `using Pkg; Pkg.update("ITensors")`, you will need to run the `TJStudies.compile()` command again to recompile the new package versions.
    """
end

function compile(;
    dir::AbstractString=default_compile_dir(),
    filename::AbstractString=default_compile_filename()
)
    if !isdir(dir)
        println("""The directory "$dir" doesn't exist yet, creating it now.""")
        println()
        mkdir(dir)
    end
    path = joinpath(dir, filename)
    println(
        """Creating the system image "$path". This may take a few minutes.""",
    )
    # include all packages in project
    create_sysimage(
        sysimage_path=path,
        precompile_execution_file=joinpath(@__DIR__, "precompile-noisyrqcs.jl"),
    )
    println(compile_note(; dir=dir, filename=filename))
    return path
end
