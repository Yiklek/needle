add_rules("plugin.compile_commands.autoupdate", { outputdir = get_config("projectdir") })
add_rules("mode.debug", "mode.release")
set_languages("cxx17")
set_optimize("fastest")

add_requires("pybind11")

function install_python_module(target)
    local install_path = path.join(target:installdir(), "needle/backend_ndarray/")
    os.mkdir(install_path)
    os.cp(target:targetfile(), install_path)
end

target("clangd", function()
    set_default(false)
    set_kind("phony")
    local cuda = get_config("cuda")
    if cuda ~= nil then
        set_configvar("CUDAToolkit_LIBRARY_ROOT", cuda)
        add_configfiles(".clangd.in")
        set_configdir("$(projectdir)")
    end
end)

target("ndarray_backend_cpu", function()
    add_rules("python.library", { soabi = true })
    add_packages("pybind11")
    set_kind("shared")
    set_group("backend")
    add_files("src/ndarray_backend_cpu.cc")
    on_install(install_python_module)
end)

target("ndarray_backend_cuda", function()
    add_rules("python.library", { soabi = true })
    add_packages("pybind11")
    set_group("backend")
    set_kind("shared")
    add_files("src/ndarray_backend_cuda.cu")
    on_install(install_python_module)
end)
