
        if [ ! $(uname) == "Darwin" ]; then
            echo "This script should be run on macOS ONLY."
            echo "It is meant to uninstall the Vulkan SDK on macOS."
            exit 1
        fi
        
        echo >> /tmp/VULKAN_UNINSTALL.log
        date >> /tmp/VULKAN_UNINSTALL.log
        
            echo "Removing /usr/local/bin/dxc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/dxc
            
            echo "Removing /usr/local/bin/spirv-objdump" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-objdump
            
            echo "Removing /usr/local/bin/gfxrecon-info" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-info
            
            echo "Removing /usr/local/bin/slangi" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/slangi
            
            echo "Removing /usr/local/bin/gfxrecon-optimize" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-optimize
            
            echo "Removing /usr/local/bin/gfx.slang" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfx.slang
            
            echo "Removing /usr/local/bin/spirv-val" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-val
            
            echo "Removing /usr/local/bin/gfxrecon-compress" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-compress
            
            echo "Removing /usr/local/bin/spirv-cross" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-cross
            
            echo "Removing /usr/local/bin/vkconfig" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/vkconfig
            
            echo "Removing /usr/local/bin/spirv-remap" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-remap
            
            echo "Removing /usr/local/bin/gfxrecon-convert" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-convert
            
            echo "Removing /usr/local/bin/gfxrecon-extract" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-extract
            
            echo "Removing /usr/local/bin/spirv-dis" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-dis
            
            echo "Removing /usr/local/bin/glslc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/glslc
            
            echo "Removing /usr/local/bin/spirv-lint" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-lint
            
            echo "Removing /usr/local/bin/spirv-opt" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-opt
            
            echo "Removing /usr/local/bin/glslang" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/glslang
            
            echo "Removing /usr/local/bin/spirv-reflect" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-reflect
            
            echo "Removing /usr/local/bin/gfxrecon-capture.py" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-capture.py
            
            echo "Removing /usr/local/bin/slang.slang" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/slang.slang
            
            echo "Removing /usr/local/bin/spirv-reflect-pp" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-reflect-pp
            
            echo "Removing /usr/local/bin/slangd" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/slangd
            
            echo "Removing /usr/local/bin/slangc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/slangc
            
            echo "Removing /usr/local/bin/spirv-cfg" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-cfg
            
            echo "Removing /usr/local/bin/spirv-lesspipe.sh" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-lesspipe.sh
            
            echo "Removing /usr/local/bin/spirv-reduce" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-reduce
            
            echo "Removing /usr/local/bin/dxc-3.7" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/dxc-3.7
            
            echo "Removing /usr/local/bin/gfxrecon-capture-vulkan.py" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-capture-vulkan.py
            
            echo "Removing /usr/local/bin/gfxrecon.py" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon.py
            
            echo "Removing /usr/local/bin/gfxrecon-replay" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/gfxrecon-replay
            
            echo "Removing /usr/local/bin/spirv-link" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-link
            
            echo "Removing /usr/local/bin/vulkaninfo" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/vulkaninfo
            
            echo "Removing /usr/local/bin/glslangValidator" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/glslangValidator
            
            echo "Removing /usr/local/bin/MoltenVKShaderConverter" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/MoltenVKShaderConverter
            
            echo "Removing /usr/local/bin/spirv-as" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/bin/spirv-as
            
            echo "Removing /usr/local/lib/libspirv-cross-util.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-util.a
            
            echo "Removing /usr/local/lib/libspirv-cross-c-shared.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-c-shared.dylib
            
            echo "Removing /usr/local/lib/libMachineIndependent.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libMachineIndependent.a
            
            echo "Removing /usr/local/lib/libshaderc_shared.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libshaderc_shared.dylib
            
            echo "Removing /usr/local/lib/libslang-rt.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libslang-rt.dylib
            
            echo "Removing /usr/local/lib/libSPIRV-Tools-diff.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools-diff.a
            
            echo "Removing /usr/local/lib/libslang-glslang.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libslang-glslang.dylib
            
            echo "Removing /usr/local/lib/libglslang.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang.dylib
            
            echo "Removing /usr/local/lib/libdxcompiler.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libdxcompiler.dylib
            
            echo "Removing /usr/local/lib/libSPIRV-Tools-lint.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools-lint.a
            
            echo "Removing /usr/local/lib/libVkLayer_khronos_validation.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_khronos_validation.dylib
            
            echo "Removing /usr/local/lib/libVkLayer_api_dump.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_api_dump.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-c-shared.0.67.0.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-c-shared.0.67.0.dylib
            
            echo "Removing /usr/local/lib/libvulkan.1.4.321.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libvulkan.1.4.321.dylib
            
            echo "Removing /usr/local/lib/libgfx.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libgfx.dylib
            
            echo "Removing /usr/local/lib/libglslang.15.4.0.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang.15.4.0.dylib
            
            echo "Removing /usr/local/lib/libVkLayer_screenshot.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_screenshot.dylib
            
            echo "Removing /usr/local/lib/libVkLayer_khronos_profiles.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_khronos_profiles.dylib
            
            echo "Removing /usr/local/lib/libvulkan.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libvulkan.dylib
            
            echo "Removing /usr/local/lib/libSPIRV-Tools-link.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools-link.a
            
            echo "Removing /usr/local/lib/libspirv-cross-glsl.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-glsl.a
            
            echo "Removing /usr/local/lib/libglslang-default-resource-limits.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang-default-resource-limits.a
            
            echo "Removing /usr/local/lib/libSPIRV-Tools-shared.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools-shared.dylib
            
            echo "Removing /usr/local/lib/libVkLayer_khronos_shader_object.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_khronos_shader_object.dylib
            
            echo "Removing /usr/local/lib/libSPVRemapper.15.4.0.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPVRemapper.15.4.0.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-reflect.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-reflect.a
            
            echo "Removing /usr/local/lib/libVkLayer_khronos_synchronization2.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_khronos_synchronization2.dylib
            
            echo "Removing /usr/local/lib/libSPIRV.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV.dylib
            
            echo "Removing /usr/local/lib/libSPIRV-Tools.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools.a
            
            echo "Removing /usr/local/lib/libslang.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libslang.dylib
            
            echo "Removing /usr/local/lib/libshaderc_combined.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libshaderc_combined.a
            
            echo "Removing /usr/local/lib/libSPVRemapper.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPVRemapper.dylib
            
            echo "Removing /usr/local/lib/libglslang-default-resource-limits.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang-default-resource-limits.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-c.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-c.a
            
            echo "Removing /usr/local/lib/libshaderc.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libshaderc.a
            
            echo "Removing /usr/local/lib/libSPIRV-Tools-reduce.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools-reduce.a
            
            echo "Removing /usr/local/lib/libSPVRemapper.15.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPVRemapper.15.dylib
            
            echo "Removing /usr/local/lib/libMoltenVK.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libMoltenVK.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-cpp.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-cpp.a
            
            echo "Removing /usr/local/lib/libSPIRV.15.4.0.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV.15.4.0.dylib
            
            echo "Removing /usr/local/lib/libOSDependent.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libOSDependent.a
            
            echo "Removing /usr/local/lib/libshaderc_shared.1.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libshaderc_shared.1.dylib
            
            echo "Removing /usr/local/lib/libglslang-default-resource-limits.15.4.0.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang-default-resource-limits.15.4.0.dylib
            
            echo "Removing /usr/local/lib/libSPIRV-Tools-opt.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV-Tools-opt.a
            
            echo "Removing /usr/local/lib/libGenericCodeGen.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libGenericCodeGen.a
            
            echo "Removing /usr/local/lib/libspirv-cross-core.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-core.a
            
            echo "Removing /usr/local/lib/libslang-glsl-module.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libslang-glsl-module.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-msl.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-msl.a
            
            echo "Removing /usr/local/lib/libglslang.15.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang.15.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-hlsl.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-hlsl.a
            
            echo "Removing /usr/local/lib/libSPIRV.15.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV.15.dylib
            
            echo "Removing /usr/local/lib/libspirv-cross-c-shared.0.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libspirv-cross-c-shared.0.dylib
            
            echo "Removing /usr/local/lib/libshaderc_util.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libshaderc_util.a
            
            echo "Removing /usr/local/lib/MoltenVK.xcframework" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/MoltenVK.xcframework
            
            echo "Removing /usr/local/lib/libglslang-default-resource-limits.15.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang-default-resource-limits.15.dylib
            
            echo "Removing /usr/local/lib/libSPVRemapper.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPVRemapper.a
            
            echo "Removing /usr/local/lib/libvulkan.1.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libvulkan.1.dylib
            
            echo "Removing /usr/local/lib/libVkLayer_gfxreconstruct.dylib" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libVkLayer_gfxreconstruct.dylib
            
            echo "Removing /usr/local/lib/libSPIRV.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libSPIRV.a
            
            echo "Removing /usr/local/lib/libglslang.a" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/libglslang.a
            
            echo "Removing /usr/local/include/dxc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/dxc
            
            echo "Removing /usr/local/include/shaderc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/shaderc
            
            echo "Removing /usr/local/include/slang" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/slang
            
            echo "Removing /usr/local/include/spirv-tools" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/spirv-tools
            
            echo "Removing /usr/local/include/spirv" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/spirv
            
            echo "Removing /usr/local/include/spirv_cross" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/spirv_cross
            
            echo "Removing /usr/local/include/glslang" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/glslang
            
            echo "Removing /usr/local/include/vk_video" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/vk_video
            
            echo "Removing /usr/local/include/MoltenVK" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/MoltenVK
            
            echo "Removing /usr/local/include/vulkan" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/include/vulkan
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools-tools" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools-tools
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_c_shared" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_c_shared
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_util" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_util
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_core" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_core
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_reflect" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_reflect
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools-link" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools-link
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools-opt" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools-opt
            
            echo "Removing /usr/local/lib/cmake/vulkan/glslang" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/glslang
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_msl" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_msl
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_cpp" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_cpp
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools-diff" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools-diff
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_glsl" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_glsl
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Headers" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Headers
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_c" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_c
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools-reduce" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools-reduce
            
            echo "Removing /usr/local/lib/cmake/vulkan/SPIRV-Tools-lint" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/SPIRV-Tools-lint
            
            echo "Removing /usr/local/lib/cmake/vulkan/spirv_cross_hlsl" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/cmake/vulkan/spirv_cross_hlsl
            
            echo "Removing /usr/local/lib/pkgconfig/SPIRV-Tools.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/SPIRV-Tools.pc
            
            echo "Removing /usr/local/lib/pkgconfig/spirv-cross-c-shared.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/spirv-cross-c-shared.pc
            
            echo "Removing /usr/local/lib/pkgconfig/SPIRV-Tools-shared.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/SPIRV-Tools-shared.pc
            
            echo "Removing /usr/local/lib/pkgconfig/vulkan.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/vulkan.pc
            
            echo "Removing /usr/local/lib/pkgconfig/shaderc.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/shaderc.pc
            
            echo "Removing /usr/local/lib/pkgconfig/SPIRV-Headers.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/SPIRV-Headers.pc
            
            echo "Removing /usr/local/lib/pkgconfig/spirv-cross-c.pc" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/lib/pkgconfig/spirv-cross-c.pc
            
            echo "Removing /usr/local/share/vulkan" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /usr/local/share/vulkan
            
            echo "Removing /Applications/vkconfig-gui.app" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /Applications/vkconfig-gui.app
            
            echo "Removing /Applications/vulkanCapsViewer.app" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /Applications/vulkanCapsViewer.app
            
            echo "Removing /Applications/vkcubepp.app" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /Applications/vkcubepp.app
            
            echo "Removing /Applications/vkcube.app" | tee -a /tmp/VULKAN_UNINSTALL.log
            rm -rf /Applications/vkcube.app
            