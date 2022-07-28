#!/bin/bash

SCRIPTS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OCCA_SOURCE_DIR="$(cd "$(dirname "${SCRIPTS_DIR}")" && pwd)"
OCCA_SOURCE_SCRIPTS_DIR="${OCCA_SOURCE_DIR}/include/occa/scripts"

#---[ Library Information ]-------------
function uniqueAddToPath {
    local path="$1"
    local dir="$2"

    if [ ! -z "$path" ]; then
        case ":$path:" in
            *":$dir:"*)         ;; # Already in the path
            *) path="$path:$dir";;
        esac
    else
        path="$dir"
    fi

    echo "$path"
}

function removeDuplicatesInPath {
    local path="$1"

    for dir_ in "${path//:/ }"; do
        if ls "$dir_" > /dev/null 2>&1; then
            path=$(uniqueAddToPath "$path" "$dir_")
        fi
    done

    echo "$path"
}

function getIncludePath {
    local path="$1"

    path=$(echo "$path:" | sed 's/\/lib[^:]*:/\/include:/g')

    path=$(removeDuplicatesInPath "$path")

    echo "$path"
}

function dirWithFileInPath {
    local path="$1"
    local filename="$2"

    if [ ! -z "$path" ]; then
        for dir_ in ${path//:/ }; do
            if ls "$dir_/$filename" > /dev/null 2>&1; then
                echo "$dir_"
                return
            fi
        done
    fi

    echo ""
}

function dirWithFileInIncludePath {
    local path=$(getIncludePath "$1")
    local filename="$2"

    if [ ! -z "$path" ]; then
        for dir_ in ${path//:/ }; do
            if ls "$dir_/$filename" > /dev/null 2>&1; then
                echo "$dir_"
                return
            fi
        done
    fi

    echo ""
}

function defaultIncludePath {
    local mergedPaths=""

    mergedPaths+=":$OCCA_INCLUDE_PATH"
    mergedPaths+=":$CPLUS_INCLUDE_PATH"
    mergedPaths+=":$C_INCLUDE_PATH"
    mergedPaths+=":$INCLUDEPATH"
    mergedPaths+=":/opt/rocm*/opencl/include"
    mergedPaths+=":/opt/rocm*/include"
    mergedPaths+=":/usr/local/cuda*/include"
    mergedPaths+=":/Developer/NVIDIA/CUDA*/include"
    mergedPaths+=":/usr/local/cuda*/targets/*/include/"
    mergedPaths+=":/opt/cuda*/include/"
    mergedPaths+=":/usr/include"

    echo "${mergedPaths}"
}

function defaultLibraryPath {
    local mergedPaths=""

    mergedPaths+=":$OCCA_LIBRARY_PATH"
    mergedPaths+=":$LD_LIBRARY_PATH"
    mergedPaths+=":$DYLD_LIBRARY_PATH"
    mergedPaths+=":/opt/rocm*/opencl/lib/*"
    mergedPaths+=":/opt/rocm*/lib"
    mergedPaths+=":/usr/local/cuda*/lib*"
    mergedPaths+=":/usr/local/cuda*/lib*/stubs"
    mergedPaths+=":/opt/cuda*/lib*"
    mergedPaths+=":/lib:/usr/lib:/usr/lib32:/usr/lib64:"
    mergedPaths+=":/usr/lib/*-gnu/"

    echo "${mergedPaths}"
}

function dirWithLibrary {
    local libName="lib$1.so"
    local result=""

    result=$(dirWithFileInPath "$(defaultLibraryPath)" "$libName")

    if [ ! -z "$result" ]; then echo "$result"; return; fi

    if hash ldconfig 2> /dev/null; then
        echo $(ldconfig -p | grep -m 1 "$libName" | sed 's/.*=>\(.*\/\).*/\1/g')
        return
    fi

    case "$(uname)" in
        Darwin)
            if ls "/System/Library/Frameworks/$1.framework" > /dev/null 2>&1; then
                echo "Is A System/Library Framework"
                return
            fi
            if ls /Library/Frameworks/$1.framework > /dev/null 2>&1; then
                echo "Is A Library Framework"
                return
            fi;;
    esac

    echo ""
}

function dirWithHeader {
    local filename="$1"
    local result=""

    result=$(dirWithFileInPath "$(defaultIncludePath)" "$filename")
    if [ ! -z "$result" ]; then echo "$result"; return; fi

    result=$(dirWithFileInIncludePath "$(defaultLibraryPath)" "$filename")

    if [ ! -z "$result" ]; then echo "$result"; return; fi

    echo ""
}

function dirsWithHeaders {
    local headers="$1"
    local path=""

    if [ ! -z $headers ]; then
        for header in "${headers//:/ }"; do
            local inc=$(dirWithHeader $header)

            if [ ! -z $inc ]; then
                path=$(uniqueAddToPath $path $inc)
            else
                echo ""
                return
            fi
        done
    fi

    echo "$path"
}

function libraryFlags {
    local libName="$1"

    local libDir=$(dirWithLibrary $libName)
    local flags=""
    local isAFramework=0

    if [ -z "$libDir" ]; then echo ""; return; fi

    if [ "$libDir" == "Is A System/Library Framework" ]; then
        flags="-framework $libName"
        isAFramework=1
    elif [ "$libDir" == "Is A Library Framework" ]; then
        flags="-F/Library/Frameworks -framework $libName"
        isAFramework=1
    else
        flags="-L$libDir -l$libName"
    fi

    echo "$flags"
}


function headerFlags {
    local headers="$1"

    local incDirs
    local flags=""

    if [ ! -z "$headers" ]; then
        incDirs=$(dirsWithHeaders "$headers")

        if [ -z "$incDirs" ]; then echo ""; return; fi

        incDirs="${incDirs%?}"      # Remove the last :
        flags="-I${incDirs//:/ -I}" # : -> -I
    fi

    echo "$flags"
}
#=======================================


#---[ Compiler Information ]------------
function compilerVendor {
    local compiler="$1"

    local b_GNU=0
    local b_LLVM=1
    local b_Intel=2
    local b_Pathscale=3
    local b_IBM=4
    local b_PGI=5
    local b_HP=6
    local b_VisualStudio=7
    local b_Cray=8
    local b_PPC=9

    local testFilename="${OCCA_SOURCE_SCRIPTS_DIR}/findCompilerVendor.cpp"
    local binaryFilename="${OCCA_SOURCE_SCRIPTS_DIR}/findCompilerVendor"

    eval "${compiler}" "${testFilename}" -o "${binaryFilename}" > /dev/null 2>&1
    if [[ "$?" -ne 0 ]]; then
        echo "Failed to build findCompilerVendor:"
        eval "${compiler}" "${testFilename}" -o "${binaryFilename}"
    fi

    eval "${binaryFilename}"
    bit="$?"

    # C/C++ Compilers
    case "$bit" in
        "${b_GNU}")          echo GCC          ;;
        "${b_LLVM}")         echo LLVM         ;;
        "${b_Intel}")        echo INTEL        ;;
        "${b_IBM}")          echo IBM          ;;
        "${b_PGI}")          echo PGI          ;;
        "${b_Pathscale}")    echo PATHSCALE    ;;
        "${b_HP}")           echo HP           ;;
        "${b_Cray}")         echo CRAY         ;;
        "${b_VisualStudio}") echo VISUALSTUDIO ;;
        "${b_PPC}")          echo POWERPC      ;;
        *)                   echo N/A          ;;
    esac
}

function compilerCpp11Flags {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM|INTEL|PGI|POWERPC)
            echo "-std=c++17";;
        CRAY)      echo "-hstd=c++17"          ;;
        IBM)       echo "-qlanglvl=extended0x" ;;
        # Unknown
        PATHSCALE) echo "-std=c++17"           ;;
        # Unknown
        HP)        echo "-std=c++17"           ;;
        *) ;;
    esac
}

function compilerReleaseFlags {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM)   echo " -O3 -march=native -D __extern_always_inline=inline" ;;
        POWERPC)    echo " -O3 -mcpu=native -mtune=native -D __extern_always_inline=inline" ;;
        INTEL)      echo " -O3 -xHost"                                         ;;
        CRAY)       echo " -O3 -h intrinsics -fast"                            ;;
        IBM)        echo " -O3 -qhot=simd"                                     ;;
        PGI)        echo " -O3 -fast -Mipa=fast,inline -Msmartalloc"           ;;
        PATHSCALE)  echo " -O3 -march=auto"                                    ;;
        HP)         echo " +O3"                                                ;;
        *)          ;;
    esac
}

function compilerDebugFlags {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        N/A) ;;
        *)   echo " -g";;
    esac
}

function compilerPicFlag {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM|INTEL|PATHSCALE|CRAY|PGI|POWERPC)
            echo "-fPIC";;
        IBM) echo "-qpic";;
        HP)  echo "+z";;
        *)   echo ""  ;;
    esac
}

function compilerSharedFlag {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM|INTEL|PATHSCALE|CRAY|PGI|POWERPC)
            echo "-shared";;
        IBM) echo "-qmkshrobj";;
        HP)  echo "-b";;
        *)   echo "";;
    esac
}

function compilerPthreadFlag {
    echo "-lpthread"
}

function compilerOpenMPFlag {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM|POWERPC) echo "-fopenmp" ;;
        INTEL|PATHSCALE)  echo "-openmp"  ;;
        CRAY)             echo ""         ;;
        IBM)              echo "-qsmp"    ;;
        PGI)              echo "-mp"      ;;
        HP)               echo "+Oopenmp" ;;
        *)                echo ""         ;;
    esac
}

function compilerSupportsOpenMP {
    local compiler="$1"
    local ompFlag=$(compilerOpenMPFlag "${compiler}")

    local filename="${OCCA_SOURCE_SCRIPTS_DIR}/compilerSupportsOpenMP.cpp"
    local binary="${OCCA_SOURCE_SCRIPTS_DIR}/compilerSupportsOpenMP"

    rm -f "${binary}"

    # Test compilation
    "${compiler}" "${ompFlag}" "${filename}" -o "${binary}" > /dev/null 2>&1

    if [[ ! -a "${binary}" ]]; then
        echo 0
        return
    fi

    if [[ "$?" -eq 0 ]]; then
        # Test binary
        "${binary}"

        if [[ "$?" -eq 0 ]]; then
            echo 1
        else
            echo 0
        fi
    else
        echo 0
    fi

    if [ ! -z "${binary}" ]; then
        rm -f "${binary}"
    fi
}
#=======================================


#---[ Fortran Compiler Information ]----
function fCompilerVendor {
    local compiler="$1"

    local filename="${SCRIPTS_DIR}/compiler/findFortranCompilerVendor.F90"
    local binary="${SCRIPTS_DIR}/compiler/findFortranCompilerVendor"

    rm -f "$binary"
    "$compiler" "$filename" -o "$binary" > /dev/null 2>&1
    if [[ ! -a "$binary" ]]; then
        echo "N/A"
        return
    fi

    vendor=$("$binary" 2>&1)
    vendor=${vendor// /}
    echo $vendor
}

function fCompilerModuleDirFlag {
    local vendor=$(fCompilerVendor "$1")

    case "$vendor" in
        GCC|CRAY)            echo "-J"       ;;
        INTEL|PGI|PATHSCALE) echo "-module"  ;;
        IBM)                 echo "-qmoddir" ;;
        *)                   echo ""         ;;
    esac
}

function fCompilerCppFlag {
    local vendor=$(fCompilerVendor "$1")

    case "$vendor" in
        GCC)   echo "-lstdc++"   ;;
        CRAY)  echo ""           ;;
        INTEL) echo "-cxxlib"    ;;
        PGI)   echo "-pgc++libs" ;;
        *)     echo ""           ;;
    esac
}

function fCompilerSupportsMPI {
    local compiler="$1"

    local filename="${SCRIPTS_DIR}/compiler/fortranCompilerSupportsMPI.f90"
    local binary="${SCRIPTS_DIR}/compiler/fortranCompilerSupportsMPI"

    rm -f "$binary"

    # Test compilation
    "$compiler" "$filename" -o "$binary" > /dev/null 2>&1

    if [[ ! -a "$binary" ]]; then
        echo 0
        return
    fi

    rm -f "$binary"
    echo 1
}
#=======================================


#---[ Commands ]------------------------
function installOcca {
    if [ -z "${PREFIX}" ]; then
        return
    fi
    if [ -d "${PREFIX}" ]; then
      echo "Warning: Install PREFIX=${PREFIX} already exists."
    else
      mkdir -p "${PREFIX}"
    fi
    cp -r bin     "${PREFIX}"
    cp -r include "${PREFIX}"
    cp -r lib     "${PREFIX}"
}
#=======================================
