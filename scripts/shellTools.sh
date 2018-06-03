# The MIT License (MIT)
#
# Copyright (c) 2014-2018 David Medina and Tim Warburton
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


OCCA_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"

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
    mergedPaths+=":/opt/rocm/opencl/include"
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
    mergedPaths+=":/opt/rocm/opencl/lib/*"
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
function getPath {
    echo "${1%/*}"
}

function stripPath {
    echo "${1##*/}"
}

function resolveRelativePath {
    local from="$1"
    local to="$2"

    if [[ "$to" == /* ]]; then
        echo "$to"
    else
        echo $(getPath "$from")/"$to"
    fi
}

function manualReadlink {
    if [[ $(command -v readlink) == "" ]]; then
        pushd `dirname "$1"` > /dev/null
        SCRIPTPATH=`pwd -P`
        popd > /dev/null
    else
        case "$(uname)" in
            Darwin) readlink    "$1";;
            *)      readlink -f "$1";;
        esac
    fi
}

function manualWhich {
    local input="$1"

    local typeOutput=$(type "$input" 2> /dev/null)

    if [[ $typeOutput == *" is hashed "* ]]; then
        local mWhich=$(type "$input" 2> /dev/null | sed "s/.*(\(.*\)).*/\1/g")
    else
        local mWhich=$(type "$input" 2> /dev/null | sed "s/.* is \(.*\)/\1/g")
    fi

    if [ ! -z "$mWhich" ]; then
        echo "$mWhich"
    else
        echo "$input"
    fi
}

function realCommand {
    local a=$(manualWhich "$1")
    local b

    case "$(uname)" in
        Darwin) b="$(manualReadlink $a)";;
        *)      b="$(manualReadlink $a)";;
    esac

    if [ -z "$b" ]; then
        echo "$a"
        return
    fi

    while [ "$a" != "$b" ]; do
        b=$(resolveRelativePath "$a" "$b")
        a=$(manualWhich "$b")

        case "$(uname)" in
            Darwin) b="$(manualReadlink $a)";;
            *)      b="$(manualReadlink $a)";;
        esac

        if [ -z "$b" ]; then
            echo "$a"
            return
        fi
    done

    echo "$a"
}

function unaliasCommand {
    typeOutput=$(type "$1" 2> /dev/null)

    aliasedTo=$(echo "$typeOutput" | grep -m 1 "$1 is aliased to" | sed "s/[^\`]*\`\([^ \t']*\)[ \t']/\1/g")

    if [ ! -z "$aliasedTo" ]; then
        echo "$aliasedTo"
        return
    fi

    echo "$1"
}

function compilerName {
    local chosenCompiler="$1"
    local realCompiler=$(realCommand "$chosenCompiler")
    local unaliasedCompiler=$(unaliasCommand "$realCompiler")
    local strippedCompiler=$(stripPath "$unaliasedCompiler")
    echo "$strippedCompiler"
}

function compilerVendor {
    local chosenCompiler="$1"
    local compiler=$(compilerName "$1")

    # Fortran Compilers
    case "$compiler" in
        gfortran*)  echo GCC      ; return;;
        ifort*)     echo INTEL    ; return;;
        ftn*)       echo CRAY     ; return;;
        xlf*)       echo IBM      ; return;;
        pgfortran*) echo PGI      ; return;;
        pathf9*)    echo PATHSCALE; return;;
    esac

    local b_GNU=0
    local b_LLVM=1
    local b_Intel=2
    local b_Pathscale=3
    local b_IBM=4
    local b_PGI=5
    local b_HP=6
    local b_VisualStudio=7
    local b_Cray=8

    local testFilename="${OCCA_DIR}/scripts/compilerVendorTest.cpp"
    local binaryFilename="${OCCA_DIR}/scripts/compilerVendorTest"

    eval "${chosenCompiler}" "${testFilename}" -o "${binaryFilename}" > /dev/null 2>&1
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
        *)                   echo N/A          ;;
    esac
}

function compilerReleaseFlags {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM)   echo "-O3 -D __extern_always_inline=inline"     ;;
        INTEL)      echo "-O3 -xHost"                               ;;
        CRAY)       echo "-O3 -h intrinsics -fast"                  ;;
        IBM)        echo "-O3 -qhot=simd"                           ;;
        PGI)        echo "-O3 -fast -Mipa=fast,inline -Msmartalloc" ;;
        PATHSCALE)  echo "-O3 -march=auto"                          ;;
        HP)         echo "+O3"                                      ;;
        *)          echo ""                                         ;;
    esac
}

function compilerDebugFlags {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        N/A)                   ;;
        *)   echo "-g" ;;
    esac
}

function compilerPicFlag {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM|INTEL|PATHSCALE|CRAY|PGI)
            echo "-fPIC";;
        IBM) echo "-qpic";;
        HP)  echo "+z";;
        *)   echo ""  ;;
    esac
}

function compilerSharedFlag {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|LLVM|INTEL|PATHSCALE|CRAY|PGI)
            echo "-shared";;
        IBM) echo "-qmkshrobj";;
        HP)  echo "-b"     ;;
        *)   echo ""       ;;
    esac
}

function compilerPthreadFlag {
    echo "-lpthread"
}

function compilerOpenMPFlag {
    local vendor=$(compilerVendor $1)

    case "$vendor" in
        GCC|LLVM)        echo "-fopenmp" ;;
        INTEL|PATHSCALE) echo "-openmp"  ;;
        CRAY)            echo ""         ;;
        IBM)             echo "-qsmp"    ;;
        PGI)             echo "-mp"      ;;
        HP)              echo "+Oopenmp" ;;
        *)               echo ""         ;;
    esac
}

function fCompilerModuleDirFlag {
    local vendor=$(compilerVendor "$1")

    case "$vendor" in
        GCC|CRAY)            echo "-J"       ;;
        INTEL|PGI|PATHSCALE) echo "-module"  ;;
        IBM)                 echo "-qmoddir" ;;
        *)                   echo ""         ;;
    esac
}

function compilerSupportsOpenMP {
    local compiler="$1"
    local vendor=$(compilerVendor "${compiler}")
    local ompFlag=$(compilerOpenMPFlag "${compiler}")

    local filename="${OCCA_DIR}"/scripts/openmpTest.cpp
    local binary="${OCCA_DIR}"/scripts/openmpTest

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

function compilerSupportsMPI {
    local compiler="$1"

    local filename="${OCCA_DIR}"/scripts/mpiTest.cpp
    local binary="${OCCA_DIR}"/scripts/mpiTest

    rm -f "${binary}"

    # Test compilation
    "${compiler}" "${filename}" -o "${binary}" > /dev/null 2>&1

    if [[ ! -a "${binary}" ]]; then
        echo 0
        return
    fi

    rm -f "${binary}"
    echo 1
}
#=======================================


#---[ System Information ]--------------
function getFieldFrom {
    local command_="$1"
    shift;
    local field="$@"

    if hash grep 2> /dev/null; then
        echo $(LC_ALL=C; $command_ | \
                   grep -i -m 1 "^$field" | \
                   sed "s/.*:[ \t]*\(.*\)/\1/g")
        return
    fi

    echo ""
}

function getLSCPUField {
    local field="$@"

    if hash lscpu 2> /dev/null; then
        getFieldFrom "lscpu" "$field"
        return
    fi

    echo ""
}

function getCPUINFOField {
    local field="$@"

    if hash cat 2> /dev/null; then
        getFieldFrom "cat /proc/cpuinfo" "$field"
        return
    fi

    echo ""
}
#=======================================


#---[ Commands ]------------------------
function installOcca {
    if [ -z "${PREFIX}" ]; then
        return
    fi
    mkdir -p "${PREFIX}"
    cp -r bin     "${PREFIX}/bin"
    cp -r include "${PREFIX}/include"
    cp -r scripts "${PREFIX}/scripts"
    cp -r lib     "${PREFIX}/lib"
}
#=======================================
