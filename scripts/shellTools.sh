#!/bin/bash

#---[ Library Information ]-------------

function uniqueAddToPath {
    local path=$1
    local dir=$2

    if [ ! -z $path ]; then
        case ":$path:" in
            *":$dir:"*)         ;; # Already in the path
            *) path="$path:$dir";;
        esac
    else
        path="$dir"
    fi

    echo $path
}

function removeDuplicatesInPath {
    local path=$1

    for dir_ in ${path//:/ }; do
        if ls $dir_ > /dev/null 2>&1; then
            path=$(uniqueAddToPath $path $dir_)
        fi
    done

    echo $path
}

function getIncludePath {
    local path=$1

    path=$(echo "$path:" | sed 's/\/lib[^:]*:/\/include:/g')

    path=$(removeDuplicatesInPath $path)

    echo $path
}

function dirWithFileInPath {
    local path=$1
    local filename=$2

    if [ ! -z $path ]; then
        for dir_ in ${path//:/ }; do
            if ls $dir_/$filename > /dev/null 2>&1; then
                echo $dir_
                return
            fi
        done
    fi

    echo ""
}

function dirWithFileInIncludePath {
    local path=$(getIncludePath $1)
    local filename=$2

    if [ ! -z $path ]; then
        for dir_ in ${path//:/ }; do
            if ls $dir_/$filename > /dev/null 2>&1; then
                echo $dir_
                return
            fi
        done
    fi

    echo ""
}

function dirWithLibrary {
    local libName="lib$1.so"
    local result=""

    local mergedLibPaths=""

    mergedLibPaths=$mergedLibPaths:"/lib:/usr/lib:/usr/lib32:/usr/lib64:"
    mergedLibPaths=$mergedLibPaths:"/usr/lib/*-gnu/"
    mergedLibPaths=$mergedLibPaths:$OCCA_LIBRARY_PATH
    mergedLibPaths=$mergedLibPaths:$LD_LIBRARY_PATH
    mergedLibPaths=$mergedLibPaths:$DYLD_LIBRARY_PATH

    result=$(dirWithFileInPath "$mergedLibPaths" $libName)

    if [ ! -z $result ]; then echo $result; return; fi

    if hash ldconfig 2> /dev/null; then
        echo $(ldconfig -p | command grep -m 1 $libName | sed 's/.*=>\(.*\/\).*/\1/g')
        return
    fi

    case "$(uname)" in
        Darwin)
            if ls /System/Library/Frameworks/$1.framework > /dev/null 2>&1; then
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

    local mergedPaths=""
    local mergedLibPaths=""

    mergedPaths=$mergedPaths:"/usr/local/cuda*/include"
    mergedPaths=$mergedPaths:"/Developer/NVIDIA/CUDA*/include"
    mergedPaths=$mergedPaths:"/usr/include"
    mergedPaths=$mergedPaths:$OCCA_INCLUDE_PATH
    mergedPaths=$mergedPaths:$CPLUS_INCLUDE_PATH
    mergedPaths=$mergedPaths:$C_INCLUDE_PATH
    mergedPaths=$mergedPaths:$INCLUDEPATH

    mergedLibPaths=$mergedLibPaths:"/usr/local/cuda*/lib*"
    mergedLibPaths=$mergedLibPaths:"/lib:/usr/lib:/usr/lib32:/usr/lib64:"
    mergedLibPaths=$mergedLibPaths:"/usr/lib/*-gnu/"
    mergedLibPaths=$mergedLibPaths:$OCCA_LIBRARY_PATH
    mergedLibPaths=$mergedLibPaths:$LD_LIBRARY_PATH
    mergedLibPaths=$mergedLibPaths:$DYLD_LIBRARY_PATH

    result=$(dirWithFileInPath "$mergedPaths" $filename)
    if [ ! -z $result ]; then echo $result; return; fi

    result=$(dirWithFileInIncludePath "$mergedLibPaths" $filename)

    if [ ! -z $result ]; then echo $result; return; fi

    echo ""
}

function dirsWithHeaders {
    local headers=$1
    local path=""

    if [ ! -z $headers ]; then
        for header in ${headers//:/ }; do
            local inc=$(dirWithHeader $header)

            if [ ! -z $inc ]; then
                path=$(uniqueAddToPath $path $inc)
            else
                echo ""
                return
            fi
        done
    fi

    echo $path
}

function libraryFlags {
    local libName=$1

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

    echo $flags
}


function headerFlags {
    local headers=$1

    local incDirs
    local flags=""

    if [ ! -z $headers ]; then
        incDirs=$(dirsWithHeaders $headers)

        if [ -z $incDirs ]; then echo ""; return; fi

        incDirs=${incDirs%?}        # Remove the last :
        flags="-I${incDirs//:/ -I}" # : -> -I
    fi

    echo $flags
}
#=======================================


#---[ Compiler Information ]------------
function mpiCompilerVendor {
    local mpiCompiler=$1
    local compiler=$($mpiCompiler --chichamanga 2>&1 > /dev/null | command grep -m 1 error | sed 's/\([^:]*\):.*/\1/g')

    echo $compiler
}

function compilerVendor {
    local compiler=$1

    case $compiler in
        mpi*) echo $(mpiCompilerVendor $compiler) ;;

        g++* | gcc*)       echo GCC          ;;
        clang*)            echo LLVM         ;;
        icc* | icpc*)      echo INTEL        ;;
        xlc*)              echo IBM          ;;
        pgcc* | pgc++*)    echo PGI          ;;
        pathcc* | pathCC*) echo PATHSCALE    ;;
        aCC*)              echo HP           ;;
        cc* | CC*)         echo CRAY         ;;
        cl*.exe*)          echo VISUALSTUDIO ;;
        *)                 echo N/A          ;;
    esac
}

function compilerReleaseFlags {
    local vendor=$(compilerVendor $1)

    case $vendor in
        GCC | LLVM) echo "-O3 -D __extern_always_inline=inline"     ;;
        INTEL)      echo "-O3 -xHost"                               ;;
        CRAY)       echo "-O3 -h intrinsics -fast"                  ;; # [-]
        IBM)        echo "-O3 -qhot=simd"                           ;; # [-]
        PGI)        echo "-O3 -fast -Mipa=fast,inline -Msmartalloc" ;; # [-]
        PATHSCALE)  echo "-O3 -march=auto"                          ;; # [-]
        HP)         echo "+O3"                                      ;; # [-]
        *)          echo ""                                         ;;
    esac
}

function compilerDebugFlags {
    local vendor=$(compilerVendor $1)

    case $vendor in
        N/A)           ;;
        *)   echo "-g" ;;
    esac
}

function compilerSharedBinaryFlags {
    local vendor=$(compilerVendor $1)

    case $vendor in
        GCC | LLVM | INTEL | PATHSCALE) echo "-fPIC -shared"          ;;
        CRAY)                           echo "-h PIC"                 ;; # [-]
        IBM)                            echo "-qpic=large -qmkshrobj" ;; # [-]
        PGI)                            echo "-fpic -shlib"           ;; # [-]
        HP)                             echo "+z -b"                  ;; # [-]
        *)                              echo ""                       ;;
    esac
}

function compilerOpenMPFlags {
    local vendor=$(compilerVendor $1)

    case $vendor in
        GCC   | LLVM)      echo "-fopenmp" ;;
        INTEL | PATHSCALE) echo "-openmp"  ;;
        CRAY)              echo ""         ;; # [-]
        IBM)               echo "-qsmp"    ;; # [-]
        PGI)               echo "-mp"      ;; # [-]
        HP)                echo "+Oopenmp" ;; # [-]
        *)                 echo ""         ;;
    esac
}

function compilerSupportsOpenMP {
    local compiler=$1
    local vendor=$(compilerVendor $compiler)
    local ompFlag=$(compilerOpenMPFlags $compiler)

    local filename=$OCCA_DIR/scripts/ompTest.cpp
    local binary=$OCCA_DIR/scripts/ompTest

    # Test compilation
    $compiler $ompFlag $filename -o $binary > /dev/null 2>&1

    if [[ $? -eq 0 ]]; then
        # Test binary
        $binary

        if [[ $? -eq 0 ]]; then
            echo 1
        else
            echo 0
        fi
    else
        echo 0
    fi

    if [ ! -z $binary ]; then
        rm -f $binary
    fi
}
#=======================================