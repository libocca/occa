#!/bin/bash

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

    result=$(dirWithFileInPath "$OCCA_LIBRARY_PATH" $libName)

    if [ ! -z $result ]; then echo $result; return; fi

    if hash ldconfig 2> /dev/null; then
        echo $(ldconfig -p | /bin/grep -m 1 $libName | sed 's/.*=>\(.*\/\).*/\1/g')
        return
    fi

    local result=$(dirWithFileInPath "$LD_LIBRARY_PATH" $libName)

    if [ ! -z $result ]; then echo $result; return; fi

    local result=$(dirWithFileInPath "$DYLD_LIBRARY_PATH" $libName)

    if [ ! -z $result ]; then echo $result; return; fi

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

    result=$(dirWithFileInPath "$OCCA_INCLUDE_PATH" $filename)

    if [ ! -z $result ]; then echo $result; return; fi

    result=$(dirWithFileInPath "$CPLUS_INCLUDE_PATH" $filename)

    if [ ! -z $result ]; then echo $result; return; fi

    result=$(dirWithFileInPath "$C_INCLUDE_PATH" $filename)

    if [ ! -z $result ]; then echo $result; return; fi

    result=$(dirWithFileInPath "$INCLUDEPATH" $filename)

    if [ ! -z $result ]; then echo $result; return; fi

    result=$(dirWithFileInIncludePath "$LD_LIBRARY_PATH" $filename)

    if [ ! -z $result ]; then echo $result; return; fi

    result=$(dirWithFileInIncludePath "$DYLD_LIBRARY_PATH" $filename)

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

function libraryAndHeaderFlags {
    local libName=$1
    local headers=$2

    local libDir=$(dirWithLibrary $libName)
    local incDirs
    local flags
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

    if [ $isAFramework -eq 1 ]; then
        echo $flags
        return
    fi

    if [ ! -z $headers ]; then
        incDirs=$(dirsWithHeaders $headers)

        if [ -z $incDirs ]; then echo ""; return; fi

        incDirs=${incDirs%?}               # Remove the last :
        flags="$flags -I${incDirs//:/ -I}" # : -> -I
    fi

    echo $flags
}
