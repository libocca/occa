#!/bin/bash

#---[ Helper Methods ]------------------
__occa_debug_echo() {
    if [ ! -z "${OCCA_DEBUG_CLI}" ]; then
        echo "$@" >&2-
    fi
}

__occa_init_command() {
    commandOptions=("${options[@]}")
    commandFlags=("${flags[@]}")
    currentFlag=""
    if [ "${#command[@]}" -ne "${#prevCommand[@]}" ]; then
        prevCommand=("${command[@]}")
        usedFlags=()
        allUsedArgs=()
        usedArgs=()
    fi
}

__occa_used_args() {
    # Only return args that were used for the current flag
    args=()
    for arg in "${allUsedArgs[@]}"; do
        if [[ "${arg}" == "${currentFlag} "* ]]; then
            args+=($(echo "${arg}" | sed "s/^${currentFlag} //g"))
        fi
    done
    echo "${args[@]}"
}

__occa_add_used_arg() {
    usedArgs+=("$1")
    # Keep track of flag-specific args
    allUsedArgs+=("${currentFlag} $1")
}

__occa_init_flag() {
    currentFlag="$1"
    usedFlags+=("$@")
    usedArgs=($(__occa_used_args))
    # Only add the command's options/flags once we
    #   have at least 1 input
    if [ "${#usedArgs[@]}" -eq 0 ]; then
        return
    fi
    options+=("${commandOptions[@]}")
    flags+=("${commandFlags[@]}")
    expansions+=(args)
}

__occa_next_input() {
    # inputs[0].strip()
    echo "${inputs[0]}" | sed 's/\s*\([^\s]*\)\s/\1/g'
}

__occa_shift_inputs() {
    inputs=("${inputs[@]:1}")
    nextInput=$(__occa_next_input)
}

__occa_reuse_flags() {
    local usedInputs=("$@")
    local inputs=("${usedFlags[@]}")
    usedFlags=($(__occa_get_unused))
}

__occa_command_name() {
    local commandPath=()
    local lastFlag=""
    for word in "${command[@]}"; do
        case "${word}" in
            -*) lastFlag="${word}";;
            *)
                if [ -z "${lastFlag}" ]; then
                    commandPath+=("${word}")
                fi
        esac
    done
    if [ -n "${lastFlag}" ]; then
        commandPath+=("${lastFlag}")
    fi
    OLD_IFS="${IFS}"
    IFS="_"
    cmd="_${commandPath[*]}"
    IFS="${OLD_IFS}"
    echo "${cmd}"
}

__occa_run_command() {
    cmd="$(__occa_command_name)"
    __occa_debug_echo "__occa_run_command | cmd : [${cmd}]"
    "${cmd}"
}

__occa_autocomplete() {
    local nextArg="${nextInput}"
    __occa_debug_echo "__occa_autocomplete | options: [$*]"
    __occa_debug_echo "__occa_autocomplete | nextArg: [${nextArg}]"
    COMPREPLY=($(compgen -W "$*" -- "${nextArg}"))
    if [ -z "${nextArg}" ]; then
        local foundNonFlag=false
        for choice in $*; do
            case "${choice}" in
                -*) ;;
                *)
                    foundNonFlag=true
                    break;;
            esac
        done
        if [ "${foundNonFlag}" = false ]; then
            COMPREPLY+=('')
        fi
    fi
    compIsDone=true
}

__occa_get_unused() {
    __occa_debug_echo "__occa_get_unused | usedInputs   : [${usedInputs[@]}]"
    __occa_debug_echo "__occa_get_unused | inputs       : [${inputs[@]}]"
    for input in "${usedInputs[@]}"; do
        for i in "${!inputs[@]}"; do
            if [ "${input}" = "${inputs[$i]}" ]; then
                unset "inputs[$i]"
                break
            fi
        done
    done
    __occa_debug_echo "__occa_get_unused | unusedInputs : [${inputs[@]}]"
    echo "${inputs[@]}"
}

__occa_unused_args() {
    local usedInputs=("${usedArgs[@]}")
    local inputs=("$@")
    __occa_get_unused
}

__occa_unused_flags() {
    local usedInputs=("${usedFlags[@]}")
    local inputs=("${flags[@]}")
    __occa_get_unused
}

__occa_input_in() {
    if [ "${#inputs[@]}" -gt 1 ]; then
        for arg in $@; do
            if [ "${arg}" = "${nextInput}" ]; then
                echo 1
                return
            fi
        done
    fi
}

__occa_compgen_with_args() {
    __occa_debug_echo "__occa_compgen_with_args | options   : [${options[@]}]"
    __occa_debug_echo "__occa_compgen_with_args | flags     : [${flags[@]}]"
    __occa_debug_echo "__occa_compgen_with_args | inputs    : [${inputs[@]}]"
    __occa_debug_echo "__occa_compgen_with_args | usedFlags : [${usedFlags[@]}]"

    # If a flag is unique, don't show it as a choice
    local unusedFlags=$(__occa_unused_flags)

    if [ $(__occa_input_in "${options[@]}" "${unusedFlags[@]}") ]; then
        command+=("${nextInput}")
        __occa_shift_inputs
        __occa_run_command
    else
        compgenOptions+=("${options[@]}" "${unusedFlags[@]}")
    fi
}

__occa_compgen_function() {
    __occa_debug_echo "__occa_compgen_function | function    : [${expansionFunction}]"
    local funcOptions=($(${expansionFunction} 2> /dev/null))
    __occa_debug_echo "__occa_compgen_function | funcOptions : [${funcOptions[@]}]"
    local unusedFuncOptions=$(__occa_unused_args "${funcOptions[@]}")
    __occa_debug_echo "__occa_compgen_function | unusedFuncOptions : [${unusedFuncOptions[@]}]"

    if [ $(__occa_input_in "${unusedFuncOptions[@]}") ]; then
        __occa_add_used_arg "${nextInput}"
        __occa_shift_inputs
        __occa_run_command
    else
        compgenOptions+=("${unusedFuncOptions[@]}")
    fi
}

__occa_compgen_file() {
    files=($(\ls))
    local unusedFiles=$(__occa_unused_args "${files[@]}")
    __occa_debug_echo "__occa_compgen_file | nextInput   : ${nextInput}"
    __occa_debug_echo "__occa_compgen_file | files       : ${files[@]}"
    __occa_debug_echo "__occa_compgen_file | unusedFiles : ${unusedFiles[@]}"

    if [ $(__occa_input_in "${unusedFiles[@]}") ]; then
        __occa_add_used_arg "${nextInput}"
        __occa_shift_inputs
        __occa_compgen
    else
        compgenOptions+=("${unusedFiles[@]}")
    fi
}

__occa_compgen() {
    __occa_debug_echo ""
    __occa_debug_echo "__occa_compgen | command    : [${command[@]}]"
    __occa_debug_echo "__occa_compgen | inputs     : [${inputs[@]}]"
    __occa_debug_echo "__occa_compgen | expansions : [${expansions[@]}]"
    __occa_debug_echo "__occa_compgen | options    : [${options[@]}]"
    __occa_debug_echo "__occa_compgen | flags      : [${flags[@]}]"

    if [ "${#inputs[@]}" -eq 0 ]; then
        COMPREPLY=("${command[${#command[@]}-1]}")
        compIsDone=true
        return
    fi

    local allCompgenOptions=()
    local compgenOptions=()

    for expansion in "${expansions[@]}"; do
        case "${expansion}" in
            args)
                __occa_compgen_with_args;;
            func)
                __occa_compgen_function;;
            same)
                unset 'command[${#command[@]}-1]'
                __occa_run_command;;
            file)
                __occa_compgen_file;;
            *) ;;
        esac
        # We found something
        if [ "${#compgenOptions[@]}" -eq 0 ]; then
            allCompgenOptions=()
            break
        fi
        __occa_debug_echo "__occa_compgen | expansion      : [${expansion}]"
        __occa_debug_echo "__occa_compgen | compgenOptions : [${compgenOptions[@]}]"
        allCompgenOptions+=("${compgenOptions[@]}")
        compgenOptions=()
    done
    if [ "${#allCompgenOptions[@]}" -gt 0 ]; then
        __occa_autocomplete "${allCompgenOptions[@]}"
    elif [ "${compIsDone}" = false ]; then
        __occa_compgen_file
    fi
}
#=======================================
