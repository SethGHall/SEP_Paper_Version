#!/bin/bash

COMMAND=${1}

PROJECT_DIR=${PWD}
BUILD_DIR=${PROJECT_DIR}/build/imaging
CONFIG_DIR=${PROJECT_DIR}/imaging_configs
DEFAULT_CONF=${CONFIG_DIR}/default.yaml
CUSTOM_CONF=${CONFIG_DIR}/custom.yaml
GAUSSIAN_CONF=${CONFIG_DIR}/gaussian.yaml
MID_CONF=${CONFIG_DIR}/full_array_nominal.yaml
GLEAM_SMALL_CONF=${CONFIG_DIR}/gleam_small.yaml
GLEAM_MEDIUM_CONF=${CONFIG_DIR}/gleam_medium.yaml
GLEAM_LARGE_CONF=${CONFIG_DIR}/gleam_large.yaml
VLA_CONF=${CONFIG_DIR}/vla.yaml
# DOCS_BUILD_DIR=${PROJECT_DIR}/build/imaging/docs/

case ${COMMAND} in

    build_and_run)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${DEFAULT_CONF} ${CUSTOM_CONF}
        cd ${PROJECT_DIR}
    ;;

    build_and_run_gaussian)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${GAUSSIAN_CONF} ${GAUSSIAN_CONF}
        cd ${PROJECT_DIR}
    ;;
	
	build_and_run_mid)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${MID_CONF} ${MID_CONF}
        cd ${PROJECT_DIR}
    ;;
	
	build_and_run_gleam_small)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${GLEAM_SMALL_CONF} ${GLEAM_SMALL_CONF}
        cd ${PROJECT_DIR}
    ;;
	
	build_and_run_gleam_medium)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${GLEAM_MEDIUM_CONF} ${GLEAM_MEDIUM_CONF}
        cd ${PROJECT_DIR}
    ;;
	
	build_and_run_gleam_large)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${GLEAM_LARGE_CONF} ${GLEAM_LARGE_CONF}
        cd ${PROJECT_DIR}
    ;;
	

    build_and_run_vla3)
        cd ${PROJECT_DIR} 
        rm -rf build
        mkdir build
        cd build && cmake ..
        make -j8
        imaging/./imaging ${VLA_CONF} ${VLA_CONF}
        cd ${PROJECT_DIR}
    ;;
	


    build)
        make build
    ;;

    run)
        echo ${DEFAULT_CONF}
        cd ${BUILD_DIR}
        ./imaging ${DEFAULT_CONF} ${CUSTOM_CONF}
    ;;

    clean)
        make clean
    ;;

    memcheck)
        cd ${BUILD_DIR}
        cuda-memcheck ./imaging ${DEFAULT_CONF} ${CUSTOM_CONF}
        cd ${PROJECT_DIR}
    ;;

    memcheck_gleam_small)
        cd ${BUILD_DIR}
        cuda-memcheck ./imaging ${GLEAM_SMALL_CONF} ${GLEAM_SMALL_CONF}
        cd ${PROJECT_DIR}
    ;;


    docs)
        make docs
        # sensible-browser 
    ;;

    *)
        echo "ERR: Unrecognized command, please review the script for valid commands..."
    ;;

esac
